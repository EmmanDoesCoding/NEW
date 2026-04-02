"""
animation_queue.py
==================
Drives the avatar through a sequence of signs with smooth transitions.

Features:
  - Smooth ease-in-out blend between every sign transition
  - Blend to/from REST pose when queue empties or starts
  - Interrupt: mid-sentence new input cancels current queue gracefully
  - Sub-frame interpolation so render fps and anim fps are decoupled
  - Thread-safe — add_sign() safe to call from NLP/background thread
  - finished_words() callback to track what has been signed

Call pattern (from AVA_panda3d.py):
    queue = AnimationQueue()
    queue.add_sign("HELLO", frames)
    queue.add_sign("YOU",   frames)

    # every animation tick (30fps):
    frame = queue.get_current_frame()
    avatar._pose(frame["pose"], frame["left_hand"], frame["right_hand"])
"""

import threading
from collections import deque
from copy import deepcopy


# ── Maths helpers ──────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _ease(t: float) -> float:
    """Cubic ease-in-out: slow start, fast middle, slow end."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def _lerp_lm(a: dict, b: dict, t: float) -> dict:
    return {
        "x": _lerp(a["x"], b["x"], t),
        "y": _lerp(a["y"], b["y"], t),
        "z": _lerp(a["z"], b["z"], t),
    }

def _lerp_frame(fa: dict, fb: dict, t: float) -> dict:
    """
    Blend two full frames.  t=0 → fa,  t=1 → fb.
    If one side has no hand data, holds the other side's pose.
    """
    out = {}
    for key in ("pose", "left_hand", "right_hand", "face"):
        la = fa.get(key) or []
        lb = fb.get(key) or []
        if la and lb and len(la) == len(lb):
            out[key] = [_lerp_lm(a, b, t) for a, b in zip(la, lb)]
        else:
            out[key] = la or lb   # hold whichever exists
    return out


# ── Rest pose ──────────────────────────────────────────────────────────────────

def _build_rest_pose() -> dict:
    """
    Neutral standing pose — arms at sides, hands relaxed.
    Coordinates are MediaPipe-normalised (same scale as your JSON files).
    """
    def lm(x, y, z): return {"x": x, "y": y, "z": z}

    # Build 33-landmark pose — start with everything at body centre
    pose = [lm(0.50, 0.50, -0.50)] * 33

    # Override key joints with natural standing values
    pose[0]  = lm(0.50, 0.33, -0.58)   # nose
    pose[7]  = lm(0.55, 0.32, -0.30)   # left ear
    pose[8]  = lm(0.47, 0.32, -0.30)   # right ear
    pose[11] = lm(0.61, 0.53, -0.13)   # left shoulder
    pose[12] = lm(0.41, 0.53, -0.12)   # right shoulder
    # Arms hanging naturally straight down at sides
    pose[13] = lm(0.65, 0.75, -0.18)   # left elbow  — close to body, mid height
    pose[14] = lm(0.36, 0.75, -0.18)   # right elbow — close to body, mid height
    pose[15] = lm(0.64, 0.90, -0.22)   # left wrist  — straight down from elbow
    pose[16] = lm(0.37, 0.90, -0.22)   # right wrist — straight down from elbow
    pose[23] = lm(0.56, 0.99,  0.00)   # left hip
    pose[24] = lm(0.44, 0.99,  0.00)   # right hip

    def fist_hand(wx, wy, wz, flip=False):
        """
        21-landmark closed fist pointing downward.
        Fingers curl downward from wrist — knuckles above, tips below.
        All X offsets are small (fist width), Y offsets go downward (+Y).
        """
        hand = [lm(wx, wy, wz)]   # [0] wrist
        # Finger columns: small X spread, curling downward in Y
        # (dx=horizontal spread, dy=downward extension)
        # Thumb tucked to side
        thumb  = [(0.012, 0.008), (0.018, 0.016), (0.016, 0.022), (0.012, 0.026)]
        # Fingers: knuckle close to wrist, tip curls back up (closed fist)
        index  = [(0.004, 0.018), (0.004, 0.030), (0.004, 0.024), (0.004, 0.018)]
        middle = [(0.000, 0.020), (0.000, 0.032), (0.000, 0.026), (0.000, 0.020)]
        ring   = [(-0.004, 0.018), (-0.004, 0.030), (-0.004, 0.024), (-0.004, 0.018)]
        pinky  = [(-0.008, 0.014), (-0.008, 0.024), (-0.008, 0.019), (-0.008, 0.014)]
        for finger in [thumb, index, middle, ring, pinky]:
            for (dx, dy) in finger:
                fx = (wx - dx) if flip else (wx + dx)
                hand.append(lm(fx, wy + dy, wz))
        return hand

    rh = fist_hand(0.37, 0.90, -0.22, flip=False)   # right wrist
    lh = fist_hand(0.64, 0.90, -0.22, flip=True)    # left wrist

    return {
        "pose":       pose,
        "left_hand":  lh,
        "right_hand": rh,
        "face":       [],
    }

REST_POSE = _build_rest_pose()


# ── Sign item ──────────────────────────────────────────────────────────────────

class _Sign:
    """One sign in the queue."""
    __slots__ = ("word", "frames", "fi")

    def __init__(self, word: str, frames: list):
        self.word   = word
        self.frames = frames
        self.fi     = 0           # current frame index

    @property
    def total(self) -> int:
        return len(self.frames)

    @property
    def finished(self) -> bool:
        return self.fi >= self.total

    @property
    def current_frame(self) -> dict:
        idx = min(self.fi, self.total - 1)
        return self.frames[idx]

    @property
    def last_frame(self) -> dict:
        return self.frames[-1] if self.frames else REST_POSE

    @property
    def first_frame(self) -> dict:
        return self.frames[0] if self.frames else REST_POSE

    def advance(self):
        self.fi += 1


# ── AnimationQueue ─────────────────────────────────────────────────────────────

class AnimationQueue:
    """
    State machine that drives the avatar through a sentence of signs.

    States:
        REST        — idle, showing REST_POSE
        BLEND_IN    — blending from rest/last-sign into first frame of new sign
        PLAYING     — playing through sign frames normally
        BLEND_OUT   — blending last frames of current sign into first of next
        BLEND_REST  — blending back to REST_POSE after queue empties
    """

    # blend_frames: frames spent on each transition (at 30fps → 8 frames = 0.27s)
    def __init__(self, blend_frames: int = 8):
        self.blend_frames = max(1, blend_frames)
        self._lock        = threading.Lock()
        self._reset()

    def _reset(self):
        """Full state reset — called on init and clear()."""
        self._queue:    deque   = deque()        # pending _Sign objects
        self._current:  _Sign | None = None      # sign currently playing
        self._state:    str     = "REST"
        self._blend_t:  float   = 0.0            # 0.0 → 1.0 blend progress
        self._blend_fa: dict    = REST_POSE       # blend source frame
        self._blend_fb: dict    = REST_POSE       # blend target frame
        self._last_frame: dict  = REST_POSE       # most recent output frame
        self._done_words: list  = []             # words finished since last poll

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_sign(self, word: str, frames: list):
        """
        Append a sign to the queue.
        Safe to call from any thread — including the NLP background thread.
        """
        if not frames:
            return
        with self._lock:
            self._queue.append(_Sign(word, frames))

    def clear(self, smooth: bool = True):
        """
        Clear the queue.
        smooth=True  → finish current sign then blend to rest (natural)
        smooth=False → stop immediately and snap to rest
        """
        with self._lock:
            self._queue.clear()
            if not smooth or self._state == "REST":
                self._reset()
            # If smooth, let the current sign finish then the queue empty path
            # will trigger BLEND_REST naturally

    def interrupt(self, new_signs: list[tuple]):
        """
        Stop everything immediately and start a new sentence.
        new_signs: [(word, frames), (word, frames), ...]
        Use when user submits a completely new sentence mid-animation.
        """
        with self._lock:
            # Blend from current pose to new sentence
            fa = self._last_frame
            self._reset()
            for word, frames in new_signs:
                if frames:
                    self._queue.append(_Sign(word, frames))
            # Start with a short blend from where we are now
            if self._queue:
                self._state    = "BLEND_IN"
                self._blend_fa = fa
                self._blend_fb = self._queue[0].first_frame
                self._blend_t  = 0.0

    def get_current_frame(self) -> dict:
        """
        Advance animation by one step and return the blended frame.
        Call this exactly once per animation tick (30fps).
        """
        with self._lock:
            frame = self._step()
            self._last_frame = frame
            return frame

    def is_idle(self) -> bool:
        """True when nothing is playing and queue is empty."""
        with self._lock:
            return self._state == "REST" and not self._queue

    def queue_length(self) -> int:
        with self._lock:
            return len(self._queue)

    def current_word(self) -> str | None:
        with self._lock:
            return self._current.word if self._current else None

    def finished_words(self) -> list[str]:
        """
        Drain and return list of words that finished playing since last call.
        Use this to update a subtitle display or log progress.
        """
        with self._lock:
            done = self._done_words[:]
            self._done_words.clear()
            return done

    # ── State machine ──────────────────────────────────────────────────────────

    def _step(self) -> dict:
        """Advance one animation frame through the state machine."""

        if self._state == "REST":
            return self._step_rest()

        elif self._state == "BLEND_IN":
            return self._step_blend(next_state="PLAYING")

        elif self._state == "PLAYING":
            return self._step_playing()

        elif self._state == "BLEND_OUT":
            return self._step_blend(next_state="PLAYING")

        elif self._state == "BLEND_REST":
            return self._step_blend(next_state="REST")

        return REST_POSE

    # ── REST ───────────────────────────────────────────────────────────────────

    def _step_rest(self) -> dict:
        if not self._queue:
            return REST_POSE

        # New sign arrived — blend into it
        self._current  = self._queue.popleft()
        self._state    = "BLEND_IN"
        self._blend_fa = REST_POSE
        self._blend_fb = self._current.first_frame
        self._blend_t  = 0.0
        return REST_POSE

    # ── Generic blend ──────────────────────────────────────────────────────────

    def _step_blend(self, next_state: str) -> dict:
        """Advance one step of a blend transition."""
        step = 1.0 / self.blend_frames
        self._blend_t = min(1.0, self._blend_t + step)
        t = _ease(self._blend_t)
        frame = _lerp_frame(self._blend_fa, self._blend_fb, t)

        if self._blend_t >= 1.0:
            # Blend finished — move to next state
            if next_state == "PLAYING":
                self._state = "PLAYING"
            elif next_state == "REST":
                self._state   = "REST"
                self._current = None

        return frame

    # ── PLAYING ────────────────────────────────────────────────────────────────

    def _step_playing(self) -> dict:
        if self._current is None or self._current.total == 0:
            self._state = "REST"
            return REST_POSE

        frame = self._current.current_frame
        self._current.advance()

        frames_left   = self._current.total - self._current.fi
        has_next      = bool(self._queue)
        near_end      = frames_left <= self.blend_frames

        if self._current.finished:
            # Sign fully played
            self._done_words.append(self._current.word)

            if has_next:
                # Immediately blend into next sign
                self._current  = self._queue.popleft()
                self._state    = "BLEND_IN"
                self._blend_fa = frame
                self._blend_fb = self._current.first_frame
                self._blend_t  = 0.0
            else:
                # Queue empty — blend back to rest
                self._state    = "BLEND_REST"
                self._blend_fa = frame
                self._blend_fb = REST_POSE
                self._blend_t  = 0.0
                self._current  = None

        elif near_end and has_next:
            # Approaching end — start blending into next sign early
            next_sign      = self._queue.popleft()
            self._state    = "BLEND_OUT"
            self._blend_fa = frame
            self._blend_fb = next_sign.first_frame
            self._blend_t  = 0.0
            # Keep current sign advancing in background during blend
            # When blend finishes, switch to next_sign
            self._current  = next_sign

        return frame


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    def dummy_sign(n: int, x_start: float = 0.5) -> list:
        """Make a dummy sign where pose[0].x moves from x_start to x_start+0.1."""
        def lm(x, y, z): return {"x": x, "y": y, "z": z}
        frames = []
        for i in range(n):
            t = i / max(1, n - 1)
            p = [lm(x_start + t * 0.1, 0.33, -0.5)] + [lm(0.5, 0.5, -0.5)] * 32
            frames.append({
                "pose":       p,
                "left_hand":  [lm(0.6, 0.88, -0.7)] * 21,
                "right_hand": [lm(0.4, 0.88, -0.7)] * 21,
                "face":       [],
            })
        return frames

    q = AnimationQueue(blend_frames=6)
    q.add_sign("HELLO", dummy_sign(20, 0.50))
    q.add_sign("YOU",   dummy_sign(15, 0.55))
    q.add_sign("EAT",   dummy_sign(18, 0.45))

    print("AnimationQueue state machine test")
    print("=" * 50)
    prev_state = ""
    for i in range(100):
        frame = q.get_current_frame()
        state = q._state
        word  = q.current_word()
        done  = q.finished_words()
        qlen  = q.queue_length()
        nose_x = frame["pose"][0]["x"]

        if state != prev_state or done:
            label = f"→ {state}" if state != prev_state else ""
            done_str = f"  DONE:{done}" if done else ""
            print(f"  f{i:03d}  state={state:12s}  word={str(word):8s}  "
                  f"q={qlen}  nose_x={nose_x:.3f}{done_str}  {label}")
            prev_state = state

    print(f"\nIdle: {q.is_idle()}")
    print("\n--- Interrupt test ---")
    q2 = AnimationQueue(blend_frames=6)
    q2.add_sign("HELLO", dummy_sign(30))
    for i in range(10):   # play 10 frames
        q2.get_current_frame()
    print(f"After 10 frames, state={q2._state}, word={q2.current_word()}")
    q2.interrupt([("GOODBYE", dummy_sign(15, 0.60))])
    print(f"After interrupt,  state={q2._state}, word={q2.current_word()}")
    for i in range(30):
        q2.get_current_frame()
    print(f"After 30 more,    state={q2._state}, idle={q2.is_idle()}")