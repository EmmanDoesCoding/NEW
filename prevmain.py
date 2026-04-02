"""
main.py
=======
Sign Language Avatar — full pipeline

Architecture:
    User types text
        → NLPProcessor  (English → ASL gloss)
        → SignMapper     (ASL gloss → JSON frames, background preload)
        → AnimationQueue (smooth sign-to-sign blending)
        → Avatar._pose() (Panda3D render at 170fps)

Real-time streaming:
    While user types, process_partial() fires on every keystroke
    and preloads sign files in background so they're ready when
    the user hits Enter.

Controls:
    Type in the text box at the bottom → Enter to sign
    SPACE   pause / resume animation
    ESC/Q   quit
    Mouse drag  orbit camera
    Scroll      zoom
"""

import sys
import os
import threading
import queue

# ── Make sure we can import from same folder ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from direct.task import Task
from direct.gui.DirectGui import DirectEntry, DirectButton, DirectLabel
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, LColor
from direct.showbase.ShowBaseGlobal import globalClock

from nlp_processor  import NLPProcessor
from sign_mapper     import SignMapper
from animation_queue import AnimationQueue

# Import Avatar from AVA_panda3d but prevent it from auto-running
import AVA_panda3d as _ava_mod
Avatar     = _ava_mod.Avatar
lerp_frame = _ava_mod.lerp_frame   # frame blending used in _main_tick


# ── Config ────────────────────────────────────────────────────────────────────
SIGNS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Signs")
ANIM_FPS     = 30       # animation playback rate
BLEND_FRAMES = 8        # frames spent blending between signs (~0.27s at 30fps)


# ── Main App ──────────────────────────────────────────────────────────────────

class SignLanguageApp(Avatar):
    """
    Extends Avatar with:
      - Text input box
      - NLP + sign mapping pipeline
      - AnimationQueue driving _tick instead of the sample data loop
      - Real-time partial processing while typing
    """

    def __init__(self):
        # Initialise the Panda3D avatar (lights, skeleton, camera, etc.)
        super().__init__()

        # ── Pipeline components ───────────────────────────────────────────────
        print("\n[Main] Initialising pipeline...")
        self.nlp    = NLPProcessor()
        self.mapper = SignMapper(SIGNS_FOLDER)
        self.aqueue = AnimationQueue(blend_frames=BLEND_FRAMES)
        print(f"[Main] Pipeline ready — {self.mapper.stats()['indexed_keys']} signs indexed")

        # Calibrate body centre from first available sign file
        self._calibrate_body_centre()

        # ── Background NLP thread ──────────────────────────────────────────────
        # We offload NLP+mapping to a thread so the render loop never stutters
        self._nlp_queue    = queue.Queue()    # text jobs to process
        self._nlp_thread   = threading.Thread(
            target=self._nlp_worker, daemon=True, name="NLPWorker")
        self._nlp_thread.start()

        # ── State ──────────────────────────────────────────────────────────────
        self._last_partial_gloss = []    # last gloss from partial processing
        self._signed_words       = []    # words signed so far this sentence
        self._input_active       = False # True when text box has focus

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_ui()

        # ── Override the tick task ─────────────────────────────────────────────
        # Remove the base Avatar tick and replace with our queue-driven one
        self.taskMgr.remove("tick")
        self.taskMgr.add(self._main_tick, "tick")

        # ── Override keyboard shortcuts so they don't conflict with text input ─
        self.accept("space",  self._toggle_pause_safe)
        self.accept("escape", self._handle_escape)
        self.accept("q",      self._handle_q)

        print("[Main] Ready — type a sentence and press Enter to sign it.\n")

    # ── Calibration ───────────────────────────────────────────────────────────

    def _calibrate_body_centre(self):
        """
        Calibrate BCX/BCZ body centre from first available sign file.
        This ensures the avatar is centred for any sign data.
        """
        import AVA_panda3d as _ava
        signs = self.mapper.available_signs()
        if not signs:
            print("[Main] No signs for calibration — using defaults (0.5)")
            return
        frames = self.mapper.get_sign(signs[0])
        if frames:
            _ava.calibrate_from_frames(frames)
            print(f"[Main] Calibrated from '{signs[0]}'")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Build the text input bar at the bottom of the screen."""

        # Input box
        self._entry = DirectEntry(
            text        = "",
            scale       = 0.048,
            pos         = (-1.0, 0, -0.90),
            width       = 36,
            numLines    = 1,
            focus       = 0,
            focusInCommand  = self._on_focus_in,
            focusOutCommand = self._on_focus_out,
            command         = self._on_enter,       # fires on Enter
            extraArgs       = [],
            frameColor      = (0.08, 0.08, 0.10, 0.92),
            text_fg         = (0.95, 0.95, 0.90, 1),
            rolloverSound   = None,
            clickSound      = None,
        )

        # "Sign it" button
        self._btn = DirectButton(
            text        = "Sign ▶",
            scale       = 0.048,
            pos         = (1.05, 0, -0.90),
            command     = self._submit_from_button,
            frameColor  = (0.15, 0.35, 0.55, 1),
            text_fg     = (1, 1, 1, 1),
            relief      = 1,
            rolloverSound = None,
            clickSound    = None,
        )

        # Prompt label
        self._prompt = OnscreenText(
            text  = "Type a sentence:",
            pos   = (-1.28, -0.83),
            scale = 0.040,
            fg    = (0.65, 0.65, 0.65, 1),
            align = TextNode.ALeft,
        )

        # Gloss display — shows the ASL word sequence being signed
        self._gloss_label = OnscreenText(
            text      = "",
            pos       = (0, 0.82),
            scale     = 0.050,
            fg        = (0.85, 0.85, 0.55, 1),
            align     = TextNode.ACenter,
            mayChange = True,
        )

        # Status line — shows current word being signed
        self._status_label = OnscreenText(
            text      = "",
            pos       = (0, -0.75),
            scale     = 0.042,
            fg        = (0.55, 0.85, 0.65, 1),
            align     = TextNode.ACenter,
            mayChange = True,
        )

        # Click the entry to focus it initially
        self._entry["focus"] = 1

    # ── Input handling ────────────────────────────────────────────────────────

    def _on_focus_in(self):
        self._input_active = True
        # Suspend SPACE/Q shortcuts while typing
        self.ignore("space")
        self.ignore("q")

    def _on_focus_out(self):
        self._input_active = False
        # Restore shortcuts when not typing
        self.accept("space", self._toggle_pause_safe)
        self.accept("q",     self._handle_q)

    def _on_entry_changed(self, text: str):
        """
        Called on every keystroke via a task that polls the entry.
        Triggers partial NLP + preloads signs in background.
        """
        if not text.strip():
            return
        # Submit partial processing job (non-blocking)
        self._nlp_queue.put(("partial", text))

    def _on_enter(self, text: str):
        """Called when user presses Enter in the text box."""
        text = text.strip()
        if not text:
            return
        self._submit(text)
        self._entry.set("")   # clear input box

    def _submit_from_button(self):
        """Called when user clicks the Sign button."""
        text = self._entry.get().strip()
        if text:
            self._submit(text)
            self._entry.set("")

    def _submit(self, text: str):
        """Submit a complete sentence to be signed."""
        print(f"\n[Main] Input: {text!r}")
        self._nlp_queue.put(("full", text))

    # ── NLP worker thread ─────────────────────────────────────────────────────

    def _nlp_worker(self):
        """
        Background thread — processes NLP jobs from the queue.
        Never blocks the render loop.
        """
        while True:
            try:
                job_type, text = self._nlp_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if job_type == "partial":
                self._process_partial(text)
            elif job_type == "full":
                self._process_full(text)

    def _process_partial(self, text: str):
        """
        Process incomplete sentence — preload signs we'll likely need.
        Called on every keystroke so must be very fast.
        """
        gloss = self.nlp.process_partial(text)
        if gloss == self._last_partial_gloss:
            return   # nothing changed
        self._last_partial_gloss = gloss

        # Preload signs we haven't loaded yet
        new_words = [w for w in gloss if not self.mapper.has_sign(w)]
        if new_words:
            self.mapper.preload(new_words)

    def _process_full(self, text: str):
        """
        Process complete sentence — build sign sequence and send to queue.
        """
        # NLP: English → ASL gloss
        gloss = self.nlp.process(text)
        if not gloss:
            print("[Main] No ASL gloss produced")
            return

        print(f"[Main] Gloss: {' '.join(gloss)}")

        # Map gloss words to frame data
        sign_pairs = self.mapper.get_signs_for_gloss(gloss)
        if not sign_pairs:
            print("[Main] No sign files found for any gloss word")
            return

        found_words = [w for w, _ in sign_pairs]
        print(f"[Main] Signing: {' → '.join(found_words)}")

        # Update gloss display
        gloss_str = "  ".join(
            f"[{w}]" if w in found_words else f"({w})"
            for w in gloss
        )
        # (UI updates must happen on main thread — we'll do it via a flag)
        self._pending_gloss_text = gloss_str
        self._signed_words       = []

        # Interrupt current animation and start new sentence
        self.aqueue.interrupt(sign_pairs)

    # ── Main tick (overrides Avatar._tick) ───────────────────────────────────

    def _main_tick(self, task):
        if self.paused:
            return Task.cont

        # Poll text entry for partial processing (every tick)
        current_text = self._entry.get()
        if current_text:
            self._nlp_queue.put(("partial", current_text))

        # Advance animation accumulator
        self._anim_accum += globalClock.getDt()
        INTERVAL = 1.0 / ANIM_FPS

        if self._anim_accum >= INTERVAL:
            self._anim_accum -= INTERVAL

            # Get next frame from animation queue
            frame = self.aqueue.get_current_frame()

            self._prev_frame = self._curr_frame
            self._curr_frame = frame

            # Track finished words + reset stale hand data on word change
            done = self.aqueue.finished_words()
            if done:
                self._signed_words.extend(done)
                # Reset last-good-hand so new sign's hand shape shows immediately
                # without inheriting the previous sign's hand pose
                self._last_lh = []
                self._last_rh = []

        # Sub-frame interpolation for smooth motion
        t     = min(1.0, self._anim_accum / max(1e-6, INTERVAL))
        frame = lerp_frame(self._prev_frame, self._curr_frame, t)

        pose = frame.get("pose", [])
        lh   = frame.get("left_hand",  [])
        rh   = frame.get("right_hand", [])

        # Filter bad z=0 fallback hand frames
        # Smart hand filter:
        # z=0 means pose-synthesised fallback with no real depth.
        # BUT some signs (e.g. About) have z=0 for ALL frames — still valid x,y.
        # So only fall back to _last hand if z=0 AND the sign has real-z frames.
        # We detect "all-z-zero sign" by checking if x,y are actually moving.
        if rh and abs(rh[0]["z"]) < 0.01:
            # Check if x or y has meaningful values (real hand, just no depth)
            has_xy = any(abs(lm["x"] - 0.5) > 0.02 or abs(lm["y"] - 0.5) > 0.02
                        for lm in rh[:3])
            if not has_xy:
                rh = self._last_rh   # truly bad frame — use last good
            # else: z=0 but x,y are real — keep rh as-is
        if lh and abs(lh[0]["z"]) < 0.01:
            has_xy = any(abs(lm["x"] - 0.5) > 0.02 or abs(lm["y"] - 0.5) > 0.02
                        for lm in lh[:3])
            if not has_xy:
                lh = self._last_lh
        if rh: self._last_rh = rh
        if lh: self._last_lh = lh

        if pose:
            self._pose(pose, lh, rh)

        # Update HUD
        fps      = globalClock.getAverageFrameRate()
        word_now = self.aqueue.current_word() or ("idle" if self.aqueue.is_idle() else "…")
        q_len    = self.aqueue.queue_length()

        self.hud.setText(
            f"render {fps:.0f}fps  |  signing: {word_now}  |  queue: {q_len}\n"
            "SPACE=pause  ESC=quit  drag=rotate  scroll=zoom"
        )

        # Update gloss label if new sentence started
        if hasattr(self, "_pending_gloss_text"):
            self._gloss_label.setText(self._pending_gloss_text)
            del self._pending_gloss_text

        # Update status label — words signed so far
        if self._signed_words:
            self._status_label.setText("✓ " + "  ".join(self._signed_words))
        elif self.aqueue.is_idle():
            self._status_label.setText("Ready — type a sentence below")

        return Task.cont

    # ── Keyboard overrides ────────────────────────────────────────────────────

    def _toggle_pause_safe(self):
        """Pause/resume — only when not typing."""
        if not self._input_active:
            self.paused = not self.paused
            print("Paused" if self.paused else "Resumed")

    def _handle_escape(self):
        if self._input_active:
            # Deselect input box
            self._entry["focus"] = 0
        else:
            sys.exit()

    def _handle_q(self):
        if not self._input_active:
            sys.exit()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Sign Language Avatar")
    print("  Type a sentence → Enter to sign it")
    print("=" * 60)

    # Check Signs folder exists
    if not os.path.exists(SIGNS_FOLDER):
        print(f"\n[WARNING] Signs folder not found: {SIGNS_FOLDER}")
        print("  Create a 'Signs' folder next to main.py and add your JSON files.")
        print("  The avatar will still run but won't be able to sign anything.\n")

    app = SignLanguageApp()
    app.run()


if __name__ == "__main__":
    main()