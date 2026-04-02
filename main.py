"""
main.py
=======
Sign Language Avatar  —  AVA
Starts with a main menu, choose Translator or Chatbot mode.

Free API: Google Gemini (no credit card needed)
Get your key at: https://aistudio.google.com/apikey

Controls (inside the app):
    Mouse drag   orbit camera
    Scroll       zoom
    ESC          back to menu / quit
    SPACE        pause / resume
"""

import sys, os, threading, queue, urllib.request, urllib.error, json as _json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from direct.task                         import Task
from direct.gui.DirectGui                import (DirectEntry, DirectButton,
                                                  DirectFrame, DirectLabel)
from direct.gui.OnscreenText             import OnscreenText
from direct.showbase.ShowBaseGlobal      import globalClock
from panda3d.core                        import TextNode, LColor, TransparencyAttrib

from nlp_processor   import NLPProcessor
from sign_mapper      import SignMapper
from animation_queue  import AnimationQueue

import AVA_panda3d as _ava_mod
Avatar                = _ava_mod.Avatar
lerp_frame            = _ava_mod.lerp_frame
calibrate_from_frames = _ava_mod.calibrate_from_frames

# ── Config ────────────────────────────────────────────────────────────────────
SIGNS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Signs")
ANIM_FPS     = 30
BLEND_FRAMES = 8

# Google Gemini — free, no credit card
# Get key: https://aistudio.google.com/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_URL     = ("https://generativelanguage.googleapis.com/v1beta/models/"
                  "gemini-2.0-flash:generateContent?key=")

CHATBOT_PROMPT = """You are AVA, a friendly sign language avatar assistant.
Answer in EXACTLY 1 to 2 short sentences.
Rules:
- Use only simple common words
- Maximum 10 words per sentence
- No contractions (say "do not" not "don't")
- Be warm and friendly"""

# ── Colour palette ────────────────────────────────────────────────────────────
# All colours as (R, G, B, A) 0-1 floats
C_BG         = (0.08, 0.09, 0.11, 1)      # near-black background
C_CARD       = (0.13, 0.14, 0.18, 1)      # card / panel
C_ACCENT1    = (0.20, 0.55, 0.90, 1)      # blue  — translator
C_ACCENT2    = (0.55, 0.25, 0.85, 1)      # purple — chatbot
C_ACCENT1_DIM= (0.10, 0.28, 0.45, 1)
C_ACCENT2_DIM= (0.28, 0.12, 0.42, 1)
C_TEXT       = (0.92, 0.93, 0.95, 1)
C_TEXT_DIM   = (0.55, 0.57, 0.62, 1)
C_SUCCESS    = (0.25, 0.82, 0.55, 1)
C_WARN       = (0.95, 0.75, 0.25, 1)
C_INPUT_BG   = (0.10, 0.11, 0.14, 1)


# ── Gemini API ────────────────────────────────────────────────────────────────

def call_gemini(question: str) -> str:
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        return "I do not have an API key. Please add your Gemini API key."
    payload = _json.dumps({
        "system_instruction": {"parts": [{"text": CHATBOT_PROMPT}]},
        "contents":           [{"parts": [{"text": question}]}],
        "generationConfig":   {"maxOutputTokens": 80, "temperature": 0.7}
    }).encode()
    req = urllib.request.Request(
        GEMINI_URL + GEMINI_API_KEY,
        data=payload, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = _json.loads(r.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return ""


# ── App ───────────────────────────────────────────────────────────────────────

class AVAApp(Avatar):

    SCREEN_MENU       = "menu"
    SCREEN_TRANSLATOR = "translator"
    SCREEN_CHATBOT    = "chatbot"

    def __init__(self):
        super().__init__()

        # pipeline
        print("\n[AVA] Loading pipeline...")
        self.nlp    = NLPProcessor()
        self.mapper = SignMapper(SIGNS_FOLDER)
        self.aqueue = AnimationQueue(blend_frames=BLEND_FRAMES)
        print(f"[AVA] {self.mapper.stats()['indexed_keys']} signs indexed")
        self._calibrate()

        # worker thread
        self._q      = queue.Queue()
        self._worker = threading.Thread(target=self._work_loop,
                                        daemon=True, name="Worker")
        self._worker.start()

        # state
        self._screen       = self.SCREEN_MENU
        self._input_active = False
        self._last_partial = []
        self._signed_words = []

        # pending UI updates (set from worker thread, applied on main thread)
        self._pend_gloss   = None
        self._pend_status  = None
        self._pend_ai      = None
        self._pend_think   = None

        # widget groups (destroyed when switching screens)
        self._ui_nodes = []

        self._build_menu()

        self.taskMgr.remove("tick")
        self.taskMgr.add(self._tick, "tick")
        self.accept("escape", self._on_escape)
        self.accept("space",  self._on_space)
        self.accept("q",      self._on_q)

    # ── Calibration ───────────────────────────────────────────────────────────
    def _calibrate(self):
        signs = self.mapper.available_signs()
        if signs:
            frames = self.mapper.get_sign(signs[0])
            if frames:
                calibrate_from_frames(frames)

    # ── UI helpers ────────────────────────────────────────────────────────────
    def _clear_ui(self):
        for node in self._ui_nodes:
            try: node.destroy()
            except: pass
        self._ui_nodes = []

    def _track(self, widget):
        """Register a widget so it gets destroyed on screen switch."""
        self._ui_nodes.append(widget)
        return widget

    def _label(self, text, pos, scale, color=C_TEXT, align=TextNode.ACenter,
               may_change=False):
        w = OnscreenText(text=text, pos=pos, scale=scale,
                         fg=color, align=align, mayChange=may_change)
        return self._track(w)

    def _btn(self, text, pos, scale, color, text_color, cmd):
        w = DirectButton(
            text=text, scale=scale, pos=pos, command=cmd,
            frameColor=color, text_fg=text_color,
            relief=1, rolloverSound=None, clickSound=None,
            text_scale=1.0, pad=(0.25, 0.15))
        return self._track(w)

    # ── MENU SCREEN ───────────────────────────────────────────────────────────
    def _build_menu(self):
        self._clear_ui()
        self._screen = self.SCREEN_MENU

        # Title
        self._label("AVA", (0, 0.70), 0.18, C_TEXT)
        self._label("Sign Language Avatar", (0, 0.55), 0.058, C_TEXT_DIM)

        # Divider line (thin rectangle)
        line = DirectFrame(
            frameSize=(-0.55, 0.55, -0.003, 0.003),
            pos=(0, 0, 0.46),
            frameColor=(0.25, 0.27, 0.32, 1))
        self._track(line)

        # Subtitle
        self._label("Choose a mode to get started",
                    (0, 0.38), 0.045, C_TEXT_DIM)

        # ── Translator card ───────────────────────────────────────────────────
        card1 = DirectFrame(
            frameSize=(-0.52, 0.52, -0.22, 0.22),
            pos=(-0.60, 0, 0.05),
            frameColor=C_CARD)
        self._track(card1)

        self._label("Translator", (-0.60, 0.24), 0.070, C_ACCENT1)
        self._label("Type any sentence and the\navatar will sign it for you.",
                    (-0.60, 0.08), 0.040, C_TEXT_DIM)

        self._btn("Start Translating  ▶",
                  (-0.60, 0, -0.12), 0.052,
                  C_ACCENT1, (1,1,1,1),
                  self._open_translator)

        # ── Chatbot card ──────────────────────────────────────────────────────
        card2 = DirectFrame(
            frameSize=(-0.52, 0.52, -0.22, 0.22),
            pos=(0.60, 0, 0.05),
            frameColor=C_CARD)
        self._track(card2)

        self._label("Chatbot", (0.60, 0.24), 0.070, C_ACCENT2)
        self._label("Ask AVA any question and she\nwill answer it through sign.",
                    (0.60, 0.08), 0.040, C_TEXT_DIM)

        self._btn("Ask AVA  ▶",
                  (0.60, 0, -0.12), 0.052,
                  C_ACCENT2, (1,1,1,1),
                  self._open_chatbot)

        # ── Bottom info ───────────────────────────────────────────────────────
        self._label("Powered by Google Gemini  •  Free API",
                    (0, -0.52), 0.036, C_TEXT_DIM)
        self._label("Press ESC to return here at any time",
                    (0, -0.60), 0.034, C_TEXT_DIM)

        # Update HUD
        self.hud.setText("AVA  —  Main Menu\nClick a mode to begin")

    # ── TRANSLATOR SCREEN ──────────────────────────────────────────────────────
    def _open_translator(self):
        self._clear_ui()
        self._screen       = self.SCREEN_TRANSLATOR
        self._signed_words = []

        # Header bar
        self._label("TRANSLATOR MODE", (0, 0.88), 0.050, C_ACCENT1)
        back = self._btn("← Menu", (-1.20, 0, 0.88), 0.042,
                         C_CARD, C_TEXT_DIM, self._build_menu)

        # Divider
        div = DirectFrame(frameSize=(-1.33, 1.33, -0.002, 0.002),
                          pos=(0, 0, 0.80), frameColor=(0.22,0.24,0.30,1))
        self._track(div)

        # Gloss display
        self._gloss_lbl = self._label("", (0, 0.70), 0.042,
                                      C_TEXT, may_change=True)

        # Status / signed words
        self._status_lbl = self._label(
            "Type a sentence below and press Enter",
            (0, -0.68), 0.040, C_SUCCESS, may_change=True)

        # Input panel
        panel = DirectFrame(
            frameSize=(-1.33, 1.33, -0.18, 0.12),
            pos=(0, 0, -0.84),
            frameColor=C_CARD)
        self._track(panel)

        self._label("Your sentence:", (-1.25, -0.74), 0.036,
                    C_TEXT_DIM, align=TextNode.ALeft)

        self._entry = DirectEntry(
            text="", scale=0.044,
            pos=(-1.10, 0, -0.90), width=42, numLines=1, focus=0,
            focusInCommand=self._focus_in,
            focusOutCommand=self._focus_out,
            command=self._on_enter, extraArgs=[],
            frameColor=C_INPUT_BG, text_fg=C_TEXT,
            rolloverSound=None, clickSound=None)
        self._track(self._entry)

        sign_btn = self._btn("Sign ▶", (1.15, 0, -0.90), 0.050,
                             C_ACCENT1, (1,1,1,1),
                             self._submit_btn)

        self._entry["focus"] = 1
        print("[AVA] → Translator mode")

    # ── CHATBOT SCREEN ─────────────────────────────────────────────────────────
    def _open_chatbot(self):
        self._clear_ui()
        self._screen       = self.SCREEN_CHATBOT
        self._signed_words = []

        # Header bar
        self._label("CHATBOT MODE", (0, 0.88), 0.050, C_ACCENT2)
        self._btn("← Menu", (-1.20, 0, 0.88), 0.042,
                  C_CARD, C_TEXT_DIM, self._build_menu)

        # Divider
        div = DirectFrame(frameSize=(-1.33, 1.33, -0.002, 0.002),
                          pos=(0, 0, 0.80), frameColor=(0.22,0.24,0.30,1))
        self._track(div)

        # AI response display area
        self._ai_lbl = self._label("", (0, 0.68), 0.042,
                                   (0.70, 0.88, 1.0, 1), may_change=True)

        # Gloss display
        self._gloss_lbl = self._label("", (0, 0.55), 0.038,
                                      C_TEXT_DIM, may_change=True)

        # Thinking / status
        self._think_lbl = self._label("", (0, -0.60), 0.042,
                                      C_WARN, may_change=True)

        self._status_lbl = self._label(
            "Ask AVA any question",
            (0, -0.68), 0.040, C_SUCCESS, may_change=True)

        # Input panel
        panel = DirectFrame(
            frameSize=(-1.33, 1.33, -0.18, 0.12),
            pos=(0, 0, -0.84),
            frameColor=C_CARD)
        self._track(panel)

        self._label("Your question:", (-1.25, -0.74), 0.036,
                    C_TEXT_DIM, align=TextNode.ALeft)

        self._entry = DirectEntry(
            text="", scale=0.044,
            pos=(-1.10, 0, -0.90), width=42, numLines=1, focus=0,
            focusInCommand=self._focus_in,
            focusOutCommand=self._focus_out,
            command=self._on_enter, extraArgs=[],
            frameColor=C_INPUT_BG, text_fg=C_TEXT,
            rolloverSound=None, clickSound=None)
        self._track(self._entry)

        self._btn("Ask ▶", (1.15, 0, -0.90), 0.050,
                  C_ACCENT2, (1,1,1,1), self._submit_btn)

        if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            self._label("⚠  Add your Gemini API key in main.py",
                        (0, 0.78), 0.038, C_WARN)

        self._entry["focus"] = 1
        print("[AVA] → Chatbot mode")

    # ── Input ─────────────────────────────────────────────────────────────────
    def _focus_in(self):
        self._input_active = True
        self.ignore("space"); self.ignore("q")

    def _focus_out(self):
        self._input_active = False
        self.accept("space", self._on_space)
        self.accept("q",     self._on_q)

    def _on_enter(self, text):
        text = text.strip()
        if text: self._submit(text)
        self._entry.set("")

    def _submit_btn(self):
        text = self._entry.get().strip()
        if text: self._submit(text)
        self._entry.set("")

    def _submit(self, text):
        if self._screen == self.SCREEN_TRANSLATOR:
            print(f"[Translator] {text!r}")
            self._q.put(("translate", text))
        elif self._screen == self.SCREEN_CHATBOT:
            print(f"[Chatbot] {text!r}")
            self._pend_think  = "AVA is thinking..."
            self._pend_status = ""
            self._q.put(("chatbot", text))

    # ── Background worker ─────────────────────────────────────────────────────
    def _work_loop(self):
        while True:
            try:
                job, text = self._q.get(timeout=1)
            except queue.Empty:
                continue
            if   job == "translate": self._do_translate(text)
            elif job == "chatbot":   self._do_chatbot(text)
            elif job == "partial":   self._do_partial(text)

    def _do_partial(self, text):
        gloss = self.nlp.process_partial(text)
        if gloss == self._last_partial: return
        self._last_partial = gloss
        new = [w for w in gloss if not self.mapper.has_sign(w)]
        if new: self.mapper.preload(new)

    def _do_translate(self, text):
        gloss = self.nlp.process(text)
        if not gloss:
            self._pend_status = "Could not process that sentence."; return
        print(f"[Translator] Gloss: {' '.join(gloss)}")
        pairs = self.mapper.get_signs_for_gloss(gloss)
        if not pairs:
            self._pend_status = "No signs found."; return
        found = [w for w, _ in pairs]
        self._pend_gloss   = "  ".join(
            f"[{w}]" if w in found else f"({w})" for w in gloss)
        self._signed_words = []
        self.aqueue.interrupt(pairs)

    def _do_chatbot(self, question):
        answer = call_gemini(question)
        self._pend_think = ""
        if not answer:
            self._pend_status = "AVA could not answer. Check your API key."; return
        print(f"[Chatbot] Answer: {answer!r}")
        self._pend_ai = f'AVA: "{answer}"'
        gloss = self.nlp.process(answer)
        if not gloss:
            self._pend_status = "Could not convert answer to signs."; return
        pairs = self.mapper.get_signs_for_gloss(gloss)
        if not pairs:
            self._pend_status = "No signs found for AVA's answer."; return
        found = [w for w, _ in pairs]
        self._pend_gloss   = "  ".join(
            f"[{w}]" if w in found else f"({w})" for w in gloss)
        self._signed_words = []
        self.aqueue.interrupt(pairs)

    # ── Main tick ─────────────────────────────────────────────────────────────
    def _tick(self, task):
        if self.paused: return Task.cont

        # Partial preload while typing
        if self._screen == self.SCREEN_TRANSLATOR:
            t = self._entry.get() if hasattr(self, "_entry") else ""
            if t: self._q.put(("partial", t))

        # Advance animation
        self._anim_accum += globalClock.getDt()
        INTERVAL = 1.0 / ANIM_FPS
        if self._anim_accum >= INTERVAL:
            self._anim_accum -= INTERVAL
            frame = self.aqueue.get_current_frame()
            self._prev_frame = self._curr_frame
            self._curr_frame = frame
            done = self.aqueue.finished_words()
            if done:
                self._signed_words.extend(done)
                self._last_lh = []; self._last_rh = []

        t     = min(1.0, self._anim_accum / max(1e-6, INTERVAL))
        frame = lerp_frame(self._prev_frame, self._curr_frame, t)

        pose = frame.get("pose", [])
        lh   = frame.get("left_hand",  [])
        rh   = frame.get("right_hand", [])

        for hand, last_attr in [(rh, "_last_rh"), (lh, "_last_lh")]:
            if hand and abs(hand[0]["z"]) < 0.01:
                has_xy = any(abs(lm["x"]-0.5)>0.02 or abs(lm["y"]-0.5)>0.02
                             for lm in hand[:3])
                if not has_xy:
                    if hand is rh: rh = self._last_rh
                    else:          lh = self._last_lh
        if rh: self._last_rh = rh
        if lh: self._last_lh = lh
        if pose: self._pose(pose, lh, rh)

        # HUD
        fps  = globalClock.getAverageFrameRate()
        word = self.aqueue.current_word() or "idle"
        self.hud.setText(
            f"render {fps:.0f}fps  |  {self._screen}  |  signing: {word}\n"
            "drag=rotate  scroll=zoom  SPACE=pause  ESC=menu")

        # Apply pending UI updates from worker thread
        if self._pend_think is not None and self._screen in (
                self.SCREEN_CHATBOT,):
            if hasattr(self, "_think_lbl"):
                self._think_lbl.setText(self._pend_think)
            self._pend_think = None

        if self._pend_ai is not None and self._screen == self.SCREEN_CHATBOT:
            if hasattr(self, "_ai_lbl"):
                self._ai_lbl.setText(self._pend_ai)
            self._pend_ai = None

        if self._pend_gloss is not None and self._screen != self.SCREEN_MENU:
            if hasattr(self, "_gloss_lbl"):
                self._gloss_lbl.setText(self._pend_gloss)
            self._pend_gloss = None

        if self._pend_status is not None and self._screen != self.SCREEN_MENU:
            if hasattr(self, "_status_lbl"):
                self._status_lbl.setText(self._pend_status)
            self._pend_status = None
        elif self._signed_words and self._screen != self.SCREEN_MENU:
            if hasattr(self, "_status_lbl"):
                self._status_lbl.setText("✓  " + "  ".join(self._signed_words))

        return Task.cont

    # ── Keyboard ──────────────────────────────────────────────────────────────
    def _on_space(self):
        if not self._input_active:
            self.paused = not self.paused

    def _on_escape(self):
        if self._input_active:
            if hasattr(self, "_entry"): self._entry["focus"] = 0
        elif self._screen != self.SCREEN_MENU:
            self._build_menu()
        else:
            sys.exit()

    def _on_q(self):
        if not self._input_active:
            sys.exit()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  AVA  —  Sign Language Avatar")
    print("=" * 55)
    if not os.path.exists(SIGNS_FOLDER):
        print(f"\n[!] Signs folder not found: {SIGNS_FOLDER}")
        print("    Create a 'Signs' folder next to main.py\n")
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("\n[!] No Gemini API key set.")
        print("    Chatbot mode needs a free key from:")
        print("    https://aistudio.google.com/apikey")
        print("    Then set: GEMINI_API_KEY=your-key  or edit main.py\n")
    AVAApp().run()

if __name__ == "__main__":
    main()