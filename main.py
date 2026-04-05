"""
main.py
=======
Sign Language Avatar  -  AVA
Starts with a main menu, choose Translator or Chatbot mode.

Free API: OpenRouter (free models available, no credit card needed)
Get your key at: https://openrouter.ai/keys

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

from nlp_processor       import NLPProcessor
from sign_mapper          import SignMapper
from animation_queue      import AnimationQueue
from tagalog_translator   import TagalogTranslator

import AVA_panda3d as _ava_mod
Avatar                = _ava_mod.Avatar
lerp_frame            = _ava_mod.lerp_frame
calibrate_from_frames = _ava_mod.calibrate_from_frames

# ── Config ────────────────────────────────────────────────────────────────────
SIGNS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Signs")
ANIM_FPS     = 30
BLEND_FRAMES = 8

# OpenRouter - free models available, no credit card needed for free tier
# Get key: https://openrouter.ai/keys
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
OPENROUTER_MODEL   = "google/gemma-3-4b-it:free"   # free model, swap as needed

CHATBOT_PROMPT = """You are AVA, a friendly sign language avatar assistant.
Answer in EXACTLY 1 to 2 short sentences.
Rules:
- Use only simple common words
- Maximum 10 words per sentence
- No contractions (say "do not" not "don't")
- Be warm and friendly"""

# ── Design tokens  (mobile-first, OLED-friendly dark palette) ─────────────────
# Backgrounds
C_BG         = (0.05, 0.05, 0.07, 1)
C_SURFACE    = (0.09, 0.10, 0.13, 1)
C_SURFACE2   = (0.12, 0.13, 0.17, 1)
# Brand accents
C_BLUE       = (0.22, 0.58, 1.00, 1)
C_GREEN      = (0.12, 0.82, 0.54, 1)
C_PURPLE     = (0.62, 0.28, 1.00, 1)
C_BLUE_DIM   = (0.07, 0.16, 0.30, 1)
C_GREEN_DIM  = (0.05, 0.22, 0.15, 1)
C_PURPLE_DIM = (0.16, 0.08, 0.28, 1)
# Text
C_TEXT       = (0.95, 0.96, 0.98, 1)
C_TEXT_DIM   = (0.50, 0.52, 0.58, 1)
C_TEXT_HINT  = (0.32, 0.34, 0.40, 1)
# Semantic
C_SUCCESS    = (0.20, 0.88, 0.56, 1)
C_WARN       = (1.00, 0.76, 0.18, 1)
C_ERROR      = (1.00, 0.38, 0.38, 1)
# Aliases
C_CARD       = C_SURFACE
C_INPUT_BG   = C_SURFACE2
C_ACCENT1    = C_BLUE
C_ACCENT2    = C_PURPLE
C_ACCENT3    = C_GREEN
C_ACCENT1_DIM= C_BLUE_DIM
C_ACCENT2_DIM= C_PURPLE_DIM
C_ACCENT3_DIM= C_GREEN_DIM
C_BORDER     = (0.18, 0.20, 0.26, 1)


# ── OpenRouter API ────────────────────────────────────────────────────────────

def call_openrouter(question: str) -> str:
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        return "I do not have an API key. Please add your OpenRouter API key."

    # Gemma (and many free models) do not support the "system" role.
    # Prepend the system prompt as the first user turn instead - works universally.
    combined = f"{CHATBOT_PROMPT}\n\nUser: {question}"

    payload = _json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": combined}
        ],
        "max_tokens": 80,
        "temperature": 0.7
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload, method="POST",
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer":  "https://github.com/AVA-sign-avatar",
            "X-Title":       "AVA Sign Language Avatar",
        })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = _json.loads(r.read())
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[OpenRouter] HTTP {e.code}: {body}")
        return ""
    except Exception as e:
        print(f"[OpenRouter] Error: {e}")
        return ""


# ── App ───────────────────────────────────────────────────────────────────────

class AVAApp(Avatar):

    SCREEN_MENU       = "menu"
    SCREEN_TRANSLATOR = "translator"
    SCREEN_CHATBOT    = "chatbot"
    SCREEN_TAGALOG    = "tagalog"

    def __init__(self):
        super().__init__()

        # Restyle the HUD inherited from Avatar to match new design
        self.hud.setPos(-1.28, 0.93)
        self.hud.setScale(0.036)
        self.hud["fg"] = (0.40, 0.42, 0.48, 1)

        # pipeline
        print("\n[AVA] Loading pipeline...")
        self.nlp    = NLPProcessor()
        self.mapper = SignMapper(SIGNS_FOLDER)
        self.aqueue = AnimationQueue(blend_frames=BLEND_FRAMES)
        self.tl     = TagalogTranslator(api_key=OPENROUTER_API_KEY,
                                        model=OPENROUTER_MODEL)
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
        self._pend_tl      = None    # Tagalog -> English translated text

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

    # ── Avatar visibility ─────────────────────────────────────────────────────
    def _hide_avatar(self):
        """Hide all geometry nodes during menu screen."""
        for node in self.G.values():
            node.hide()

    def _show_avatar(self):
        """Restore avatar visibility when entering a mode."""
        # Nodes will be shown by _pose() on next tick - just un-hide all
        for node in self.G.values():
            node.show()

    # ── UI helpers ────────────────────────────────────────────────────────────
    def _clear_ui(self):
        for node in self._ui_nodes:
            try: node.destroy()
            except: pass
        self._ui_nodes = []

    def _track(self, widget):
        self._ui_nodes.append(widget)
        return widget

    def _label(self, text, pos, scale, color=C_TEXT, align=TextNode.ACenter,
               may_change=False):
        w = OnscreenText(text=text, pos=pos, scale=scale,
                         fg=color, align=align, mayChange=may_change)
        return self._track(w)

    def _btn(self, text, pos, scale, color, text_color, cmd):
        """Pill-shaped primary action button."""
        w = DirectButton(
            text=text, scale=scale, pos=pos, command=cmd,
            frameColor=color, text_fg=text_color,
            relief=1, rolloverSound=None, clickSound=None,
            text_scale=1.0, pad=(0.40, 0.18),
            frameSize=None)
        return self._track(w)

    def _ghost_btn(self, text, pos, scale, color, cmd):
        """Small ghost/text-only back button."""
        w = DirectButton(
            text=text, scale=scale, pos=pos, command=cmd,
            frameColor=(0.12, 0.13, 0.17, 0.85),
            text_fg=color,
            relief=1, rolloverSound=None, clickSound=None,
            text_scale=1.0, pad=(0.30, 0.14))
        return self._track(w)

    def _panel(self, x, y, w, h, color):
        """Flat filled rectangle panel."""
        hw, hh = w / 2, h / 2
        f = DirectFrame(frameSize=(-hw, hw, -hh, hh),
                        pos=(x, 0, y), frameColor=color)
        return self._track(f)

    def _hline(self, y, color=C_BORDER):
        """Full-width 1px separator line."""
        f = DirectFrame(frameSize=(-2.0, 2.0, -0.002, 0.002),
                        pos=(0, 0, y), frameColor=color)
        return self._track(f)

    def _badge(self, text, x, y, color):
        """Small coloured pill label (mode badge)."""
        # background pill
        self._track(DirectFrame(
            frameSize=(-0.14, 0.14, -0.030, 0.030),
            pos=(x, 0, y),
            frameColor=(*color[:3], 0.18)))
        self._label(text, (x, y - 0.008), 0.032, color)

    # ── MENU SCREEN ───────────────────────────────────────────────────────────
    def _build_menu(self):
        self._clear_ui()
        self._screen = self.SCREEN_MENU
        self._hide_avatar()
        self.aqueue.clear(smooth=False)
        self.setBackgroundColor(*C_BG)

        # ── Full backdrop ──────────────────────────────────────────────────────
        self._panel(0, 0, 4.0, 2.6, C_BG)

        # ── Subtle gradient top glow  (blue tint strip) ───────────────────────
        self._track(DirectFrame(
            frameSize=(-2.0, 2.0, -0.003, 0.003),
            pos=(0, 0, 0.97), frameColor=C_BLUE))
        self._track(DirectFrame(
            frameSize=(-2.0, 2.0, -0.18, 0.0),
            pos=(0, 0, 1.10),
            frameColor=(0.09, 0.17, 0.36, 0.28)))

        # ── Logo / title area ──────────────────────────────────────────────────
        # "AVA" large wordmark
        self._label("AVA", (0, 0.70), 0.22, C_TEXT)
        # Accent underline beneath AVA
        self._track(DirectFrame(
            frameSize=(-0.22, 0.22, -0.006, 0.006),
            pos=(0, 0, 0.62), frameColor=C_BLUE))
        self._label("Sign Language Avatar", (0, 0.50), 0.048, C_TEXT_DIM)
        self._label("Choose a mode to begin",
                    (0, 0.40), 0.034, C_TEXT_HINT)

        # ── Mode cards  ── stacked vertically, mobile-style ───────────────────
        #   Each card: dark surface + coloured left-edge bar + icon emoji + text

        def mode_card(cy, accent, dim, icon, title, desc, btn_label, cmd):
            # Card spans full width, fixed height
            self._track(DirectFrame(
                frameSize=(-1.55, 1.55, -0.175, 0.175),
                pos=(0, 0, cy), frameColor=dim))
            # Left accent bar
            self._track(DirectFrame(
                frameSize=(-1.55, -1.46, -0.175, 0.175),
                pos=(0, 0, cy), frameColor=accent))
            # Icon - absolute screen pos
            self._label(icon,  (-1.28, cy + 0.055), 0.075, accent)
            # Title - left-aligned, absolute
            self._label(title, (-0.92, cy + 0.065), 0.050, C_TEXT,
                        align=TextNode.ALeft)
            # Description - left-aligned, absolute
            self._label(desc,  (-0.92, cy - 0.060), 0.032, C_TEXT_DIM,
                        align=TextNode.ALeft)
            # Pill button - right side
            self._btn(btn_label, (1.22, 0, cy), 0.038,
                      accent, (0.03, 0.03, 0.05, 1), cmd)

        mode_card( 0.15, C_BLUE,   C_BLUE_DIM,
                  "[EN]", "TRANSLATOR",
                  "Type English  ->  AVA signs it live",
                  "Start >", self._open_translator)

        mode_card(-0.20, C_GREEN,  C_GREEN_DIM,
                  "[TL]", "TAGALOG",
                  "Type Filipino  ->  AVA signs it",
                  "Magsimula >", self._open_tagalog)

        mode_card(-0.55, C_PURPLE, C_PURPLE_DIM,
                  "[AI]", "CHATBOT",
                  "Ask AVA  ->  she signs the answer",
                  "Ask AVA >", self._open_chatbot)

        # ── Bottom status bar ─────────────────────────────────────────────────
        self._hline(-0.80, C_BORDER)
        self._panel(0, -0.92, 4.0, 0.24, (0.06, 0.07, 0.09, 1))
        self._label("Powered by OpenRouter  |  English & Tagalog  |  openrouter.ai/keys",
                    (0, -0.89), 0.028, C_TEXT_HINT)
        self._label("ESC = quit   SPACE = pause   drag = orbit camera",
                    (0, -0.97), 0.026, C_TEXT_HINT)

        self.hud.setText("")

    # ── Shared mode chrome ────────────────────────────────────────────────────
    def _build_mode_chrome(self, title, accent, badge_text):
        """
        Draw the common top-bar and bottom input panel chrome for all modes.
        Returns nothing - widgets are tracked automatically.
        """
        self.setBackgroundColor(*C_BG)

        # ── Top bar ───────────────────────────────────────────────────────────
        self._panel(0, 0.91, 4.0, 0.19, C_SURFACE)
        self._hline(0.82, accent)   # thin coloured underline beneath top bar

        # Back button - left side of top bar
        self._ghost_btn("<  Menu", (-1.18, 0, 0.91), 0.040,
                        C_TEXT_DIM, self._build_menu)

        # Mode title - centred
        self._label(title, (0, 0.88), 0.048, accent)

        # ── Bottom input panel ────────────────────────────────────────────────
        self._panel(0, -0.88, 4.0, 0.26, C_SURFACE)
        self._hline(-0.76, C_BORDER)

    # ── TRANSLATOR SCREEN ──────────────────────────────────────────────────────
    def _open_translator(self):
        self._clear_ui()
        self._screen       = self.SCREEN_TRANSLATOR
        self._signed_words = []
        self._show_avatar()

        self._build_mode_chrome("TRANSLATOR", C_BLUE, "EN")

        # ── Info cards (upper area, above avatar) ─────────────────────────────
        # Gloss pill row
        self._panel(0, 0.68, 2.80, 0.10, C_BLUE_DIM)
        self._gloss_lbl = self._label("ASL gloss will appear here",
                                      (0, 0.665), 0.036, C_TEXT_DIM,
                                      may_change=True)

        # Status row
        self._status_lbl = self._label(
            "Type a sentence below and press Enter",
            (0, -0.66), 0.036, C_SUCCESS, may_change=True)

        # ── Input row (inside bottom panel) ───────────────────────────────────
        self._label("English:", (-1.28, -0.815), 0.034,
                    C_BLUE, align=TextNode.ALeft)

        self._entry = DirectEntry(
            text="", scale=0.042,
            pos=(-1.10, 0, -0.895), width=40, numLines=1, focus=0,
            focusInCommand=self._focus_in,
            focusOutCommand=self._focus_out,
            command=self._on_enter, extraArgs=[],
            frameColor=C_SURFACE2, text_fg=C_TEXT,
            rolloverSound=None, clickSound=None)
        self._track(self._entry)

        self._btn("Sign >", (1.18, 0, -0.895), 0.044,
                  C_BLUE, (0.03, 0.03, 0.05, 1), self._submit_btn)

        self._entry["focus"] = 1
        print("[AVA] -> Translator mode")

    # ── CHATBOT SCREEN ─────────────────────────────────────────────────────────
    def _open_chatbot(self):
        self._clear_ui()
        self._screen       = self.SCREEN_CHATBOT
        self._signed_words = []
        self._show_avatar()

        self._build_mode_chrome("CHATBOT", C_PURPLE, "AI")

        # ── Response card ─────────────────────────────────────────────────────
        self._panel(0, 0.68, 2.80, 0.10, C_PURPLE_DIM)
        self._ai_lbl = self._label("", (0, 0.665), 0.036,
                                   (0.80, 0.65, 1.0, 1), may_change=True)

        # Gloss row
        self._panel(0, 0.55, 2.80, 0.08, C_SURFACE)
        self._gloss_lbl = self._label("", (0, 0.538), 0.032,
                                      C_TEXT_DIM, may_change=True)

        # Thinking / status
        self._think_lbl = self._label("", (0, -0.62), 0.036,
                                      C_WARN, may_change=True)
        self._status_lbl = self._label(
            "Ask AVA anything - she'll sign the answer",
            (0, -0.66), 0.034, C_SUCCESS, may_change=True)

        if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
            self._label("[!]  No API key - add OPENROUTER_API_KEY in main.py",
                        (0, 0.76), 0.033, C_WARN)

        # ── Input row ─────────────────────────────────────────────────────────
        self._label("Question:", (-1.28, -0.815), 0.034,
                    C_PURPLE, align=TextNode.ALeft)

        self._entry = DirectEntry(
            text="", scale=0.042,
            pos=(-1.10, 0, -0.895), width=40, numLines=1, focus=0,
            focusInCommand=self._focus_in,
            focusOutCommand=self._focus_out,
            command=self._on_enter, extraArgs=[],
            frameColor=C_SURFACE2, text_fg=C_TEXT,
            rolloverSound=None, clickSound=None)
        self._track(self._entry)

        self._btn("Ask >", (1.18, 0, -0.895), 0.044,
                  C_PURPLE, (0.04, 0.02, 0.07, 1), self._submit_btn)

        self._entry["focus"] = 1
        print("[AVA] -> Chatbot mode")

    # ── TAGALOG SCREEN ─────────────────────────────────────────────────────────
    def _open_tagalog(self):
        self._clear_ui()
        self._screen       = self.SCREEN_TAGALOG
        self._signed_words = []
        self._show_avatar()

        self._build_mode_chrome("TAGALOG", C_GREEN, "TL")

        # ── Translation result card ───────────────────────────────────────────
        self._panel(0, 0.68, 2.80, 0.10, C_GREEN_DIM)
        self._tl_lbl = self._label("", (0, 0.665), 0.036,
                                   C_GREEN, may_change=True)

        # Gloss row
        self._panel(0, 0.55, 2.80, 0.08, C_SURFACE)
        self._gloss_lbl = self._label("", (0, 0.538), 0.032,
                                      C_TEXT_DIM, may_change=True)

        # Thinking / status
        self._think_lbl = self._label("", (0, -0.62), 0.036,
                                      C_WARN, may_change=True)
        self._status_lbl = self._label(
            "I-type ang Tagalog na pangungusap",
            (0, -0.66), 0.034, C_SUCCESS, may_change=True)

        # ── Input row ─────────────────────────────────────────────────────────
        self._label("Tagalog:", (-1.28, -0.815), 0.034,
                    C_GREEN, align=TextNode.ALeft)

        self._entry = DirectEntry(
            text="", scale=0.042,
            pos=(-1.10, 0, -0.895), width=40, numLines=1, focus=0,
            focusInCommand=self._focus_in,
            focusOutCommand=self._focus_out,
            command=self._on_enter, extraArgs=[],
            frameColor=C_SURFACE2, text_fg=C_TEXT,
            rolloverSound=None, clickSound=None)
        self._track(self._entry)

        self._btn("I-sign >", (1.18, 0, -0.895), 0.044,
                  C_GREEN, (0.02, 0.06, 0.04, 1), self._submit_btn)

        self._entry["focus"] = 1
        print("[AVA] -> Tagalog mode")

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
        elif self._screen == self.SCREEN_TAGALOG:
            print(f"[Tagalog] {text!r}")
            self._pend_think  = "Nagsasalin..."   # "Translating..." in Filipino
            self._pend_status = ""
            self._q.put(("tagalog", text))

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
            elif job == "tagalog":   self._do_tagalog(text)

    def _do_partial(self, text):
        gloss = self.nlp.process_partial(text)
        if gloss == self._last_partial: return
        self._last_partial = gloss
        new = [w for w in gloss if not self.mapper.has_sign(w)]
        if new: self.mapper.preload(new)

    def _do_tagalog(self, tagalog_text):
        english, method = self.tl.translate(tagalog_text)
        self._pend_think = ""
        if not english:
            self._pend_status = "Hindi ma-translate. Subukan ulit."; return
        print(f"[Tagalog] '{tagalog_text}' -> '{english}'  [{method}]")
        self._pend_tl = f"EN: {english}"
        # Feed the English result into the normal translation pipeline
        gloss = self.nlp.process(english)
        if not gloss:
            self._pend_status = "Could not convert to signs."; return
        pairs = self.mapper.get_signs_for_gloss(gloss)
        if not pairs:
            self._pend_status = "No signs found for that sentence."; return
        found = [w for w, _ in pairs]
        self._pend_gloss   = "  ".join(
            f"[{w}]" if w in found else f"({w})" for w in gloss)
        self._signed_words = []
        self.aqueue.interrupt(pairs)

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
        answer = call_openrouter(question)
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
        word = self.aqueue.current_word() or "-"
        self.hud.setText(
            f"{fps:.0f} fps  |  signing: {word}  |  drag=orbit  scroll=zoom  SPACE=pause")

        # Apply pending UI updates from worker thread
        if self._pend_think is not None and self._screen in (
                self.SCREEN_CHATBOT, self.SCREEN_TAGALOG):
            if hasattr(self, "_think_lbl"):
                self._think_lbl.setText(self._pend_think)
            self._pend_think = None

        if self._pend_tl is not None and self._screen == self.SCREEN_TAGALOG:
            if hasattr(self, "_tl_lbl"):
                self._tl_lbl.setText(self._pend_tl)
            self._pend_tl = None

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
                self._status_lbl.setText("OK  " + "  ".join(self._signed_words))

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
    print("  AVA  -  Sign Language Avatar")
    print("=" * 55)
    if not os.path.exists(SIGNS_FOLDER):
        print(f"\n[!] Signs folder not found: {SIGNS_FOLDER}")
        print("    Create a 'Signs' folder next to main.py\n")
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        print("\n[!] No OpenRouter API key set.")
        print("    Chatbot mode needs a free key from:")
        print("    https://openrouter.ai/keys")
        print("    Then set: OPENROUTER_API_KEY=your-key  or edit main.py\n")
    AVAApp().run()

if __name__ == "__main__":
    main()