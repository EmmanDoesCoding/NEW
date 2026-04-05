"""
api.py  —  AVA Sign Language REST API Backend
==============================================
FastAPI server that exposes AVA's pipeline to any frontend (React Native, web, etc.)

Endpoints:
    POST /translate   — English text  → gloss + animation frames
    POST /tagalog     — Tagalog text  → English → gloss + animation frames
    POST /chatbot     — Question      → AI answer + gloss + animation frames
    GET  /health      — Liveness check
    GET  /signs       — List all available sign words in the library

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Or for production:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations

import os
import json
import threading
import urllib.request
import urllib.error
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Local AVA modules ──────────────────────────────────────────────────────────
from nlp_processor import NLPProcessor
from sign_mapper   import SignMapper
from tagalog_translator import TagalogTranslator

# ── Config ─────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.environ.get("OPENROUTER_MODEL", "google/gemma-3-4b-it:free")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

SIGNS_FOLDER = os.environ.get(
    "SIGNS_FOLDER",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Signs"),
)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AVA Sign Language API",
    description=(
        "Real-time sign language avatar backend. "
        "Converts English / Tagalog text to ASL gloss and MediaPipe animation frames."
    ),
    version="1.0.0",
)

# Allow all origins for development; lock this down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Module singletons (initialised once at startup) ────────────────────────────
_nlp:        NLPProcessor      | None = None
_mapper:     SignMapper         | None = None
_translator: TagalogTranslator | None = None


@app.on_event("startup")
def _startup():
    global _nlp, _mapper, _translator
    print("[AVA API] Initialising modules …")
    _nlp        = NLPProcessor()
    _mapper     = SignMapper(signs_folder=SIGNS_FOLDER)
    _translator = TagalogTranslator(
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
    )
    print("[AVA API] Ready.")


# ── Request / Response models ──────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text:    str = Field(..., description="English text to sign")
    partial: bool = Field(
        False,
        description="If true, skip the last incomplete word (real-time typing mode)",
    )

class TagalogRequest(BaseModel):
    text: str = Field(..., description="Tagalog text to sign")

class ChatbotRequest(BaseModel):
    message: str = Field(..., description="User question for the AI chatbot")
    history: list[dict] = Field(
        default_factory=list,
        description=(
            "Conversation history as [{role, content}] for multi-turn context. "
            "Roles must be 'user' or 'assistant'."
        ),
    )

class SignWord(BaseModel):
    word:   str
    frames: list[dict]

class TranslateResponse(BaseModel):
    input_text:   str
    english_text: str | None = None   # filled for /tagalog
    gloss:        list[str]
    signs:        list[SignWord]
    missing:      list[str]           # gloss words with no sign file
    frame_count:  int
    duration_ms:  float               # estimated playback time at 30fps

class ChatbotResponse(BaseModel):
    question:    str
    answer:      str
    gloss:       list[str]
    signs:       list[SignWord]
    missing:     list[str]
    frame_count: int
    duration_ms: float


# ── Shared pipeline helper ─────────────────────────────────────────────────────

def _gloss_to_signs(gloss: list[str]) -> tuple[list[SignWord], list[str]]:
    """
    Map a list of ASL gloss words to (word, frames) pairs.
    Returns (signs, missing) where missing = words that had no matching file.
    """
    signs:   list[SignWord] = []
    missing: list[str]      = []
    for word in gloss:
        frames = _mapper.get_sign(word)
        if frames:
            signs.append(SignWord(word=word, frames=frames))
        else:
            missing.append(word)
    return signs, missing


def _estimate_duration(signs: list[SignWord]) -> float:
    """Estimated playback duration in milliseconds at 30fps (+ 270ms blend per sign)."""
    total_frames = sum(len(s.frames) for s in signs)
    blend_ms     = len(signs) * 270.0   # ~8 frames at 30fps per transition
    return round((total_frames / 30.0) * 1000.0 + blend_ms, 1)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check."""
    return {
        "status":    "ok",
        "signs_indexed": _mapper.stats()["indexed_keys"] if _mapper else 0,
    }


@app.get("/signs")
def list_signs():
    """Return all available sign words in the library."""
    if not _mapper:
        raise HTTPException(503, "SignMapper not initialised")
    return {"signs": _mapper.available_signs()}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    """
    Convert English text to ASL gloss and return animation frames.

    The 'partial' flag enables real-time mode: the last incomplete word is
    skipped so the avatar can start signing before the user finishes typing.
    """
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")

    t0 = time.perf_counter()

    if req.partial:
        gloss = _nlp.process_partial(req.text)
    else:
        gloss = _nlp.process(req.text)

    signs, missing = _gloss_to_signs(gloss)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[/translate] '{req.text[:60]}' → {gloss} ({elapsed:.1f}ms)")

    return TranslateResponse(
        input_text=req.text,
        gloss=gloss,
        signs=signs,
        missing=missing,
        frame_count=sum(len(s.frames) for s in signs),
        duration_ms=_estimate_duration(signs),
    )


@app.post("/tagalog", response_model=TranslateResponse)
def tagalog(req: TagalogRequest):
    """
    Convert Tagalog text to English, then to ASL gloss + animation frames.
    Uses local lookup table first; falls back to OpenRouter API when confidence
    is below 50%.
    """
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")

    t0 = time.perf_counter()

    # Layer 1/2: Tagalog → English  (returns (english, method))
    english, _method = _translator.translate(req.text)

    # English → ASL gloss
    gloss = _nlp.process(english)

    signs, missing = _gloss_to_signs(gloss)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[/tagalog] '{req.text[:60]}' → '{english}' → {gloss} ({elapsed:.1f}ms)")

    return TranslateResponse(
        input_text=req.text,
        english_text=english,
        gloss=gloss,
        signs=signs,
        missing=missing,
        frame_count=sum(len(s.frames) for s in signs),
        duration_ms=_estimate_duration(signs),
    )


@app.post("/chatbot", response_model=ChatbotResponse)
def chatbot(req: ChatbotRequest):
    """
    Send a question to the Gemma AI via OpenRouter, then sign the response.

    Conversation history can be passed in for multi-turn dialogue.
    Note: Gemma does not support the 'system' role — the system prompt is
    prepended to the first user message automatically.
    """
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty")
    if not OPENROUTER_API_KEY:
        raise HTTPException(503, "OPENROUTER_API_KEY not configured")

    t0 = time.perf_counter()

    # ── Build message list ─────────────────────────────────────────────────────
    SYSTEM_PROMPT = (
        "You are AVA, a friendly sign language assistant. "
        "Answer clearly and concisely in 1-3 sentences. "
        "Do not use bullet points, markdown, or special characters. "
        "Plain text only."
    )

    messages: list[dict] = []

    # Copy history, validating roles
    for msg in req.history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})

    # Prepend system prompt to first user message (Gemma requirement)
    new_user_content = req.message
    if not messages:
        new_user_content = f"{SYSTEM_PROMPT}\n\n{req.message}"

    messages.append({"role": "user", "content": new_user_content})

    # ── Call OpenRouter ────────────────────────────────────────────────────────
    payload = json.dumps({
        "model":    OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": 256,
    }).encode("utf-8")

    request_obj = urllib.request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        answer = data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise HTTPException(502, f"OpenRouter error {e.code}: {body[:300]}")
    except urllib.error.URLError as e:
        raise HTTPException(504, f"Network error: {e.reason}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise HTTPException(502, f"Unexpected API response: {e}")

    # ── Convert answer to ASL ──────────────────────────────────────────────────
    gloss          = _nlp.process(answer)
    signs, missing = _gloss_to_signs(gloss)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[/chatbot] '{req.message[:60]}' → '{answer[:80]}' → {gloss} ({elapsed:.1f}ms)")

    return ChatbotResponse(
        question=req.message,
        answer=answer,
        gloss=gloss,
        signs=signs,
        missing=missing,
        frame_count=sum(len(s.frames) for s in signs),
        duration_ms=_estimate_duration(signs),
    )


# ── Optional: partial translate via WebSocket (for real-time keystroke mode) ──
# Uncomment if you want a WebSocket endpoint instead of polling /translate?partial=true

# from fastapi import WebSocket
# @app.websocket("/ws/translate")
# async def ws_translate(ws: WebSocket):
#     await ws.accept()
#     try:
#         while True:
#             text = await ws.receive_text()
#             gloss = _nlp.process_partial(text)
#             signs, missing = _gloss_to_signs(gloss)
#             payload = {
#                 "gloss": gloss,
#                 "signs": [{"word": s.word, "frames": s.frames} for s in signs],
#                 "missing": missing,
#             }
#             await ws.send_text(json.dumps(payload))
#     except Exception:
#         pass


# ── Dev entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )