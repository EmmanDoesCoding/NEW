"""
sign_mapper.py
==============
Maps ASL gloss words → animation frames from your Signs/ folder.
Optimised for 1000+ JSON files.

Key design decisions:
  - Index built at startup (just filenames, no file loading) — fast init
  - LRU cache: max 100 signs in RAM (~11 MB) — old signs evicted automatically
  - Fuzzy match only runs on cache miss — stays fast at scale
  - Background preloader thread warms cache ahead of animation
  - Thread-safe throughout

Signs folder:  Signs/  (same directory as this file)
Filename format: CamelCase  →  Hello.json, ThankYou.json
JSON format: array of frame objects  [{pose:[...], left_hand:[...], ...}, ...]
"""

import os
import json
import threading
import re
from collections import OrderedDict
from difflib import get_close_matches


# ── Config ─────────────────────────────────────────────────────────────────────
SIGNS_FOLDER  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Signs")
FUZZY_CUTOFF  = 0.72    # 0.0–1.0, higher = stricter fuzzy matching
LRU_MAX_SIZE  = 100     # max sign files kept in RAM simultaneously


# ── LRU Cache ──────────────────────────────────────────────────────────────────
class _LRUCache:
    """
    Simple LRU cache using OrderedDict.
    Evicts least-recently-used entry when max size is reached.
    NOT thread-safe on its own — caller must hold a lock.
    """
    def __init__(self, maxsize: int):
        self._maxsize = maxsize
        self._data: OrderedDict = OrderedDict()

    def get(self, key: str):
        if key not in self._data:
            return None
        self._data.move_to_end(key)   # mark as recently used
        return self._data[key]

    def put(self, key: str, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._maxsize:
            evicted_key, _ = self._data.popitem(last=False)
            # print(f"[SignMapper] LRU evicted: {evicted_key}")

    def __contains__(self, key: str):
        return key in self._data

    def __len__(self):
        return len(self._data)


# ── SignMapper ─────────────────────────────────────────────────────────────────
class SignMapper:
    """
    Maps ASL gloss words to animation frame lists.
    Handles 1000+ JSON files efficiently with LRU caching.
    """

    def __init__(self, signs_folder: str = SIGNS_FOLDER,
                 lru_size: int = LRU_MAX_SIZE):
        self.signs_folder = signs_folder
        self._index: dict[str, str] = {}   # "HELLO" → full filepath
        self._cache = _LRUCache(lru_size)
        self._lock  = threading.Lock()

        self._build_index()

    # ── Index ──────────────────────────────────────────────────────────────────

    def _build_index(self):
        """
        Scan Signs/ folder and build gloss → filepath mapping.
        Only reads filenames — no JSON loading at this stage.
        Completes in <50ms even for 1000+ files.
        """
        if not os.path.exists(self.signs_folder):
            print(f"[SignMapper] ERROR: Signs folder not found:")
            print(f"            {self.signs_folder}")
            print(f"            Create a 'Signs' folder next to sign_mapper.py")
            return

        count = 0
        for fname in os.listdir(self.signs_folder):
            if not fname.lower().endswith(".json"):
                continue

            stem = os.path.splitext(fname)[0]          # "Hello"
            path = os.path.join(self.signs_folder, fname)

            # Register under multiple lookup keys for maximum hit rate
            keys = self._stem_to_keys(stem)
            for key in keys:
                if key not in self._index:             # first file wins
                    self._index[key] = path

            count += 1

        total_keys = len(self._index)
        print(f"[SignMapper] {count} sign files indexed  "
              f"({total_keys} lookup keys)  "
              f"LRU cache: {self._cache._maxsize} slots")

    @staticmethod
    def _stem_to_keys(stem: str) -> list[str]:
        """
        Generate all lookup keys from a filename stem.

        Hello       → [HELLO]
        ThankYou    → [THANKYOU, THANK-YOU, THANK_YOU]
        EatFood     → [EATFOOD, EAT-FOOD, EAT_FOOD]
        """
        upper = stem.upper()
        # Split CamelCase into parts
        parts = re.sub(r'([A-Z][a-z]+)', r' \1', stem).split()
        parts = [p.upper() for p in parts if p]

        keys = [upper]                          # THANKYOU
        if len(parts) > 1:
            keys.append("-".join(parts))        # THANK-YOU
            keys.append("_".join(parts))        # THANK_YOU
        return keys

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_sign(self, word: str) -> list | None:
        """
        Return animation frames for a gloss word.
        Returns None if not found after fuzzy matching.

        Lookup order:
            1. LRU cache  (no I/O, <0.01ms)
            2. Exact index match  (no I/O, <0.1ms)
            3. Fuzzy index match  (~0.5ms for 1000 keys)
            4. Load JSON file  (I/O, ~5–20ms first time)
        """
        word = word.upper().strip()
        if not word:
            return None

        with self._lock:
            # 1. LRU cache hit
            cached = self._cache.get(word)
            if cached is not None:
                return cached

            # 2. Exact index match
            if word in self._index:
                return self._load_to_cache(word, self._index[word])

            # 3. Fuzzy match
            fuzzy_key = self._fuzzy_match(word)
            if fuzzy_key:
                print(f"[SignMapper] '{word}' → fuzzy '{fuzzy_key}'")
                frames = self._load_to_cache(fuzzy_key, self._index[fuzzy_key])
                # Cache under original word too for instant future lookups
                self._cache.put(word, frames)
                return frames

            print(f"[SignMapper] '{word}' — no match found")
            return None

    def get_signs_for_gloss(self, gloss_words: list[str]) -> list[tuple]:
        """
        Map a full gloss word list to (word, frames) pairs.
        Words with no sign are skipped.

        Returns: [("HELLO", [frame,...]), ("YOU", [frame,...]), ...]
        """
        result = []
        for word in gloss_words:
            frames = self.get_sign(word)
            if frames:
                result.append((word, frames))
            else:
                print(f"[SignMapper] Skipping '{word}' — no sign file")
        return result

    def preload(self, words: list[str]):
        """
        Load and cache a list of words in a background thread.
        Call this as soon as NLP produces the next batch of words —
        they'll be in cache by the time the animation needs them.
        """
        def _worker():
            for w in words:
                self.get_sign(w)
            print(f"[SignMapper] Preloaded: {words}")

        t = threading.Thread(target=_worker, daemon=True, name="SignPreload")
        t.start()

    def has_sign(self, word: str) -> bool:
        """
        Quick check if a sign exists — no file loading, no fuzzy match.
        Safe to call very frequently.
        """
        word = word.upper().strip()
        with self._lock:
            return (self._cache.get(word) is not None
                    or word in self._index)

    def has_sign_fuzzy(self, word: str) -> bool:
        """Check with fuzzy matching — slightly slower than has_sign."""
        word = word.upper().strip()
        with self._lock:
            return (self._cache.get(word) is not None
                    or word in self._index
                    or bool(self._fuzzy_match(word)))

    def available_signs(self) -> list[str]:
        """Sorted list of all indexed gloss words."""
        with self._lock:
            return sorted(set(self._index.keys()))

    def stats(self) -> dict:
        with self._lock:
            return {
                "indexed_keys":  len(self._index),
                "cached_signs":  len(self._cache),
                "cache_max":     self._cache._maxsize,
                "folder":        self.signs_folder,
            }

    # ── Internals ──────────────────────────────────────────────────────────────

    def _load_to_cache(self, key: str, filepath: str) -> list:
        frames = _load_json(filepath)
        self._cache.put(key, frames)
        return frames

    def _fuzzy_match(self, word: str) -> str | None:
        """
        Find closest indexed key using difflib.
        ~0.5ms for 1000 keys — acceptable for a cache miss.
        Called WITHOUT lock held check — caller must hold lock.
        """
        candidates = list(self._index.keys())
        if not candidates:
            return None
        matches = get_close_matches(word, candidates, n=1, cutoff=FUZZY_CUTOFF)
        return matches[0] if matches else None


# ── JSON loading (module-level, no class needed) ───────────────────────────────

def _load_json(filepath: str) -> list:
    """
    Load and validate a sign JSON file.
    Handles format A (direct array) and format B (dict with 'frames' key).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[SignMapper] JSON error in {os.path.basename(filepath)}: {e}")
        return []
    except Exception as e:
        print(f"[SignMapper] Load error {os.path.basename(filepath)}: {e}")
        return []

    # Normalise to a list of frames
    if isinstance(raw, list):
        frames = raw
    elif isinstance(raw, dict) and "frames" in raw:
        frames = raw["frames"]
    elif isinstance(raw, dict) and "pose" in raw:
        frames = [raw]   # single frame
    else:
        print(f"[SignMapper] Unknown format: {os.path.basename(filepath)}")
        return []

    return _validate(frames, filepath)


def _validate(frames: list, filepath: str) -> list:
    """
    Validate frames — each must have a non-empty pose list.
    Fills missing hand/face keys with empty lists.
    """
    valid = []
    for f in frames:
        if not isinstance(f, dict):
            continue
        if not f.get("pose"):
            continue   # pose required
        f.setdefault("left_hand",  [])
        f.setdefault("right_hand", [])
        f.setdefault("face",       [])
        valid.append(f)

    if not valid:
        print(f"[SignMapper] No valid frames in {os.path.basename(filepath)}")
    return valid


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, time

    folder = sys.argv[1] if len(sys.argv) > 1 else SIGNS_FOLDER
    print(f"Testing SignMapper\nFolder: {folder}\n")

    mapper = SignMapper(folder)
    print(f"\nStats: {mapper.stats()}")

    signs = mapper.available_signs()
    print(f"\nSample signs ({min(10, len(signs))} of {len(signs)}):")
    for s in signs[:10]:
        print(f"  {s}")

    # Test lookups
    print(f"\nLookup tests:")
    # Test real signs from index + common fuzzy cases
    test_words = signs[:5] + ["HELO", "THANKYOU", "THANK-YOU", "ZZZNOTEXIST"]
    for word in test_words:
        t0     = time.perf_counter()
        frames = mapper.get_sign(word)
        ms     = (time.perf_counter() - t0) * 1000
        status = f"{len(frames)} frames" if frames else "NOT FOUND"
        print(f"  {word:25s} → {status:15s} ({ms:.1f}ms)")

    # Test that second lookup hits cache (should be near 0ms)
    print(f"\nCache hit test (same words again):")
    for word in test_words[:5]:
        t0     = time.perf_counter()
        frames = mapper.get_sign(word)
        ms     = (time.perf_counter() - t0) * 1000
        status = f"{len(frames)} frames" if frames else "NOT FOUND"
        print(f"  {word:25s} → {status:15s} ({ms:.2f}ms)  ← should be <0.1ms")