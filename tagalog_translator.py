"""
tagalog_translator.py
=====================
Converts Tagalog text → English text.

Two-layer approach:
  1. Word-level lookup table  — instant, no network, covers ~400 common words
  2. OpenRouter API fallback  — used when confidence is low (few words matched)

The module is self-contained.  main.py passes in the API key and model at
call time so nothing needs to be configured here.

Usage:
    from tagalog_translator import TagalogTranslator
    t = TagalogTranslator(api_key="sk-or-...", model="google/gemma-3-4b-it:free")
    english = t.translate("Kumusta ka?")   # → "How are you?"
"""

import re
import urllib.request
import urllib.error
import json
import threading


# ── Word-level lookup table ────────────────────────────────────────────────────
# Keys: lowercase Tagalog words / common roots
# Values: English equivalent (single word or short phrase)
#
# Tagalog grammar note: many words are formed by adding affixes to a root.
# We store both the root and the most common derived forms for maximum coverage.

WORD_MAP: dict[str, str] = {

    # ── Greetings / courtesy ──────────────────────────────────────────────────
    "kumusta":      "how are you",
    "kamusta":      "how are you",
    "magandang":    "good",
    "umaga":        "morning",
    "hapon":        "afternoon",
    "gabi":         "evening",
    "tanghali":     "noon",
    "oo":           "yes",
    "hindi":        "no",
    "salamat":      "thank you",
    "maraming":     "many",        # maraming salamat = thank you very much
    "walang":       "no",          # walang anuman = you're welcome
    "anuman":       "problem",
    "paumanhin":    "sorry",
    "pasensya":     "sorry",
    "patawad":      "forgive",
    "pakiusap":     "please",
    "po":           "",            # politeness particle — drop in English
    "ho":           "",            # politeness particle — drop
    "na":           "",            # aspect marker — drop
    "nga":          "",            # emphasis particle — drop
    "ba":           "",            # question marker — drop
    "daw":          "",            # hearsay marker — drop
    "raw":          "",            # hearsay marker — drop
    "pala":         "",            # realization marker — drop
    "ang":          "",            # subject marker — drop
    "ng":           "of",          # object/genitive marker
    "sa":           "in",          # locative/dative marker
    "ni":           "of",          # genitive marker (personal)
    "kay":          "to",          # dative (personal)
    "mga":          "",            # plural marker — drop
    "nang":         "when",        # conjunction/adverb marker
    "para":         "for",
    "hanggang":     "until",
    "habang":       "while",
    "kung":         "if",
    "kahit":        "even if",
    "palagi":       "always",
    "lagi":         "always",
    "talaga":       "really",
    "lang":         "only",
    "lamang":       "only",
    "din":          "also",
    "rin":          "also",
    "na naman":     "again",
    "naman":        "too",
    "kaya":         "so",
    "pero":         "but",
    "at":           "and",
    "o":            "or",
    "kasi":         "because",

    # ── Pronouns ──────────────────────────────────────────────────────────────
    "ako":          "I",
    "ikaw":         "you",
    "ka":           "you",
    "siya":         "he",          # also she/it — context-free
    "kami":         "we",          # exclusive we
    "tayo":         "we",          # inclusive we
    "kayo":         "you all",
    "sila":         "they",
    "ko":           "my",
    "mo":           "your",
    "niya":         "his",
    "namin":        "our",
    "natin":        "our",
    "ninyo":        "your",
    "nila":         "their",
    "akin":         "mine",
    "iyo":          "yours",
    "kanya":        "his",
    "atin":         "ours",
    "kanila":       "theirs",
    "ito":          "this",
    "iyan":         "that",
    "iyon":         "that",
    "dito":         "here",
    "diyan":        "there",
    "doon":         "there",
    "sino":         "who",
    "ano":          "what",
    "saan":         "where",
    "kailan":       "when",
    "bakit":        "why",
    "paano":        "how",
    "ilan":         "how many",
    "magkano":      "how much",

    # ── Common verbs (root forms + common conjugations) ───────────────────────
    "gusto":        "want",
    "ayaw":         "do not want",
    "kailangan":    "need",
    "mahal":        "love",
    "galit":        "angry",
    "takot":        "afraid",
    "alam":         "know",
    "hindi alam":   "do not know",
    "punta":        "go",
    "pumunta":      "went",
    "pupunta":      "will go",
    "uwi":          "go home",
    "umuwi":        "went home",
    "uuwi":         "will go home",
    "kain":         "eat",
    "kumain":       "ate",
    "kakain":       "will eat",
    "inom":         "drink",
    "uminom":       "drank",
    "iinom":        "will drink",
    "tulog":        "sleep",
    "natulog":      "slept",
    "matulog":      "sleep",
    "maglaro":      "play",
    "naglaro":      "played",
    "maglalaro":    "will play",
    "trabaho":      "work",
    "nagtatrabaho": "working",
    "magtrabaho":   "work",
    "aral":         "study",
    "mag-aral":     "study",
    "nag-aral":     "studied",
    "mag-aaral":    "will study",
    "sulat":        "write",
    "sumulat":      "wrote",
    "magsusulat":   "will write",
    "basa":         "read",
    "magbasa":      "read",
    "nagbasa":      "read",
    "makinig":      "listen",
    "nakinig":      "listened",
    "tumingin":     "look",
    "tumingala":    "look up",
    "makita":       "see",
    "nakita":       "saw",
    "marinig":      "hear",
    "narinig":      "heard",
    "umalis":       "leave",
    "aalis":        "will leave",
    "dumating":     "arrive",
    "darating":     "will arrive",
    "bumalik":      "return",
    "babalik":      "will return",
    "tumayo":       "stand",
    "umupo":        "sit",
    "higa":         "lie down",
    "takbo":        "run",
    "tumakbo":      "ran",
    "lakad":        "walk",
    "lumakad":      "walked",
    "lumapit":      "approach",
    "lumayo":       "move away",
    "dalhin":       "bring",
    "kumuha":       "get",
    "ibigay":       "give",
    "bigay":        "give",
    "ibenta":       "sell",
    "bilhin":       "buy",
    "binili":       "bought",
    "bayad":        "pay",
    "magbayad":     "pay",
    "nagtanong":    "asked",
    "tanong":       "ask",
    "magtanong":    "ask",
    "sumagot":      "answer",
    "sagot":        "answer",
    "magsalita":    "speak",
    "nagsalita":    "spoke",
    "magsasalita":  "will speak",
    "sabihin":      "say",
    "sinabi":       "said",
    "sasabihin":    "will say",
    "makinig":      "listen",
    "tawag":        "call",
    "tumawag":      "called",
    "tutawag":      "will call",
    "manood":       "watch",
    "nanood":       "watched",
    "manoonood":    "will watch",
    "luto":         "cook",
    "magluto":      "cook",
    "nagluto":      "cooked",
    "magluluto":    "will cook",
    "linis":        "clean",
    "maglinis":     "clean",
    "naglinis":     "cleaned",
    "hugasan":      "wash",
    "hinugasan":    "washed",
    "maghugas":     "wash",
    "tulungan":     "help",
    "tumulong":     "help",
    "natulungan":   "helped",
    "isipin":       "think",
    "naisip":       "thought",
    "mag-isip":     "think",
    "maintindihan": "understand",
    "naintindihan": "understood",
    "kalimutan":    "forget",
    "nakalimutan":  "forgot",
    "alalahanin":   "remember",
    "naaalala":     "remember",
    "magbigay":     "give",
    "kunin":        "take",
    "kinuha":       "took",
    "hawak":        "hold",
    "humawak":      "hold",
    "buksan":       "open",
    "binuksan":     "opened",
    "isara":        "close",
    "isinara":      "closed",
    "hanapin":      "find",
    "nahanap":      "found",
    "maghanap":     "search",
    "tanggalin":    "remove",
    "ilagay":       "put",
    "inilagay":     "placed",
    "simulan":      "start",
    "sinimulan":    "started",
    "tapusin":      "finish",
    "tapos":        "done",
    "matuto":       "learn",
    "natuto":       "learned",
    "magturo":      "teach",
    "nagturo":      "taught",

    # ── Common adjectives ─────────────────────────────────────────────────────
    "mabuti":       "good",
    "masama":       "bad",
    "maganda":      "beautiful",
    "pangit":       "ugly",
    "malaki":       "big",
    "maliit":       "small",
    "mataas":       "tall",
    "mababa":       "short",
    "mahaba":       "long",
    "maikli":       "short",
    "mabigat":      "heavy",
    "magaan":       "light",
    "mainit":       "hot",
    "malamig":      "cold",
    "mabilis":      "fast",
    "mabagal":      "slow",
    "matanda":      "old",
    "bata":         "young",
    "bago":         "new",
    "luma":         "old",
    "marami":       "many",
    "kaunti":       "few",
    "lahat":        "all",
    "wala":         "none",
    "may":          "have",
    "mayroon":      "have",
    "masaya":       "happy",
    "malungkot":    "sad",
    "pagod":        "tired",
    "gutom":        "hungry",
    "uhaw":         "thirsty",
    "malusog":      "healthy",
    "may sakit":    "sick",
    "masakit":      "painful",
    "malakas":      "strong",
    "mahina":       "weak",
    "mayaman":      "rich",
    "mahirap":      "poor",
    "matalino":     "smart",
    "tanga":        "foolish",
    "masipag":      "hardworking",
    "tamad":        "lazy",
    "maingay":      "noisy",
    "tahimik":      "quiet",
    "malinis":      "clean",
    "marumi":       "dirty",
    "buo":          "whole",
    "sira":         "broken",
    "tama":         "correct",
    "mali":         "wrong",

    # ── Time ──────────────────────────────────────────────────────────────────
    "ngayon":       "now",
    "bukas":        "tomorrow",
    "kahapon":      "yesterday",
    "mamaya":       "later",
    "kanina":       "earlier",
    "lagi":         "always",
    "minsan":       "sometimes",
    "madalas":      "often",
    "bihira":       "rarely",
    "hindi kailanman": "never",
    "dati":         "before",
    "maagang":      "early",
    "huli":         "late",
    "linggo":       "sunday",
    "lunes":        "monday",
    "martes":       "tuesday",
    "miyerkules":   "wednesday",
    "huwebes":      "thursday",
    "biyernes":     "friday",
    "sabado":       "saturday",
    "enero":        "january",
    "pebrero":      "february",
    "marso":        "march",
    "abril":        "april",
    "mayo":         "may",
    "hunyo":        "june",
    "hulyo":        "july",
    "agosto":       "august",
    "setyembre":    "september",
    "oktubre":      "october",
    "nobyembre":    "november",
    "disyembre":    "december",
    "oras":         "time",
    "araw":         "day",
    "gabi":         "night",
    "linggo":       "week",
    "buwan":        "month",
    "taon":         "year",

    # ── Common nouns ──────────────────────────────────────────────────────────
    "tao":          "person",
    "lalaki":       "man",
    "babae":        "woman",
    "bata":         "child",
    "pamilya":      "family",
    "ina":          "mother",
    "nanay":        "mother",
    "ama":          "father",
    "tatay":        "father",
    "kapatid":      "sibling",
    "kuya":         "older brother",
    "ate":          "older sister",
    "bunso":        "youngest",
    "lolo":         "grandfather",
    "lola":         "grandmother",
    "tito":         "uncle",
    "tita":         "aunt",
    "pinsan":       "cousin",
    "kaibigan":     "friend",
    "kapitbahay":   "neighbor",
    "guro":         "teacher",
    "estudyante":   "student",
    "doktor":       "doctor",
    "nars":         "nurse",
    "pulis":        "police",
    "sundalo":      "soldier",
    "manggagawa":   "worker",
    "aso":          "dog",
    "pusa":         "cat",
    "ibon":         "bird",
    "isda":         "fish",
    "bahay":        "house",
    "silid":        "room",
    "kusina":       "kitchen",
    "banyo":        "bathroom",
    "kwarto":       "room",
    "sala":         "living room",
    "eskwela":      "school",
    "opisina":      "office",
    "ospital":      "hospital",
    "simbahan":     "church",
    "palengke":     "market",
    "tindahan":     "store",
    "parke":        "park",
    "daan":         "road",
    "kotse":        "car",
    "bus":          "bus",
    "tren":         "train",
    "eroplano":     "airplane",
    "barko":        "ship",
    "pagkain":      "food",
    "tubig":        "water",
    "gatas":        "milk",
    "kape":         "coffee",
    "tsaa":         "tea",
    "bigas":        "rice",
    "kanin":        "rice",
    "tinapay":      "bread",
    "karne":        "meat",
    "gulay":        "vegetable",
    "prutas":       "fruit",
    "damit":        "clothes",
    "sapatos":      "shoes",
    "pera":         "money",
    "trabaho":      "work",
    "libro":        "book",
    "papel":        "paper",
    "lapis":        "pencil",
    "bolpen":       "pen",
    "telepono":     "phone",
    "kompyuter":    "computer",
    "langit":       "sky",
    "ulan":         "rain",
    "araw":         "sun",
    "buwan":        "moon",
    "bituin":       "star",
    "dagat":        "sea",
    "bundok":       "mountain",
    "ilog":         "river",
    "kagubatan":    "forest",
    "lupa":         "land",

    # ── Numbers ───────────────────────────────────────────────────────────────
    "isa":          "one",
    "dalawa":       "two",
    "tatlo":        "three",
    "apat":         "four",
    "lima":         "five",
    "anim":         "six",
    "pito":         "seven",
    "walo":         "eight",
    "siyam":        "nine",
    "sampu":        "ten",
    "labing-isa":   "eleven",
    "labing-dalawa":"twelve",
    "dalawampu":    "twenty",
    "tatlumpu":     "thirty",
    "isang daan":   "one hundred",
    "isang libo":   "one thousand",

    # ── Common phrases (multi-word keys matched first) ────────────────────────
    "mahal kita":       "I love you",
    "kumusta ka":       "how are you",
    "ayos lang":        "I am fine",
    "hindi ko alam":    "I do not know",
    "saan ka pupunta":  "where are you going",
    "ano ang pangalan mo": "what is your name",
    "ano ang pangalan niya": "what is his name",
    "gutom na ako":     "I am hungry",
    "pagod na ako":     "I am tired",
    "tulog na":         "time to sleep",
    "kain na":          "time to eat",
    "uwi na":           "go home now",
    "ingat ka":         "take care",
    "mahal kita":       "I love you",
    "miss kita":        "I miss you",
    "nandito ako":      "I am here",
    "nandoon siya":     "he is there",
    "tulad ng":         "like",
    "dahil sa":         "because of",
    "para sa":          "for",
    "kasama ko":        "with me",
    "kasama mo":        "with you",
}

# Sort phrase keys by length descending so longer phrases match before shorter ones
_SORTED_PHRASES = sorted(
    [(k, v) for k, v in WORD_MAP.items() if " " in k],
    key=lambda x: len(x[0]), reverse=True
)
_WORD_ONLY_MAP = {k: v for k, v in WORD_MAP.items() if " " not in k}


# ── Translator class ───────────────────────────────────────────────────────────

class TagalogTranslator:
    """
    Translates Tagalog text to English.

    Strategy:
      1. Try phrase-level substitution (multi-word patterns first).
      2. Fall back to word-by-word lookup for remaining tokens.
      3. Compute a confidence score: fraction of tokens that were matched.
      4. If confidence >= threshold (default 0.5), return the local translation.
      5. Otherwise (or if api_key is set and force_api=True) call OpenRouter.

    The API call uses the same key and model already configured in main.py
    so no additional credentials are needed.
    """

    TRANSLATE_PROMPT = (
        "Translate the following Tagalog text to English. "
        "Return ONLY the English translation, nothing else. "
        "Keep it natural and simple. Text: "
    )

    def __init__(self,
                 api_key: str = "",
                 model: str   = "google/gemma-3-4b-it:free",
                 confidence_threshold: float = 0.50):
        self.api_key   = api_key
        self.model     = model
        self.threshold = confidence_threshold
        self._lock     = threading.Lock()

    def translate(self, text: str) -> tuple[str, str]:
        """
        Translate Tagalog text to English.

        Returns (english_text, method) where method is "local" or "api".
        Raises no exceptions — returns ("", "error") on failure.
        """
        text = text.strip()
        if not text:
            return ("", "local")

        with self._lock:
            local, confidence = self._local_translate(text)

        print(f"[Tagalog] local='{local}'  confidence={confidence:.2f}")

        if confidence >= self.threshold:
            return (local, "local")

        # Low confidence — try API if key is available
        if self.api_key and self.api_key != "YOUR_OPENROUTER_API_KEY_HERE":
            api_result = self._api_translate(text)
            if api_result:
                return (api_result, "api")

        # Fall back to whatever the local translator produced
        return (local if local.strip() else text, "local")

    # ── Local translation ──────────────────────────────────────────────────────

    def _local_translate(self, text: str) -> tuple[str, float]:
        """
        Rule-based word/phrase lookup.
        Returns (translated_text, confidence_0_to_1).
        """
        lower = text.lower()
        lower = re.sub(r"[^\w\s\-]", " ", lower)

        # Step 1: replace known multi-word phrases first
        for phrase, english in _SORTED_PHRASES:
            if phrase in lower:
                lower = lower.replace(phrase, english)

        # Step 2: word-by-word lookup
        tokens = lower.split()
        if not tokens:
            return ("", 0.0)

        matched   = 0
        out_parts = []
        for tok in tokens:
            mapped = _WORD_ONLY_MAP.get(tok)
            if mapped is not None:
                matched += 1
                if mapped:          # skip empty-string particles
                    out_parts.append(mapped)
            else:
                out_parts.append(tok)   # keep as-is (may be English loanword)

        confidence = matched / len(tokens)
        result     = " ".join(out_parts).strip()

        # Basic cleanup: capitalise first letter
        if result:
            result = result[0].upper() + result[1:]

        return (result, confidence)

    # ── API translation ────────────────────────────────────────────────────────

    def _api_translate(self, text: str) -> str:
        """Call OpenRouter to translate. Returns empty string on failure."""
        prompt  = f"{self.TRANSLATE_PROMPT}{text}"
        payload = json.dumps({
            "model":    self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens":  120,
            "temperature": 0.3,    # lower temp = more literal translation
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload, method="POST",
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer":  "https://github.com/AVA-sign-avatar",
                "X-Title":       "AVA Sign Language Avatar",
            })
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
                result = data["choices"][0]["message"]["content"].strip()
                # Strip any quotes the model may wrap around the translation
                result = result.strip('"\'')
                print(f"[Tagalog] API translation: '{result}'")
                return result
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"[Tagalog] API HTTP {e.code}: {body}")
            return ""
        except Exception as e:
            print(f"[Tagalog] API error: {e}")
            return ""


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = TagalogTranslator()   # no API key — local only

    tests = [
        "Kumusta ka?",
        "Gusto ko kumain ng pagkain.",
        "Mahal kita.",
        "Saan ka pupunta bukas?",
        "Hindi ko alam ang sagot.",
        "Pagod na ako ngayon.",
        "Magandang umaga!",
        "Kailangan ko ng tulong.",
        "Ano ang pangalan mo?",
        "Maraming salamat po.",
        "Ingat ka palagi.",
        "Natulog na ang bata.",
    ]

    print("Tagalog → English  (local only)")
    print("=" * 55)
    for sentence in tests:
        english, method = t.translate(sentence)
        print(f"  TL: {sentence}")
        print(f"  EN: {english}  [{method}]")
        print()