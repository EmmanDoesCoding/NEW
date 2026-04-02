"""
nlp_processor.py
================
Converts English text → ASL gloss word list.

No external dependencies — pure Python stdlib only.

ASL grammar rules applied (in order):
  1.  Expand contractions         (don't → do not)
  2.  Strip punctuation
  3.  Tokenize
  4.  Drop filler words            (a, an, the, be-verbs, conjunctions)
  5.  Normalize verb tenses        (eating → EAT, went → GO)
  6.  Map English → ASL gloss      (hi → HELLO, thanks → THANK-YOU)
  7.  Move time expressions first  (ASL topic-comment structure)
  8.  Question handling:
        WH-question  → move WH word to end  (ASL non-manual marker)
        Y/N question → append YES-NO
  9.  Deduplicate adjacent identical words  (THANK-YOU YOU → THANK-YOU)
  10. process_partial: skip last incomplete word while user still typing

Usage:
    nlp = NLPProcessor()
    gloss = nlp.process("Hi, want to eat?")
    # → ["HELLO", "YOU", "EAT", "WANT"]

    # While user is typing (real-time):
    partial = nlp.process_partial("I want to go to sch")
    # → ["I", "WANT", "GO"]   (skips incomplete "sch")
"""

import re
import threading


# ─────────────────────────────────────────────────────────────────────────────
# Lookup tables
# ─────────────────────────────────────────────────────────────────────────────

# Contractions to expand before any other processing
CONTRACTIONS: dict[str, str] = {
    "i'm":      "i am",
    "you're":   "you are",
    "he's":     "he is",
    "she's":    "she is",
    "it's":     "it is",
    "we're":    "we are",
    "they're":  "they are",
    "i've":     "i have",
    "you've":   "you have",
    "we've":    "we have",
    "they've":  "they have",
    "i'll":     "i will",
    "you'll":   "you will",
    "he'll":    "he will",
    "she'll":   "she will",
    "we'll":    "we will",
    "they'll":  "they will",
    "i'd":      "i would",
    "you'd":    "you would",
    "he'd":     "he would",
    "she'd":    "she would",
    "we'd":     "we would",
    "they'd":   "they would",
    "can't":    "cannot",
    "won't":    "will not",
    "don't":    "do not",
    "doesn't":  "does not",
    "didn't":   "did not",
    "isn't":    "is not",
    "aren't":   "are not",
    "wasn't":   "was not",
    "weren't":  "were not",
    "haven't":  "have not",
    "hasn't":   "has not",
    "hadn't":   "had not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't":"should not",
    "mustn't":  "must not",
    "let's":    "let us",
    "that's":   "that is",
    "there's":  "there is",
    "what's":   "what is",
    "where's":  "where is",
    "who's":    "who is",
    "how's":    "how is",
    "wanna":    "want to",
    "gonna":    "going to",
    "gotta":    "got to",
    "kinda":    "kind of",
    "sorta":    "sort of",
}

# Words dropped entirely — not signed in ASL
DROP_WORDS: set[str] = {
    # Articles
    "a", "an", "the",
    # Linking / auxiliary be-verbs (ASL omits these)
    "is", "are", "am", "was", "were", "be", "been", "being",
    # Auxiliary do (question form handled separately)
    "do", "does", "did",
    # Infinitive marker
    "to",
    # Conjunctions (ASL uses spatial grammar instead)
    "and", "but", "or", "so", "yet", "nor",
    # Filler
    "very", "just", "really", "quite", "rather", "also",
    # Prepositions that have no direct sign and just pad sentences
    "of", "in", "on", "at", "by", "as", "up",
}

# WH question words (moved to END of sentence in ASL)
WH_WORDS: set[str] = {
    "what", "where", "when", "who", "why", "how", "which", "whose",
}

# Time / temporal expressions (moved to START of sentence in ASL)
TIME_WORDS: set[str] = {
    "today", "tomorrow", "yesterday", "now", "later", "soon", "early",
    "morning", "afternoon", "evening", "night", "tonight",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "always", "never", "sometimes", "often", "usually", "already",
    "before", "after", "ago", "recently", "soon", "later",
    "daily", "weekly", "monthly", "yearly",
    "past", "future", "recently",
}

# English word/phrase → ASL gloss mapping
# Handles tense variants, synonyms, and common phrases
WORD_MAP: dict[str, str] = {

    # ── Greetings ──
    "hello":        "HELLO",
    "hi":           "HELLO",
    "hey":          "HELLO",
    "greetings":    "HELLO",
    "goodbye":      "GOODBYE",
    "bye":          "GOODBYE",
    "farewell":     "GOODBYE",
    "good":         "GOOD",
    "morning":      "MORNING",
    "afternoon":    "AFTERNOON",
    "evening":      "EVENING",
    "night":        "NIGHT",
    "goodnight":    "GOOD-NIGHT",

    # ── Courtesy ──
    "please":       "PLEASE",
    "thank":        "THANK-YOU",
    "thanks":       "THANK-YOU",
    "thankyou":     "THANK-YOU",
    "welcome":      "WELCOME",
    "sorry":        "SORRY",
    "apologize":    "SORRY",
    "excuse":       "EXCUSE-ME",
    "pardon":       "EXCUSE-ME",

    # ── Yes / No / Negation ──
    "yes":          "YES",
    "yeah":         "YES",
    "yep":          "YES",
    "yup":          "YES",
    "no":           "NO",
    "nope":         "NO",
    "not":          "NOT",
    "never":        "NEVER",
    "nothing":      "NOTHING",
    "nobody":       "NOBODY",
    "nowhere":      "NOWHERE",
    "cannot":       "CANNOT",

    # ── Pronouns ──
    "i":            "I",
    "me":           "ME",
    "my":           "MY",
    "mine":         "MINE",
    "myself":       "MYSELF",
    "you":          "YOU",
    "your":         "YOUR",
    "yours":        "YOURS",
    "yourself":     "YOURSELF",
    "he":           "HE",
    "him":          "HIM",
    "his":          "HIS",
    "himself":      "HIMSELF",
    "she":          "SHE",
    "her":          "HER",
    "hers":         "HERS",
    "herself":      "HERSELF",
    "it":           "IT",
    "its":          "ITS",
    "itself":       "ITSELF",
    "we":           "WE",
    "us":           "US",
    "our":          "OUR",
    "ours":         "OURS",
    "ourselves":    "OURSELVES",
    "they":         "THEY",
    "them":         "THEM",
    "their":        "THEIR",
    "theirs":       "THEIRS",
    "themselves":   "THEMSELVES",

    # ── Common verbs (normalised to base form) ──
    "want":         "WANT",  "wants":    "WANT",  "wanted":  "WANT",
    "need":         "NEED",  "needs":    "NEED",  "needed":  "NEED",
    "like":         "LIKE",  "likes":    "LIKE",  "liked":   "LIKE",
    "love":         "LOVE",  "loves":    "LOVE",  "loved":   "LOVE",
    "hate":         "HATE",  "hates":    "HATE",  "hated":   "HATE",
    "have":         "HAVE",  "has":      "HAVE",  "had":     "HAVE",
    "get":          "GET",   "gets":     "GET",   "got":     "GET",
    "give":         "GIVE",  "gives":    "GIVE",  "gave":    "GIVE",
    "take":         "TAKE",  "takes":    "TAKE",  "took":    "TAKE",
    "make":         "MAKE",  "makes":    "MAKE",  "made":    "MAKE",
    "go":           "GO",    "goes":     "GO",    "went":    "GO",    "going": "GO",
    "come":         "COME",  "comes":    "COME",  "came":    "COME",  "coming":"COME",
    "see":          "SEE",   "sees":     "SEE",   "saw":     "SEE",   "seen": "SEE",
    "look":         "LOOK",  "looks":    "LOOK",  "looked":  "LOOK",
    "watch":        "WATCH", "watches":  "WATCH", "watched": "WATCH",
    "hear":         "HEAR",  "hears":    "HEAR",  "heard":   "HEAR",
    "know":         "KNOW",  "knows":    "KNOW",  "knew":    "KNOW",
    "think":        "THINK", "thinks":   "THINK", "thought": "THINK",
    "feel":         "FEEL",  "feels":    "FEEL",  "felt":    "FEEL",
    "say":          "SAY",   "says":     "SAY",   "said":    "SAY",
    "tell":         "TELL",  "tells":    "TELL",  "told":    "TELL",
    "ask":          "ASK",   "asks":     "ASK",   "asked":   "ASK",
    "help":         "HELP",  "helps":    "HELP",  "helped":  "HELP",
    "work":         "WORK",  "works":    "WORK",  "worked":  "WORK",
    "play":         "PLAY",  "plays":    "PLAY",  "played":  "PLAY",
    "eat":          "EAT",   "eats":     "EAT",   "ate":     "EAT",   "eating":"EAT",
    "drink":        "DRINK", "drinks":   "DRINK", "drank":   "DRINK",
    "sleep":        "SLEEP", "sleeps":   "SLEEP", "slept":   "SLEEP",
    "walk":         "WALK",  "walks":    "WALK",  "walked":  "WALK",
    "run":          "RUN",   "runs":     "RUN",   "ran":     "RUN",
    "sit":          "SIT",   "sits":     "SIT",   "sat":     "SIT",
    "stand":        "STAND", "stands":   "STAND", "stood":   "STAND",
    "wait":         "WAIT",  "waits":    "WAIT",  "waited":  "WAIT",
    "buy":          "BUY",   "buys":     "BUY",   "bought":  "BUY",
    "pay":          "PAY",   "pays":     "PAY",   "paid":    "PAY",
    "learn":        "LEARN", "learns":   "LEARN", "learned": "LEARN",
    "teach":        "TEACH", "teaches":  "TEACH", "taught":  "TEACH",
    "meet":         "MEET",  "meets":    "MEET",  "met":     "MEET",
    "talk":         "TALK",  "talks":    "TALK",  "talked":  "TALK",
    "call":         "CALL",  "calls":    "CALL",  "called":  "CALL",
    "write":        "WRITE", "writes":   "WRITE", "wrote":   "WRITE",
    "read":         "READ",  "reads":    "READ",
    "drive":        "DRIVE", "drives":   "DRIVE", "drove":   "DRIVE",
    "live":         "LIVE",  "lives":    "LIVE",  "lived":   "LIVE",
    "finish":       "FINISH","finishes":  "FINISH","finished":"FINISH",
    "start":        "START", "begins":   "START", "began":   "START",
    "try":          "TRY",   "tries":    "TRY",   "tried":   "TRY",
    "use":          "USE",   "uses":     "USE",   "used":    "USE",
    "find":         "FIND",  "finds":    "FIND",  "found":   "FIND",
    "forget":       "FORGET","forgets":  "FORGET","forgot":  "FORGET",
    "remember":     "REMEMBER","remembers":"REMEMBER","remembered":"REMEMBER",
    "understand":   "UNDERSTAND","understands":"UNDERSTAND","understood":"UNDERSTAND",
    "decide":       "DECIDE","decides":  "DECIDE","decided": "DECIDE",
    "agree":        "AGREE", "agrees":   "AGREE", "agreed":  "AGREE",
    "disagree":     "DISAGREE",
    "hurt":         "HURT",  "hurts":    "HURT",
    "lose":         "LOSE",  "loses":    "LOSE",  "lost":    "LOSE",
    "win":          "WIN",   "wins":     "WIN",   "won":     "WIN",
    "show":         "SHOW",  "shows":    "SHOW",  "showed":  "SHOW",
    "bring":        "BRING", "brings":   "BRING", "brought": "BRING",
    "leave":        "LEAVE", "leaves":   "LEAVE", "left":    "LEAVE",
    "arrive":       "ARRIVE","arrives":  "ARRIVE","arrived": "ARRIVE",
    "visit":        "VISIT", "visits":   "VISIT", "visited": "VISIT",
    "change":       "CHANGE","changes":  "CHANGE","changed": "CHANGE",
    "open":         "OPEN",  "opens":    "OPEN",  "opened":  "OPEN",
    "close":        "CLOSE", "closes":   "CLOSE", "closed":  "CLOSE",

    # ── Modals ──
    "can":          "CAN",
    "cannot":       "CANNOT",
    "could":        "COULD",
    "will":         "WILL",
    "would":        "WOULD",
    "should":       "SHOULD",
    "must":         "MUST",
    "may":          "MAY",
    "might":        "MIGHT",

    # ── Adjectives / descriptors ──
    "good":         "GOOD",
    "bad":          "BAD",
    "great":        "GREAT",
    "wonderful":    "WONDERFUL",
    "terrible":     "TERRIBLE",
    "happy":        "HAPPY",
    "sad":          "SAD",
    "angry":        "ANGRY",
    "scared":       "SCARED",
    "afraid":       "SCARED",
    "surprised":    "SURPRISED",
    "excited":      "EXCITED",
    "bored":        "BORED",
    "tired":        "TIRED",
    "sick":         "SICK",
    "fine":         "FINE",
    "okay":         "FINE",
    "ok":           "FINE",
    "beautiful":    "BEAUTIFUL",
    "ugly":         "UGLY",
    "big":          "BIG",
    "large":        "BIG",
    "small":        "SMALL",
    "little":       "SMALL",
    "tall":         "TALL",
    "short":        "SHORT",
    "fast":         "FAST",
    "quick":        "FAST",
    "slow":         "SLOW",
    "hot":          "HOT",
    "warm":         "WARM",
    "cold":         "COLD",
    "cool":         "COOL",
    "new":          "NEW",
    "old":          "OLD",
    "young":        "YOUNG",
    "easy":         "EASY",
    "hard":         "HARD",
    "difficult":    "HARD",
    "important":    "IMPORTANT",
    "different":    "DIFFERENT",
    "same":         "SAME",
    "wrong":        "WRONG",
    "right":        "RIGHT",
    "correct":      "CORRECT",
    "true":         "TRUE",
    "false":        "FALSE",
    "ready":        "READY",
    "busy":         "BUSY",
    "free":         "FREE",
    "full":         "FULL",
    "empty":        "EMPTY",
    "late":         "LATE",
    "early":        "EARLY",
    "far":          "FAR",
    "near":         "NEAR",
    "alone":        "ALONE",
    "together":     "TOGETHER",
    "lost":         "LOST",
    "safe":         "SAFE",
    "careful":      "CAREFUL",
    "funny":        "FUNNY",
    "serious":      "SERIOUS",
    "quiet":        "QUIET",
    "loud":         "LOUD",
    "clean":        "CLEAN",
    "dirty":        "DIRTY",
    "hungry":       "HUNGRY",
    "thirsty":      "THIRSTY",

    # ── Quantifiers ──
    "much":         "MUCH",
    "many":         "MANY",
    "more":         "MORE",
    "most":         "MOST",
    "less":         "LESS",
    "few":          "FEW",
    "some":         "SOME",
    "any":          "ANY",
    "all":          "ALL",
    "none":         "NONE",
    "both":         "BOTH",
    "enough":       "ENOUGH",
    "again":        "AGAIN",
    "another":      "ANOTHER",
    "every":        "EVERY",
    "each":         "EACH",

    # ── Demonstratives / location ──
    "this":         "THIS",
    "that":         "THAT",
    "these":        "THESE",
    "those":        "THOSE",
    "here":         "HERE",
    "there":        "THERE",
    "where":        "WHERE",
    "inside":       "INSIDE",
    "outside":      "OUTSIDE",
    "above":        "ABOVE",
    "below":        "BELOW",
    "left":         "LEFT",
    "right":        "RIGHT",
    "front":        "FRONT",
    "behind":       "BEHIND",
    "between":      "BETWEEN",

    # ── Question words ──
    "what":         "WHAT",
    "when":         "WHEN",
    "who":          "WHO",
    "why":          "WHY",
    "how":          "HOW",
    "which":        "WHICH",

    # ── Time ──
    "today":        "TODAY",
    "tomorrow":     "TOMORROW",
    "yesterday":    "YESTERDAY",
    "now":          "NOW",
    "later":        "LATER",
    "soon":         "SOON",
    "before":       "BEFORE",
    "after":        "AFTER",
    "always":       "ALWAYS",
    "sometimes":    "SOMETIMES",
    "often":        "OFTEN",
    "never":        "NEVER",
    "already":      "ALREADY",
    "still":        "STILL",
    "yet":          "YET",
    "ago":          "AGO",
    "during":       "DURING",

    # ── Numbers (words) ──
    "one":          "1",
    "two":          "2",
    "three":        "3",
    "four":         "4",
    "five":         "5",
    "six":          "6",
    "seven":        "7",
    "eight":        "8",
    "nine":         "9",
    "ten":          "10",
    "first":        "FIRST",
    "second":       "SECOND",
    "third":        "THIRD",
    "last":         "LAST",

    # ── Places ──
    "home":         "HOME",
    "house":        "HOUSE",
    "school":       "SCHOOL",
    "hospital":     "HOSPITAL",
    "store":        "STORE",
    "shop":         "STORE",
    "church":       "CHURCH",
    "work":         "WORK",
    "office":       "OFFICE",
    "city":         "CITY",
    "town":         "TOWN",
    "country":      "COUNTRY",
    "world":        "WORLD",

    # ── People ──
    "person":       "PERSON",
    "people":       "PEOPLE",
    "man":          "MAN",
    "woman":        "WOMAN",
    "boy":          "BOY",
    "girl":         "GIRL",
    "child":        "CHILD",
    "children":     "CHILDREN",
    "baby":         "BABY",
    "adult":        "ADULT",
    "family":       "FAMILY",
    "mother":       "MOTHER",
    "mom":          "MOTHER",
    "mum":          "MOTHER",
    "father":       "FATHER",
    "dad":          "FATHER",
    "parents":      "PARENTS",
    "brother":      "BROTHER",
    "sister":       "SISTER",
    "son":          "SON",
    "daughter":     "DAUGHTER",
    "husband":      "HUSBAND",
    "wife":         "WIFE",
    "partner":      "PARTNER",
    "friend":       "FRIEND",
    "enemy":        "ENEMY",
    "teacher":      "TEACHER",
    "student":      "STUDENT",
    "doctor":       "DOCTOR",
    "nurse":        "NURSE",
    "police":       "POLICE",
    "boss":         "BOSS",

    # ── Things / objects ──
    "food":         "FOOD",
    "water":        "WATER",
    "milk":         "MILK",
    "coffee":       "COFFEE",
    "tea":          "TEA",
    "bread":        "BREAD",
    "money":        "MONEY",
    "car":          "CAR",
    "bus":          "BUS",
    "train":        "TRAIN",
    "plane":        "PLANE",
    "phone":        "PHONE",
    "book":         "BOOK",
    "paper":        "PAPER",
    "pen":          "PEN",
    "key":          "KEY",
    "door":         "DOOR",
    "window":       "WINDOW",
    "table":        "TABLE",
    "chair":        "CHAIR",
    "bed":          "BED",
    "clothes":      "CLOTHES",
    "shoes":        "SHOES",
    "bag":          "BAG",
    "computer":     "COMPUTER",

    # ── Misc ──
    "name":         "NAME",
    "age":          "AGE",
    "time":         "TIME",
    "day":          "DAY",
    "week":         "WEEK",
    "month":        "MONTH",
    "year":         "YEAR",
    "hour":         "HOUR",
    "minute":       "MINUTE",
    "number":       "NUMBER",
    "color":        "COLOR",
    "colour":       "COLOR",
    "language":     "LANGUAGE",
    "sign":         "SIGN",
    "help":         "HELP",
    "problem":      "PROBLEM",
    "idea":         "IDEA",
    "question":     "QUESTION",
    "answer":       "ANSWER",
    "information":  "INFORMATION",
    "news":         "NEWS",
    "story":        "STORY",
    "reason":       "REASON",
    "way":          "WAY",
    "thing":        "THING",
    "place":        "PLACE",
    "with":         "WITH",
    "without":      "WITHOUT",
    "for":          "FOR",
    "about":        "ABOUT",
    "because":      "BECAUSE",
    "if":           "IF",
    "maybe":        "MAYBE",
    "perhaps":      "MAYBE",
    "more":         "MORE",
    "less":         "LESS",
}


# ─────────────────────────────────────────────────────────────────────────────
# NLPProcessor
# ─────────────────────────────────────────────────────────────────────────────

class NLPProcessor:
    """
    Converts English sentences to ASL gloss word lists.
    Rule-based — no external libraries required.
    Thread-safe.
    """

    def __init__(self):
        self._lock = threading.Lock()

    # ── Public ────────────────────────────────────────────────────────────────

    def process(self, text: str) -> list[str]:
        """
        Convert a complete English sentence to ASL gloss.
        "Hi, want to eat?" → ["HELLO", "YOU", "EAT", "WANT"]
        """
        with self._lock:
            return self._run(text, partial=False)

    def process_partial(self, text: str) -> list[str]:
        """
        Convert an incomplete sentence (user still typing).
        Skips the last word if it looks incomplete (no trailing space/punct).
        Safe to call on every keystroke — very fast.
        """
        with self._lock:
            return self._run(text, partial=True)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _run(self, raw: str, partial: bool) -> list[str]:
        text = raw.strip()
        if not text:
            return []

        # Track whether original input ends with a space (word complete signal)
        trailing_space = raw.endswith((" ", "	"))

        # ── Step 1: detect question type BEFORE any stripping ────────────────
        is_question  = text.rstrip().endswith("?")
        lower        = text.lower()
        wh_found     = None
        for w in WH_WORDS:
            if re.search(rf'\b{w}\b', lower):
                wh_found = w
                break
        is_wh = is_question and wh_found is not None
        is_yn = is_question and not is_wh

        # ── Step 2: expand contractions + common phrases ────────────────────
        text = self._expand(text.lower())
        # Multi-word phrases → single gloss token before splitting
        text = re.sub(r'thank you',    'THANK-YOU', text)
        text = re.sub(r'excuse me',    'EXCUSE-ME', text)
        text = re.sub(r'good morning', 'GOOD-MORNING', text)
        text = re.sub(r'good night',   'GOOD-NIGHT', text)
        text = re.sub(r'good afternoon','GOOD-AFTERNOON', text)
        text = re.sub(r'good evening', 'GOOD-EVENING', text)
        text = re.sub(r'i love you',   'I LOVE YOU', text)
        text = re.sub(r'how are you',  'HOW YOU', text)

        # ── Step 3: strip punctuation (preserve hyphens in gloss tokens) ────────
        # First protect already-mapped tokens with hyphens
        text = re.sub(r"[^\w\s\-]", " ", text)   # keep hyphens
        text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)  # remove stray hyphens

        # ── Step 4: tokenize ──────────────────────────────────────────────────
        tokens = text.split()

        # ── Step 5: partial — drop last token if word is still being typed ──────
        if partial and tokens and not trailing_space:
            tokens = tokens[:-1]   # last word has no space after it = incomplete
        if not tokens:
            return []

        # ── Step 6: map each token to ASL gloss + drop fillers ────────────────
        gloss = []
        for tok in tokens:
            if tok in DROP_WORDS:
                continue
            mapped = WORD_MAP.get(tok)
            if mapped:
                gloss.append(mapped)
            else:
                # Unknown word — keep as-is uppercased
                # (sign_mapper will try fuzzy match later)
                gloss.append(tok.upper())

        # ── Step 7: move time expressions to front ────────────────────────────
        time_part  = [w for w in gloss if w.lower() in TIME_WORDS]
        other_part = [w for w in gloss if w.lower() not in TIME_WORDS]
        gloss = time_part + other_part

        # ── Step 8: question restructuring ────────────────────────────────────
        if is_wh and wh_found:
            wh_gloss = WORD_MAP.get(wh_found, wh_found.upper())
            # Remove WH word from wherever it ended up
            gloss = [w for w in gloss if w != wh_gloss]
            # Append at end (ASL non-manual marker raised eyebrows at end)
            gloss.append(wh_gloss)
        elif is_yn:
            gloss.append("YES-NO")

        # ── Step 9: dedup adjacent identical tokens ───────────────────────────
        gloss = self._dedup(gloss)

        return gloss

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _expand(text: str) -> str:
        """Replace contractions with full forms."""
        for contraction, expanded in CONTRACTIONS.items():
            text = re.sub(
                rf'\b{re.escape(contraction)}\b',
                expanded,
                text
            )
        return text

    @staticmethod
    def _dedup(tokens: list[str]) -> list[str]:
        """Remove consecutive duplicate tokens."""
        if not tokens:
            return []
        result = [tokens[0]]
        for tok in tokens[1:]:
            if tok != result[-1]:
                result.append(tok)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    nlp = NLPProcessor()

    tests = [
        # (input, expected_output_comment)
        ("Hi, want to eat?",                   "HELLO WANT EAT YES-NO  (implicit YOU dropped — added by user if needed)"),
        ("I love you",                         "I LOVE YOU"),
        ("What is your name?",                 "YOUR NAME WHAT"),
        ("I don't want to go to school today", "TODAY I NOT WANT GO SCHOOL"),
        ("She is very happy",                  "SHE HAPPY"),
        ("Can you help me please?",            "CAN YOU HELP ME PLEASE YES-NO"),
        ("I'm going to the store tomorrow",    "TOMORROW I GO STORE"),
        ("Where do you live?",                 "YOU LIVE WHERE"),
        ("Thank you very much",                "THANK-YOU"),
        ("I am tired and sick",                "I TIRED SICK"),
        ("Do you understand me?",              "YOU UNDERSTAND ME YES-NO"),
        ("I need water now",                   "NOW I NEED WATER"),
        ("She doesn't like it",               "SHE NOT LIKE IT"),
        ("When does the bus arrive?",          "BUS ARRIVE WHEN"),
        ("I will meet you tomorrow",           "TOMORROW I WILL MEET YOU"),
    ]

    print("NLP Processor — Full Sentence Tests")
    print("=" * 60)
    for text, comment in tests:
        result = nlp.process(text)
        print(f"  IN:  {text}")
        print(f"  OUT: {' '.join(result)}")
        print(f"  EXP: {comment}")
        print()

    print()
    print("Partial (real-time typing) tests:")
    print("=" * 60)
    partials = [
        "I want to go to sch",    # incomplete last word
        "tomorrow I need ",       # trailing space = last word complete
        "What is your ",          # WH question building up
        "hel",                    # single incomplete word
    ]
    for p in partials:
        result = nlp.process_partial(p)
        print(f"  typing: {repr(p)}")
        print(f"  gloss:  {result}")
        print()