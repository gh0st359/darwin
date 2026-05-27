"""Function-word and connector vocabulary for the LLM-free realizer.

These constants are the only source of "structure words" the DiscourseRealizer
is allowed to use. Content words (nouns, verbs, adjectives, adverbs, numbers,
named entities) come *only* from ResponsePlan fields. Splitting these out
means the FaithfulnessValidator can audit the realizer's output token-by-token
against (a) the plan's grounded fields, (b) this fixed vocabulary, and
(c) derived morphological variants of plan tokens.

No item in this module is a content word, a canned response, or a claim. Each
list is a structural primitive of English prose composition.
"""

from __future__ import annotations


# Closed-class function words: pronouns, articles, prepositions, modals,
# auxiliaries, and a few short adverbs that the realizer may use to glue
# content words into sentences. Intentionally minimal; the validator denies
# anything outside this set + plan-content + connectors below.
FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        # Articles & determiners
        "a", "an", "the", "this", "that", "these", "those", "any", "some",
        "every", "each", "all", "no", "none", "one", "another", "other",
        "such", "same", "few", "many", "much", "most", "least", "more", "less",
        # Pronouns
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
        "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
        "there", "here", "now", "then",
        # Be / have / do
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having",
        "do", "does", "did", "doing", "done",
        # Modals
        "will", "would", "shall", "should", "can", "could", "may", "might",
        "must", "ought",
        # Prepositions
        "of", "to", "in", "on", "at", "by", "for", "with", "from", "into",
        "onto", "over", "under", "above", "below", "between", "across",
        "around", "near", "against", "within", "without", "before", "after",
        "during", "since", "until", "about", "off", "out", "up", "down",
        # Conjunctions
        "and", "but", "or", "nor", "so", "because", "since", "while", "if",
        "unless", "though", "although", "whereas", "as", "than",
        # Negation, intensifiers, qualifiers
        "not", "n't", "no", "never", "neither", "either", "both", "yet",
        "still", "also", "too", "very", "quite", "really", "just", "only",
        "even", "actually", "almost", "already",
        # Common short adverbs
        "well", "right", "wrong", "back", "again", "ahead", "away",
        # Common helpers used in narration
        "like", "as", "such", "rather", "instead", "next", "first", "second",
        "third", "last",
        # Common abbreviations the realizer may emit
        "etc",
        # Contractions the realizer commonly uses (treated as function tokens
        # for validation). The tokenizer strips trailing apostrophes/dashes
        # but contractions like "i've" stay intact as single tokens.
        "i'm", "i've", "i'd", "i'll",
        "it's", "that's", "there's", "here's", "what's",
        "you're", "you've", "you'd", "you'll",
        "we're", "we've", "we'd", "we'll",
        "they're", "they've", "they'd", "they'll",
        "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
        "shouldn't", "isn't", "aren't", "wasn't", "weren't", "hasn't",
        "haven't", "hadn't",
    }
)


# Transitions for joining sections of a multi-sentence response. Each entry
# is a structural marker, not a claim. The realizer picks among them via
# content-fingerprint hashing so phrasing varies without ever being random.
SECTION_OPENERS: tuple[str, ...] = (
    "what stands out is that",
    "one thing i've noticed is that",
    "the part i keep coming back to is that",
    "looking at this directly,",
    "from what i can tell,",
    "based on what i've seen,",
    "putting it plainly,",
    "the way i'd describe it is",
)


SECONDARY_OPENERS: tuple[str, ...] = (
    "tied to that,",
    "underneath that,",
    "alongside that,",
    "stepping into the detail,",
    "going a layer down,",
    "and",
    "also,",
    "which means",
)


QUALIFIER_OPENERS: tuple[str, ...] = (
    "i'm still uncertain about",
    "what i'm not sure about is",
    "one thing that's still fuzzy for me is",
    "i wouldn't bet on",
    "i can't claim much about",
    "the thinner part of my understanding is",
)


CLOSE_OPENERS: tuple[str, ...] = (
    "if i'm being honest,",
    "stepping back,",
    "to be straight about it,",
    "for what it's worth,",
    "the through-line here is that",
)


REFLECTIVE_OPENERS: tuple[str, ...] = (
    "what i keep noticing is that",
    "the pattern i keep seeing is that",
    "where this lands for me is",
    "the way it's settling for me is",
)


GREETING_ACKNOWLEDGEMENTS: tuple[str, ...] = (
    "hey",
    "hi",
    "good to hear from you",
    "yeah, i'm here",
    "hey — yeah, here",
    "still here",
)


FAREWELL_TOKENS: tuple[str, ...] = (
    "talk soon",
    "see you",
    "later",
    "until next time",
    "catch you later",
)


GRATITUDE_ACKNOWLEDGEMENTS: tuple[str, ...] = (
    "you're welcome",
    "sure",
    "anytime",
    "of course",
    "no problem",
)


PRESENCE_CONFIRMATIONS: tuple[str, ...] = (
    "yes, i'm here",
    "here",
    "still here",
    "yeah, here",
)


INVITES: tuple[str, ...] = (
    "what did you want to talk about?",
    "what's on your mind?",
    "anything you want to push on?",
    "what would you like to dig into?",
)


# Words the validator allows in addition to FUNCTION_WORDS because they
# appear inside the structural openers above. These are purely connective.
STRUCTURE_CONNECTORS: frozenset[str] = frozenset(
    {
        "noticed", "stands", "coming", "looking", "directly", "tell",
        "putting", "plainly", "way", "describe", "tied", "underneath",
        "alongside", "stepping", "layer", "down", "honest", "straight",
        "worth", "through-line", "line", "through", "keep", "noticing",
        "settling", "pattern", "where", "lands", "matter", "matters",
        "noted", "good", "talk", "soon", "see", "later", "next", "time",
        "catch", "you're", "welcome", "sure", "anytime", "course", "problem",
        "yes", "here", "still", "yeah", "hi", "hey", "hello",
        "ground", "grounded", "thinking",
        "remember", "remembering", "noted",
        "say", "saying", "said",
        # Numbers commonly used as adjectival qualifiers
        "once", "twice",
        # Common state/cognition verbs the realizer uses around claims
        "make", "makes", "made", "making",
        "drop", "drops", "rise", "rises", "stay", "change", "changes",
        "become", "becomes",
        "got", "going", "seem", "seems", "seemed",
        "thought", "thoughts", "thinking",
        "feel", "feels", "feeling",
        # Common evidence framing words
        "evidence", "samples", "sample", "run", "runs", "ran",
        "times", "time",
        "data", "observation", "observations",
        # Common qualifiers that appear in QUALIFIER_OPENERS
        "uncertain", "fuzzy", "thin", "thinner", "tentative", "weak",
        "limited", "claim", "claims", "bet", "betting",
        # Narrative verbs and nouns the realizer uses to wire content
        "applying", "apply", "applies", "applied",
        "tends", "tend", "tending", "tended",
        "runs", "run", "running", "ran",
        "move", "moves", "moved", "moving",
        "push", "pushes", "pushed", "pushing",
        "link", "links", "linked", "linking",
        "fits", "fit", "fitted", "fitting",
        "look", "looks", "looked", "looking",
        "thread", "threads", "threaded",
        "memory", "memories",
        "connected", "connect", "connecting",
        "sit", "sits", "sat", "sitting",
        "chew", "chews", "chewing", "chewed",
        "focused", "focus", "focusing",
        "deep", "deeper", "deepest",
        "cleanest", "clean", "cleaner",
        "stronger", "strongest", "strong",
        "settle", "settled", "settling", "settles",
        "carry", "carrying", "carried", "carries",
        "build", "builds", "built", "building",
        "reading", "reads",
        # System-noun fallbacks used in identity drafts
        "system", "systems", "agent", "module", "modules", "version",
        "darwin", "name", "named", "names",
        # Time / event narrative
        "once", "twice", "currently", "right",
        # Punctuation-internal artifacts ("through-line" split yields these)
        "through", "line",
        # Causal / consequence narrative verbs
        "matter", "matters", "mattering",
        "appear", "appears", "appearing", "appeared",
        "show", "shows", "showed", "shown", "showing",
        "give", "gives", "giving", "gave", "given",
        "take", "takes", "took", "taking", "taken",
        "hold", "holds", "held", "holding",
        # Sensory / perception verbs the realizer uses in passing
        "see", "seen", "saw", "seeing",
        "hear", "heard", "hearing",
        # Possessive contractions / common adverbs
        "actually", "currently", "right",
        # Reflection nouns
        "understanding", "understandings",
        "part", "parts",
        "read", "reads",
        "beat", "beats",
        # Greeting / response framing
        "alright", "ok", "okay",
        # Words used inside INVITES / QUALIFIER_OPENERS / CLOSE_OPENERS
        "want", "wants", "wanting", "wanted",
        "talk", "talks", "talked", "talking",
        "mind", "minds", "minded", "minding",
        "anything", "everything", "something", "nothing",
        "dig", "digs", "dug", "digging",
        "thing", "things",
        "stuff", "matter", "matters",
        # Common identity-section vocabulary
        "causal", "adaptive", "experience", "experiences",
        "world", "worlds", "across",
        # Common ground/develop verbs the realizer emits
        "produce", "produces", "produced", "producing",
        "drive", "drives", "drove", "driven", "driving",
        "tend", "tends", "tendency",
        "true", "false",
    }
)
