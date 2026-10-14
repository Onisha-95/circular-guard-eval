"""
Automated Knowledge Extraction Engine — Rules R01-R11.
No spaCy dependency. Pure regex + structure signals.
In the original prototype tutor these facts were written to a graph database; in this release
exp/build_kbs.py records them to JSONL files instead.
"""
import re


def auto_generate_negation(claim: str) -> str:
    """Rule-based negation generator — no LLM."""
    rules = [
        (r'\bbefore\b',    'after'),
        (r'\bafter\b',     'before'),
        (r'\bTrue\b',      'False'),
        (r'\bFalse\b',     'True'),
        (r'\balways\b',    'never'),
        (r'\bnever\b',     'always'),
        (r'\bmust\b',      'must not'),
        (r'\brequires\b',  'does not require'),
        (r'\b(is|are|was|were)\b',                      r'\1 NOT'),
        (r'\b(checks|runs|executes|returns|stops|starts|evaluates|increments)\b', r'\1 NOT'),
    ]
    for pattern, replacement in rules:
        negated = re.sub(pattern, replacement, claim, count=1)
        if negated != claim:
            return negated
    return f"It is NOT the case that: {claim}"


STOP_WORDS = {
    "the","and","for","this","that","with","are","was","you","can","has",
    "its","not","but","from","they","will","have","been","more","also",
    "when","then","what","how","why","which","each","any","all","use",
    "used","using","one","two","get","set","new","old","first","last",
    "next","into","out","about","over","just","like","some","other",
}

# Generic/structural labels that must NEVER become concept nodes
_INVALID_STARTS = (
    "slide", "chapter", "overview", "summary", "introduction",
    "conclusion", "agenda", "outline", "review", "table of",
    "figure", "notes:", "ocr_text",
)
_INVALID_EXACT = {
    "question", "answer", "solution", "definition", "recall",
    "objectives", "today", "next", "last", "first", "final",
    "page", "section", "module", "unit", "week", "day",
    "part", "topic", "item", "point", "step",
}

def _valid(term: str) -> bool:
    if not term: return False
    t = term.strip().lower()
    if len(t) < 3 or len(t) > 60: return False
    if t in STOP_WORDS: return False
    if t in _INVALID_EXACT: return False
    if t.replace(" ", "").replace("_", "").isdigit(): return False
    if all(c in ".,;:!?()[]{}\'\"-_" for c in t): return False
    # Reject generic structural headings (slide_N, overview, chapter 3, etc.)
    if any(t.startswith(s) for s in _INVALID_STARTS): return False
    # Reject date strings like "february 10 math 1112 sec. 54"
    import re as _re
    if _re.match(r"^(january|february|march|april|may|june|july|august|"
                 r"september|october|november|december)\b", t):
        return False
    # Reject strings with too few alphabetic characters (mostly numbers/symbols)
    alpha_ratio = sum(c.isalpha() for c in t) / max(len(t), 1)
    if alpha_ratio < 0.5: return False
    return True


class KnowledgeExtractionEngine:

    def __init__(self, kg_client):
        self.kg = kg_client

    def extract_and_write(self, parsed_doc: dict) -> dict:
        source      = parsed_doc["source_file"]
        course_id   = parsed_doc.get("course_id")
        module_name = parsed_doc.get("module_name")
        counts = {"concepts": 0, "facts": 0, "relationships": 0, "examples": 0}

        # R01 — bold terms -> concept nodes
        for term in parsed_doc.get("all_bold_terms", []):
            if _valid(term):
                if self.kg.merge_concept(term.lower().strip(), source):
                    counts["concepts"] += 1

        for section in parsed_doc["sections"]:
            heading = section["heading"].strip()
            content = section["content"].strip()
            loc     = section["location"]

            # R01b — heading as concept
            if _valid(heading):
                if self.kg.merge_concept(heading.lower(), source, loc):
                    counts["concepts"] += 1

            # R01c — section bold terms
            for term in section.get("bold_terms", []):
                if _valid(term):
                    if self.kg.merge_concept(term.lower().strip(), source, loc):
                        counts["concepts"] += 1

            # R02 — ALL-CAPS words as concept candidates
            for w in content.split():
                cleaned = w.strip(".,;:()[]!?\"'")
                if cleaned.isupper() and 2 < len(cleaned) < 25 and _valid(cleaned):
                    self.kg.merge_concept(cleaned.lower(), source, loc)

            # R03 — IS_A: "X is a/an/defined as Y"
            for m in re.finditer(
                r'([A-Za-z][\w ]{2,28}?) (?:is a|is an|is defined as|refers to) ([A-Za-z][\w ]{2,28})',
                content, re.IGNORECASE):
                a, b = m.group(1).strip().lower(), m.group(2).strip().lower()
                if _valid(a) and _valid(b):
                    self.kg.merge_concept(a, source, loc)
                    self.kg.merge_concept(b, source, loc)
                    if self.kg.merge_relationship(a, "IS_A", b, source): counts["relationships"] += 1

            # R04 — CONSISTS_OF
            for m in re.finditer(
                r'([A-Za-z][\w ]{1,25}) (?:consists of|is made up of|contains|includes|has) ([\w ,]+)',
                content, re.IGNORECASE):
                src = m.group(1).strip().lower()
                for part in [p.strip().lower() for p in m.group(2).split(",") if p.strip()]:
                    if _valid(part) and _valid(src):
                        self.kg.merge_concept(src, source, loc)
                        self.kg.merge_concept(part, source, loc)
                        if self.kg.merge_relationship(src, "CONSISTS_OF", part, source): counts["relationships"] += 1

            # R05 — REQUIRES
            for m in re.finditer(
                r'([A-Za-z][\w ]{1,25}) (?:requires|needs|depends on|must have) ([A-Za-z][\w ]{1,25})',
                content, re.IGNORECASE):
                a, b = m.group(1).strip().lower(), m.group(2).strip().lower()
                if _valid(a) and _valid(b):
                    self.kg.merge_concept(a, source, loc); self.kg.merge_concept(b, source, loc)
                    if self.kg.merge_relationship(a, "REQUIRES", b, source): counts["relationships"] += 1

            # R06 — LEADS_TO
            for m in re.finditer(
                r'([A-Za-z][\w ]{1,25}) (?:leads to|results in|produces|causes) ([A-Za-z][\w ]{1,25})',
                content, re.IGNORECASE):
                a, b = m.group(1).strip().lower(), m.group(2).strip().lower()
                if _valid(a) and _valid(b):
                    self.kg.merge_concept(a, source, loc); self.kg.merge_concept(b, source, loc)
                    if self.kg.merge_relationship(a, "LEADS_TO", b, source): counts["relationships"] += 1

            # R07 — sentences as facts for heading concept
            if _valid(heading):
                for sent in self._sentences(content):
                    if self.kg.create_fact(heading.lower(), sent, source, loc): counts["facts"] += 1

            # R08 — "For example..." -> Example
            if _valid(heading):
                for m in re.finditer(r'[Ff]or example[,:]?\s+(.+?)(?:[.!]|$)', content):
                    ex = m.group(1).strip()
                    if len(ex) > 10:
                        self.kg.create_example(heading.lower(), ex, source, loc)
                        counts["examples"] += 1

            # R09 — Note/Important -> HIGH priority fact
            if _valid(heading):
                for m in re.finditer(
                    r'(?:Note|Important|Remember|Key point|Warning)[:\s]+(.+?)(?:[.!]|$)',
                    content, re.IGNORECASE):
                    txt = m.group(1).strip()
                    if len(txt) > 10:
                        if self.kg.create_fact(heading.lower(), txt, source, loc, "HIGH"): counts["facts"] += 1

            # R10 — Explicit negation signals -> store as negation-flagged fact
            if _valid(heading):
                for m in re.finditer(
                    r'(?:NOT|never|incorrect|wrong|cannot|do not|does not)\s+(.+?)(?:[.!]|$)',
                    content, re.IGNORECASE):
                    neg = m.group(0).strip()
                    if len(neg) > 15:
                        self.kg.create_fact(heading.lower(), neg, source, loc, "HIGH", is_negation=True)

            # R11 — CONTRASTS_WITH: "X vs Y" / "unlike X"
            if _valid(heading):
                for m in re.finditer(
                    r'(?:unlike|compared to|versus|vs\.?)\s+([A-Za-z][\w ]{1,25})',
                    content, re.IGNORECASE):
                    other = m.group(1).strip().lower()
                    if _valid(other):
                        self.kg.merge_concept(other, source, loc)
                        if self.kg.merge_relationship(heading.lower(), "CONTRASTS_WITH", other, source):
                            counts["relationships"] += 1

        return counts

    def _sentences(self, text: str) -> list:
        """Split text into meaningful sentence-length chunks."""
        raw = re.split(r'(?<=[.!?])\s+', text)
        bullets = re.split(r'\
|•|·|–|^\s*-\s', text)
        out = []
        for s in raw + bullets:
            s = s.strip().strip("•·–-").strip()
            if 20 < len(s) < 350:
                out.append(s)
        return out