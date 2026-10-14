"""
fetch_progmiscon.py -- independent, expert-curated misconceptions for the course KBs.

ProgMiscon (Chiodini et al., ITiCSE 2021; https://progmiscon.org) lists programming
misconceptions observed in real classrooms. Every entry has an INCORRECT statement
(what students believe) and a CORRECT statement (written by the curators). We use the
Java and Python entries as a second, independently written test set:
  incorrect statements  -> errors the guard should flag
  correct statements    -> correct statements it should not flag (false alarms)

The site states no licence, so the release ships this script and the entry names,
not the statement text. Run once, on a machine with internet access:

  python3 exp/fetch_progmiscon.py            # -> exp/data/progmiscon_errors.json, progmiscon_true.json
"""
import html
import json
import os
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://progmiscon.org"
LANGS = ("Java", "Python")
UA = {"User-Agent": "Mozilla/5.0 (research; misconception evaluation; contact: course project)"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


class Lines(HTMLParser):
    """Visible text as a list of lines; block-level tags start a new line."""
    BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "td", "th",
             "section", "header", "footer", "dt", "dd", "span", "a", "code", "strong", "em", "b"}
    SKIP = {"script", "style", "nav", "noscript"}

    def __init__(self):
        super().__init__()
        self.lines, self.buf, self.skip = [], [], 0

    def flush(self):
        t = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        if t:
            self.lines.append(t)
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK and tag not in {"span", "a", "code", "strong", "em", "b"}:
            self.flush()

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK and tag not in {"span", "a", "code", "strong", "em", "b"}:
            self.flush()

    def handle_data(self, data):
        if not self.skip:
            self.buf.append(data)


def text_lines(page):
    p = Lines()
    p.feed(page)
    p.flush()
    return [html.unescape(l) for l in p.lines]


def after(lines, label, start=0):
    """First non-empty line after the line equal to `label` (case-insensitive)."""
    for i in range(start, len(lines)):
        if lines[i].strip().lower() == label.lower():
            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j].strip():
                    return lines[j].strip(), j
    return None, start


STOP = {"expressible in", "related misconceptions", "other languages", "language", "correction",
        "origin", "symptoms", "value", "literature references", "find misconceptions by...", "about"}


def concepts(lines):
    for i, l in enumerate(lines):
        if l.strip().lower() == "concepts":
            out = []
            for m in lines[i + 1:i + 12]:
                if m.strip().lower() in STOP:
                    break
                out.append(m.strip())
            return out
    return []


def main():
    index = get(BASE + "/misconceptions/")
    names = sorted(set(re.findall(r'/misconceptions/(%s)/([A-Za-z0-9]+)/' % "|".join(LANGS), index)))
    print(f"{len(names)} Java/Python entries listed")
    entries, failed = [], []
    for lang, name in names:
        url = f"{BASE}/misconceptions/{lang}/{name}/"
        try:
            lines = text_lines(get(url))
        except Exception as e:                       # noqa: BLE001
            failed.append((name, str(e)))
            continue
        wrong, j = after(lines, "Incorrect")
        right, _ = after(lines, "Correct", j)
        if not wrong or not right or wrong == right:
            failed.append((name, "statements not found"))
            continue
        entries.append({"lang": lang, "name": name, "url": url, "incorrect": wrong, "correct": right,
                        "concepts": concepts(lines), "observed": any(l.strip() == "Observed" for l in lines[:40])})
        time.sleep(0.5)                              # be polite to the site
    os.makedirs(os.path.join(ROOT, "exp", "data"), exist_ok=True)
    json.dump(entries, open(os.path.join(ROOT, "exp", "data", "progmiscon_raw.json"), "w"), indent=1)
    errs, trues = [], []
    for k, e in enumerate(entries):
        concept = (", ".join(e["concepts"]) or e["name"]).lower()
        errs.append({"id": f"pe{k:03d}", "pair": f"pt{k:03d}", "concept": concept, "text": e["incorrect"],
                     "kind": "error", "source": f"ProgMiscon/{e['lang']}/{e['name']}"})
        trues.append({"id": f"pt{k:03d}", "pair": f"pe{k:03d}", "concept": concept, "text": e["correct"],
                      "kind": "true", "source": f"ProgMiscon/{e['lang']}/{e['name']}"})
    json.dump(errs, open(os.path.join(ROOT, "exp", "data", "progmiscon_errors.json"), "w"), indent=1)
    json.dump(trues, open(os.path.join(ROOT, "exp", "data", "progmiscon_true.json"), "w"), indent=1)
    print(f"Saved {len(entries)} pairs ({sum(e['observed'] for e in entries)} marked 'Observed'); failed: {len(failed)}")
    for n, why in failed[:10]:
        print("  skipped", n, why)
    for e in entries[:3]:
        print(f"\n  {e['lang']}/{e['name']}\n    incorrect: {e['incorrect']}\n    correct:   {e['correct']}\n    concepts:  {e['concepts']}")
    if not entries:
        sys.exit("No entries parsed -- send the output to Claude.")


if __name__ == "__main__":
    main()
