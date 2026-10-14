"""
build_fever.py -- a public second testbed from FEVER (Thorne et al., NAACL 2018).

Why FEVER: it is public (Wikipedia, CC BY-SA), it gives both error types the
paper needs without writing any items, and its errors sit at a third distance
from the KB:
  circular      rule negation of a stored fact             (closest to the KB)
  FEVER REFUTES a human annotator's edit of a source sentence
  course set    misconceptions written from topics           (furthest)
and its SUPPORTS claims are true statements for the false-contradiction rate.

What it builds (all from the same Wikipedia sentences):
  exp/kbs/fever_sentence.jsonl  one fact per sentence (FEVER's own segmentation)
  exp/kbs/fever_fragment.jsonl  windows of 3 sentences   (the fragmented-KB arm)
  exp/kbs/fever_llm_only.jsonl  KCA proposals, no gate   (needs an LLM)
  exp/kbs/fever_kca.jsonl       KCA proposals + NLI gate (needs an LLM + RoBERTa)
  exp/data/fever_errors.json    150 REFUTES claims (concept = evidence page)
  exp/data/fever_true.json      150 SUPPORTS claims
The KCA arms use the KCA code unchanged (core/hybrid_extractor.py): each window
of sentences is treated like a slide whose title is the page title.

USAGE (on the machine with internet):
  curl -LO https://fever.ai/download/fever/shared_task_dev.jsonl
  curl -LO https://fever.ai/download/fever/wiki-pages.zip          # ~1.7 GB, read in place
  export OPENROUTER_API_KEY=...                                     # or OPENAI_API_KEY
  python3 exp/build_fever.py --dev shared_task_dev.jsonl --wiki wiki-pages.zip \
         --llm openai/gpt-4o-mini@https://openrouter.ai/api/v1
  (use --llm gpt-4o-mini to call OpenAI directly; --skip-kca builds only the model-free arms)
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

from exp.kb_store import new_fact, write_kb  # noqa: E402
from exp.build_kbs import kb_stats  # noqa: E402

PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}",
       "``": '"', "''": '"', "--": "-"}


def clean(s):
    for k, v in PTB.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+([,.;:!?%)\]])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    s = re.sub(r"\s+'s\b", "'s", s)
    return re.sub(r"\s+", " ", s).strip()


def page_title(pid):
    return clean(pid.replace("_", " ")).lower()


# ------------------------------------------------------------------ sampling
def sample_claims(dev_path, n_each, seed):
    """REFUTES and SUPPORTS claims whose every evidence set uses one page, one
    claim per page, so each item has exactly one 'concept'."""
    by_label = {"REFUTES": [], "SUPPORTS": []}
    for line in open(dev_path, encoding="utf-8"):
        d = json.loads(line)
        if d.get("label") not in by_label:
            continue
        pages = {e[2] for es in d["evidence"] for e in es if e[2]}
        if len(pages) != 1:
            continue
        sent_ids = sorted({e[3] for es in d["evidence"] for e in es if e[3] is not None})
        by_label[d["label"]].append({"claim": d["claim"], "page": pages.pop(), "sent_ids": sent_ids,
                                     "fever_id": d["id"]})
    rng = random.Random(seed)
    out, used = {}, set()
    for lab in ("REFUTES", "SUPPORTS"):
        pool = by_label[lab][:]
        rng.shuffle(pool)
        pick = []
        for c in pool:
            if c["page"] in used:
                continue
            pick.append(c)
            used.add(c["page"])
            if len(pick) == n_each:
                break
        out[lab] = pick
    return out


def load_pages(wiki_zip, wanted):
    """Stream the wiki-pages archive and keep only the pages we need."""
    pages = {}
    with zipfile.ZipFile(wiki_zip) as z:
        bad = 0
        for name in sorted(z.namelist()):
            base = os.path.basename(name)
            # skip macOS metadata (__MACOSX/, ._foo.jsonl) added when the zip was re-packed on a Mac
            if not name.endswith(".jsonl") or "__MACOSX" in name or base.startswith("._"):
                continue
            with z.open(name) as fh:
                for raw in fh:
                    if not raw.strip():          # the dump contains blank lines
                        continue
                    try:
                        d = json.loads(raw)
                    except json.JSONDecodeError:
                        bad += 1
                        continue
                    if d.get("id") in wanted:
                        lines = {}
                        for row in (d.get("lines") or "").split("\n"):
                            parts = row.split("\t")
                            if len(parts) >= 2 and parts[0].isdigit() and parts[1].strip():
                                lines[int(parts[0])] = clean(parts[1])
                        pages[d["id"]] = lines
            if len(pages) == len(wanted):
                break
    if bad:
        print(f"  skipped {bad} unreadable lines in {wiki_zip}")
    return pages


# -------------------------------------------------------------- KCA via LLM
def make_llm(spec):
    """spec: model or model@base_url (OpenAI-compatible, e.g. OpenRouter). Cached."""
    model, _, base = spec.partition("@")
    from openai import OpenAI
    if base and "openrouter" in base:
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    else:
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY (or OPENAI_API_KEY).")
    client = OpenAI(api_key=key, base_url=base or None)
    os.makedirs(os.path.join(ROOT, "exp", "cache"), exist_ok=True)
    cpath = os.path.join(ROOT, "exp", "cache", "kca_fever__" + re.sub(r"[^A-Za-z0-9._-]+", "_", model) + ".jsonl")
    cache = {}
    if os.path.exists(cpath):
        for l in open(cpath, encoding="utf-8"):
            d = json.loads(l)
            cache[d["k"]] = d["v"]

    def call(prompt):
        k = hashlib.sha1((model + "\x00" + prompt).encode()).hexdigest()
        if k in cache:
            return cache[k]
        from exp.nli_backends import with_retries
        r = with_retries(lambda: client.chat.completions.create(
            model=model, temperature=0, messages=[{"role": "user", "content": prompt}]))
        v = r.choices[0].message.content or ""
        cache[k] = v
        with open(cpath, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"k": k, "v": v}) + "\n")
        return v
    return call


def kca_arms(pages, keep, llm_call, window=4, gate_model="roberta-large-mnli", threshold=0.65):
    from core.hybrid_extractor import (SlideChunk, propose_claims_llm, is_junk_claim, dedupe_claims,
                                       build_candidate_premises, NLIValidator)
    if os.environ.get("EXP_TEST_STUB_GATE"):          # tests only: no model download
        class _G:
            def validate_best(self, claim, prem):
                t = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
                sc = [len(t(claim) & t(p)) / max(len(t(claim)), 1) for p in prem]
                i = max(range(len(sc)), key=lambda j: sc[j])
                return sc[i] >= threshold, sc[i], i
        gate = _G()
    else:
        gate = NLIValidator(model_name=gate_model, entail_threshold=threshold)
    acc, allp = [], []
    for pid, lines in pages.items():
        title = page_title(pid)
        ids = keep[pid]
        for w in range(0, len(ids), window):
            sents = [lines[i] for i in ids[w:w + window] if i in lines]
            if not sents:
                continue
            ch = SlideChunk(slide_num=w // window + 1, title=title,
                            bullets=[{"text": s, "bold": False} for s in sents], source_file=pid)
            props = dedupe_claims([p for p in propose_claims_llm(ch, llm_call) if not is_junk_claim(p["claim"])])
            prem = build_candidate_premises(ch)
            for p in props:
                ok, prob, _ = gate.validate_best(p["claim"], prem)
                rec = (title, p["claim"], pid, f"window {w // window + 1}")
                allp.append(rec)
                if ok:
                    acc.append(rec + (round(prob, 4),))
    llm_only = [new_fact(t, c, f, loc, "fever_llm_only") for t, c, f, loc in allp]
    kca = [new_fact(t, c, f, loc, "fever_kca", nli_entail_prob=pr) for t, c, f, loc, pr in acc]
    return llm_only, kca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True, help="shared_task_dev.jsonl")
    ap.add_argument("--wiki", required=True, help="wiki-pages.zip (read in place)")
    ap.add_argument("--n", type=int, default=150, help="claims per label")
    ap.add_argument("--lead", type=int, default=12, help="sentences kept from the top of each page")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm", default=None, help="model or model@base_url for the KCA arms")
    ap.add_argument("--skip-kca", action="store_true")
    args = ap.parse_args()

    claims = sample_claims(args.dev, args.n, args.seed)
    wanted = {c["page"] for lab in claims.values() for c in lab}
    print(f"Sampled {len(claims['REFUTES'])} REFUTES + {len(claims['SUPPORTS'])} SUPPORTS claims "
          f"over {len(wanted)} pages; reading pages ...")
    pages = load_pages(args.wiki, wanted)
    print(f"  found {len(pages)}/{len(wanted)} pages")

    # KB text: the page's lead sentences plus every evidence sentence
    keep = {}
    ev = {}
    for lab in claims.values():
        for c in lab:
            ev.setdefault(c["page"], set()).update(c["sent_ids"])
    for pid, lines in pages.items():
        keep[pid] = sorted(set(i for i in lines if i < args.lead) | (ev.get(pid, set()) & set(lines)))

    sent, frag = [], []
    for pid, lines in pages.items():
        t = page_title(pid)
        ids = keep[pid]
        for i in ids:
            sent.append(new_fact(t, lines[i], pid, f"sentence {i}", "fever_sentence"))
        for w in range(0, len(ids), 3):
            chunk = " ".join(lines[i] for i in ids[w:w + 3])
            frag.append(new_fact(t, chunk, pid, f"window {w // 3 + 1}", "fever_fragment"))
    arms = {"fever_sentence": sent, "fever_fragment": frag}
    if not args.skip_kca:
        if not args.llm:
            raise SystemExit("--llm is required for the KCA arms (or pass --skip-kca).")
        arms["fever_llm_only"], arms["fever_kca"] = kca_arms(pages, keep, make_llm(args.llm))

    stats_path = os.path.join(ROOT, "exp", "kbs", "kb_stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    for arm, facts in arms.items():
        write_kb(os.path.join(ROOT, "exp", "kbs", f"{arm}.jsonl"), facts)
        stats[arm] = kb_stats(facts)
        print(f"  {arm:16s} {stats[arm]}")
    json.dump(stats, open(stats_path, "w"), indent=2)

    def items(lab, kind, prefix):
        return [{"id": f"{prefix}{i:03d}", "concept": page_title(c["page"]), "text": clean(c["claim"]),
                 "kind": kind, "fever_id": c["fever_id"], "in_dev55": False, "dup_cluster": f"f{prefix}{i:03d}"}
                for i, c in enumerate(claims[lab]) if c["page"] in pages]
    json.dump(items("REFUTES", "error", "e"), open(os.path.join(ROOT, "exp", "data", "fever_errors.json"), "w"),
              indent=1, ensure_ascii=False)
    json.dump(items("SUPPORTS", "true", "t"), open(os.path.join(ROOT, "exp", "data", "fever_true.json"), "w"),
              indent=1, ensure_ascii=False)
    print("Saved exp/data/fever_errors.json and fever_true.json")


if __name__ == "__main__":
    main()
