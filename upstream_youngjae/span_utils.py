"""Locate each A/G/D factor's T5 token span inside the AGD caption.

Two definitions, and they must agree:

verbatim
    find the phrase string from `caption.json["phrases"]` inside the AGD caption
    and take the tokens overlapping it.  Simple, but the captions are not always
    literal concatenations -- 33/100 samples have at least one phrase that is
    split or reworded (D in 24/100), which silently yields an empty span.

diff
    tokenise the AGD caption and the caption with factor f deleted (GD / AD / AG)
    and take the tokens of AGD that no longer align.  This is exactly the set of
    tokens the causal reference `v_remove(f) = f(z_t, AGD\\f)` removes, so the
    thing we blind and the thing we compare against are the same by construction.

`factor_spans` uses `diff` and reports how far the two disagree.
"""
import difflib

REMOVE = {"A": "GD", "G": "AD", "D": "AG"}


def _encode(tok, text, max_len):
    return tok(text, padding="max_length", max_length=max_len, truncation=True,
               return_offsets_mapping=True)


def verbatim_span(enc, prompt, phrase):
    c0 = prompt.find(phrase)
    if c0 < 0:
        return None
    c1 = c0 + len(phrase)
    return [j for j, (a, b) in enumerate(enc["offset_mapping"])
            if enc["attention_mask"][j] == 1 and b > a and a < c1 and b > c0]


def diff_span(tok, ids_full, alt_text):
    """Indices of `ids_full` that do not survive the alignment to `alt_text`."""
    ids_alt = tok(alt_text, add_special_tokens=True)["input_ids"]
    sm = difflib.SequenceMatcher(a=ids_full, b=ids_alt, autojunk=False)
    keep = set()
    for op, i1, i2, _, _ in sm.get_opcodes():
        if op == "equal":
            keep.update(range(i1, i2))
    return [i for i in range(len(ids_full)) if i not in keep]


def factor_spans(tok, captions, phrases, text_field="text_gtD", max_len=512,
                 factors=("A", "G", "D"), verbose=True):
    """Return {A,G,D,base,pad} -> token indices, plus a report dict."""
    prompt = captions["AGD"][text_field]
    enc = _encode(tok, prompt, max_len)
    n_real = sum(enc["attention_mask"])
    ids_full = enc["input_ids"][:n_real]

    spans, report = {}, {"prompt": prompt, "n_real_tokens": n_real, "factors": {}}
    for f in factors:
        d = diff_span(tok, ids_full, captions[REMOVE[f]][text_field])
        v = verbatim_span(enc, prompt, phrases[f])
        spans[f] = d
        report["factors"][f] = {
            "n_diff": len(d),
            "n_verbatim": None if v is None else len(v),
            "verbatim_found": v is not None,
            "jaccard": (None if v is None else
                        round(len(set(d) & set(v)) / max(len(set(d) | set(v)), 1), 3)),
            "text": tok.decode([ids_full[i] for i in d]) if d else "",
        }
    # A token can fall in two factors' diff spans when removing one factor also
    # reworded another (the captions are not strict concatenations: 24/100
    # samples).  Such tokens go to a separate `shared` bucket -- kept out of
    # every factor so the write decomposition stays additive and blinding one
    # factor cannot blind another -- and the size of that bucket is reported so
    # heavily entangled samples can be filtered out of the statistics.
    from collections import Counter
    count = Counter(i for f in factors for i in spans[f])
    shared = sorted(i for i, n in count.items() if n > 1)
    for f in factors:
        spans[f] = [i for i in spans[f] if count[i] == 1]
        report["factors"][f]["n_exclusive"] = len(spans[f])
    spans["shared"] = shared
    taken = set(count)
    report["overlap"] = len(shared)
    report["overlap_frac"] = round(len(shared) / max(len(taken), 1), 3)
    spans["base"] = [i for i in range(n_real) if i not in taken]
    spans["pad"] = list(range(n_real, max_len))
    if verbose:
        for f in factors:
            r = report["factors"][f]
            print(f"  span {f}: n={r['n_exclusive']:3d} of {r['n_diff']} "
                  f"(verbatim {r['n_verbatim']}, jaccard {r['jaccard']})  {r['text'][:70]}")
        print(f"  span base: n={len(spans['base'])}  shared: {len(shared)} "
              f"({report['overlap_frac']:.0%} of factor tokens)")
    return spans, report
