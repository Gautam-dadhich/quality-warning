"""Complaint-aspect lexicon for appliance reviews.

Each aspect is a case-insensitive regular expression applied to
title + text. A review can match several aspects. The lexicon is kept
small and transparent on purpose so it can be printed as a table in the
paper and audited by a reader.
"""
from __future__ import annotations

import pandas as pd

ASPECTS: dict[str, str] = {
    # product stops functioning
    "stopped_working": (
        r"\b(?:stopped working|stop working|quit working|died|dead|"
        r"does(?:n'?t| not) work|did(?:n'?t| not) work|not working|"
        r"won'?t (?:turn on|start|work|power)|broke down|malfunction\w*|"
        r"defective|faulty|failed|failure)\b"
    ),
    "leak": r"\b(?:leak\w*|drip(?:s|ped|ping)?|flood\w*|water everywhere)\b",
    "noise": r"\b(?:nois[ey]\w*|noise|loud\w*|rattl\w*|squeak\w*|squeal\w*|grind(?:s|ing)?|buzz\w*|vibrat\w*)\b",
    "broken_damaged": r"\b(?:broke|broken|crack(?:s|ed|ing)?|shatter\w*|dented|damag\w*|snapped|bent)\b",
    "missing_parts": r"\b(?:missing|incomplete|not included|no instructions)\b",
    "fit_compat": (
        r"\b(?:does(?:n'?t| not) fit|did(?:n'?t| not) fit|wrong size|wrong part|"
        r"not compatible|incompatible|too (?:small|big|large|short))\b"
    ),
    # 'burn' restricted to verb forms so "burner" (stove part) does not match
    "heat_electrical": (
        r"\b(?:overheat\w*|burn(?:t|ed|ing|s)|smok(?:e|ed|ing|y)|melt(?:s|ed|ing)?|"
        r"spark(?:s|ed|ing)?|shock(?:ed|s)?|caught fire|on fire|short(?:ed)? out|trip(?:s|ped) the)\b"
    ),
    "return_refund": r"\b(?:return(?:ed|ing)?|refund\w*|replacement|send(?:ing)? (?:it )?back|sent (?:it )?back|warranty)\b",
}


def tag_aspects(title: pd.Series, text: pd.Series) -> pd.DataFrame:
    """Return a 0/1 DataFrame with one column per aspect (asp_<name>)."""
    doc = (title.fillna("") + " . " + text.fillna("")).str.lower()
    out = {}
    for name, pattern in ASPECTS.items():
        out[f"asp_{name}"] = doc.str.contains(pattern, regex=True, na=False).astype("int8")
    df = pd.DataFrame(out, index=title.index)
    df["asp_any"] = (df.sum(axis=1) > 0).astype("int8")
    return df
