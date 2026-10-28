"""Per-report language and structure profile (no report text is written out).

Output: reports/eda/report_profile.csv with one row per study:
  language (lingua), confidence, runner-up language, dominant script,
  length statistics, section markers, duplicate-report group, and flags for
  truncation risk under the old 1200-character prompt limit.
"""
import argparse
import hashlib
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from lingua import LanguageDetectorBuilder

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]

# Section headers seen across languages (case-insensitive, matched as words).
IMPRESSION_RE = re.compile(
    r"\b(impression|conclusion|conclusi[oó]n|impresi[oó]n|diagn[oó]stico|"
    r"conclusie|samenvatting|beoordeling|indruk|beurteilung|zusammenfassung|"
    r"sonu[cç]|kanaat|yorum|συμπέρασμα|γνωμάτευση|εντύπωση|заключение|"
    r"закључак|zaklju[cč]ak|mi[sš]ljenje|résumé|avis|opinion)\b", re.I)
FINDINGS_RE = re.compile(
    r"\b(findings|hallazgos|resultados|bevindingen|befund|bulgular|"
    r"ευρήματα|находки|nalaz|résultats|constatations)\b", re.I)
HISTORY_RE = re.compile(
    r"\b(history|clinical (information|indication)|indication|antecedentes|"
    r"indicaci[oó]n|klinische (inlichtingen|gegevens)|vraagstelling|"
    r"anamnese|fragestellung|klinik|[öo]yk[üu]|endikasyon|ιστορικό|"
    r"анамнез|anamneza|indication|renseignements)\b", re.I)
COMPARISON_RE = re.compile(
    r"\b(comparison|compared (with|to)|prior|previous|comparaci[oó]n|"
    r"previo|vergelijking|eerdere|voorgaande|vergleich|voraufnahme|"
    r"karşılaştırma|[öo]nceki|σύγκριση|προηγούμεν|сравнени|предыдущ|"
    r"prethodn|usporedb|comparaison|précédent)\b", re.I)


def dominant_script(text):
    counts = {}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        script = name.split(" ")[0]
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "NONE", 0.0
    script, c = max(counts.items(), key=lambda kv: kv[1])
    return script, c / sum(counts.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="D:/Coding/kaggle/RSNA/data/Raw_dicom/train.csv")
    ap.add_argument("--out", default="D:/Coding/kaggle/RSNA/reports/eda/report_profile.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

    rows = []
    for uid, rep in zip(df.StudyInstanceUID, df.Report.fillna("")):
        text = str(rep)
        confs = detector.compute_language_confidence_values(text)
        top = confs[0] if confs else None
        second = confs[1] if len(confs) > 1 else None
        script, script_frac = dominant_script(text)
        norm = re.sub(r"\s+", " ", text.strip().lower())
        rows.append({
            "StudyInstanceUID": uid,
            "lang": top.language.name.title() if top else "UNKNOWN",
            "lang_conf": round(top.value, 4) if top else 0.0,
            "lang2": second.language.name.title() if second else None,
            "lang2_conf": round(second.value, 4) if second else 0.0,
            "script": script,
            "script_frac": round(script_frac, 3),
            "n_chars": len(text),
            "n_words": len(text.split()),
            "n_lines": text.count("\n") + 1,
            "has_impression_hdr": bool(IMPRESSION_RE.search(text)),
            "has_findings_hdr": bool(FINDINGS_RE.search(text)),
            "has_history_hdr": bool(HISTORY_RE.search(text)),
            "has_comparison_ref": bool(COMPARISON_RE.search(text)),
            "has_date_token": "[DATE]" in text,
            "truncated_at_1200": len(text) > 1200,
            "frac_lost_at_1200": round(max(0, len(text) - 1200) / max(len(text), 1), 3),
            "non_ascii_frac": round(sum(ord(c) > 127 for c in text) / max(len(text), 1), 3),
            "replacement_char": "\ufffd" in text,
            "mojibake_suspect": bool(re.search(r"Ã.|Â.|Ð.|Ñ.", text)),
            "report_hash": hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16],
            "is_gold": bool(df.loc[df.StudyInstanceUID == uid, TARGETS].notna().all(axis=1).iloc[0]),
        })
    out = pd.DataFrame(rows)
    dup_sizes = out.groupby("report_hash").StudyInstanceUID.transform("size")
    out["dup_group_size"] = dup_sizes
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    print(out.lang.value_counts().to_string())
    print("\nscript:", out.script.value_counts().to_dict())
    print("low-confidence (<0.5):", int((out.lang_conf < 0.5).sum()))
    print("gold languages:", out[out.is_gold].lang.value_counts().to_dict())
    print("duplicate report groups (size>1):", int((out.dup_group_size > 1).sum()),
          "rows in", out[out.dup_group_size > 1].report_hash.nunique(), "groups")
    print("truncated at 1200 chars:", int(out.truncated_at_1200.sum()),
          f"({out.truncated_at_1200.mean():.1%}); gold: {out[out.is_gold].truncated_at_1200.mean():.1%}")


if __name__ == "__main__":
    main()
