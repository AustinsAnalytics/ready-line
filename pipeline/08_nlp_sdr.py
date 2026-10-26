"""Stage 8 - text mining 600k+ FAA Service Difficulty Reports.

A. Auto-coding: predict the JASC system chapter from the mechanic's free-text write-up
   (temporal split: train 2016-2023, test 2024-2025), with a leakage ablation that masks
   maintenance-manual references (e.g. "IAW AMM 33-51-38" literally contains the chapter).
B. Data-quality reconciliation: confident disagreements between model and coded chapter.
C. Root-cause discovery on engine write-ups: NMF topic model -> failure-mode taxonomy,
   risk-weighted Pareto (volume x flight-interruption rate) and corrective-action mix.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB

from common import ENGINE_CHAPTERS, JASC_CHAPTERS, PROCESSED, SEED, WAREHOUSE, Timer, export

REF = re.compile(r"\b(?:AMM|SRM|IPC|CMM|TSM|FIM|WDM|MM|ATA|CH|CHAPTER|TASK|IAW|PER|REF)?\s*\d{2}-\d{2}(?:-\d{2,3})*(?:-\d{3})*\b")
ATA_WORD = re.compile(r"\bATA\s*\d{2,4}\b")
IDS = re.compile(r"\b[A-Z]{0,2}\d{5,}\b")
TOKEN = r"(?u)\b[A-Z][A-Z0-9/]+\b"


def mask(text: pd.Series) -> pd.Series:
    t = text.str.upper().str.replace(REF, " MANUALREF ", regex=True)
    return t.str.replace(ATA_WORD, " MANUALREF ", regex=True)


def scrub(t: str, n: int = 260) -> str:
    """Display-safe snippet: hide employee / document id numbers, trim length."""
    t = IDS.sub("#####", t)
    return (t[: n - 1] + "…") if len(t) > n else t


def load() -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    df = con.execute("""
        SELECT report_id, year, jasc, chapter, is_engine, interruption, stage_group, action,
               aircraft_make, aircraft_model, part_name, text
        FROM fact_sdr
    """).df()
    con.close()
    return df


# --------------------------------------------------------------------------- A/B
def classifier(df: pd.DataFrame) -> dict:
    counts = df[df.year <= 2023].chapter.value_counts()
    keep = counts[counts >= 300].index
    d = df[df.chapter.isin(keep)].copy()
    d["masked"] = mask(d.text)
    tr, te = d[d.year <= 2023], d[d.year >= 2024]
    print(f"  {len(keep)} chapters; train {len(tr):,}, test {len(te):,}")

    results, models = {}, {}
    for variant, col in (("raw", "text"), ("masked", "masked")):
        vec = TfidfVectorizer(token_pattern=TOKEN, lowercase=False, ngram_range=(1, 2), min_df=5,
                              max_features=200_000, sublinear_tf=True, dtype=np.float32)
        Xtr = vec.fit_transform(tr[col].str.upper())
        Xte = vec.transform(te[col].str.upper())
        nb = ComplementNB(alpha=0.3).fit(Xtr, tr.chapter)
        sgd = SGDClassifier(loss="modified_huber", alpha=1e-6, max_iter=25, tol=None,
                            random_state=SEED, n_jobs=-1).fit(Xtr, tr.chapter)
        for name, m in (("Naive Bayes", nb), ("Linear (SGD)", sgd)):
            pred = m.predict(Xte)
            proba = m.predict_proba(Xte)
            top3 = np.argsort(-proba, axis=1)[:, :3]
            cls = m.classes_
            top3_acc = np.mean([y in cls[t] for y, t in zip(te.chapter.values, top3)])
            results[f"{variant}|{name}"] = {
                "variant": variant, "model": name, "accuracy": accuracy_score(te.chapter, pred),
                "macro_f1": f1_score(te.chapter, pred, average="macro"),
                "weighted_f1": f1_score(te.chapter, pred, average="weighted"), "top3": top3_acc}
            print(f"  {variant:6s} {name:12s} acc {results[f'{variant}|{name}']['accuracy']:.3f} "
                  f"macroF1 {results[f'{variant}|{name}']['macro_f1']:.3f} top3 {top3_acc:.3f}")
        models[variant] = (vec, sgd, Xte)

    # Headline model = masked linear model (it cannot cheat off manual references).
    vec, sgd, Xte = models["masked"]
    proba = sgd.predict_proba(Xte)
    pred = sgd.classes_[proba.argmax(1)]
    conf = proba.max(1)
    te = te.assign(pred=pred, conf=conf)

    per_class = []
    f1s = f1_score(te.chapter, te.pred, labels=sgd.classes_, average=None)
    for c, f in zip(sgd.classes_, f1s):
        per_class.append({"chapter": c, "title": JASC_CHAPTERS.get(c, c), "f1": f, "n": int((te.chapter == c).sum())})
    per_class.sort(key=lambda r: -r["n"])

    top = [r["chapter"] for r in per_class[:12]]
    sub = te[te.chapter.isin(top)]
    cm = confusion_matrix(sub.chapter, sub.pred, labels=top, normalize="true")

    # Highest-weight terms per chapter (what the model "reads").
    terms = np.array(vec.get_feature_names_out())
    top_terms = {}
    for i, c in enumerate(sgd.classes_):
        if c in top:
            w = sgd.coef_[i]
            top_terms[c] = [t for t in terms[np.argsort(-w)[:40]] if t != "MANUALREF"][:8]

    # B. Reconciliation: confident disagreements = candidate miscodes for analyst review.
    dis = te[(te.pred != te.chapter) & (te.conf >= 0.9)]
    rate = len(dis) / len(te)
    flows = (dis.groupby(["chapter", "pred"]).size().sort_values(ascending=False).head(10).reset_index(name="n"))
    examples = []
    for _, r in dis.sample(min(400, len(dis)), random_state=SEED).iterrows():
        if 80 < len(r.text) < 300 and r.chapter in JASC_CHAPTERS and r.pred in JASC_CHAPTERS:
            examples.append({"coded": r.chapter, "coded_title": JASC_CHAPTERS[r.chapter], "pred": r.pred,
                             "pred_title": JASC_CHAPTERS[r.pred], "conf": r.conf, "text": scrub(r.text)})
        if len(examples) >= 24:
            break
    return {"results": list(results.values()), "per_class": per_class, "confusion": {"labels": top, "matrix": cm.tolist(),
            "titles": [JASC_CHAPTERS.get(c, c) for c in top]}, "top_terms": top_terms,
            "miscode": {"rate": rate, "n": len(dis), "n_test": len(te), "flows": flows.to_dict("records"),
                        "examples": examples},
            "n_train": len(tr), "n_test": len(te), "n_classes": len(keep)}


# ----------------------------------------------------------------------------- C
DOMAIN_STOP = {
    "ENG", "ENGINE", "ENGINES", "NO", "NR", "NUMBER", "REPLACED", "REMOVED", "REPLACE", "INSTALLED", "IAW", "AMM",
    "REF", "PER", "FOUND", "INSPECTION", "INSPECTED", "OPS", "CHECK", "CHECKED", "CHECKS", "GOOD", "LH", "RH",
    "LEFT", "RIGHT", "AIRCRAFT", "A/C", "AC", "SHIP", "FLT", "FLIGHT", "CREW", "REPORTED", "DURING", "MANUALREF",
    "TASK", "CARD", "WO", "NRC", "DISC", "DISCREPANCY", "ACTION", "CORRECTIVE", "TAKEN", "ITEM", "AT", "MX", "OK",
    "SAT", "SATISFACTORY", "NORMAL", "WORK", "PERFORMED", "STATION", "SDR", "REMOVE", "NEW", "SERVICEABLE",
    "INSTALL", "TROUBLESHOOT", "TROUBLESHOOTING", "TROUBLESHOT", "FAULT", "ISOLATION", "FIM", "TSM", "MM",
    "CMM", "IPC", "SRM", "POSITION", "POS", "ACCOMPLISHED", "COMPLIED", "DUE", "IN", "OUT", "ON", "OFF", "THE",
    "AND", "WITH", "FOR", "WAS", "IS", "WERE", "BE", "TO", "OF", "UNIT", "UNITS", "PART", "PARTS", "SN", "PN",
    "S/N", "P/N", "1", "2", "LOG", "PAGE", "MAINT", "MAINTENANCE", "INFO", "REV", "SEE", "IAE", "CFM", "GE",
    "PW", "RR", "NOTE", "UPON", "AFTER", "PRIOR", "ALSO", "INTO", "FROM", "BY", "OPERATIONAL", "TEST", "TESTED",
    # operator boilerplate and form fields
    "YES", "GROUNDED", "NOTED", "DEFECTS", "THIS", "SERVICE", "NOT", "ALL", "HAS", "WHILE", "SKY", "SKYCARD", "REO",
    "RO", "T/C", "MSG", "NBR", "DEFERRAL", "DEFERRED", "ECONOMIC", "FAULTMSG", "B737", "PLS", "MSR", "JOB", "AN",
    "TYPE", "COMPLIED", "DATED", "MAJOR", "WERE", "THERE", "NONE", "WAS", "ARE", "HAD", "PERFORMING", "PERFORMED",
    # flight outcomes (measured separately through the precautionary-procedure code)
    "EMERGENCY", "DECLARED", "LANDED", "SAFELY", "INCIDENT", "WITHOUT", "DIVERT", "DIVERTED", "RETURNED", "RETURN",
    "LANDING", "UNEVENTFUL", "ATC", "PILOT", "PILOTS", "ROUTE", "GATE",
}


TOPIC_NAMES_PATH = os.path.join(os.path.dirname(__file__), "topic_names.csv")


def topics(df: pd.DataFrame, k: int = 16) -> dict:
    e = df[df.is_engine].copy()
    e["masked"] = mask(e.text)
    stop = list(ENGLISH_STOP_WORDS.union({w.lower() for w in DOMAIN_STOP}).union(DOMAIN_STOP))
    vec = TfidfVectorizer(token_pattern=TOKEN, lowercase=False, ngram_range=(1, 2), min_df=8, max_df=0.4,
                          max_features=30_000, sublinear_tf=True, stop_words=stop, dtype=np.float32)
    X = vec.fit_transform(e.masked.str.upper())
    nmf = NMF(n_components=k, init="nndsvda", random_state=SEED, max_iter=600, beta_loss="frobenius")
    W = nmf.fit_transform(X)
    H = nmf.components_
    terms = np.array(vec.get_feature_names_out())
    e["topic"] = W.argmax(1)
    e["strength"] = W.max(1) / np.maximum(W.sum(1), 1e-9)

    names = {}
    if os.path.exists(TOPIC_NAMES_PATH):
        names = pd.read_csv(TOPIC_NAMES_PATH, dtype={"topic": int}).set_index("topic")["name"].to_dict()

    out = []
    years = sorted(e.year.unique())
    for t in range(k):
        g = e[e.topic == t]
        top_terms = [w for w in terms[np.argsort(-H[t])[:12]]][:10]
        ex = g.sort_values("strength", ascending=False)
        ex = ex[(ex.text.str.len() > 70) & (ex.text.str.len() < 320)].head(40).sample(min(3, len(ex)), random_state=SEED)
        actions = g.action.value_counts(normalize=True).to_dict()
        jasc = g.jasc.value_counts().head(3).to_dict()
        by_year = g.groupby("year").size().reindex(years, fill_value=0)
        out.append({
            "topic": t, "name": names.get(t, " / ".join(top_terms[:3]).title()), "terms": top_terms,
            "n": len(g), "share": len(g) / len(e), "interruption_rate": g.interruption.mean(),
            "inflight_share": (~g.stage_group.isin(["Inspection / maintenance", "Other / unknown"])).mean(),
            "actions": actions, "jasc": jasc, "by_year": by_year.tolist(),
            "examples": [scrub(x) for x in ex.text.tolist()],
        })
    out.sort(key=lambda r: -r["n"] * r["interruption_rate"])
    total_impact = sum(r["n"] * r["interruption_rate"] for r in out)
    cum = 0.0
    for r in out:
        cum += r["n"] * r["interruption_rate"]
        r["impact"] = r["n"] * r["interruption_rate"]
        r["cum_impact_share"] = cum / total_impact
    e[["report_id", "year", "topic", "interruption", "action"]].to_parquet(PROCESSED / "engine_topics.parquet")
    return {"k": k, "n": len(e), "years": [int(y) for y in years], "topics": out,
            "reconstruction_err": float(nmf.reconstruction_err_)}


def l100(df: pd.DataFrame) -> dict:
    g = df[df.aircraft_make == "LKHEED"]
    ch = g.chapter.value_counts().head(8)
    return {"n": len(g), "models": g.aircraft_model.value_counts().head(4).to_dict(),
            "chapters": [{"chapter": c, "title": JASC_CHAPTERS.get(c, c), "n": int(n)} for c, n in ch.items()],
            "interruption_rate": g.interruption.mean(), "years": [int(g.year.min()), int(g.year.max())]}


def main() -> None:
    with Timer("load"):
        df = load()
    with Timer("classifier"):
        clf = classifier(df)
    with Timer("topics"):
        top = topics(df)
    for r in top["topics"]:
        print(f"  [{r['topic']:2d}] n={r['n']:5d} intr={r['interruption_rate']:.2f} {r['name'][:40]:40s} | {' '.join(r['terms'][:8])}")
    export("nlp", {"classifier": clf, "topics": top, "l100": l100(df)}, n=4)


if __name__ == "__main__":
    main()
