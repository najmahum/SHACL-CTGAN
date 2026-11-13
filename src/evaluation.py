import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.metrics import pairwise_distances_argmin_min

from ctgan_module import RAW_FEATURES, STRING_COLS, DISCRETE_COLS

DATA = "data"
N_BINS = 20

DISCRETE_NUMERIC = [c for c in DISCRETE_COLS if c in RAW_FEATURES and c not in STRING_COLS]
CONTINUOUS = [c for c in RAW_FEATURES if c not in DISCRETE_COLS]
NUMERIC_FEATURES = [c for c in RAW_FEATURES if c not in STRING_COLS]


def load(name):
    df = pd.read_csv(os.path.join(DATA, name), low_memory=False)
    for c in CONTINUOUS + DISCRETE_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ------------------------------------------------------------------
# Fidelitas: KS dan JSD (A = knowledge injection, B = baseline)
# ------------------------------------------------------------------
def ks_stat(real, synth):
    r, s = real.dropna(), synth.dropna()
    return float(ks_2samp(r, s).statistic) if len(r) and len(s) else np.nan


def _jsd(p, q):
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))  # basis 2 -> [0, 1]


def jsd_continuous(real, synth):
    r, s = real.dropna().to_numpy(), synth.dropna().to_numpy()
    if len(r) == 0 or len(s) == 0:
        return np.nan
    edges = np.unique(np.quantile(r, np.linspace(0, 1, N_BINS + 1))).astype(float)
    if len(edges) < 3:  # hampir konstan
        edges = np.array([-np.inf, np.median(r), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    p, _ = np.histogram(r, bins=edges)
    q, _ = np.histogram(s, bins=edges)
    return _jsd(p.astype(float), q.astype(float))


def _cat(series, numeric):
    s = series.dropna()
    return (s.round().astype(int) if numeric else s).astype(str)


def jsd_categorical(real, synth, numeric):
    r, s = _cat(real, numeric), _cat(synth, numeric)
    if len(r) == 0 or len(s) == 0:
        return np.nan
    cats = sorted(set(r) | set(s))
    p = r.value_counts().reindex(cats, fill_value=0).to_numpy(float)
    q = s.value_counts().reindex(cats, fill_value=0).to_numpy(float)
    return _jsd(p, q)


def column_metrics(real, synth, col):
    if col in CONTINUOUS:
        return ks_stat(real[col], synth[col]), jsd_continuous(real[col], synth[col])
    return np.nan, jsd_categorical(real[col], synth[col], numeric=col not in STRING_COLS)


def fidelity_table(real_test, real_train, synth_a, synth_b):
    rows = []
    for col in RAW_FEATURES:
        ks_ref, jsd_ref = column_metrics(real_test, real_train, col)
        ks_a, jsd_a = column_metrics(real_test, synth_a, col)
        ks_b, jsd_b = column_metrics(real_test, synth_b, col)
        rows.append({
            "column": col, "type": "continuous" if col in CONTINUOUS else "discrete",
            "KS_ref": ks_ref, "KS_A": ks_a, "KS_B": ks_b,
            "JSD_ref": jsd_ref, "JSD_A": jsd_a, "JSD_B": jsd_b,
            "missing_real": real_test[col].isna().mean(),
            "missing_A": synth_a[col].isna().mean(),
            "missing_B": synth_b[col].isna().mean(),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# SHACL: compliance rate per constraint dan keseluruhan
# ------------------------------------------------------------------
def shacl_tables(detail):
    per_shape = pd.DataFrame([
        {"dataset": k, "shape": s, **m}
        for k, v in detail.items() for s, m in v["per_shape"].items()
    ])
    return per_shape


# ------------------------------------------------------------------
# DCR: jarak Euclidean ke record asli terdekat (real_train)
# Numerik: min-max dari real_train, NaN -> median. String: one-hot.
# ------------------------------------------------------------------
class DCR:
    def __init__(self, ref):
        self.num = NUMERIC_FEATURES
        self.cat = [c for c in RAW_FEATURES if c in STRING_COLS]
        self.lo, self.hi = ref[self.num].min(), ref[self.num].max()
        self.med = ref[self.num].median()
        self.cat_cols = pd.get_dummies(ref[self.cat].astype(str)).columns
        self.ref_X = self._encode(ref)

    def _encode(self, df):
        span = (self.hi - self.lo).replace(0, 1)
        num = ((df[self.num].fillna(self.med) - self.lo) / span).clip(0, 1)
        cat = pd.get_dummies(df[self.cat].astype(str)).reindex(columns=self.cat_cols, fill_value=0)
        return np.hstack([num.to_numpy(np.float32), cat.to_numpy(np.float32)])

    def distances(self, df):
        _, d = pairwise_distances_argmin_min(self._encode(df), self.ref_X)
        return d


def dcr_summary(name, d):
    return {
        "dataset": name, "DCR_mean": d.mean(), "DCR_median": np.median(d),
        "DCR_p5": np.quantile(d, 0.05), "DCR_min": d.min(),
        "pct_exact_copy": float((d < 1e-6).mean()),
    }


def main():
    real_test, real_train = load("real_test.csv"), load("real_train.csv")
    synth_a, synth_b = load("synthetic_A.csv"), load("synthetic_B.csv")

    # 1. Fidelitas
    fid = fidelity_table(real_test, real_train, synth_a, synth_b)
    fid.to_csv(f"{DATA}/results_fidelity_per_column.csv", index=False)
    cont, disc = fid[fid["type"] == "continuous"], fid[fid["type"] == "discrete"]
    summary = pd.DataFrame([
        {"scenario": name, "KS_mean_continuous": cont[f"KS_{k}"].mean(),
         "JSD_mean_all": fid[f"JSD_{k}"].mean(),
         "JSD_mean_continuous": cont[f"JSD_{k}"].mean(),
         "JSD_mean_discrete": disc[f"JSD_{k}"].mean()}
        for name, k in [("Ref (real_train vs real_test)", "ref"),
                        ("A_knowledge_injection", "A"), ("B_baseline", "B")]
    ])
    summary.to_csv(f"{DATA}/results_fidelity_summary.csv", index=False)

    # 2. SHACL
    with open(f"{DATA}/results_shacl_detail.json", encoding="utf-8") as f:
        detail = json.load(f)
    shacl = shacl_tables(detail)
    shacl.to_csv(f"{DATA}/results_shacl_per_shape.csv", index=False)

    # 3. DCR (real_test sebagai pembanding: jarak data asli yang tidak dilatihkan)
    dcr = DCR(real_train)
    dcr_table = pd.DataFrame([
        dcr_summary("Real_test", dcr.distances(real_test)),
        dcr_summary("A_knowledge_injection", dcr.distances(synth_a)),
        dcr_summary("B_baseline", dcr.distances(synth_b)),
    ])
    dcr_table.to_csv(f"{DATA}/results_dcr.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Fidelitas (makin kecil makin mirip data asli) ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== Compliance rate per constraint ===")
    print(shacl.pivot(index="shape", columns="dataset", values="compliance_rate").round(4).to_string())
    print("\n=== Compliance rate keseluruhan (Violation) ===")
    print({k: round(v["compliance_rate"], 4) for k, v in detail.items()})
    print("\n=== DCR ===")
    print(dcr_table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()