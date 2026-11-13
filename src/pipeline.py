import json
import os
import sys

import pandas as pd

from ctgan_module import RAW_FEATURES, DERIVED_FEATURES, train_and_sample_ctgan
from semantic_enrichment import add_derived
from tabular_RDF import NUMERIC_COLS
from validator_shacl import validate_dataframe

SOURCE_CSV = "data/application_train_clean.csv"   # keluaran notebook pra-pemrosesan
CSV_SEP = ";"
N_TEST = 10000
N_SYNTH = 10000
SEED = 42
EPOCHS = int(os.environ.get("EPOCHS", 300))
BATCH_SIZE = 500       # kelipatan 10 (pac)
SHAPES_TTL = "shapes/constraints.ttl"
TRAIN_CSV, TEST_CSV = "data/real_train.csv", "data/real_test.csv"
ENRICHED_CSV = "data/real_train_enriched.csv"


def prepare():
    """Test = N_TEST baris, train = sisanya. Split identik di setiap run karena SEED tetap."""
    df = pd.read_csv(SOURCE_CSV, sep=CSV_SEP, low_memory=False)
    for c in NUMERIC_COLS:
        before = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().sum() > before:
            print(f"{c}: NaN bertambah {df[c].isna().sum() - before}")
    df = df[["SK_ID_CURR"] + RAW_FEATURES].sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.iloc[N_TEST:].to_csv(TRAIN_CSV, index=False)
    df.iloc[:N_TEST].to_csv(TEST_CSV, index=False)
    print(f"Data asli: train={len(df) - N_TEST}, test={N_TEST}")


def _real_train():
    if not os.path.exists(TRAIN_CSV):
        prepare()
    return pd.read_csv(TRAIN_CSV, low_memory=False)


def scenario_a():
    """Skenario 1 (A): knowledge injection. RDF -> SPARQL -> fitur turunan -> CTGAN."""
    real_train = _real_train()
    if os.path.exists(ENRICHED_CSV):
        enriched = pd.read_csv(ENRICHED_CSV, low_memory=False)
    else:
        enriched = add_derived(real_train)
        enriched.to_csv(ENRICHED_CSV, index=False)
    synth = train_and_sample_ctgan(
        enriched, feature_cols=RAW_FEATURES + DERIVED_FEATURES, epochs=EPOCHS,
        batch_size=BATCH_SIZE, num_samples=N_SYNTH, seed=SEED, model_path="models/ctgan_A.pkl",
    )
    synth.to_csv("data/synthetic_A_with_derived.csv", index=False)
    synth[RAW_FEATURES].to_csv("data/synthetic_A.csv", index=False)


def scenario_b():
    """Skenario 2 (B): baseline, atribut asli saja, tanpa RDF/SPARQL."""
    synth = train_and_sample_ctgan(
        _real_train(), feature_cols=RAW_FEATURES, epochs=EPOCHS, batch_size=BATCH_SIZE,
        num_samples=N_SYNTH, seed=SEED, model_path="models/ctgan_B.pkl",
    )
    synth.to_csv("data/synthetic_B.csv", index=False)


def validate():
    """Validasi SHACL pada atribut asli: data asli (pembanding), A, dan B."""
    datasets = {
        "Real": (pd.read_csv(TEST_CSV, low_memory=False), {"id_col": "SK_ID_CURR", "uri_prefix": "real"}),
        "A": (pd.read_csv("data/synthetic_A.csv", low_memory=False), {"uri_prefix": "synth_a"}),
        "B": (pd.read_csv("data/synthetic_B.csv", low_memory=False), {"uri_prefix": "synth_b"}),
    }
    detail = {
        k: validate_dataframe(df, SHAPES_TTL, report_path=f"data/report_validation_{k}.txt", **kw)
        for k, (df, kw) in datasets.items()
    }
    names = {"Real": "Real_test", "A": "A_knowledge_injection", "B": "B_baseline"}
    summary = pd.DataFrame([
        {"scenario": names[k], **{m: v[m] for m in
         ("n_applications", "n_violating_apps", "n_warning_apps", "compliance_rate")}}
        for k, v in detail.items()
    ])
    summary.to_csv("data/results_shacl.csv", index=False)
    print(summary.to_string(index=False))
    with open("data/results_shacl_detail.json", "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)


STEPS = {"prepare": prepare, "A": scenario_a, "B": scenario_b, "validate": validate}

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for step in sys.argv[1:] or STEPS:
        STEPS[step]()