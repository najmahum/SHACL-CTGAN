import os

import numpy as np
import pandas as pd
import torch
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

from semantic_enrichment import DERIVED_COLUMNS
from tabular_RDF import RAW_MAPPING, CATEGORICAL_INT_COLS

# Skenario 1 (knowledge, Synthetic A) : RAW_FEATURES + DERIVED_FEATURES
# Skenario 2 (baseline,  Synthetic B) : RAW_FEATURES
# Kolom ID (SK_ID_CURR) sengaja tidak masuk daftar mana pun.
RAW_FEATURES = [c for c in RAW_MAPPING if c != "SK_ID_CURR"]
DERIVED_FEATURES = [col for _, col, _ in DERIVED_COLUMNS]
DERIVED_DISCRETE = ["CNT_ADULT", "CITY_MISMATCH_COUNT", "REGION_MISMATCH_COUNT"]

STRING_COLS = [c for c in RAW_FEATURES if RAW_MAPPING[c][2] == "string"]
BOOLEAN_COLS = [c for c in RAW_FEATURES if RAW_MAPPING[c][2] == "boolean"]
DISCRETE_COLS = STRING_COLS + BOOLEAN_COLS + CATEGORICAL_INT_COLS + DERIVED_DISCRETE
# Hitungan bertipe integer (umur, hari, jumlah anak, dst.): dibulatkan setelah sampling
INTEGER_COLS = [
    c for c in RAW_FEATURES
    if RAW_MAPPING[c][2] == "integer" and c not in DISCRETE_COLS
]

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_sample_ctgan(
    df,
    feature_cols,
    discrete_cols=DISCRETE_COLS,
    epochs=300,
    batch_size=500,
    num_samples=None,
    seed=42,
    model_path=None,
    output_csv_path=None,
):
    """
    Melatih CTGAN (SDV) pada feature_cols dan menghasilkan data sintetis.

    Sengaja TIDAK ada constraint SDV dan TIDAK ada imputasi manual:
      - constraint akan membuat batasan langsung pada kolom, padahal
        yang diuji adalah pengaruh fitur turunan;
      - NaN ditangani SDV sendiri (nilai pengganti + indikator missing),
        jadi nilai kosong pada EXT_SOURCE_* tidak diubah.

    Semua hyperparameter dan seed sama untuk kedua skenario;
    satu-satunya perbedaan adalah feature_cols.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ada di DataFrame: {missing}")

    train = df[feature_cols].copy()

    # Tipe nullable pandas (Int64, boolean) -> float agar aman untuk SDV
    for c in train.columns:
        if str(train[c].dtype) in ("Int64", "boolean"):
            train[c] = train[c].astype("float64")

    discrete = [c for c in discrete_cols if c in train.columns]
    numeric_discrete = [c for c in discrete if pd.api.types.is_numeric_dtype(train[c])]

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train)
    for c in discrete:
        metadata.update_column(column_name=c, sdtype="categorical")

    set_seed(seed)
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        batch_size=batch_size,  # harus kelipatan pac (default 10)
        verbose=True,
    )
    synthesizer.fit(train)

    if model_path:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        if os.path.exists(model_path):
            os.remove(model_path)
        synthesizer.save(model_path)

    set_seed(seed)
    n_rows = num_samples if num_samples is not None else len(train)
    synthetic = synthesizer.sample(num_rows=n_rows)

    # Kolom diskrit numerik dikembalikan ke integer
    for c in INTEGER_COLS:
        if c in synthetic.columns:
            synthetic[c] = synthetic[c].round().astype("Int64")

    if output_csv_path:
        out_dir = os.path.dirname(output_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        synthetic.to_csv(output_csv_path, index=False)
        print(f"CTGAN sample tersimpan ke {output_csv_path} ({len(synthetic)} baris)")

    return synthetic