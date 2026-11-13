import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tabular_RDF import RAW_MAPPING, df_to_rdf

PREFIXES = """
PREFIX : <http://www.semanticweb.org/najma/ontologies/homecredit/>
PREFIX fibo-app: <https://spec.edmcouncil.org/fibo/ontology/LOAN/LoansGeneral/LoanApplications/>
"""

MARRIED = '"Married", "Civil marriage"'
UNMARRIED = '"Single / not married", "Separated", "Widow"'

# (variabel SPARQL, kolom keluaran, tipe)
DERIVED_COLUMNS = [
    ("installmentToIncome", "INSTALLMENT_TO_INCOME_RATIO", "float"),
    ("tenor", "ESTIMATED_TENOR_MONTHS", "float"),
    ("age", "AGE_YEARS", "float"),
    ("empYears", "EMPLOYMENT_YEARS", "float"),
    ("ageAtId", "AGE_AT_ID_PUBLISH", "float"),
    ("ageAtReg", "AGE_AT_REGISTRATION", "float"),
    ("adults", "CNT_ADULT", "int"),
    ("cityCount", "CITY_MISMATCH_COUNT", "int"),
    ("regionCount", "REGION_MISMATCH_COUNT", "int"),
]

OUT_VARS = " ".join(f"?{v}" for v, _, _ in DERIVED_COLUMNS)

QUERY = f"""{PREFIXES}
SELECT ?app {OUT_VARS}
WHERE {{
    ?app a fibo-app:LoanApplication ;
         :SubmittedBy ?applicant .
    OPTIONAL {{ ?app :hasCreditAmount ?credit }}
    OPTIONAL {{ ?app :hasAnnuityAmount ?annuity }}
    OPTIONAL {{ ?app :hasLoanType ?loanType }}
    OPTIONAL {{ ?applicant :hasIncomeAmount ?income }}
    OPTIONAL {{ ?applicant :hasAgeInDays ?birth }}
    OPTIONAL {{ ?applicant :hasEmploymentDurationInDays ?emp }}
    OPTIONAL {{ ?applicant :hasDaysIDPublish ?idPublish }}
    OPTIONAL {{ ?applicant :hasRegistrationDaysBeforeApplication ?registration }}
    OPTIONAL {{ ?applicant :hasFamilyStatus ?status }}
    OPTIONAL {{
        ?applicant :hasResidenceInformation ?res .
        OPTIONAL {{ ?res :hasRegistrationLivingCityDifference ?cRL }}
        OPTIONAL {{ ?res :hasRegistrationWorkCityDifference ?cRW }}
        OPTIONAL {{ ?res :hasLivingWorkCityDifference ?cLW }}
        OPTIONAL {{ ?res :hasRegistrationLivingRegionDifference ?rRL }}
        OPTIONAL {{ ?res :hasRegistrationWorkRegionDifference ?rRW }}
        OPTIONAL {{ ?res :hasLivingWorkRegionDifference ?rLW }}
    }}

    # ?nothing sengaja tidak pernah terikat; BIND yang error atau memakai ?nothing menjadi unbound (NaN)
    BIND(IF(?income > 0, ?annuity / ?income, ?nothing) AS ?installmentToIncome)
    BIND(IF(STR(?loanType) = "Cash loans" && ?annuity > 0, ?credit / ?annuity, ?nothing) AS ?tenor)
    BIND(-?birth / 365.25 AS ?age)
    BIND(IF(?emp <= 0, -?emp / 365.25, ?nothing) AS ?empYears)
    BIND((?idPublish - ?birth) / 365.25 AS ?ageAtId)
    BIND((?registration - ?birth) / 365.25 AS ?ageAtReg)
    BIND(IF(STR(?status) IN ({MARRIED}), 2,
            IF(STR(?status) IN ({UNMARRIED}), 1, ?nothing)) AS ?adults)
    BIND(?cRL + ?cRW + ?cLW AS ?cityCount)
    BIND(?rRL + ?rRW + ?rLW AS ?regionCount)
}}
"""

# Hanya kolom yang dibaca query yang dikonversi ke RDF (menghemat memori)
_used = set(re.findall(r":(has\w+|SubmittedBy)", QUERY))
ENRICH_MAPPING = {c: m for c, m in RAW_MAPPING.items() if m[1] in _used}


def _convert(value, kind):
    if value is None:
        return np.nan
    v = float(value.toPython())
    return int(round(v)) if kind == "int" else v


def _run(graph):
    records = []
    for row in graph.query(QUERY):
        rec = {"ENTITY_ID": int(str(row["app"]).rsplit("_", 1)[-1])}
        for var, col, kind in DERIVED_COLUMNS:
            rec[col] = _convert(row[var], kind)
        records.append(rec)
    return pd.DataFrame(records)


def _enrich_chunk(chunk, id_col):
    return _run(df_to_rdf(chunk, ENRICH_MAPPING, id_col, None))


def add_derived(df, id_col="SK_ID_CURR", chunk_size=5000, n_jobs=4):
    """Data asli -> RDF -> SPARQL -> kolom turunan digabung ke df (chunk diproses paralel)."""
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_enrich_chunk)(df.iloc[i:i + chunk_size], id_col)
        for i in range(0, len(df), chunk_size)
    )
    derived = pd.concat(parts, ignore_index=True)
    for _, col, kind in DERIVED_COLUMNS:
        if kind == "float":
            derived[col] = derived[col].round(6)

    out = df.copy().assign(ENTITY_ID=df[id_col].astype(int)).merge(derived, on="ENTITY_ID", how="left")
    if len(out) != len(df):
        raise ValueError("Jumlah baris berubah setelah merge: ada ID ganda.")
    print(f"SPARQL: {len(derived)} aplikasi, {len(DERIVED_COLUMNS)} fitur turunan")
    return out.drop(columns="ENTITY_ID")