import os
from decimal import Decimal

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

HC = Namespace("http://www.semanticweb.org/najma/ontologies/homecredit/")
LOAN_APPLICATION = URIRef(
    "https://spec.edmcouncil.org/fibo/ontology/LOAN/LoansGeneral/LoanApplications/LoanApplication"
)

NODE_CLASSES = {
    "app": LOAN_APPLICATION,
    "applicant": HC.Applicant,
    "ext": HC.ExternalAssessment,
    "res": HC.ResidenceInformation,
    "bureau": HC.CreditBureauHistory,
    "contact": HC.ContactInformation,
    "prop": HC.PropertyInformation,
    "social": HC.SocialSurroundings,
}

NODE_LINKS = [
    ("app", HC.SubmittedBy, "applicant"),
    ("app", HC.hasExternalAssessment, "ext"),
    ("applicant", HC.hasResidenceInformation, "res"),
    ("applicant", HC.hasCreditBureauHistory, "bureau"),
    ("applicant", HC.hasContactInformation, "contact"),
    ("applicant", HC.hasPropertyInfo, "prop"),
    ("applicant", HC.hasSocialSurroundings, "social"),
]

TYPE_MAPPING = {
    "integer": XSD.integer,
    "float": XSD.float,
    "decimal": XSD.decimal,
    "string": XSD.string,
    "boolean": XSD.boolean,
}

# (kolom, node, properti, tipe) -- domain mengikuti Ontology.ttl
_RAW_SPEC = [
    # LoanApplication
    ("SK_ID_CURR", "app", "hasApplicationID", "integer"),
    ("TARGET", "app", "hasCreditRiskStatus", "integer"),
    ("NAME_CONTRACT_TYPE", "app", "hasLoanType", "string"),
    ("AMT_CREDIT", "app", "hasCreditAmount", "float"),
    ("AMT_ANNUITY", "app", "hasAnnuityAmount", "float"),
    ("AMT_GOODS_PRICE", "app", "hasGoodsPrice", "float"),
    ("NAME_TYPE_SUITE", "app", "hasCompanionship", "string"),
    ("WEEKDAY_APPR_PROCESS_START", "app", "hasApplicationStartDay", "string"),
    ("HOUR_APPR_PROCESS_START", "app", "hasApplicationStartHour", "integer"),
    # Applicant
    ("CODE_GENDER", "applicant", "hasApplicantGender", "string"),
    ("FLAG_OWN_CAR", "applicant", "hasCarOwnershipStatus", "string"),
    ("FLAG_OWN_REALTY", "applicant", "hasRealtyOwnershipStatus", "string"),
    ("CNT_CHILDREN", "applicant", "hasNumberOfChildren", "integer"),
    ("AMT_INCOME_TOTAL", "applicant", "hasIncomeAmount", "float"),
    ("NAME_INCOME_TYPE", "applicant", "hasIncomeSourceType", "string"),
    ("NAME_EDUCATION_TYPE", "applicant", "hasEducationLevel", "string"),
    ("NAME_FAMILY_STATUS", "applicant", "hasFamilyStatus", "string"),
    ("NAME_HOUSING_TYPE", "applicant", "hasHousingType", "string"),
    ("DAYS_BIRTH", "applicant", "hasAgeInDays", "integer"),
    ("DAYS_EMPLOYED", "applicant", "hasEmploymentDurationInDays", "integer"),
    ("DAYS_REGISTRATION", "applicant", "hasRegistrationDaysBeforeApplication", "float"),
    ("DAYS_ID_PUBLISH", "applicant", "hasDaysIDPublish", "float"),
    ("OWN_CAR_AGE", "applicant", "hasCarAge", "float"),
    ("OCCUPATION_TYPE", "applicant", "hasOccupationType", "string"),
    ("CNT_FAM_MEMBERS", "applicant", "hasNumberOfFamilyMembers", "float"),
    ("ORGANIZATION_TYPE", "applicant", "hasOrganizationType", "string"),
    # ContactInformation
    ("FLAG_MOBIL", "contact", "hasContactFlagMobile", "boolean"),
    ("FLAG_EMP_PHONE", "contact", "hasContactFlagEmployerPhone", "boolean"),
    ("FLAG_WORK_PHONE", "contact", "hasContactFlagPhone", "boolean"),
    ("FLAG_CONT_MOBILE", "contact", "hasContactFlagMobileReachable", "boolean"),
    ("FLAG_PHONE", "contact", "hasContactFlagHomePhone", "boolean"),
    ("FLAG_EMAIL", "contact", "hasContactFlagEmail", "boolean"),
    ("DAYS_LAST_PHONE_CHANGE", "contact", "hasLastPhoneChangeDays", "float"),
    # ResidenceInformation
    ("REGION_POPULATION_RELATIVE", "res", "hasRegionPopulationRelative", "float"),
    ("REGION_RATING_CLIENT", "res", "hasRegionRating", "integer"),
    ("REGION_RATING_CLIENT_W_CITY", "res", "hasRegionRatingCity", "integer"),
    ("REG_REGION_NOT_LIVE_REGION", "res", "hasRegistrationLivingRegionDifference", "integer"),
    ("REG_REGION_NOT_WORK_REGION", "res", "hasRegistrationWorkRegionDifference", "integer"),
    ("LIVE_REGION_NOT_WORK_REGION", "res", "hasLivingWorkRegionDifference", "integer"),
    ("REG_CITY_NOT_LIVE_CITY", "res", "hasRegistrationLivingCityDifference", "integer"),
    ("REG_CITY_NOT_WORK_CITY", "res", "hasRegistrationWorkCityDifference", "integer"),
    ("LIVE_CITY_NOT_WORK_CITY", "res", "hasLivingWorkCityDifference", "integer"),
    # ExternalAssessment
    ("EXT_SOURCE_1", "ext", "hasExternalScore1", "float"),
    ("EXT_SOURCE_2", "ext", "hasExternalScore2", "float"),
    ("EXT_SOURCE_3", "ext", "hasExternalScore3", "float"),
    # CreditBureauHistory
    ("AMT_REQ_CREDIT_BUREAU_HOUR", "bureau", "hasBureauInquiryCountHour", "float"),
    ("AMT_REQ_CREDIT_BUREAU_DAY", "bureau", "hasBureauInquiryCountDay", "float"),
    ("AMT_REQ_CREDIT_BUREAU_WEEK", "bureau", "hasBureauInquiryCountWeek", "float"),
    ("AMT_REQ_CREDIT_BUREAU_MON", "bureau", "hasBureauInquiryCountMonth", "float"),
    ("AMT_REQ_CREDIT_BUREAU_QRT", "bureau", "hasBureauInquiryCountQuarter", "float"),
    ("AMT_REQ_CREDIT_BUREAU_YEAR", "bureau", "hasBureauInquiryCountYear", "float"),
    # SocialSurroundings
    ("OBS_30_CNT_SOCIAL_CIRCLE", "social", "hasSocialCircleObservation30", "float"),
    ("DEF_30_CNT_SOCIAL_CIRCLE", "social", "hasSocialCircleDefault30", "float"),
    ("OBS_60_CNT_SOCIAL_CIRCLE", "social", "hasSocialCircleObservation60", "float"),
    ("DEF_60_CNT_SOCIAL_CIRCLE", "social", "hasSocialCircleDefault60", "float"),
    # PropertyInformation (kolom non-AVG/MODE/MEDI)
    ("FONDKAPREMONT_MODE", "prop", "hasFondkapremontMode", "string"),
    ("HOUSETYPE_MODE", "prop", "hasHouseTypeMode", "string"),
    ("TOTALAREA_MODE", "prop", "hasTotalAreaMode", "float"),
    ("WALLSMATERIAL_MODE", "prop", "hasWallsMaterialMode", "string"),
    ("EMERGENCYSTATE_MODE", "prop", "hasEmergencyStateMode", "string"),
]

# PropertyInformation: 14 ukuran bangunan x 3 statistik (AVG, MODE, MEDI)
_BUILDING = [
    ("APARTMENTS", "Apartments"), ("BASEMENTAREA", "BasementArea"),
    ("YEARS_BEGINEXPLUATATION", "YearsBeginExpluatation"), ("YEARS_BUILD", "YearsBuild"),
    ("COMMONAREA", "CommonArea"), ("ELEVATORS", "Elevators"), ("ENTRANCES", "Entrances"),
    ("FLOORSMAX", "FloorsMax"), ("FLOORSMIN", "FloorsMin"), ("LANDAREA", "LandArea"),
    ("LIVINGAPARTMENTS", "LivingApartments"), ("LIVINGAREA", "LivingArea"),
    ("NONLIVINGAPARTMENTS", "NonLivingApartments"), ("NONLIVINGAREA", "NonLivingArea"),
]
for _col, _name in _BUILDING:
    for _suffix, _camel in [("AVG", "Avg"), ("MODE", "Mode"), ("MEDI", "Medi")]:
        _RAW_SPEC.append((f"{_col}_{_suffix}", "prop", f"has{_name}{_camel}", "float"))

# FLAG_DOCUMENT_2 ... 21 (domain LoanApplication)
for _i in range(2, 22):
    _RAW_SPEC.append((f"FLAG_DOCUMENT_{_i}", "app", f"hasDocumentFlag{_i}", "boolean"))

RAW_MAPPING = {col: (node, prop, typ) for col, node, prop, typ in _RAW_SPEC}
assert len(RAW_MAPPING) == 122, f"Jumlah kolom {len(RAW_MAPPING)}, seharusnya 122"

# Kolom bertipe integer yang sebenarnya kategori (bukan hitungan/ukuran)
CATEGORICAL_INT_COLS = [
    "TARGET", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
]

# Kolom yang harus berupa angka saat dibaca dari CSV
NUMERIC_COLS = [c for c, (_, _, t) in RAW_MAPPING.items() if t in ("integer", "float", "boolean")]

def _to_literal(value, data_type):
    """Konversi nilai pandas/numpy menjadi rdflib Literal bertipe XSD."""
    xsd_type = TYPE_MAPPING.get(data_type, XSD.string)

    if data_type == "integer":
        py_val = int(round(float(value)))
    elif data_type == "float":
        py_val = float(value)
    elif data_type == "decimal":
        py_val = Decimal(str(round(float(value), 6)))
    elif data_type == "boolean":
        if isinstance(value, str):
            py_val = value.strip().lower() in ("1", "true", "y", "yes")
        else:
            py_val = bool(int(value))
    else:
        py_val = str(value)

    return Literal(py_val, datatype=xsd_type)


def _node_uri(uri_prefix, node_type, entity_id):
    return HC[f"{uri_prefix}_{node_type}_{entity_id}"]


def df_to_rdf(
    df,
    column_map,
    id_col,
    output_ttl_path,
    uri_prefix="app",
    node_links=NODE_LINKS,
):
    """
    Mengubah DataFrame menjadi RDF.

    Satu baris -> satu LoanApplication + satu Applicant (selalu dibuat),
    ditambah node lain hanya jika baris tersebut punya minimal satu nilai
    untuk node itu. Nilai NaN tidak dibuat triple-nya (dianggap missing).

    column_map : {kolom: (node, nama_properti, tipe)}
    id_col     : kolom ID; jika None atau tidak ada, dipakai indeks baris.
    """
    
    g = Graph()
    g.bind("", HC)

    use_id_col = id_col is not None and id_col in df.columns

    for idx, row in df.iterrows():
        if use_id_col and pd.notna(row[id_col]):
            entity_id = int(row[id_col])
        else:
            entity_id = idx

        used_nodes = {"app", "applicant"}

        for col, (node_type, prop_name, data_type) in column_map.items():
            if col not in df.columns or pd.isna(row[col]):
                continue

            subject = _node_uri(uri_prefix, node_type, entity_id)
            g.add((subject, HC[prop_name], _to_literal(row[col], data_type)))
            used_nodes.add(node_type)

        for node_type in used_nodes:
            g.add((
                _node_uri(uri_prefix, node_type, entity_id),
                RDF.type,
                NODE_CLASSES[node_type],
            ))

        for parent, prop, child in node_links:
            if parent in used_nodes and child in used_nodes:
                g.add((
                    _node_uri(uri_prefix, parent, entity_id),
                    prop,
                    _node_uri(uri_prefix, child, entity_id),
                ))

    if output_ttl_path:
        out_dir = os.path.dirname(output_ttl_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        g.serialize(destination=output_ttl_path, format="turtle")
        print(f"{len(g)} triple tersimpan ke {output_ttl_path}")
    return g


def csv_to_rdf(csv_path, column_map, id_col, output_ttl_path, uri_prefix="app", sep=","):
    df = pd.read_csv(csv_path, sep=sep)
    return df_to_rdf(df, column_map, id_col, output_ttl_path, uri_prefix)