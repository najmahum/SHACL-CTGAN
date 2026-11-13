import os

from pyshacl import validate
from rdflib import Graph, Namespace, RDF, URIRef

from tabular_RDF import RAW_MAPPING, df_to_rdf

SH = Namespace("http://www.w3.org/ns/shacl#")


def _short(term):
    return str(term).rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _entity_id(node):
    # node bernama {prefix}_{tipe}_{id}: id di akhir memetakan node mana pun ke satu aplikasi
    return _short(node).rsplit("_", 1)[-1]


def validate_dataframe(df, shapes_path, id_col=None, uri_prefix="synth",
                       chunk_size=2000, report_path=None):
    """
    Mengubah df (atribut asli) ke RDF per chunk lalu memvalidasinya terhadap shapes.

    Compliance rate per constraint = (N - aplikasi yang gagal pada shape itu) / N.
    Compliance rate keseluruhan    = (N - aplikasi dengan >= 1 Violation) / N.
    Warning dilaporkan per shape tetapi tidak mengurangi compliance keseluruhan.
    """
    shapes_graph = Graph().parse(shapes_path, format="turtle")
    severity = {
        _short(s): "Warning" if (s, SH.severity, SH.Warning) in shapes_graph else "Violation"
        for s in shapes_graph.subjects(RDF.type, SH.NodeShape) if isinstance(s, URIRef)
    }
    failed = {name: set() for name in severity}

    for i in range(0, len(df), chunk_size):
        data_graph = df_to_rdf(df.iloc[i:i + chunk_size], RAW_MAPPING, id_col, None,
                               uri_prefix=uri_prefix)
        _, results, _ = validate(
            data_graph=data_graph, shacl_graph=shapes_graph,
            inference="none", abort_on_first=False, advanced=True,
        )
        for r in results.subjects(RDF.type, SH.ValidationResult):
            name = _short(results.value(r, SH.sourceShape))
            failed.setdefault(name, set()).add(_entity_id(results.value(r, SH.focusNode)))
            severity.setdefault(
                name, "Violation" if results.value(r, SH.resultSeverity) == SH.Violation else "Warning"
            )

    n = len(df)
    violating = set().union(*[ids for k, ids in failed.items() if severity[k] == "Violation"])
    warning = set().union(*[ids for k, ids in failed.items() if severity[k] == "Warning"])
    summary = {
        "n_applications": n,
        "n_violating_apps": len(violating),
        "n_warning_apps": len(warning),
        "compliance_rate": (n - len(violating)) / n,
        "per_shape": {
            k: {"severity": severity[k], "n_failed": len(ids), "compliance_rate": (n - len(ids)) / n}
            for k, ids in failed.items()
        },
    }

    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        lines = [
            f"Jumlah aplikasi      : {n}",
            f"Aplikasi melanggar   : {len(violating)}",
            f"Aplikasi ber-warning : {len(warning)}",
            f"Compliance rate      : {summary['compliance_rate']:.4f}",
            "",
            "Per constraint:",
            *[f"  {k:<35} {v['severity']:<9} gagal={v['n_failed']:<6} CR={v['compliance_rate']:.4f}"
              for k, v in sorted(summary["per_shape"].items())],
        ]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return summary