import requests, json
from urllib.parse import quote_plus

# 1. pull triples (limit / filter if you like)
SPARQL = """
SELECT ?s ?p ?o WHERE { ?s ?p ?o . }
"""
url = "http://localhost:9999/blazegraph/namespace/dkg/sparql"
resp = requests.post(
    url,
    data={"query": SPARQL, "format": "application/sparql-results+json"},
    timeout=120,
)
rows = resp.json()["results"]["bindings"]

triples = [
    (
        r["s"]["value"],
        r["p"]["value"],
        r["o"]["value"],
    )
    for r in rows
]

# optional: filter only your own namespace / last 10k triples
# triples = [t for t in triples if t[0].startswith("uuid:")]

# 2. send to the micro-service
requests.post(
    "http://localhost:8001/train",
    headers={"content-type": "application/json"},
    json={
        "triples": [
            {"head": h, "relation": r, "tail": t} for h, r, t in triples
        ],
        "model_name": "ComplEx",
        "dimension": 200,
        "epochs": 30,
    },
    timeout=300,
).raise_for_status()

print("✓ train job submitted:", len(triples), "triples")
