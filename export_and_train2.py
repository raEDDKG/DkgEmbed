#!/usr/bin/env python3
import requests, json, os
import sys

SPARQL_ENDPOINT = "http://localhost:9999/blazegraph/namespace/dkg/sparql"
KGSVC           = "http://localhost:8001"

# 1⃣  Pull triples (add filters/LIMIT if graph is huge)
SPARQL = """
SELECT ?s ?p ?o WHERE {
  ?s ?p ?o .
}
"""
resp = requests.post(
    SPARQL_ENDPOINT,
    headers={"Accept": "application/sparql-results+json"},
    data={"query": SPARQL},
    timeout=300,
)

if not resp.ok:
    print("SPARQL error:", resp.status_code, resp.text[:200])
    sys.exit(1)
    
rows = resp.json()["results"]["bindings"]

triples = [
    {"head": r["s"]["value"], "relation": r["p"]["value"], "tail": r["o"]["value"]}
    for r in rows
]
print("Exported", len(triples), "triples")

# 2⃣  Send to /train
payload = {
    "triples":     triples,
    "model_name":  "ComplEx",
    "dimension":   200,
    "epochs":      30,
    "batch_size":  1024
}
train_res = requests.post(f"{KGSVC}/train",
                          headers={"content-type":"application/json"},
                          json=payload,
                          timeout=900)
train_res.raise_for_status()
print("Train result:", train_res.json())
