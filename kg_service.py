#!/usr/bin/env python3
"""
kg_service.py

Knowledge Graph Embedding Microservice (FastAPI)
- Trains entity (kg_embeddings) and relation (kg_rel_embeddings) vectors using PyKEEN.
- Stores vectors in Postgres tables via pgvector.
- Scores triples with a TransE-style scoring function.

Environment Variables (in a .env file):
  PG_DSN     : PostgreSQL DSN (e.g., postgresql://user:pass@host:port/dbname)

Usage:
  uvicorn kg_service:app --reload --port 8001
"""
import os
import hashlib
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import psycopg2, psycopg2.extras as pe
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instantiation
app = FastAPI(
    title="Knowledge Graph Embedding Service",
    description="Train and score KG embeddings (entities + relations) via PyKEEN",
    version="0.1.0"
)

# Database connection setup
PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    logger.error("Environment variable PG_DSN is required but not set.")
    raise RuntimeError("PG_DSN environment variable is required")
PG_CONN = psycopg2.connect(PG_DSN)

# Pydantic request models
class TripleItem(BaseModel):
    head: str = Field(
        ..., 
        description="IRI of the head entity",
        json_schema_extra={"example": "ual:EntityA"},
    )
    relation: str = Field(
        ..., 
        description="IRI of the predicate/relation",
        json_schema_extra={"example": "http://schema.org/mentions"},
    )
    tail: str = Field(
        ..., 
        description="IRI of the tail entity",
        json_schema_extra={"example": "ual:EntityB"},
    )

class TrainRequest(BaseModel):
    triples: list[TripleItem]
    model_name: str = Field("ComplEx", description="PyKEEN embedding model name")
    dimension: int = Field(200, gt=0, description="Embedding dimensionality")
    epochs: int = Field(25, gt=0, description="Number of training epochs")
    batch_size: int = Field(512, gt=0, description="Batch size for training")

class ScoreRequest(BaseModel):
    head: str = Field(
        ...,
        description="IRI of the head entity",
        json_schema_extra={"example": "ual:EntityA"},
    )
    relation: str = Field(
        ...,
        description="IRI of the predicate/relation",
        json_schema_extra={"example": "http://schema.org/mentions"},
    )
    tail: str = Field(
        ...,
        description="IRI of the tail entity",
        json_schema_extra={"example": "ual:EntityB"},
    )

# Endpoint: Train KG embeddings
@app.post("/train", summary="Train entity + relation embeddings")
def train_graph(req: TrainRequest):
    # Prepare triples matrix
    triples = [(t.head, t.relation, t.tail) for t in req.triples]
    tf = TriplesFactory.from_labeled_triples(np.array(triples, dtype=object))

    # 80 / 10 / 10 split for train / valid / test
    train_tf, valid_tf, test_tf = tf.split(
        ratios=[0.8, 0.1, 0.1],
        random_state=42
    )

    # Run training pipeline
    result = pipeline(
        model=req.model_name,
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        model_kwargs=dict(embedding_dim=req.dimension),
        training_kwargs=dict(num_epochs=req.epochs, batch_size=req.batch_size),
    )

    def to_numpy(rep):
        # PyKEEN wrapper
        if hasattr(rep, "get_in_canonical_shape"):
            return (
                rep.get_in_canonical_shape()  # torch.Tensor [n, dim]
                .detach()
                .cpu()
                .numpy()
            )
        # Plain torch.nn.Embedding
        return rep.weight.detach().cpu().numpy()


    # Extract learned vectors
    ent_vecs = to_numpy(result.model.entity_representations[0])
    rel_vecs = to_numpy(result.model.relation_representations[0])



    # ent_vecs = (
    #     result.model.entity_representations[0]          # Embedding wrapper
    #     .get_in_canonical_shape()                       # → torch.Tensor [n_entities, dim]
    #     .detach()                                       # drop graph
    #     .cpu()
    #     .numpy()
    # )

    # rel_vecs = (
    #     result.model.relation_representations[0]
    #     .get_in_canonical_shape()
    #     .detach()
    #     .cpu()
    #     .numpy()
    # )

    # Compute a snapshot hash (entities+relations)
    combined = ent_vecs.tobytes() + rel_vecs.tobytes()
    sha = hashlib.sha256(combined).hexdigest()[:16]
    snapshot_ual = f"did:dkg:model:{req.model_name.lower()}-{sha}"

    # Upsert entity vectors
    ents = tf.entity_id_to_label
    ent_rows = [(ents[i], ent_vecs[i].tolist(), snapshot_ual) for i in range(len(ents))]
    # Upsert relation vectors
    rels = tf.relation_id_to_label
    rel_rows = [(rels[i], rel_vecs[i].tolist(), snapshot_ual) for i in range(len(rels))]

    with PG_CONN, PG_CONN.cursor() as cur:
        # Entities
        pe.execute_values(
            cur,
            """
            INSERT INTO kg_embeddings(uri, vec, snapshot_ual)
            VALUES %s
            ON CONFLICT (uri) DO UPDATE
              SET vec = EXCLUDED.vec,
                  snapshot_ual = EXCLUDED.snapshot_ual
            """,
            ent_rows
        )
        # Relations
        pe.execute_values(
            cur,
            """
            INSERT INTO kg_rel_embeddings(relation_uri, vec, snapshot_ual)
            VALUES %s
            ON CONFLICT (relation_uri) DO UPDATE
              SET vec = EXCLUDED.vec,
                  snapshot_ual = EXCLUDED.snapshot_ual
            """,
            rel_rows
        )

    logger.info(f"Trained {req.model_name} [{snapshot_ual}]: {len(ents)} entities, {len(rels)} relations")
    return {"snapshot_ual": snapshot_ual, "entities": len(ents), "relations": len(rels)}

# Endpoint: Score a triple (TransE-style)
@app.post("/score", summary="Score a triple for link-prediction")
def score_triple(q: ScoreRequest):
    # Fetch vectors from PG
    with PG_CONN, PG_CONN.cursor() as cur:
        # Head
        cur.execute("SELECT vec FROM kg_embeddings WHERE uri = %s", (q.head,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, detail=f"Entity vector not found for head: {q.head}")
        hv = np.array(row[0])
        # Relation
        cur.execute("SELECT vec FROM kg_rel_embeddings WHERE relation_uri = %s", (q.relation,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, detail=f"Relation vector not found for: {q.relation}")
        rv = np.array(row[0])
        # Tail
        cur.execute("SELECT vec FROM kg_embeddings WHERE uri = %s", (q.tail,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, detail=f"Entity vector not found for tail: {q.tail}")
        tv = np.array(row[0])

    # TransE score = -||h + r - t||
    score = float(-np.linalg.norm(hv + rv - tv))
    return {"score": score}

# Run via `uvicorn kg_service:app --reload --port 8001`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("kg_service:app", host="0.0.0.0", port=int(os.getenv("PORT", 8001)), reload=True)
