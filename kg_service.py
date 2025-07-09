#!/usr/bin/env python3
# kg_service.py
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
import torch
import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.nn.representation import Embedding
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from typing import Literal

# Load environment
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format='{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}')
logger = logging.getLogger(__name__)

# FastAPI app instantiation
app = FastAPI(
    title="Knowledge Graph Embedding Service",
    description="Train and score KG embeddings (entities + relations) via PyKEEN",
    version="0.2.0"
)

# Database connection pooling
PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    logger.error("Environment variable PG_DSN is required but not set.")
    raise RuntimeError("PG_DSN environment variable is required")

PG_IVFFLAT_PROBES = int(os.getenv("PG_IVFFLAT_PROBES", 10))

db_pool = SimpleConnectionPool(minconn=1, maxconn=20, dsn=PG_DSN)

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
    max_block: int = Field(..., description="The maximum block height from the input triples")

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

class NearestNeighborsRequest(BaseModel):
    vec: list[float] = Field(..., description="The vector to find neighbors for")
    k: int = Field(20, gt=0, description="The number of nearest neighbors to return")
    metric: Literal["cosine", "l2"] = Field("cosine", description="The distance metric to use")

class HealthResponse(BaseModel):
    status: str
    db_pool: dict[str, int]


@app.get("/health", response_model=HealthResponse, summary="Check service health")
def health_check():
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        # Note: _used and _pool are internal attributes of SimpleConnectionPool
        return {
            "status": "ok",
            "db_pool": {
                "used": len(db_pool._used),  # type: ignore
                "free": len(db_pool._pool),  # type: ignore
                "max": db_pool.maxconn,
            }
        }
    except psycopg2.Error as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database connection error")
    finally:
        if conn:
            db_pool.putconn(conn)


@app.post("/nearest_neighbors", summary="Find nearest neighbor entities")
def nearest_neighbors(req: NearestNeighborsRequest):
    if req.metric not in ("cosine", "l2"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid metric. Must be 'cosine' or 'l2'."
        )
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            # Check vector dimension
            cur.execute("SELECT ndim(vec) FROM kg_embeddings LIMIT 1")
            db_dim_row = cur.fetchone()
            if db_dim_row and len(req.vec) != db_dim_row[0]:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid vector dimension. Expected {db_dim_row[0]}, got {len(req.vec)}."
                )

            cur.execute(f"SET ivfflat.probes = {PG_IVFFLAT_PROBES}")
            if req.metric == "cosine":
                operator = "<=>"
            else:  # l2
                operator = "<->"

            query = f"SELECT uri, vec {operator} %s AS distance FROM kg_embeddings ORDER BY distance LIMIT %s"
            cur.execute(query, (req.vec, req.k))
            neighbors = [{"uri": row[0], "distance": float(row[1])} for row in cur.fetchall()]
            return {"neighbors": neighbors}
    except psycopg2.Error as e:
        logger.error(f"Nearest neighbor search failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database query error")
    finally:
        if conn:
            db_pool.putconn(conn)


# Endpoint: Train KG embeddings
@app.post("/train", summary="Train entity + relation embeddings")
def train_graph(req: TrainRequest):
    if not req.triples:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot train on empty triples list.")

    # Prepare triples matrix
    triples = [(t.head, t.relation, t.tail) for t in req.triples]
    tf = TriplesFactory.from_labeled_triples(np.array(triples, dtype=object))

    # 80 / 10 / 10 split for train / valid / test
    train_tf, valid_tf, test_tf = tf.split(ratios=[0.8, 0.1, 0.1], random_state=42)

    # Run training pipeline
    try:
        result = pipeline(
            model=req.model_name,
            training=train_tf,
            validation=valid_tf,
            testing=test_tf,
            model_kwargs=dict(embedding_dim=req.dimension),
            training_kwargs=dict(num_epochs=req.epochs, batch_size=req.batch_size),
        )
    except Exception as e:
        logger.error(f"PyKEEN pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"PyKEEN training failed: {e}")

    def to_numpy(representation: Embedding) -> np.ndarray:
        """Extract numpy array from a PyKEEN representation object."""
        # The representation object can be called to get all embeddings.
        if hasattr(representation, "__call__"):
            # Calling with `indices=None` returns the full tensor.
            all_embeddings = representation(indices=None)
            return all_embeddings.detach().cpu().numpy()
        elif hasattr(representation, "weight"):
            return representation.weight.detach().cpu().numpy()
        else:
            raise TypeError(f"Unsupported representation type: {type(representation)}")

    # Extract learned vectors
    ent_vecs = to_numpy(result.model.entity_representations[0])
    rel_vecs = to_numpy(result.model.relation_representations[0])
    
    # Compute snapshot hashes
    triples_bytes = json.dumps([t.dict() for t in req.triples], sort_keys=True).encode('utf-8')
    sha_triples = hashlib.sha256(triples_bytes).hexdigest()
    weights_bytes = ent_vecs.tobytes() + rel_vecs.tobytes()
    sha_weights = hashlib.sha256(weights_bytes).hexdigest()
    snapshot_ual = f"did:dkg:model:{req.model_name.lower()}-{sha_weights[:16]}"

    # Generate Knowledge Asset snapshot and log it
    snapshot_ka = {
        "@context": "https://schema.org/",
        "@id": snapshot_ual,
        "@type": "ModelSnapshot",
        "embedding_dimension": req.dimension,
        "num_entities": len(tf.entity_id_to_label),
        "num_relations": len(tf.relation_id_to_label),
        "prov:generatedAtTime": datetime.now(timezone.utc).isoformat(),
        "sha256_weights": sha_weights,
        "sha256_triples": sha_triples,
    }
    logger.info(f"Generated Knowledge Asset snapshot: {json.dumps(snapshot_ka)}")

    # Upsert vectors
    ents = tf.entity_id_to_label
    ent_rows = [(ents[i], ent_vecs[i].tolist(), snapshot_ual) for i in range(len(ents))]
    rels = tf.relation_id_to_label
    rel_rows = [(rels[i], rel_vecs[i].tolist(), snapshot_ual) for i in range(len(rels))]

    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            # Entities
            execute_values(
                cur,
                """
                INSERT INTO kg_embeddings(uri, vec, snapshot_ual)
                VALUES %s
                ON CONFLICT (uri) DO UPDATE
                  SET vec = EXCLUDED.vec,
                      snapshot_ual = EXCLUDED.snapshot_ual,
                      updated_at = now()
                """,
                ent_rows
            )
            # Relations
            execute_values(
                cur,
                """
                INSERT INTO kg_rel_embeddings(relation_uri, vec, snapshot_ual)
                VALUES %s
                ON CONFLICT (relation_uri) DO UPDATE
                  SET vec = EXCLUDED.vec,
                      snapshot_ual = EXCLUDED.snapshot_ual,
                      updated_at = now()
                """,
                rel_rows
            )
        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Database upsert failed: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database upsert failed")
    finally:
        if conn:
            db_pool.putconn(conn)

    logger.info(f"Trained {req.model_name} [{snapshot_ual}]: {len(ents)} entities, {len(rels)} relations")
    logger.info(f"Training snapshot: {snapshot_ual}")
    return {"snapshot_ual": snapshot_ual, "entities": len(ents), "relations": len(rels), "max_block": req.max_block}

# Endpoint: Score a triple (TransE-style)
@app.post("/score", summary="Score a triple for link-prediction")
def score_triple(q: ScoreRequest):
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            # Head
            cur.execute("SELECT vec FROM kg_embeddings WHERE uri = %s", (q.head,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Entity vector not found for head: {q.head}")
            hv = np.array(row[0])
            # Relation
            cur.execute("SELECT vec FROM kg_rel_embeddings WHERE relation_uri = %s", (q.relation,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Relation vector not found for: {q.relation}")
            rv = np.array(row[0])
            # Tail
            cur.execute("SELECT vec FROM kg_embeddings WHERE uri = %s", (q.tail,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Entity vector not found for tail: {q.tail}")
            tv = np.array(row[0])
    except psycopg2.Error as e:
        logger.error(f"Score triple DB error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database query error")
    finally:
        if conn:
            db_pool.putconn(conn)

    # TransE score = -||h + r - t||
    score = float(-np.linalg.norm(hv + rv - tv))
    return {"score": score}
# Run via `uvicorn kg_service:app --reload --port 8001`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("kg_service:app", host="0.0.0.0", port=int(os.getenv("PORT", 8001)), reload=True)
