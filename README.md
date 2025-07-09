# DkgEmbed

This project contains a set of services and scripts to create and maintain knowledge graph embeddings from an OriginTrail Decentralized Knowledge Graph (DKG).

## Components

1.  **`kg_service.py`**: A FastAPI microservice that:
    *   Trains knowledge graph embeddings using PyKEEN.
    *   Stores the learned vectors in a PostgreSQL database with the `pgvector` extension.
    *   Exposes endpoints to score triple plausibility (`/score`) and find nearest neighbor entities (`/nearest_neighbors`).

2.  **`export_and_train.py`**: A command-line script that:
    *   Incrementally exports triples from a Blazegraph SPARQL endpoint.
    *   Submits the triples to the `kg_service.py` for training.
    *   Tracks the last processed block height in a PostgreSQL table (`kg_meta`).

3.  **`autoTag.ts`**: A TypeScript sketch for a downstream enrichment plugin that:
    *   Uses the `/nearest_neighbors` and `/score` endpoints to discover and validate new, plausible triples.

## Database Schema

The system requires a PostgreSQL database with the `pgvector` extension enabled. The following tables are used:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE kg_embeddings (
  uri TEXT PRIMARY KEY,
  vec vector(1024) NOT NULL,
  snapshot_ual TEXT NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE kg_rel_embeddings (
  relation_uri TEXT PRIMARY KEY,
  vec vector(1024) NOT NULL,
  snapshot_ual TEXT NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE kg_meta (
  key TEXT PRIMARY KEY,
  val TEXT NOT NULL
);

-- Indexes for performance
CREATE INDEX kg_ent_vec_cos_idx ON kg_embeddings USING ivfflat (vec vector_cosine_ops) WITH (lists = 100);
CREATE INDEX kg_ent_vec_l2_idx ON kg_embeddings USING ivfflat (vec vector_l2_ops) WITH (lists = 100);
```

### Vector Dimension Future-Proofing

The `vec` columns in `kg_embeddings` and `kg_rel_embeddings` are defined as `vector(1024)` to provide flexibility for future model changes without requiring an immediate data migration. If you need to increase the embedding dimension beyond this limit (e.g., to 2048), you can run the following DDL command:

```sql
ALTER TABLE kg_embeddings ALTER COLUMN vec TYPE vector(2048);
ALTER TABLE kg_rel_embeddings ALTER COLUMN vec TYPE vector(2048);
```

After altering the table, you will need to rebuild the `ivfflat` indexes.