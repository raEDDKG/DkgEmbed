# export_and_train.py
import os
import sys
import time
import logging
import argparse
import requests
import psycopg2
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format='{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}')
logger = logging.getLogger(__name__)

# --- Database Functions ---
PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    logger.error("Environment variable PG_DSN is required but not set.")
    raise RuntimeError("PG_DSN environment variable is required")

def get_last_block(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT val FROM kg_meta WHERE key = 'last_block_height'")
        row = cur.fetchone()
        return int(row[0]) if row else 0

def set_last_block(conn, block_height):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO kg_meta (key, val) VALUES ('last_block_height', %s) ON CONFLICT (key) DO UPDATE SET val = EXCLUDED.val",
            (str(block_height),)
        )
    conn.commit()

# --- SPARQL & Training Functions ---
SPARQL_TEMPLATE = """
PREFIX dkg: <https://ontology.origintrail.io/dkg/1.0#>
SELECT ?s ?p ?o ?block
WHERE {
  ?ka dkg:publishedAtBlock ?block .
  FILTER(?block > %s)
  GRAPH ?ka { ?s ?p ?o }
}
ORDER BY ?block
LIMIT %s
"""

def fetch_triples(last_block, batch_size):
    sparql_query = SPARQL_TEMPLATE % (last_block, batch_size)
    url = "http://localhost:9999/blazegraph/namespace/dkg/sparql"
    try:
        resp = requests.post(
            url,
            data={"query": sparql_query, "format": "application/sparql-results+json"},
            timeout=120,
        )
        resp.raise_for_status()
        rows = resp.json()["results"]["bindings"]
        if not rows:
            return [], 0
        
        triples = [(r["s"]["value"], r["p"]["value"], r["o"]["value"]) for r in rows]
        max_block = max(int(r["block"]["value"]) for r in rows)
        return triples, max_block
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to query Blazegraph: {e}")
        return None, 0

def submit_train_job(triples, max_block, batch_size):
    retries = 5
    delay = 5
    for i in range(retries):
        try:
            resp = requests.post(
                "http://localhost:8001/train",
                headers={"content-type": "application/json"},
                json={
                    "triples": [{"head": h, "relation": r, "tail": t} for h, r, t in triples],
                    "model_name": "ComplEx",
                    "dimension": 200,
                    "epochs": 30,
                    "batch_size": batch_size,
                    "max_block": max_block,
                },
                timeout=300,
            )
            if resp.status_code == 503:
                logger.warning(f"Training service returned 503. Retrying in {delay}s... ({i+1}/{retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            resp.raise_for_status()
            logger.info(f"✓ Train job submitted and completed: {len(triples)} triples, max_block: {max_block}")
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to submit train job: {e}")
            if i < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                logger.error("Giving up after multiple retries.")
                return None
    return None

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Export triples from Blazegraph and submit for training.")
    parser.add_argument("--since-block", type=int, help="Override the last processed block height from the database.")
    parser.add_argument("--batch-size", type=int, default=100000, help="Number of triples to process per batch.")
    args = parser.parse_args()

    try:
        conn = psycopg2.connect(PG_DSN)
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        sys.exit(1)

    last_block = args.since_block if args.since_block is not None else get_last_block(conn)
    logger.info(f"Starting export from block height: {last_block}")

    total_triples_processed = 0
    while True:
        logger.info(f"Fetching next batch of triples after block {last_block}...")
        triples, max_block_in_batch = fetch_triples(last_block, args.batch_size)

        if triples is None: # Error case
            logger.error("Halting due to error fetching triples.")
            break

        if not triples:
            logger.info("No new triples found. Exiting.")
            break

        logger.info(f"Fetched {len(triples)} new triples up to block {max_block_in_batch}.")
        
        result = submit_train_job(triples, max_block_in_batch, args.batch_size)

        if result:
            last_block = result["max_block"]
            set_last_block(conn, last_block)
            total_triples_processed += len(triples)
            logger.info(f"Successfully processed batch. New last_block is {last_block}.")
            logger.info(f"Training snapshot: {result['snapshot_ual']}")
        else:
            logger.error("Failed to process batch after multiple retries. Halting.")
            break
            
        # If we received fewer triples than the batch size, we are likely at the end
        if len(triples) < args.batch_size:
            logger.info("Processed the last batch of triples.")
            break

    logger.info(f"Export and train job finished. Total triples processed: {total_triples_processed}")
    conn.close()

if __name__ == "__main__":
    main()
