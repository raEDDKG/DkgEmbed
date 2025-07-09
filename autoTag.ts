//autoTag.ts
import axios from 'axios';
import 'dotenv/config';

const KG_SERVICE_URL = process.env.KG_SERVICE_URL || 'http://localhost:8001';

interface ScoreResponse {
  score: number;
}

interface Neighbor {
  uri: string;
  distance: number;
}

interface NearestNeighborsResponse {
  neighbors: Neighbor[];
}

const NEIGHBOR_DISTANCE_THRESHOLD = parseFloat(process.env.NEIGHBOR_DISTANCE_THRESHOLD || "0.4");

/**
 * Checks for missing links for a given post and, if plausible,
 * publishes a patch Knowledge-Asset to add the link.
 *
 * @param postId The full UAL of the post, e.g., "uuid:POST-1".
 * @param promptVec The vector representation of the post's content.
 */
async function autoTag(postId: string, promptVec: number[]): Promise<void> {
  console.log(`[autoTag] Starting process for post: ${postId}`);

  try {
    // 1. Find candidate tail entities using the nearest_neighbors endpoint
    const nnResponse = await axios.post<NearestNeighborsResponse>(
      `${KG_SERVICE_URL}/nearest_neighbors`,
      {
        vec: promptVec,
        k: 20, // Number of candidates to check
        metric: 'cosine',
      }
    );

    const candidates = nnResponse.data.neighbors;
    console.log(`[autoTag] Found ${candidates.length} raw candidates for ${postId}`);

    // 2. Filter candidates by distance and score them
    for (const candidate of candidates) {
      if (candidate.distance > NEIGHBOR_DISTANCE_THRESHOLD) {
        continue;
      }

      const triple = {
        head: postId,
        relation: 'http://schema.org/about',
        tail: candidate.uri,
      };

      try {
        const scoreResponse = await axios.post<ScoreResponse>(
          `${KG_SERVICE_URL}/score`,
          triple
        );

        const { score } = scoreResponse.data;
        console.log(`[autoTag] Score for (${postId}, about, ${candidate.uri}): ${score.toFixed(4)}`);

        // 3. If score is high, publish a patch KA (placeholder)
        // TODO: Check if the triple already exists in the DKG before publishing.
        // The SPARQL pattern is: ASK { <head> <relation> <tail> }.
        if (score >= 0.8) {
          console.log(`[autoTag] PLAUSIBLE LINK FOUND: (${triple.head}, ${triple.relation}, ${triple.tail}) with score ${score}.`);
          // Placeholder for publishing the patch Knowledge Asset
          // e.g., publishPatchKA(triple);
        }
      } catch (error) {
        if (axios.isAxiosError(error) && error.response?.status === 404) {
          // This is expected if a candidate vector doesn't exist, so we can ignore it.
        } else {
          console.error(`[autoTag] Error scoring triple for candidate ${candidate.uri}:`, error);
        }
      }
    }
  } catch (error) {
    console.error(`[autoTag] Failed to get nearest neighbors for post ${postId}:`, error);
  }

  console.log(`[autoTag] Finished process for post: ${postId}`);
}

// Example usage:
// const exampleVector = Array(200).fill(0).map(() => Math.random());
// autoTag('uuid:some-post-id', exampleVector);