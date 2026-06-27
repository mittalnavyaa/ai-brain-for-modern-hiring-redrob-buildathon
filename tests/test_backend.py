import unittest

from app import build_dataset, rank_candidates


class BackendIntegrationTests(unittest.TestCase):
    def test_rank_candidates_returns_ranked_results(self):
        results = rank_candidates("Senior DevOps engineer with Kubernetes and AWS", limit=5)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["rank"], 1)
        self.assertIn("score", results[0])

    def test_build_dataset_returns_candidate_rows(self):
        rows = build_dataset(limit=5)
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("candidate_id", rows[0])


if __name__ == "__main__":
    unittest.main()
