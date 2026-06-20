import os
from typing import Iterator, Dict, Any, List, Set

import orjson
from flashtext import KeywordProcessor


def memory_mapped_stream(jsonl_path: str) -> Iterator[Dict[str, Any]]:
    """
    Streams individual candidate JSON records from a raw JSONL file.
    """

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Target data file not found: {jsonl_path}"
        )

    with open(jsonl_path, "rb") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue

            yield orjson.loads(line)


class TechnicalTaxonomyExtractor:
    """
    Deterministic technical skill extraction using FlashText.
    """

    def __init__(self, trie_path: str = "data/dictionary_trie.pkl"):
        self.trie_path = trie_path
        self.processor = KeywordProcessor(case_sensitive=False)

        self.foundational_ml: Set[str] = {
            "vector databases",
            "pinecone",
            "faiss",
            "milvus",
            "embeddings",
            "custom rankers",
            "ndcg",
            "mrr",
            "backpropagation",
            "pytorch",
        }

        self.wrapper_hype: Set[str] = {
            "langchain",
            "openai api",
            "chatgpt api",
            "wrapper development",
            "wrappers",
            "prompt engineering",
        }

        self.evaluation_metrics: Set[str] = {
            "ndcg",
            "mrr",
            "a/b testing",
            "map",
            "precision-at-k",
        }

        self._build_trie()

    def _build_trie(self) -> None:
        """
        Build FlashText keyword trie in memory.
        """

        self.processor = KeywordProcessor(case_sensitive=False)

        for term in self.foundational_ml:
            self.processor.add_keyword(
                term,
                f"foundational:{term}"
            )

        for term in self.wrapper_hype:
            self.processor.add_keyword(
                term,
                f"wrapper:{term}"
            )

        for term in self.evaluation_metrics:
            self.processor.add_keyword(
                term,
                f"eval:{term}"
            )

    def build_and_serialize_trie(self) -> None:
        """
        Compatibility method retained for older code.

        FlashText KeywordProcessor objects cannot be pickled,
        so we simply rebuild the trie in memory.
        """

        self._build_trie()

        os.makedirs(
            os.path.dirname(self.trie_path),
            exist_ok=True
        )

        print(
            f"[Parser] FlashText trie built successfully."
        )

    def load_trie(self) -> None:
        """
        Compatibility method retained for older code.
        """

        self._build_trie()

    def extract_skills(
        self,
        text_corpus: str
    ) -> Dict[str, List[str]]:
        """
        Extract technical taxonomy phrases.
        """

        matches = self.processor.extract_keywords(text_corpus)

        extracted = {
            "foundational": [],
            "wrapper": [],
            "eval": [],
        }

        for match in matches:
            category, term = match.split(":", 1)
            extracted[category].append(term)

        return extracted