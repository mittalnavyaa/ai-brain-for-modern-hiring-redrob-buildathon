import os
import pickle
from typing import Iterator, Dict, Any, List, Set
import orjson
from flashtext2 import KeywordProcessor

def memory_mapped_stream(jsonl_path: str) -> Iterator[Dict[str, Any]]:
    """
    Streams individual candidate JSON records from a raw JSONL file.
    Utilizes Rust-backed SIMD parsing to keep memory footprints < 1.2 GB.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Target data file not found: {jsonl_path}")
        
    with open(jsonl_path, "rb") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue
            # Yield parsed record instantly using SIMD acceleration
            yield orjson.loads(line)

class TechnicalTaxonomyExtractor:
    """
    Tracks and matches custom skill domains using the Aho-Corasick algorithm 
    via flashtext2 for constant-time deterministic matching.
    """
    def __init__(self, trie_path: str = "data/dictionary_trie.pkl"):
        self.trie_path = trie_path
        self.processor = KeywordProcessor(case_sensitive=False)
        
        # Define taxonomic categories mapped out by system design
        self.foundational_ml: Set[str] = {
            "vector databases", "pinecone", "faiss", "milvus", "embeddings", 
            "custom rankers", "ndcg", "mrr", "backpropagation", "pytorch"
        }
        self.wrapper_hype: Set[str] = {
            "langchain", "openai api", "chatgpt api", "wrapper development", 
            "wrappers", "prompt engineering"
        }
        self.evaluation_metrics: Set[str] = {
            "ndcg", "mrr", "a/b testing", "map", "precision-at-k"
        }

    def build_and_serialize_trie(self) -> None:
        """Builds the FlashText prefix tree and serializes it to disk."""
        # Populate custom terms into processor
        for term in self.foundational_ml:
            self.processor.add_keyword(term, f"foundational:{term}")
        for term in self.wrapper_hype:
            self.processor.add_keyword(term, f"wrapper:{term}")
        for term in self.evaluation_metrics:
            self.processor.add_keyword(term, f"eval:{term}")

        # Ensure the storage directory exists
        os.makedirs(os.path.dirname(self.trie_path), exist_ok=True)
        with open(self.trie_path, "wb") as f:
            pickle.dump(self.processor, f)
        print(f"[Parser] Taxonomic FlashText Dictionary Trie serialized to {self.trie_path}")

    def load_trie(self) -> None:
        """Loads a pre-compiled structural dictionary trie from disk."""
        if not os.path.exists(self.trie_path):
            self.build_and_serialize_trie()
        else:
            with open(self.trie_path, "rb") as f:
                self.processor = pickle.load(f)

    def extract_skills(self, text_corpus: str) -> Dict[str, List[str]]:
        """Extracts technical taxonomy phrases in a single deterministic pass."""
        matches = self.processor.extract_keywords(text_corpus)
        extracted = {"foundational": [], "wrapper": [], "eval": []}
        
        for match in matches:
            category, term = match.split(":", 1)
            extracted[category].append(term)
            
        return extracted