import argparse

from sentence_transformers import SentenceTransformer
from core.vector_store import VectorStore
from core.util import make_chunks, load_yaml

def init_vector_store(config) -> VectorStore:
    chunks = make_chunks(config["doc_path"], chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"])
    embedding_model = SentenceTransformer(config["embedding_model"], config["backend"])
    vector_store = VectorStore(embedding_model, backend=config["backend"])
    vector_store.add_chunks(chunks)
    vector_store.generate_hash_buckets(config["n_bits"])
    vector_store.index(threshold=config["deduplication_threshold"], method=config["index_method"])
    return vector_store

def main(cfg_path):
    config = load_yaml(cfg_path)
    vector_store = init_vector_store(config)
    print("RED_RAG Vector Store is ready!")

    while(True):
        query = input("Enter a search query: ")
        res = vector_store.search(query, k=config["search_top_k"])
        print(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RED_RAG")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file")
    args = parser.parse_args()
    main(args.config)
