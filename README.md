<h2 align="center">
  RED_RAG
</h2>

RED_RAG is a fast and lightweight semantic deduplicator and retriever for Retrieval Augmented Generation (RAG).
It uses random projection-based hashing to group similar embeddings and eliminate duplicate embeddings above a certain 
threshold using cosine similarity. RED_RAG is faster than SemHash and SemDeDup while achieving similar deduplication
performance.

RED_RAG currently supports two backends:
1. **CUDA:** Written in native CUDA that leverages Nvidia CUDA Toolkits.
2. **Pytorch:** For running on CPUs and other Pytorch backends.
## Quickstart
Install the required packages
```bash
pip install -r requirements.txt
```

Initialize the RED_RAG vector store with an embedding model. Set backend to either 