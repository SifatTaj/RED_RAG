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

1. Install the required packages
```bash
pip install -r requirements.txt
```

2. Initialize the RED_RAG vector store with an embedding model. Set backend to either `"cuda"` or `"pytorch"`:

```python
from sentence_transformers import SentenceTransformer

# Initialize an embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initilize the vector store
vector_store = VectorStore(embedding_model, backend="pytorch")    
```

3. Select a document in plain text to embed and add to the vector store. Set `chunk_size` and `chunk_overlap` according to your preference:

```python
from core.util import make_chunks

# Create text chunks
chunks = make_chunks("example_doc/wixqa.txt", chunk_size=100, chunk_overlap=20)

# Add the chunks to the vector store
vector_store.add_chunks(chunks)
```

4. Now compute the hash using random projections and group the embeddings to their corresponding buckets. Set the number of bits in `nbits` for the hashes:

```python
vector_store.generate_hash_buckets(n_bits=4)
```

5. Finally, deduplicate and index the embeddings. Set the similarity threshold for deduplication in `threshold`. Currently, the buckets can be indexed with either `"FlatL2"` or `"HNSW"`. Set the indexing method in `method`:

```python
vector_store.index(threshold=0.9, method="FlatL2")
```

6. The vector store can be now be queried. Set the number of retrieved chunks in `k`:

```python
res = vector_store.search(query, k=4)
print(res)
```

Alternatively, the `config.yaml` file can be configured and launched using `red_rag.py`:

```yaml
doc_path: "example_doc/wixqa.txt"                         # path to the document in plain text
chunk_size: 100                                           # size of each text chunk for embedding
chunk_overlap: 20                                         # number of overlapping characters for each chunk
embedding_model: "sentence-transformers/all-MiniLM-L6-v2" # name or path to the embedding model
backend: "pytorch"                                        # choose between "pytorch" and "cuda" backend
index_method: "FlatL2"                                    # supports "FlatL2" and "HNSW"
deduplication_threshold: 0.9                              # similarity threshold for deduplication
search_top_k: 10                                          # number of retrieved chunks
n_bits: 4                                                 # number of bits for hashing
```

## CUDA Backend

The CUDA files in `core/cuda` need to be compiled into shared objects before the CUDA backend can be used. The CUDA backend should provide better performance.

### CUDA requirements (Linux)

To build and run the CUDA backend you need a working CUDA development environment. Minimum/recommended items:

- NVIDIA driver compatible with your CUDA Toolkit version (install driver from NVIDIA).  
- CUDA Toolkit (nvcc, runtime, dev headers). Recommended: CUDA 11.7+ (11.8 or 12.x commonly used). Match the toolkit version with any prebuilt libtorch you use.
- cuBLAS and cuBLASLt (shipped with the CUDA Toolkit) — required for GEMM/FP32/FP8 paths.
- nvcc in PATH and the CUDA libraries in LD_LIBRARY_PATH:
  - export PATH=/usr/local/cuda/bin:$PATH
  - export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
- CMake >= 3.18 (required for modern CUDA + libtorch CMake integration).
- A host compiler compatible with your CUDA/toolkit and libtorch (GCC 9–12 typically safe; check libtorch docs for ABI compatibility).

- A CUDA-capable NVIDIA GPU with compute capability >= 6.0 is recommended for good performance.

**Use the Makefile to compile the CUDA files:**
```bash
make
```