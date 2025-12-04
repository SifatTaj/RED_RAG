import ctypes
import os

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from numpy.ctypeslib import ndpointer


class VectorStore:
    """
    Manages the storage, retrieval, and deduplication of embeddings.

    The VectorStore class is designed to store embedding vectors, organize them into hash buckets,
    and identify duplicate embeddings based on similarity thresholds. It uses either PyTorch or custom CUDA
    backends for embedding calculation and similarity checks. The stored embeddings can be indexed and
    searched using either FlatL2 or HNSW.

    :ivar embedding_model: The model used to compute the embeddings.
    :type embedding_model: SentenceTransformer (Supports
    :ivar backend: Backend used for computations, either 'cuda' or 'pytorch'.
    :type backend: str
    """
    def __init__(self,
                 embedding_model: SentenceTransformer,
                 backend: str,
                 ):

        self.bucket_indices = {}
        self.all_index = None
        self.chunks = []
        self.embeddings = None
        self.n_embeddings = 0
        self.embedding_model = embedding_model
        self.dim = embedding_model.get_sentence_embedding_dimension() \
            if isinstance(self.embedding_model, SentenceTransformer) \
            else self.embedding_model.dim
        self.projections = None
        self.buckets = {}
        self.bucket_chunks = {}
        self.bucket_embs = {}
        self.unique_chunks = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backend = backend

        print(f"Initalized RED_RAG vector store with {backend} backend.")

    def add_chunks(self, chunks):
        """
        Embeds and adds the text chunks to the vector store.

        :param chunks: A list of strings to be added to the vector store.
        :type chunks: list[str]
        :return: None
        """

        print(f"Adding {len(chunks)} chunks to vector store...")

        self.chunks.extend(chunks)

        if self.backend == "pytorch":
            self.embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True).to(self.device)
        elif self.backend == "cuda":
            self.embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)

        self.n_embeddings = len(self.embeddings)
        print(f"{self.n_embeddings} embeddings added to the vector store.")

    def _generate_hash_buckets_cuda(self, nbits):
        """
        Computes nbits hash of each embedding using the CUDA backend.

        This method uses a compiled CUDA shared library to compute hash buckets for
        given embeddings and projections, based on the specified number of bits. The
        CUDA library is loaded dynamically at runtime. The function prepares
        appropriate argument types for the operations and delegates the computation
        to the CUDA library.

        :param nbits: Number of bits to be used for hashing.
        :type nbits: int
        :return: Array of computed hash bucket indices.
        :rtype: numpy.ndarray
        """
        lib_path = os.path.abspath("core/cuda/lib/hash_bucket.so")
        lib = ctypes.CDLL(lib_path)

        lib.compute_hash.argtypes = [
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        hashes = np.zeros(self.n_embeddings, dtype=np.int32)

        lib.compute_hash(
            self.embeddings.astype(np.float32),
            self.projections.cpu().numpy().astype(np.float32),
            hashes,
            self.n_embeddings,
            self.dim,
            nbits
        )
        return hashes

    def _generate_hash_buckets_pytorch(self):
        """
        Computes hash buckets using the embeddings and projection matrices.

        This method computes the dot product of the embeddings and projections using torch.matmul().
        Depending on the available device, this operation can use either "cpu" or "cuda".

        :rtype: list[int]
        :return: A list of integers corresponding to computed hash buckets.
        """
        hashes = torch.matmul(self.embeddings, self.projections)
        hashes = hashes > 0
        hashes = hashes.cpu().numpy().astype(int)
        hashes = [int(''.join(h.astype(str)), 2) for h in hashes]
        return hashes

    def generate_hash_buckets(self, nbits):
        """
        Computes nbits hash of each embedding and puts it in the corresponding hash bucket.

        The function works by creating random projections of a certain number of bits (`nbits`),
        then generating hash buckets based on the specified backend. The embeddings are categorized
        into the hash buckets, which are stored along with their chunks and embeddings.

        :param nbits: Number of bits used for the hash projections.
        :type nbits: int
        :return: None
        """

        self.projections = torch.rand(self.dim, nbits, device=self.device, dtype=torch.float32) - .5

        if self.backend == "pytorch":
            hashes = self._generate_hash_buckets_pytorch()
        elif self.backend == "cuda":
            hashes = self._generate_hash_buckets_cuda(nbits)

        for i, h in enumerate(hashes):
            if h not in self.buckets.keys():
                self.buckets[h] = []
                self.bucket_chunks[h] = []
                self.bucket_embs[h] = []

            self.buckets[h].append(i)
            self.bucket_chunks[h].append(self.chunks[i])
            self.bucket_embs[h].append(self.embeddings[i])

        print(f"{len(self.buckets)} hash buckets created.")

    def _similarity_check_cuda(self, threshold):
        """
        Checks for similarity between embeddings using a custom CUDA shared library. The
        method computes a cosine similarity matrix for the embeddings within each bucket and
        identifies pairs of indices where the similarity exceeds the given threshold.

        :param threshold: A float value representing the similarity threshold. Only pairs
            of embeddings whose cosine similarity exceeds this threshold will be considered
            as duplicates.
        :return: A set of indices representing all detected duplicates based on the
            similarity threshold.
        """

        all_dupes = set()

        lib_path = os.path.abspath("core/cuda/lib/cos_sim.so")
        lib = ctypes.CDLL(lib_path)

        for k in self.buckets:
            bucket = self.buckets[k]
            embs = [self.embeddings[i] for i in bucket]

            bucket_size = len(embs)
            embs = np.concatenate(embs).astype(np.float32)
            sim_matrix = np.zeros(bucket_size ** 2, dtype=np.float32)

            lib.compute_sim_matrix.argtypes = [
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int,
                ctypes.c_int
            ]

            lib.compute_sim_matrix(embs, sim_matrix, bucket_size, self.dim)
            sim_matrix = torch.from_numpy(sim_matrix)
            sim_matrix = sim_matrix.reshape(bucket_size, bucket_size)

            upper_tri_matrix = torch.triu(sim_matrix, diagonal=1)
            indices = torch.nonzero(upper_tri_matrix > threshold, as_tuple=False)
            all_dupes.update([bucket[j] for (i, j) in indices])

        return all_dupes

    def _similarity_check_pytorch(self, threshold):
        """
        Computes the similarity matrix using the Pytorch backend.

        :param threshold: The similarity threshold value (between 0 and 1) above
                          which embeddings are considered similar.
        :type threshold: float
        :return: A set of indices representing duplicates within the embeddings
                 based on the provided threshold.
        :rtype: set
        """

        all_dupes = set()

        for k in self.buckets:
            bucket = self.buckets[k]
            embs = [self.embeddings[i] for i in bucket]
            embs = torch.stack(embs)

            embs = F.normalize(embs, p=2, dim=1)
            similarities = torch.mm(embs, embs.transpose(0, 1))

            upper_tri_matrix = torch.triu(similarities, diagonal=1)
            indices = torch.nonzero(upper_tri_matrix > threshold, as_tuple=False)
            all_dupes.update([bucket[j] for (i, j) in indices])

        return all_dupes

    def index(self, threshold, method="FlatL2"):
        """
        Deduplicates and indexes data using the specified similarity threshold and indexing method.
        This method processes embeddings to remove duplicates using either CUDA or PyTorch as the
        backend and indexes the remaining unique embeddings with FAISS.

        It checks for similarities between embedded vectors stored in buckets and
        identifies duplicates based on a specified similarity threshold. The method
        compares embeddings using cosine similarity, normalizes them, and calculates
        pairwise similarities to detect duplicates.

        The indexing method can be specified as either "FlatL2" or "HNSW". After deduplication,
        the reduction percentage is printed.

        :param threshold: Float value specifying the similarity threshold for deduplication.
        :param method: String specifying the type of FAISS indexing method to use. Defaults to "FlatL2".
        :return: None
        """

        print(f"Deduplicating and indexing buckets with {method}...")

        self.bucket_embs = None
        self.bucket_chunks = None

        if self.backend == "cuda":
            all_dupes = self._similarity_check_cuda(threshold)
        elif self.backend == "pytorch":
            all_dupes = self._similarity_check_pytorch(threshold)

        unique_embeddings = [emb for i, emb in enumerate(self.embeddings) if i not in all_dupes]
        self.unique_chunks = [chunk for i, chunk in enumerate(self.chunks) if i not in all_dupes]

        if method == "FlatL2":
            self.all_index = faiss.IndexFlatL2(self.dim)
        elif method == "HNSW":
            self.all_index = faiss.IndexHNSW(self.dim, 32, faiss.METRIC_L2)

        if self.backend == "cuda":
            unique_embeddings = np.stack(unique_embeddings)
        elif self.backend == "pytorch":
            unique_embeddings = torch.stack(unique_embeddings).cpu()

        self.all_index.add(unique_embeddings)

        reduction = (len(all_dupes) / self.n_embeddings) * 100
        print(f"Deduplication: {reduction:.2f}%")

    def search(self, query, k=3):
        """
        Searches for the top-k most similar chunks to the provided query using the specified
        embedding model and a search index. This function embeds the query, performs
        a search in the index, and retrieves and returns the top-k chunks with the smallest
        distances to the query.

        :param query: The input query string to search for similar chunks.
        :type query: str
        :param k: The number of top similar chunks to retrieve. Defaults to 3.
        :type k: int, optional
        :return: A list of the top-k chunks that are most similar to the provided query based
            on the embedding similarity.
        :rtype: list
        """
        query_embed = self.embedding_model.encode([query])
        distances, indices = self.all_index.search(query_embed, k)
        return [self.unique_chunks[i] for i in indices[0]]
