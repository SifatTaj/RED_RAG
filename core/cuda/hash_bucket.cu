#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "filter_hash.cuh"

// Error checking macros for CUDA and cuBLAS
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    const cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

/**
 * @brief Computes hash buckets for input embeddings using random projections.
 * @param h_embeddings Host pointer to input embeddings (size: N x dim).
 * @param h_projections Host pointer to random projection matrix (size: nbits x dim).
 * @param h_hashes Host pointer to output hash buckets (size: N).
 * @param N Number of input embeddings.
 * @param dim Dimension of each embedding.
 * @param nbits Number of bits for the hash.
 */

extern "C" {
    void compute_hash(float *h_embeddings, float *h_projections, unsigned int *h_hashes, int N, int dim, int nbits) {

        // Input Sizes
        size_t embedding_size = N * dim * sizeof(float);
        size_t projection_size = nbits * dim * sizeof(float);
        size_t output_size = N * nbits * sizeof(float);
        size_t hash_size = N * sizeof(int);

        // Device Pointers
        float *d_embeddings, *d_projections, *d_output;
        unsigned int *d_hashes;
        CHECK_CUDA(cudaMalloc(&d_embeddings, embedding_size));
        CHECK_CUDA(cudaMalloc(&d_projections, projection_size));
        CHECK_CUDA(cudaMalloc(&d_output, output_size));
        CHECK_CUDA(cudaMalloc(&d_hashes, hash_size));

        // Copy Input to Device
        CHECK_CUDA(cudaMemcpy(d_embeddings, h_embeddings, embedding_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_projections, h_projections, projection_size, cudaMemcpyHostToDevice));
        
        // Create cuBLAS Handle
        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));

        // Set alpha and beta for SGEMM
        float alpha = 1.0f;
        float beta = 0.0f;

        // Compute hashes using cuBLAS SGEMM
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N,        // No transpose on first matrix (projections)
            CUBLAS_OP_N,        // Transpose on second matrix (embeddings)
            nbits, N, dim,      // n = nbits, m = N, k = dim
            &alpha,
            d_projections, nbits,        
            d_embeddings, dim,        
            &beta,
            d_output, nbits
        ));

        // Launch filtering kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N * nbits + threadsPerBlock - 1) / threadsPerBlock;
        filter_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_hashes, N, nbits);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy result back to Host
        CHECK_CUDA(cudaMemcpy(h_hashes, d_hashes, hash_size, cudaMemcpyDeviceToHost));

        // Memory cleanup
        cudaFree(d_embeddings);
        cudaFree(d_projections);
        cudaFree(d_output);
        cublasDestroy(handle);
    }
}