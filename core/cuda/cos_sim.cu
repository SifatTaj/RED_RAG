#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "l2_norm_kernel.cuh"

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
 * @brief Computes the cosine similarity matrix inplace for a hash bucket using CUDA and cuBLAS.
 * @param h_bucket Host vector containing N vectors of dimension D (size N*D).
 * @param h_sim_matrix Host vector containing the NxN cosine similarity matrix.
 * @param N Number of vector embeddings.
 * @param D Dimension of each vector embeddings.
 */

extern "C" {
    void compute_sim_matrix(float *h_bucket, float *h_sim_matrix, int N, int D) {

        size_t input_size = N * D * sizeof(float);
        size_t output_size = N * N * sizeof(float);

        // Device Pointers
        float *d_bucket, *d_sim_matrix;
        CHECK_CUDA(cudaMalloc(&d_bucket, input_size));
        CHECK_CUDA(cudaMalloc(&d_sim_matrix, output_size));

        // Copy Input to Device
        CHECK_CUDA(cudaMemcpy(d_bucket, h_bucket, input_size, cudaMemcpyHostToDevice));

        // Step 1: Normalize Vectors
        int threadsPerBlock = 1024;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        
        // Launch L2 Normalization Kernel
        l2_normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_bucket, N, D);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Step 2: Compute Cosine Similarity (Matrix Mul)

        // Create cuBLAS Handle
        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));

        // Set alpha and beta for SGEMM
        float alpha = 1.0f;
        float beta = 0.0f;

        // We want C = A * A^T (where A is NxD).
        // cuBLAS assumes Column-Major. 
        // If we pass our Row-Major array 'A' to cuBLAS, cuBLAS reads it as A^T (DxN).
        // So, to compute (N x N) output, we technically tell cuBLAS to compute:
        // (A^T)^T * (A^T) which results in A * A^T.
        // This means we use OP_T on the first arg and OP_N on the second.
        
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_T,        // Transpose the first matrix (viewed as col-major)
            CUBLAS_OP_N,        // No transpose on second matrix
            N, N, D,            // m, n, k
            &alpha,
            d_bucket, D,        // LDA (Leading Dimension of A) is D because row-major trick
            d_bucket, D,        // LDB
            &beta,
            d_sim_matrix, N     // LDC
        ));

        // Copy result back to Host
        CHECK_CUDA(cudaMemcpy(h_sim_matrix, d_sim_matrix, output_size, cudaMemcpyDeviceToHost));

        // Memory cleanup
        cudaFree(d_bucket);
        cudaFree(d_sim_matrix);
        cublasDestroy(handle);
    }
}