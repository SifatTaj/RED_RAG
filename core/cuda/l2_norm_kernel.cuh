/**
 * @brief L2 Normalization Kernel
 * @param d_bucket Pointer to the bucket of vectors in device memory
 * @param N Number of vectors
 * @param D Dimensions per vector
 * @note Each vector is normalized in place
 * @note Uses fast inverse square root (rsqrtf) for efficiency
 * @note Adds a small epsilon to avoid division by zero
 */

__global__ void l2_normalize_kernel(float* d_bucket, int N, int D) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < N) {
        float sum_sq = 0.0f;
        int offset = row_idx * D;

        // Calculate Sum of Squares
        for (int i = 0; i < D; ++i) {
            float val = d_bucket[offset + i];
            sum_sq += val * val;
        }

        // Calculate Inverse Norm (rsqrt is fast inverse square root)
        float inv_norm = rsqrtf(sum_sq + 1e-12f); 

        // Scale the vector
        for (int i = 0; i < D; ++i) {
            d_bucket[offset + i] *= inv_norm;
        }
    }
}