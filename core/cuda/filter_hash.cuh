/**
 * @brief CUDA kernel to filter hash values.
 * @param d_hashes Pointer to the device array of hash values.
 * @param N Number of data points.
 * @param nbits Number of bits per hash.
 */

__global__ void filter_hash_kernel(float *d_output, unsigned int *d_hashes, int N, int nbits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int offset = idx * nbits;
        unsigned int hash = 0;

        for (int hash_index = 0; hash_index < nbits; hash_index++) {
            // Set hash_index bit to 1 if > 0
            if (d_output[offset + hash_index] > 0) {
                hash |= (1 << (nbits - hash_index - 1));
            }
        }
        d_hashes[idx] = hash;
    }
}