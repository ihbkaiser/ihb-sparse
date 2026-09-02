#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace {

using BFloat16 = __nv_bfloat16;

__device__ __forceinline__ float bf16_to_float(const BFloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ BFloat16 float_to_bf16(const float value) {
    return __float2bfloat16(value);
}

__device__ __forceinline__ float block_max(float value, float* shared, int threads) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ __forceinline__ float block_sum(float value, float* shared, int threads) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void retrieval_softmax_kernel(
    const BFloat16* __restrict__ query,
    const BFloat16* __restrict__ landmarks,
    BFloat16* __restrict__ output,
    int batch_size,
    int query_heads,
    int kv_heads,
    int groups,
    int chunks,
    int head_dim) {
    const int work = blockIdx.x;
    const int group_work = kv_heads * groups;
    const int batch_idx = work / group_work;
    const int head_group = work % group_work;
    const int kv_head = head_group / groups;
    const int group = head_group % groups;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) {
        return;
    }

    extern __shared__ float shared[];
    float* scores = shared;
    float* max_reduce = scores + chunks;
    float* sum_reduce = max_reduce + blockDim.x;
    const BFloat16* q_ptr = query + (batch_idx * query_heads + kv_head * groups + group) * head_dim;
    const BFloat16* landmark_ptr = landmarks + (batch_idx * kv_heads + kv_head) * chunks * head_dim;
    BFloat16* output_ptr = output + (batch_idx * kv_heads + kv_head) * groups * chunks + group * chunks;

    float local_max = -INFINITY;
    for (int chunk = tid; chunk < chunks; chunk += blockDim.x) {
        float score = 0.0f;
        const BFloat16* key_ptr = landmark_ptr + chunk * head_dim;
        for (int dim = 0; dim < head_dim; ++dim) {
            score += bf16_to_float(q_ptr[dim]) * bf16_to_float(key_ptr[dim]);
        }
        scores[chunk] = score * 0.08838834764831843f; // 1/sqrt(128), official scale
        local_max = fmaxf(local_max, scores[chunk]);
    }
    const float max_score = block_max(local_max, max_reduce, blockDim.x);

    float local_sum = 0.0f;
    for (int chunk = tid; chunk < chunks; chunk += blockDim.x) {
        local_sum += expf(scores[chunk] - max_score);
    }
    const float sum_score = block_sum(local_sum, sum_reduce, blockDim.x);

    for (int chunk = tid; chunk < chunks; chunk += blockDim.x) {
        output_ptr[chunk] = float_to_bf16(expf(scores[chunk] - max_score) / sum_score);
    }
}

__global__ void gather_gemm_rope_kernel(
    const BFloat16* __restrict__ u,
    const BFloat16* __restrict__ sv,
    const int64_t* __restrict__ position_ids,
    const BFloat16* __restrict__ cos_sin_cache,
    BFloat16* __restrict__ output,
    int tokens,
    int heads,
    int selected_tokens,
    int rank,
    int head_dim) {
    const int work = blockIdx.x;
    const int head = work / selected_tokens;
    const int selected = work % selected_tokens;
    const int tid = threadIdx.x;

    if (head >= heads) {
        return;
    }

    extern __shared__ float shared[];
    float* u_row = shared;
    float* raw = shared + rank;

    for (int r = tid; r < rank; r += blockDim.x) {
        const int64_t position = position_ids[head * selected_tokens + selected];
        u_row[r] = bf16_to_float(u[position * rank + r]);
    }
    __syncthreads();

    if (tid < head_dim) {
        float value = 0.0f;
        for (int r = 0; r < rank; ++r) {
            value += u_row[r] * bf16_to_float(sv[(head * rank + r) * head_dim + tid]);
        }
        raw[tid] = value;
    }
    __syncthreads();

    if (tid < head_dim / 2) {
        const int64_t position = position_ids[head * selected_tokens + selected];
        const BFloat16* rope = cos_sin_cache + position * head_dim;
        const float x1 = raw[tid];
        const float x2 = raw[tid + head_dim / 2];
        const float cosine = bf16_to_float(rope[tid]);
        const float sine = bf16_to_float(rope[tid + head_dim / 2]);
        BFloat16* output_ptr = output + (head * selected_tokens + selected) * head_dim;
        output_ptr[tid] = float_to_bf16(x1 * cosine - x2 * sine);
        output_ptr[tid + head_dim / 2] = float_to_bf16(x2 * cosine + x1 * sine);
    }
}

void check_cuda_bfloat16(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be bfloat16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

} // namespace

torch::Tensor retrieval_softmax_cuda(
    torch::Tensor query,
    torch::Tensor landmarks) {
    check_cuda_bfloat16(query, "query");
    check_cuda_bfloat16(landmarks, "landmarks");
    TORCH_CHECK(query.dim() == 3, "query must have shape [batch, query_heads, head_dim]");
    TORCH_CHECK(landmarks.dim() == 4, "landmarks must have shape [batch, kv_heads, chunks, head_dim]");
    TORCH_CHECK(query.size(0) == landmarks.size(0), "query and landmarks batch sizes must match");
    TORCH_CHECK(query.size(2) == landmarks.size(3), "query and landmark head dimensions must match");
    TORCH_CHECK(query.size(1) % landmarks.size(1) == 0, "query heads must be divisible by KV heads");
    TORCH_CHECK(query.size(2) > 0, "head dimension must be positive");

    const int batch_size = query.size(0);
    const int query_heads = query.size(1);
    const int kv_heads = landmarks.size(1);
    const int groups = query_heads / kv_heads;
    const int chunks = landmarks.size(2);
    const int head_dim = query.size(2);
    constexpr int threads = 256;
    TORCH_CHECK(chunks <= 16384, "landmark chunk count exceeds fused kernel shared-memory limit");
    auto output = torch::empty({batch_size, kv_heads, groups, chunks}, query.options());
    const auto stream = at::cuda::getCurrentCUDAStream(query.get_device()).stream();
    retrieval_softmax_kernel<<<batch_size * kv_heads * groups, threads, (chunks + 2 * threads) * sizeof(float), stream>>>(
        reinterpret_cast<const BFloat16*>(query.data_ptr<at::BFloat16>()),
        reinterpret_cast<const BFloat16*>(landmarks.data_ptr<at::BFloat16>()),
        reinterpret_cast<BFloat16*>(output.data_ptr<at::BFloat16>()),
        batch_size, query_heads, kv_heads, groups, chunks, head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor gather_gemm_rope_cuda(
    torch::Tensor u,
    torch::Tensor sv,
    torch::Tensor position_ids,
    torch::Tensor cos_sin_cache) {
    check_cuda_bfloat16(u, "u");
    check_cuda_bfloat16(sv, "sv");
    check_cuda_bfloat16(cos_sin_cache, "cos_sin_cache");
    TORCH_CHECK(position_ids.is_cuda(), "position_ids must be a CUDA tensor");
    TORCH_CHECK(position_ids.scalar_type() == torch::kInt64, "position_ids must be int64");
    TORCH_CHECK(position_ids.is_contiguous(), "position_ids must be contiguous");
    TORCH_CHECK(u.dim() == 2, "u must have shape [tokens, rank]");
    TORCH_CHECK(sv.dim() == 3, "sv must have shape [heads, rank, head_dim]");
    TORCH_CHECK(position_ids.dim() == 2, "position_ids must have shape [heads, selected_tokens]");
    TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must have shape [max_position, head_dim]");
    TORCH_CHECK(u.size(1) == sv.size(1), "u and sv rank dimensions must match");
    TORCH_CHECK(sv.size(0) == position_ids.size(0), "sv and position_ids head dimensions must match");
    TORCH_CHECK(sv.size(2) == cos_sin_cache.size(1), "sv and RoPE dimensions must match");
    TORCH_CHECK(sv.size(2) % 2 == 0, "RoPE head dimension must be even");
    TORCH_CHECK(sv.size(2) <= 1024, "RoPE head dimension exceeds the fused kernel limit");

    const int tokens = u.size(0);
    const int heads = sv.size(0);
    const int selected_tokens = position_ids.size(1);
    const int rank = u.size(1);
    const int head_dim = sv.size(2);
    auto output = torch::empty({heads, selected_tokens, head_dim}, u.options());
    const auto stream = at::cuda::getCurrentCUDAStream(u.get_device()).stream();
    gather_gemm_rope_kernel<<<heads * selected_tokens, head_dim, (rank + head_dim) * sizeof(float), stream>>>(
        reinterpret_cast<const BFloat16*>(u.data_ptr<at::BFloat16>()),
        reinterpret_cast<const BFloat16*>(sv.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int64_t>(),
        reinterpret_cast<const BFloat16*>(cos_sin_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<BFloat16*>(output.data_ptr<at::BFloat16>()),
        tokens, heads, selected_tokens, rank, head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
