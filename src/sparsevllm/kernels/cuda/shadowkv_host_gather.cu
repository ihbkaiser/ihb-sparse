#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <vector>

namespace {

// The offset reorder and two-task copy below are adapted from
// ShadowKV's kernels/gather_copy.cu and kernels/copy.cuh
// (Apache-2.0, ByteDance Ltd., revision e51904c).  Sparse-vLLM keeps the
// same hit-prefix/miss-suffix contract but uses one pinned chunk pointer per
// (request, KV-head), which matches the head-indexed ShadowKV payload.

constexpr int kShadowKVOffsetTableSize = 2048;

__global__ void gather_host_kernel(
    const std::uint64_t* __restrict__ source_ptrs,
    const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ lengths,
    std::uint16_t* __restrict__ output,
    int batch,
    int width,
    int heads,
    int head_dim) {
  const int64_t vectors_per_row = static_cast<int64_t>(heads) * head_dim / 8;
  const int64_t total = static_cast<int64_t>(batch) * width * vectors_per_row;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = linear / vectors_per_row;
    const int64_t vector = linear - row * vectors_per_row;
    const int batch_idx = static_cast<int>(row / width);
    const int position = positions[row];
    const int length = lengths[batch_idx];
    auto* dst = reinterpret_cast<uint4*>(output) + linear;
    if (position < 0 || position >= length) {
      *dst = make_uint4(0, 0, 0, 0);
      continue;
    }
    const auto* src = reinterpret_cast<const uint4*>(source_ptrs[batch_idx]);
    const int64_t source_vector =
        static_cast<int64_t>(position) * vectors_per_row + vector;
    *dst = src[source_vector];
  }
}

__global__ void gather_host_scalar_kernel(
    const std::uint64_t* __restrict__ source_ptrs,
    const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ lengths,
    std::uint16_t* __restrict__ output,
    int batch,
    int width,
    int elements_per_row) {
  const int64_t total = static_cast<int64_t>(batch) * width * elements_per_row;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = linear / elements_per_row;
    const int element = static_cast<int>(linear - row * elements_per_row);
    const int batch_idx = static_cast<int>(row / width);
    const int position = positions[row];
    auto* dst = output + linear;
    if (position < 0 || position >= lengths[batch_idx]) {
      *dst = 0;
      continue;
    }
    const auto* src = reinterpret_cast<const std::uint16_t*>(source_ptrs[batch_idx]);
    *dst = src[static_cast<int64_t>(position) * elements_per_row + element];
  }
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float value);

template <typename scalar_t>
__global__ void gather_host_per_head_kernel(
    const std::uint64_t* __restrict__ source_ptrs,
    const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ lengths,
    scalar_t* __restrict__ output,
    int batch,
    int heads,
    int width,
    int head_dim) {
  const int64_t total = static_cast<int64_t>(batch) * heads * width * head_dim;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int element = static_cast<int>(linear % head_dim);
    const int64_t row = linear / head_dim;
    const int token = static_cast<int>(row % width);
    const int64_t head_row = row / width;
    const int head = static_cast<int>(head_row % heads);
    const int batch_idx = static_cast<int>(head_row / heads);
    const int position = positions[
        (static_cast<int64_t>(batch_idx) * heads + head) * width + token];
    const int length = lengths[static_cast<int64_t>(batch_idx) * heads + head];
    if (position < 0 || position >= length) {
      output[linear] = float_to_scalar<scalar_t>(0.0f);
      continue;
    }
    const auto* source = reinterpret_cast<const scalar_t*>(source_ptrs[batch_idx]);
    output[linear] = source[
        static_cast<int64_t>(position) * heads * head_dim
        + static_cast<int64_t>(head) * head_dim
        + element];
  }
}

// GPU-cache mode stores exact post-RoPE KV as [batch, source_width, heads, D],
// while the explicit decode payload is [batch, heads, selected_width, D].
// Doing the lookup directly avoids the graph-hostile temporary tensors that
// torch.permute(...).gather(...).where(...) creates on every decode step.
template <typename scalar_t>
__global__ void gather_gpu_cache_per_head_kernel(
    const scalar_t* __restrict__ source,
    const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ lengths,
    scalar_t* __restrict__ output,
    int batch,
    int source_width,
    int heads,
    int width,
    int head_dim) {
  const int64_t total = static_cast<int64_t>(batch) * heads * width * head_dim;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int element = static_cast<int>(linear % head_dim);
    const int64_t row = linear / head_dim;
    const int token = static_cast<int>(row % width);
    const int64_t head_row = row / width;
    const int head = static_cast<int>(head_row % heads);
    const int batch_idx = static_cast<int>(head_row / heads);
    const int position = positions[
        (static_cast<int64_t>(batch_idx) * heads + head) * width + token];
    scalar_t* destination = output + linear;
    if (position < 0 || position >= lengths[batch_idx] || position >= source_width) {
      *destination = float_to_scalar<scalar_t>(0.0f);
      continue;
    }
    const int64_t source_index =
        (static_cast<int64_t>(batch_idx) * source_width + position) * heads * head_dim
        + static_cast<int64_t>(head) * head_dim + element;
    *destination = source[source_index];
  }
}

template <typename scalar_t>
__global__ void gather_gpu_cache_per_head_kv_kernel(
    const scalar_t* __restrict__ source_k,
    const scalar_t* __restrict__ source_v,
    const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ lengths,
    scalar_t* __restrict__ output_k,
    scalar_t* __restrict__ output_v,
    int batch,
    int source_width,
    int heads,
    int width,
    int head_dim) {
  const int64_t total = static_cast<int64_t>(batch) * heads * width * head_dim;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int element = static_cast<int>(linear % head_dim);
    const int64_t row = linear / head_dim;
    const int token = static_cast<int>(row % width);
    const int64_t head_row = row / width;
    const int head = static_cast<int>(head_row % heads);
    const int batch_idx = static_cast<int>(head_row / heads);
    const int position = positions[
        (static_cast<int64_t>(batch_idx) * heads + head) * width + token];
    scalar_t* destination_k = output_k + linear;
    scalar_t* destination_v = output_v + linear;
    if (position < 0 || position >= lengths[batch_idx] || position >= source_width) {
      *destination_k = float_to_scalar<scalar_t>(0.0f);
      *destination_v = float_to_scalar<scalar_t>(0.0f);
      continue;
    }
    const int64_t source_index =
        (static_cast<int64_t>(batch_idx) * source_width + position) * heads * head_dim
        + static_cast<int64_t>(head) * head_dim + element;
    *destination_k = source_k[source_index];
    *destination_v = source_v[source_index];
  }
}

template <int MAP_SIZE>
__global__ void reorder_shadowkv_chunk_offsets_kernel(
    const std::int64_t* __restrict__ cached_positions,
    const std::int64_t* __restrict__ current_positions,
    std::int64_t* __restrict__ reordered_positions,
    std::int32_t* __restrict__ offsets,
    std::int32_t* __restrict__ counts) {
  __shared__ std::int64_t map_keys[kShadowKVOffsetTableSize];
  __shared__ std::int32_t map_values[kShadowKVOffsetTableSize];
  __shared__ std::int32_t sort_keys[MAP_SIZE];
  __shared__ std::int64_t sort_positions[MAP_SIZE];
  __shared__ std::int32_t sort_offsets[MAP_SIZE];
  __shared__ std::int32_t hit_count;
  __shared__ std::int32_t hit_write;
  __shared__ std::int32_t miss_write;

  const int tid = static_cast<int>(threadIdx.x);
  for (int index = tid; index < kShadowKVOffsetTableSize; index += MAP_SIZE) {
    map_keys[index] = -1;
    map_values[index] = -1;
  }
  if (tid == 0) {
    hit_count = 0;
    hit_write = 0;
    miss_write = 0;
  }
  __syncthreads();

  const int block = static_cast<int>(blockIdx.x);
  const int64_t base = static_cast<int64_t>(block) * MAP_SIZE;
  const std::int64_t old_key = cached_positions[base + tid];
  if (old_key >= 0) {
    unsigned int slot = static_cast<unsigned int>(old_key) &
        (kShadowKVOffsetTableSize - 1);
    while (true) {
      const auto existing = atomicCAS(
          reinterpret_cast<unsigned long long*>(&map_keys[slot]),
          static_cast<unsigned long long>(-1),
          static_cast<unsigned long long>(old_key));
      if (existing == static_cast<unsigned long long>(-1) ||
          existing == static_cast<unsigned long long>(old_key)) {
        map_values[slot] = tid;
        break;
      }
      slot = (slot + 1) & (kShadowKVOffsetTableSize - 1);
    }
  }
  __syncthreads();

  const std::int64_t current_key = current_positions[base + tid];
  int old_offset = -1;
  if (current_key >= 0) {
    unsigned int slot = static_cast<unsigned int>(current_key) &
        (kShadowKVOffsetTableSize - 1);
    while (true) {
      const auto key = map_keys[slot];
      if (key == current_key) {
        old_offset = map_values[slot];
        break;
      }
      if (key == -1) break;
      slot = (slot + 1) & (kShadowKVOffsetTableSize - 1);
    }
  }
  const bool hit = old_offset >= 0;
  if (hit) atomicAdd(&hit_count, 1);
  __syncthreads();

  const int output_index = hit
      ? atomicAdd(&hit_write, 1)
      : atomicAdd(&miss_write, 1);
  const int reordered_index = hit ? output_index : hit_count + output_index;
  // Sort the hit prefix by source offset.  For a unique set of offsets this
  // gives source_offset[i] >= i, which makes the D2D prefix safe to gather
  // in-place.  Misses remain after the prefix and keep their current key as
  // the host-chunk offset.
  sort_keys[reordered_index] = hit ? old_offset : INT_MAX;
  sort_positions[reordered_index] = current_key;
  sort_offsets[reordered_index] = hit ? old_offset : static_cast<int>(current_key);
  __syncthreads();

  for (int size = 2; size <= MAP_SIZE; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      const int partner = tid ^ stride;
      if (partner > tid) {
        const bool ascending = (tid & size) == 0;
        const int left_key = sort_keys[tid];
        const int right_key = sort_keys[partner];
        if ((left_key > right_key) == ascending) {
          sort_keys[tid] = right_key;
          sort_keys[partner] = left_key;
          const auto left_position = sort_positions[tid];
          sort_positions[tid] = sort_positions[partner];
          sort_positions[partner] = left_position;
          const auto left_offset = sort_offsets[tid];
          sort_offsets[tid] = sort_offsets[partner];
          sort_offsets[partner] = left_offset;
        }
      }
      __syncthreads();
    }
  }
  reordered_positions[base + tid] = sort_positions[tid];
  offsets[base + tid] = sort_offsets[tid];
  if (tid == 0) counts[block] = hit_count;
}

constexpr int kShadowKVCopyChunkBatch = 32;

__device__ inline void shadowkv_signal_arrive(std::uint32_t* signal) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    // Toggle 0 <-> 1.  The two block-specialized tasks can therefore arrive
    // in either order and the host task waits for the other task to finish.
    atomicInc(signal, 1);
  }
}

__device__ inline void shadowkv_signal_wait_reset(std::uint32_t* signal) {
  if (threadIdx.x == 0) {
    while (atomicCAS(signal, 0, 0) != 0) {
    }
  }
  __syncthreads();
}

template <int MAP_SIZE>
__global__ void gather_copy_with_offsets_kernel(
    const std::uint64_t* __restrict__ host_source_ptrs,
    std::uint16_t* __restrict__ device_values_buffer,
    std::uint16_t* __restrict__ temp_buffer,
    const std::int32_t* __restrict__ offsets,
    const std::int32_t* __restrict__ counts,
    std::uint32_t* __restrict__ signals,
    int blocks,
    int cpu_chunk_count,
    int gpu_chunk_count,
    int chunk_size,
    int head_dim) {
  const int block = static_cast<int>(blockIdx.x) / 2;
  const bool host_task = (blockIdx.x & 1) == 0;
  if (block >= blocks) return;

  const int vectors_per_chunk = chunk_size * head_dim / 8;
  extern __shared__ unsigned char shared[];
  auto* shared_offsets = reinterpret_cast<std::int32_t*>(shared);
  auto* shared_count = shared_offsets + MAP_SIZE;
  // Reserve four int32 entries so the uint4 tile is naturally aligned.
  auto* shared_data = reinterpret_cast<uint4*>(shared_count + 4);
  for (int index = static_cast<int>(threadIdx.x); index < MAP_SIZE;
       index += MAP_SIZE) {
    shared_offsets[index] = offsets[static_cast<int64_t>(block) * MAP_SIZE + index];
  }
  if (threadIdx.x == 0) shared_count[0] = counts[block];
  __syncthreads();

  const int hit_count = shared_count[0];
  const int start = host_task ? hit_count : 0;
  const int end = host_task ? MAP_SIZE : hit_count;
  const auto* host_source = reinterpret_cast<const uint4*>(host_source_ptrs[block]);
  auto* device_values = reinterpret_cast<uint4*>(device_values_buffer) +
      static_cast<int64_t>(block) * gpu_chunk_count * vectors_per_chunk;
  auto* temp_values = reinterpret_cast<uint4*>(temp_buffer) +
      static_cast<int64_t>(block) * gpu_chunk_count * vectors_per_chunk;
  const auto* device_source = device_values;

  // Process several chunks per shared-memory tile, matching the original
  // ShadowKV block-specialized copy kernel while supporting arbitrary
  // chunk_size * head_dim that is 128-bit aligned.
  for (int base = start; base < end; base += kShadowKVCopyChunkBatch) {
    const int tile_items = min(kShadowKVCopyChunkBatch, end - base);
    const int tile_vectors = tile_items * vectors_per_chunk;
    for (int linear = static_cast<int>(threadIdx.x); linear < tile_vectors;
         linear += MAP_SIZE) {
      const int item = linear / vectors_per_chunk;
      const int vector = linear - item * vectors_per_chunk;
      const int source_chunk = shared_offsets[base + item];
      const int source_limit = host_task ? cpu_chunk_count : gpu_chunk_count;
      uint4 value = make_uint4(0, 0, 0, 0);
      if (source_chunk >= 0 && source_chunk < source_limit) {
        const auto* source = host_task
            ? host_source + static_cast<int64_t>(source_chunk) * vectors_per_chunk
            : device_source + static_cast<int64_t>(source_chunk) * vectors_per_chunk;
        value = source[vector];
      }
      shared_data[linear] = value;
    }
    __syncthreads();
    auto* destination = host_task ? temp_values : device_values;
    for (int linear = static_cast<int>(threadIdx.x); linear < tile_vectors;
         linear += MAP_SIZE) {
      const int item = linear / vectors_per_chunk;
      const int vector = linear - item * vectors_per_chunk;
      destination[static_cast<int64_t>(base + item) * vectors_per_chunk + vector] =
          shared_data[linear];
    }
    __syncthreads();
  }

  // D2D reads may source chunks that the H2D task will overwrite.  The host
  // task stages misses in temp, signals arrival, waits for the D2D task to
  // toggle the signal back to zero, and only then commits the miss suffix.
  // The D2D task can signal and return immediately.  This is the same
  // two-block protocol used by ShadowKV's gather_copy_var_midpoint_BP:
  // atomicInc(signal, 1) toggles 0 <-> 1, so either task may arrive first.
  if (host_task) {
    shadowkv_signal_arrive(signals + block);
    shadowkv_signal_wait_reset(signals + block);
    for (int base = hit_count; base < MAP_SIZE; base += kShadowKVCopyChunkBatch) {
      const int tile_items = min(kShadowKVCopyChunkBatch, MAP_SIZE - base);
      const int tile_vectors = tile_items * vectors_per_chunk;
      for (int linear = static_cast<int>(threadIdx.x); linear < tile_vectors;
           linear += MAP_SIZE) {
        const int item = linear / vectors_per_chunk;
        const int vector = linear - item * vectors_per_chunk;
        shared_data[linear] = temp_values[
            static_cast<int64_t>(base + item) * vectors_per_chunk + vector];
      }
      __syncthreads();
      for (int linear = static_cast<int>(threadIdx.x); linear < tile_vectors;
           linear += MAP_SIZE) {
        const int item = linear / vectors_per_chunk;
        const int vector = linear - item * vectors_per_chunk;
        device_values[
            static_cast<int64_t>(base + item) * vectors_per_chunk + vector] =
            shared_data[linear];
      }
      __syncthreads();
    }
  } else {
    shadowkv_signal_arrive(signals + block);
  }
}

template <int MAP_SIZE>
void configure_gather_copy_with_offsets_kernel(int chunk_size, int head_dim) {
  TORCH_CHECK(chunk_size > 0 && head_dim > 0 &&
                  (static_cast<int64_t>(chunk_size) * head_dim) % 8 == 0,
              "ShadowKV offset copy chunk dimensions are invalid.");
  const int vectors_per_chunk = chunk_size * head_dim / 8;
  const size_t shared_bytes =
      static_cast<size_t>(MAP_SIZE) * sizeof(std::int32_t)
      + 4 * sizeof(std::int32_t)
      + static_cast<size_t>(kShadowKVCopyChunkBatch) * vectors_per_chunk
          * sizeof(uint4);
  const auto status = cudaFuncSetAttribute(
      gather_copy_with_offsets_kernel<MAP_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_bytes));
  TORCH_CHECK(
      status == cudaSuccess,
      "ShadowKV offset-copy shared-memory configuration failed for map_size=",
      MAP_SIZE,
      " shared_bytes=",
      shared_bytes,
      ": ",
      cudaGetErrorString(status));
}

void configure_gather_copy_with_offsets(
    int chunk_size, int head_dim, int map_size) {
  if (map_size == 128) {
    configure_gather_copy_with_offsets_kernel<128>(chunk_size, head_dim);
  } else if (map_size == 256) {
    configure_gather_copy_with_offsets_kernel<256>(chunk_size, head_dim);
  } else if (map_size == 512) {
    configure_gather_copy_with_offsets_kernel<512>(chunk_size, head_dim);
  } else if (map_size == 1024) {
    configure_gather_copy_with_offsets_kernel<1024>(chunk_size, head_dim);
  } else {
    TORCH_CHECK(false, "ShadowKV offset map_size must be one of 128, 256, 512, 1024.");
  }
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t value);

template <>
__device__ inline float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <>
__device__ inline float scalar_to_float<__half>(__half value) {
  return __half2float(value);
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float value);

template <>
__device__ inline __nv_bfloat16 float_to_scalar<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <>
__device__ inline __half float_to_scalar<__half>(float value) {
  return __float2half(value);
}

template <typename scalar_t>
__global__ void apply_rope_inplace_kernel(
    scalar_t* __restrict__ output,
    const float* __restrict__ cos_sin,
    const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ lengths,
    int batch,
    int width,
    int heads,
    int head_dim) {
  const int pair = static_cast<int>(threadIdx.x);
  const int half_dim = head_dim / 2;
  if (pair >= half_dim) {
    return;
  }
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int batch_idx = static_cast<int>(row / width);
  const int token_idx = static_cast<int>(row - static_cast<int64_t>(batch_idx) * width);
  const int position = positions[row];
  scalar_t* token = output + row * static_cast<int64_t>(heads) * head_dim;
  if (position < 0 || position >= lengths[batch_idx]) {
    for (int head = 0; head < heads; ++head) {
      scalar_t* head_ptr = token + static_cast<int64_t>(head) * head_dim;
      head_ptr[pair] = float_to_scalar<scalar_t>(0.0f);
      head_ptr[pair + half_dim] = float_to_scalar<scalar_t>(0.0f);
    }
    return;
  }
  // Sparse-vLLM stores [cos, sin] as [head_dim / 2, head_dim / 2].
  const float* rope = cos_sin + static_cast<int64_t>(position) * head_dim;
  for (int head = 0; head < heads; ++head) {
    scalar_t* head_ptr = token + static_cast<int64_t>(head) * head_dim;
    const float x1 = scalar_to_float<scalar_t>(head_ptr[pair]);
    const float x2 = scalar_to_float<scalar_t>(head_ptr[pair + half_dim]);
    const float c = rope[pair];
    const float s = rope[pair + half_dim];
    head_ptr[pair] = float_to_scalar<scalar_t>(x1 * c - x2 * s);
    head_ptr[pair + half_dim] = float_to_scalar<scalar_t>(x2 * c + x1 * s);
  }
}

void check_source_tensors(const std::vector<torch::Tensor>& sources, int batch) {
  TORCH_CHECK(static_cast<int>(sources.size()) == batch,
              "ShadowKV host gather source count must equal batch size.");
  for (const auto& source : sources) {
    TORCH_CHECK(source.device().is_cpu(), "ShadowKV gather sources must be CPU tensors.");
    TORCH_CHECK(source.is_contiguous(), "ShadowKV gather sources must be contiguous.");
    TORCH_CHECK(source.is_pinned(), "ShadowKV gather sources must be pinned CPU tensors.");
    TORCH_CHECK(source.scalar_type() == torch::kBFloat16 ||
                    source.scalar_type() == torch::kHalf,
                "ShadowKV host gather supports BF16 and FP16 sources.");
    TORCH_CHECK(source.dim() == 3, "ShadowKV gather sources must have shape [tokens, heads, dim].");
  }
}

void set_host_pointers(
    const std::vector<torch::Tensor>& sources,
    torch::Tensor pointer_table) {
  TORCH_CHECK(pointer_table.is_cuda(), "ShadowKV pointer table must be CUDA resident.");
  TORCH_CHECK(pointer_table.scalar_type() == torch::kInt64 && pointer_table.dim() == 1,
              "ShadowKV pointer table must be a CUDA int64 vector.");
  const int batch = static_cast<int>(sources.size());
  check_source_tensors(sources, batch);
  TORCH_CHECK(pointer_table.numel() >= batch,
              "ShadowKV pointer table is smaller than the active batch.");

  std::vector<std::uint64_t> device_ptrs(batch);
  for (int index = 0; index < batch; ++index) {
    void* device_pointer = nullptr;
    const auto status = cudaHostGetDevicePointer(
        &device_pointer, sources[index].data_ptr(), 0);
    TORCH_CHECK(status == cudaSuccess,
                "cudaHostGetDevicePointer failed for ShadowKV source: ",
                cudaGetErrorString(status));
    device_ptrs[index] = reinterpret_cast<std::uint64_t>(device_pointer);
  }
  auto stream = at::cuda::getCurrentCUDAStream(pointer_table.get_device());
  const auto status = cudaMemcpyAsync(
      pointer_table.data_ptr<int64_t>(),
      device_ptrs.data(),
      static_cast<size_t>(batch) * sizeof(std::uint64_t),
      cudaMemcpyHostToDevice,
      stream.stream());
  TORCH_CHECK(status == cudaSuccess,
              "ShadowKV pointer-table upload failed: ",
              cudaGetErrorString(status));
}

void reorder_shadowkv_chunk_offsets(
    torch::Tensor cached_positions,
    torch::Tensor current_positions,
    torch::Tensor reordered_positions,
    torch::Tensor offsets,
    torch::Tensor counts,
    int batch,
    int heads,
    int map_size) {
  TORCH_CHECK(cached_positions.is_cuda() && current_positions.is_cuda() &&
                  reordered_positions.is_cuda() && offsets.is_cuda() &&
                  counts.is_cuda(),
              "ShadowKV offset metadata must be CUDA tensors.");
  TORCH_CHECK(cached_positions.scalar_type() == torch::kLong &&
                  current_positions.scalar_type() == torch::kLong &&
                  reordered_positions.scalar_type() == torch::kLong &&
                  offsets.scalar_type() == torch::kInt &&
                  counts.scalar_type() == torch::kInt,
              "ShadowKV offset metadata has an invalid dtype.");
  TORCH_CHECK(cached_positions.dim() == 3 && current_positions.sizes() == cached_positions.sizes() &&
                  reordered_positions.sizes() == cached_positions.sizes(),
              "ShadowKV offset position tensors must be [batch, heads, map_size].");
  TORCH_CHECK(cached_positions.size(0) == batch && cached_positions.size(1) == heads &&
                  cached_positions.size(2) == map_size &&
                  offsets.numel() >= static_cast<int64_t>(batch) * heads * map_size &&
                  counts.numel() >= static_cast<int64_t>(batch) * heads,
              "ShadowKV offset metadata dimensions disagree.");
  TORCH_CHECK(cached_positions.is_contiguous() && current_positions.is_contiguous() &&
                  reordered_positions.is_contiguous() && offsets.is_contiguous() &&
                  counts.is_contiguous(),
              "ShadowKV offset metadata must be contiguous.");
  const dim3 grid(static_cast<unsigned int>(batch * heads));
  const dim3 block(static_cast<unsigned int>(map_size));
  auto stream = at::cuda::getCurrentCUDAStream(cached_positions.get_device());
  if (map_size == 128) {
    reorder_shadowkv_chunk_offsets_kernel<128><<<grid, block, 0, stream.stream()>>>(
        cached_positions.data_ptr<int64_t>(), current_positions.data_ptr<int64_t>(),
        reordered_positions.data_ptr<int64_t>(), offsets.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>());
  } else if (map_size == 256) {
    reorder_shadowkv_chunk_offsets_kernel<256><<<grid, block, 0, stream.stream()>>>(
        cached_positions.data_ptr<int64_t>(), current_positions.data_ptr<int64_t>(),
        reordered_positions.data_ptr<int64_t>(), offsets.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>());
  } else if (map_size == 512) {
    reorder_shadowkv_chunk_offsets_kernel<512><<<grid, block, 0, stream.stream()>>>(
        cached_positions.data_ptr<int64_t>(), current_positions.data_ptr<int64_t>(),
        reordered_positions.data_ptr<int64_t>(), offsets.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>());
  } else if (map_size == 1024) {
    reorder_shadowkv_chunk_offsets_kernel<1024><<<grid, block, 0, stream.stream()>>>(
        cached_positions.data_ptr<int64_t>(), current_positions.data_ptr<int64_t>(),
        reordered_positions.data_ptr<int64_t>(), offsets.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>());
  } else {
    TORCH_CHECK(false, "ShadowKV offset map_size must be one of 128, 256, 512, 1024.");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_copy_with_offsets(
    torch::Tensor host_source_ptrs,
    torch::Tensor device_values,
    torch::Tensor temp,
    torch::Tensor offsets,
    torch::Tensor counts,
    torch::Tensor signals,
    int batch,
    int heads,
    int cpu_chunk_count,
    int gpu_chunk_count,
    int chunk_size,
    int head_dim,
    int map_size) {
  TORCH_CHECK(host_source_ptrs.is_cuda() && device_values.is_cuda() &&
                  temp.is_cuda() && offsets.is_cuda() && counts.is_cuda() &&
                  signals.is_cuda(),
              "ShadowKV offset copy inputs must be CUDA tensors.");
  TORCH_CHECK(host_source_ptrs.scalar_type() == torch::kLong &&
                  host_source_ptrs.dim() == 1 &&
                  (device_values.scalar_type() == torch::kBFloat16 ||
                   device_values.scalar_type() == torch::kHalf),
              "ShadowKV offset copy requires BF16 or FP16 device values and int64 pointers.");
  TORCH_CHECK(device_values.is_contiguous() && temp.is_contiguous() &&
                  offsets.is_contiguous() && counts.is_contiguous() &&
                  signals.is_contiguous(),
              "ShadowKV offset copy buffers must be contiguous.");
  const int blocks = batch * heads;
  TORCH_CHECK(host_source_ptrs.numel() >= blocks && offsets.numel() >= static_cast<int64_t>(blocks) * map_size &&
                  counts.numel() >= blocks && signals.numel() >= blocks,
              "ShadowKV offset copy metadata is smaller than the block grid.");
  TORCH_CHECK(cpu_chunk_count > 0 && gpu_chunk_count > 0 && chunk_size > 0 &&
                  head_dim > 0 && (static_cast<int64_t>(chunk_size) * head_dim) % 8 == 0,
              "ShadowKV offset copy chunk dimensions are invalid.");
  TORCH_CHECK(device_values.numel() >= static_cast<int64_t>(blocks) * gpu_chunk_count * chunk_size * head_dim,
              "ShadowKV offset copy output is too small.");
  TORCH_CHECK(temp.numel() >= static_cast<int64_t>(blocks) * gpu_chunk_count * chunk_size * head_dim,
              "ShadowKV offset copy temporary buffer is too small.");
  const int vectors_per_chunk = chunk_size * head_dim / 8;
  const size_t shared_bytes =
      static_cast<size_t>(map_size) * sizeof(std::int32_t)
      + 4 * sizeof(std::int32_t)
      + static_cast<size_t>(kShadowKVCopyChunkBatch) * vectors_per_chunk
          * sizeof(uint4);
  auto stream = at::cuda::getCurrentCUDAStream(device_values.get_device());
  const dim3 grid(static_cast<unsigned int>(blocks * 2));
  const dim3 block(static_cast<unsigned int>(map_size));
  if (map_size == 128) {
    gather_copy_with_offsets_kernel<128><<<grid, block, shared_bytes, stream.stream()>>>(
        reinterpret_cast<const std::uint64_t*>(host_source_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<std::uint16_t*>(device_values.data_ptr()),
        reinterpret_cast<std::uint16_t*>(temp.data_ptr()), offsets.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(), reinterpret_cast<std::uint32_t*>(signals.data_ptr<int32_t>()),
        blocks, cpu_chunk_count, gpu_chunk_count, chunk_size, head_dim);
  } else if (map_size == 256) {
    gather_copy_with_offsets_kernel<256><<<grid, block, shared_bytes, stream.stream()>>>(
        reinterpret_cast<const std::uint64_t*>(host_source_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<std::uint16_t*>(device_values.data_ptr()),
        reinterpret_cast<std::uint16_t*>(temp.data_ptr()), offsets.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(), reinterpret_cast<std::uint32_t*>(signals.data_ptr<int32_t>()),
        blocks, cpu_chunk_count, gpu_chunk_count, chunk_size, head_dim);
  } else if (map_size == 512) {
    gather_copy_with_offsets_kernel<512><<<grid, block, shared_bytes, stream.stream()>>>(
        reinterpret_cast<const std::uint64_t*>(host_source_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<std::uint16_t*>(device_values.data_ptr()),
        reinterpret_cast<std::uint16_t*>(temp.data_ptr()), offsets.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(), reinterpret_cast<std::uint32_t*>(signals.data_ptr<int32_t>()),
        blocks, cpu_chunk_count, gpu_chunk_count, chunk_size, head_dim);
  } else if (map_size == 1024) {
    gather_copy_with_offsets_kernel<1024><<<grid, block, shared_bytes, stream.stream()>>>(
        reinterpret_cast<const std::uint64_t*>(host_source_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<std::uint16_t*>(device_values.data_ptr()),
        reinterpret_cast<std::uint16_t*>(temp.data_ptr()), offsets.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(), reinterpret_cast<std::uint32_t*>(signals.data_ptr<int32_t>()),
        blocks, cpu_chunk_count, gpu_chunk_count, chunk_size, head_dim);
  } else {
    TORCH_CHECK(false, "ShadowKV offset map_size must be one of 128, 256, 512, 1024.");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_host(
    torch::Tensor pointer_table,
    torch::Tensor positions,
    torch::Tensor lengths,
    torch::Tensor output) {
  TORCH_CHECK(pointer_table.is_cuda() && positions.is_cuda() && lengths.is_cuda() &&
                  output.is_cuda(),
              "ShadowKV gather inputs and output must be CUDA tensors.");
  TORCH_CHECK(pointer_table.scalar_type() == torch::kInt64 && positions.scalar_type() == torch::kInt32 &&
                  lengths.scalar_type() == torch::kInt32,
              "ShadowKV gather metadata has an invalid dtype.");
  TORCH_CHECK(pointer_table.dim() == 1 && positions.dim() == 2 && lengths.dim() == 1 &&
                  output.dim() == 4,
              "ShadowKV gather expects pointer_table[B], positions[B,W], lengths[B], output[B,W,H,D].");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16 || output.scalar_type() == torch::kHalf,
              "ShadowKV host gather supports BF16 and FP16 outputs.");
  const int batch = static_cast<int>(positions.size(0));
  const int width = static_cast<int>(positions.size(1));
  const int heads = static_cast<int>(output.size(2));
  const int head_dim = static_cast<int>(output.size(3));
  TORCH_CHECK(pointer_table.numel() >= batch && lengths.numel() == batch,
              "ShadowKV gather metadata batch dimensions disagree.");
  TORCH_CHECK(output.size(0) == batch && output.size(1) == width &&
                  output.is_contiguous() && positions.is_contiguous() &&
                  lengths.is_contiguous() && pointer_table.is_contiguous(),
              "ShadowKV gather tensors must be contiguous and shape-compatible.");
  TORCH_CHECK(head_dim > 0, "ShadowKV gather requires a positive head dimension.");

  const int elements_per_row = heads * head_dim;
  const int64_t work = static_cast<int64_t>(batch) * width * elements_per_row / 8;
  const int blocks = static_cast<int>(std::min<int64_t>((work + 255) / 256, 65535));
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  if (elements_per_row % 8 == 0) {
    gather_host_kernel<<<blocks, 256, 0, stream.stream()>>>(
        reinterpret_cast<const std::uint64_t*>(pointer_table.data_ptr<int64_t>()),
        positions.data_ptr<int32_t>(),
        lengths.data_ptr<int32_t>(),
        reinterpret_cast<std::uint16_t*>(output.data_ptr()),
        batch, width, heads, head_dim);
  } else {
    const int64_t scalar_work = static_cast<int64_t>(batch) * width * elements_per_row;
    const int scalar_blocks = static_cast<int>(std::min<int64_t>((scalar_work + 255) / 256, 65535));
    gather_host_scalar_kernel<<<scalar_blocks, 256, 0, stream.stream()>>>(
        reinterpret_cast<const std::uint64_t*>(pointer_table.data_ptr<int64_t>()),
        positions.data_ptr<int32_t>(),
        lengths.data_ptr<int32_t>(),
        reinterpret_cast<std::uint16_t*>(output.data_ptr()),
        batch, width, elements_per_row);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_host_per_head(
    torch::Tensor pointer_table,
    torch::Tensor positions,
    torch::Tensor lengths,
    torch::Tensor output) {
  TORCH_CHECK(pointer_table.is_cuda() && positions.is_cuda() && lengths.is_cuda() &&
                  output.is_cuda(),
              "ShadowKV per-head gather inputs and output must be CUDA tensors.");
  TORCH_CHECK(pointer_table.scalar_type() == torch::kInt64 &&
                  positions.scalar_type() == torch::kInt32 &&
                  lengths.scalar_type() == torch::kInt32,
              "ShadowKV per-head gather metadata has an invalid dtype.");
  TORCH_CHECK(pointer_table.dim() == 1 && positions.dim() == 3 &&
                  lengths.dim() == 2 && output.dim() == 4,
              "ShadowKV per-head gather expects pointer[B], positions[B,H,W], "
              "lengths[B,H], output[B,H,W,D].");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16 || output.scalar_type() == torch::kHalf,
              "ShadowKV per-head gather supports BF16 and FP16 outputs.");
  const int batch = static_cast<int>(positions.size(0));
  const int heads = static_cast<int>(positions.size(1));
  const int width = static_cast<int>(positions.size(2));
  const int head_dim = static_cast<int>(output.size(3));
  TORCH_CHECK(pointer_table.numel() >= batch &&
                  lengths.sizes() == torch::IntArrayRef({batch, heads}) &&
                  output.sizes() == torch::IntArrayRef({batch, heads, width, head_dim}) &&
                  pointer_table.is_contiguous() && positions.is_contiguous() &&
                  lengths.is_contiguous() && output.is_contiguous(),
              "ShadowKV per-head gather dimensions or strides disagree.");
  const int64_t work = static_cast<int64_t>(batch) * heads * width * head_dim;
  const int blocks = static_cast<int>(std::min<int64_t>((work + 255) / 256, 65535));
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  if (output.scalar_type() == torch::kBFloat16) {
    gather_host_per_head_kernel<__nv_bfloat16><<<blocks, 256, 0, stream.stream()>>>(
        reinterpret_cast<const std::uint64_t*>(pointer_table.data_ptr<int64_t>()),
        positions.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        batch, heads, width, head_dim);
  } else {
    gather_host_per_head_kernel<__half><<<blocks, 256, 0, stream.stream()>>>(
        reinterpret_cast<const std::uint64_t*>(pointer_table.data_ptr<int64_t>()),
        positions.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr()),
        batch, heads, width, head_dim);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_gpu_cache_per_head(
    torch::Tensor source,
    torch::Tensor positions,
    torch::Tensor lengths,
    torch::Tensor output) {
  TORCH_CHECK(source.is_cuda() && positions.is_cuda() && lengths.is_cuda() &&
                  output.is_cuda(),
              "ShadowKV GPU-cache gather inputs must be CUDA tensors.");
  TORCH_CHECK(source.scalar_type() == torch::kBFloat16 ||
                  source.scalar_type() == torch::kHalf,
              "ShadowKV GPU-cache gather supports BF16 and FP16 sources.");
  TORCH_CHECK(source.scalar_type() == output.scalar_type() &&
                  positions.scalar_type() == torch::kInt32 &&
                  lengths.scalar_type() == torch::kInt32,
              "ShadowKV GPU-cache gather has an invalid dtype.");
  TORCH_CHECK(source.dim() == 4 && positions.dim() == 3 &&
                  lengths.dim() == 1 && output.dim() == 4,
              "ShadowKV GPU-cache gather expects source[B,W,H,D], "
              "positions[B,H,O], lengths[B], output[B,H,O,D].");
  const int batch = static_cast<int>(source.size(0));
  const int source_width = static_cast<int>(source.size(1));
  const int heads = static_cast<int>(source.size(2));
  const int head_dim = static_cast<int>(source.size(3));
  const int width = static_cast<int>(positions.size(2));
  TORCH_CHECK(positions.size(0) == batch && positions.size(1) == heads &&
                  lengths.numel() == batch && output.sizes() ==
                  torch::IntArrayRef({batch, heads, width, head_dim}) &&
                  source.is_contiguous() && positions.is_contiguous() &&
                  lengths.is_contiguous() && output.is_contiguous(),
              "ShadowKV GPU-cache gather dimensions or strides disagree.");
  TORCH_CHECK(source_width > 0 && head_dim > 0,
              "ShadowKV GPU-cache gather requires positive source dimensions.");
  const int64_t work = static_cast<int64_t>(batch) * heads * width * head_dim;
  const int blocks = static_cast<int>(std::min<int64_t>((work + 255) / 256, 65535));
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  if (source.scalar_type() == torch::kBFloat16) {
    gather_gpu_cache_per_head_kernel<__nv_bfloat16><<<blocks, 256, 0, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(source.data_ptr()),
        positions.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        batch, source_width, heads, width, head_dim);
  } else {
    gather_gpu_cache_per_head_kernel<__half><<<blocks, 256, 0, stream.stream()>>>(
        reinterpret_cast<const __half*>(source.data_ptr()),
        positions.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr()),
        batch, source_width, heads, width, head_dim);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_gpu_cache_per_head_kv(
    torch::Tensor source_k,
    torch::Tensor source_v,
    torch::Tensor positions,
    torch::Tensor lengths,
    torch::Tensor output_k,
    torch::Tensor output_v) {
  TORCH_CHECK(source_k.is_cuda() && source_v.is_cuda() && positions.is_cuda() &&
                  lengths.is_cuda() && output_k.is_cuda() && output_v.is_cuda(),
              "ShadowKV fused GPU-cache gather inputs must be CUDA tensors.");
  TORCH_CHECK(
      (source_k.scalar_type() == torch::kBFloat16 ||
       source_k.scalar_type() == torch::kHalf) &&
          source_v.scalar_type() == source_k.scalar_type() &&
          output_k.scalar_type() == source_k.scalar_type() &&
          output_v.scalar_type() == source_k.scalar_type() &&
          positions.scalar_type() == torch::kInt32 &&
          lengths.scalar_type() == torch::kInt32,
      "ShadowKV fused GPU-cache gather has an invalid dtype.");
  TORCH_CHECK(
      source_k.dim() == 4 && source_v.sizes() == source_k.sizes() &&
          positions.dim() == 3 && lengths.dim() == 1 && output_k.dim() == 4 &&
          output_v.sizes() == output_k.sizes(),
      "ShadowKV fused GPU-cache gather expects source/outputs with matching shapes.");
  const int batch = static_cast<int>(source_k.size(0));
  const int source_width = static_cast<int>(source_k.size(1));
  const int heads = static_cast<int>(source_k.size(2));
  const int head_dim = static_cast<int>(source_k.size(3));
  const int width = static_cast<int>(positions.size(2));
  TORCH_CHECK(
      positions.size(0) == batch && positions.size(1) == heads &&
          lengths.numel() == batch &&
          output_k.sizes() == torch::IntArrayRef({batch, heads, width, head_dim}) &&
          source_k.is_contiguous() && source_v.is_contiguous() &&
          positions.is_contiguous() && lengths.is_contiguous() &&
          output_k.is_contiguous() && output_v.is_contiguous(),
      "ShadowKV fused GPU-cache gather dimensions or strides disagree.");
  TORCH_CHECK(source_width > 0 && head_dim > 0,
              "ShadowKV fused GPU-cache gather requires positive dimensions.");
  const int64_t work = static_cast<int64_t>(batch) * heads * width * head_dim;
  const int blocks = static_cast<int>(std::min<int64_t>((work + 255) / 256, 65535));
  auto stream = at::cuda::getCurrentCUDAStream(output_k.get_device());
  if (source_k.scalar_type() == torch::kBFloat16) {
    gather_gpu_cache_per_head_kv_kernel<__nv_bfloat16>
        <<<blocks, 256, 0, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(source_k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(source_v.data_ptr()),
            positions.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(output_k.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output_v.data_ptr()), batch,
            source_width, heads, width, head_dim);
  } else {
    gather_gpu_cache_per_head_kv_kernel<__half>
        <<<blocks, 256, 0, stream.stream()>>>(
            reinterpret_cast<const __half*>(source_k.data_ptr()),
            reinterpret_cast<const __half*>(source_v.data_ptr()),
            positions.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(output_k.data_ptr()),
            reinterpret_cast<__half*>(output_v.data_ptr()), batch,
            source_width, heads, width, head_dim);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_gemm_rope(
    torch::Tensor pointer_table,
    torch::Tensor positions,
    torch::Tensor lengths,
    torch::Tensor sv_bmm,
    torch::Tensor cos_sin,
    torch::Tensor u_workspace,
    torch::Tensor output) {
  TORCH_CHECK(pointer_table.is_cuda() && positions.is_cuda() && lengths.is_cuda() &&
                  sv_bmm.is_cuda() && cos_sin.is_cuda() && u_workspace.is_cuda() &&
                  output.is_cuda(),
              "ShadowKV fused reconstruction inputs must be CUDA tensors.");
  TORCH_CHECK(pointer_table.scalar_type() == torch::kInt64 &&
                  positions.scalar_type() == torch::kInt32 &&
                  lengths.scalar_type() == torch::kInt32,
              "ShadowKV fused reconstruction metadata has an invalid dtype.");
  TORCH_CHECK(pointer_table.dim() == 1 && positions.dim() == 2 && lengths.dim() == 1 &&
                  sv_bmm.dim() == 4 && cos_sin.dim() == 2 && u_workspace.dim() == 4 &&
                  output.dim() == 4,
              "ShadowKV fused reconstruction expects pointer[B], positions[B,K], "
              "lengths[B], sv[B,R,H,D], cos_sin[P,2D], u[B,K,1,R], output[B,K,H,D].");
  TORCH_CHECK(cos_sin.scalar_type() == torch::kFloat,
              "ShadowKV fused reconstruction requires a float32 RoPE cache.");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16 || output.scalar_type() == torch::kHalf,
              "ShadowKV fused reconstruction supports BF16 and FP16 output.");
  TORCH_CHECK(u_workspace.scalar_type() == output.scalar_type() &&
                  sv_bmm.scalar_type() == output.scalar_type(),
              "ShadowKV fused reconstruction requires matching U/SV/output dtypes.");
  const int batch = static_cast<int>(positions.size(0));
  const int width = static_cast<int>(positions.size(1));
  const int rank = static_cast<int>(sv_bmm.size(1));
  const int heads = static_cast<int>(sv_bmm.size(2));
  const int head_dim = static_cast<int>(sv_bmm.size(3));
  TORCH_CHECK(pointer_table.numel() >= batch && lengths.numel() == batch,
              "ShadowKV fused reconstruction batch metadata disagrees.");
  TORCH_CHECK(u_workspace.size(0) == batch && u_workspace.size(1) == width &&
                  u_workspace.size(2) == 1 && u_workspace.size(3) == rank,
              "ShadowKV fused reconstruction U workspace shape disagrees.");
  TORCH_CHECK(output.size(0) == batch && output.size(1) == width &&
                  output.size(2) == heads && output.size(3) == head_dim,
              "ShadowKV fused reconstruction output shape disagrees.");
  TORCH_CHECK(sv_bmm.is_contiguous() && cos_sin.is_contiguous() &&
                  u_workspace.is_contiguous() && output.is_contiguous() &&
                  positions.is_contiguous() && lengths.is_contiguous() &&
                  pointer_table.is_contiguous(),
              "ShadowKV fused reconstruction tensors must be contiguous.");
  gather_host(pointer_table, positions, lengths, u_workspace);

  // Keep the BMM in the same stream and write directly into the graph-stable
  // reconstruction buffer.  sv_bmm is staged as [B,R,H,D] so this reshape is
  // contiguous and does not allocate or transpose inside CUDA Graph capture.
  auto u_matrix = u_workspace.view({batch, width, rank});
  auto sv_matrix = sv_bmm.view({batch, rank, heads * head_dim});
  auto output_matrix = output.view({batch, width, heads * head_dim});
  at::bmm_out(output_matrix, u_matrix, sv_matrix);
  if (cos_sin.size(0) == 0) {
    return;
  }

  const int64_t rows = static_cast<int64_t>(batch) * width;
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  const int threads = std::min(1024, std::max(1, head_dim / 2));
  const dim3 grid(static_cast<unsigned int>(rows));
  if (output.scalar_type() == torch::kBFloat16) {
    apply_rope_inplace_kernel<__nv_bfloat16><<<grid, threads, 0, stream.stream()>>>(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        cos_sin.data_ptr<float>(),
        positions.data_ptr<int32_t>(),
        lengths.data_ptr<int32_t>(),
        batch,
        width,
        heads,
        head_dim);
  } else {
    apply_rope_inplace_kernel<__half><<<grid, threads, 0, stream.stream()>>>(
        reinterpret_cast<__half*>(output.data_ptr()),
        cos_sin.data_ptr<float>(),
        positions.data_ptr<int32_t>(),
        lengths.data_ptr<int32_t>(),
        batch,
        width,
        heads,
        head_dim);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("set_host_pointers", &set_host_pointers);
  module.def("reorder_shadowkv_chunk_offsets", &reorder_shadowkv_chunk_offsets);
  module.def("configure_gather_copy_with_offsets", &configure_gather_copy_with_offsets);
  module.def("gather_copy_with_offsets", &gather_copy_with_offsets);
  module.def("gather_host", &gather_host);
  module.def("gather_host_per_head", &gather_host_per_head);
  module.def("gather_gpu_cache_per_head", &gather_gpu_cache_per_head);
  module.def("gather_gpu_cache_per_head_kv", &gather_gpu_cache_per_head_kv);
  module.def("gather_gemm_rope", &gather_gemm_rope);
}
