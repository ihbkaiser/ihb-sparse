/*
 * ShadowKV CUTLASS integration.
 *
 * CUTLASS remains an external BSD-3-Clause dependency.  The kernels below
 * use its strided-batched GEMM for the two dense products and keep the host
 * gather/RoPE glue repository-owned.  All buffers are supplied by the cache
 * manager, so the operations do not allocate on the decode hot path.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "shadowkv_cutlass/batch_gemm_softmax.h"
#include "shadowkv_cutlass/gemm_universal_batch_gather_indices.h"

namespace {

template <typename scalar_t>
__device__ inline float to_float(scalar_t value);
template <>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
template <>
__device__ inline float to_float<__half>(__half value) {
  return __half2float(value);
}
template <typename scalar_t>
__device__ inline scalar_t from_float(float value);
template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}
template <>
__device__ inline __half from_float<__half>(float value) {
  return __float2half(value);
}

template <typename scalar_t>
__global__ void gather_host_kernel(
    const std::uint64_t* __restrict__ source_ptrs,
    const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ lengths,
    scalar_t* __restrict__ output,
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
    if (position < 0 || position >= lengths[batch_idx]) {
      output[linear] = from_float<scalar_t>(0.0f);
      continue;
    }
    const auto* source = reinterpret_cast<const scalar_t*>(source_ptrs[batch_idx]);
    output[linear] = source[static_cast<int64_t>(position) * elements_per_row + element];
  }
}

__global__ void gather_host_vector_kernel(
    const std::uint64_t* __restrict__ source_ptrs,
    const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ lengths,
    std::uint16_t* __restrict__ output,
    int batch,
    int width,
    int vectors_per_row) {
  const int64_t total = static_cast<int64_t>(batch) * width * vectors_per_row;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = linear / vectors_per_row;
    const int vector = static_cast<int>(linear - row * vectors_per_row);
    const int batch_idx = static_cast<int>(row / width);
    const int position = positions[row];
    auto* destination = reinterpret_cast<uint4*>(output) + linear;
    if (position < 0 || position >= lengths[batch_idx]) {
      *destination = make_uint4(0, 0, 0, 0);
      continue;
    }
    const auto* source = reinterpret_cast<const uint4*>(source_ptrs[batch_idx]);
    *destination = source[static_cast<int64_t>(position) * vectors_per_row + vector];
  }
}

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
      output[linear] = from_float<scalar_t>(0.0f);
      continue;
    }
    const auto* source = reinterpret_cast<const scalar_t*>(source_ptrs[batch_idx]);
    output[linear] = source[
        static_cast<int64_t>(position) * heads * head_dim
        + static_cast<int64_t>(head) * head_dim
        + element];
  }
}

template <typename scalar_t>
__global__ void apply_rope_kernel(
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
  if (pair >= half_dim) return;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int batch_idx = static_cast<int>(row / width);
  const int position = positions[row];
  scalar_t* token = output + row * static_cast<int64_t>(heads) * head_dim;
  if (position < 0 || position >= lengths[batch_idx]) {
    for (int head = 0; head < heads; ++head) {
      scalar_t* ptr = token + static_cast<int64_t>(head) * head_dim;
      ptr[pair] = from_float<scalar_t>(0.0f);
      ptr[pair + half_dim] = from_float<scalar_t>(0.0f);
    }
    return;
  }
  // Sparse-vLLM stores [cos, sin] as [head_dim / 2, head_dim / 2], so the
  // packed row width is head_dim rather than 2 * head_dim.
  const float* rope = cos_sin + static_cast<int64_t>(position) * head_dim;
  for (int head = 0; head < heads; ++head) {
    scalar_t* ptr = token + static_cast<int64_t>(head) * head_dim;
    const float x1 = to_float(ptr[pair]);
    const float x2 = to_float(ptr[pair + half_dim]);
    const float c = rope[pair];
    const float s = rope[pair + half_dim];
    ptr[pair] = from_float<scalar_t>(x1 * c - x2 * s);
    ptr[pair + half_dim] = from_float<scalar_t>(x2 * c + x1 * s);
  }
}

template <typename scalar_t>
__global__ void softmax_rows_kernel(scalar_t* values, int rows, int columns) {
  __shared__ float reduction[256];
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  if (row >= rows) return;
  float maximum = -FLT_MAX;
  for (int column = lane; column < columns; column += blockDim.x) {
    maximum = fmaxf(maximum, to_float(values[static_cast<int64_t>(row) * columns + column]));
  }
  reduction[lane] = maximum;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width >>= 1) {
    if (lane < width) reduction[lane] = fmaxf(reduction[lane], reduction[lane + width]);
    __syncthreads();
  }
  maximum = reduction[0];
  float total = 0.0f;
  for (int column = lane; column < columns; column += blockDim.x) {
    total += __expf(to_float(values[static_cast<int64_t>(row) * columns + column]) - maximum);
  }
  reduction[lane] = total;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width >>= 1) {
    if (lane < width) reduction[lane] += reduction[lane + width];
    __syncthreads();
  }
  total = reduction[0];
  for (int column = lane; column < columns; column += blockDim.x) {
    values[static_cast<int64_t>(row) * columns + column] = from_float<scalar_t>(
        __expf(to_float(values[static_cast<int64_t>(row) * columns + column]) - maximum) / total);
  }
}

void check_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), name,
              " must be a contiguous CUDA tensor.");
}

void set_host_pointers(
    const std::vector<torch::Tensor>& sources,
    torch::Tensor pointer_table) {
  check_cuda(pointer_table, "pointer_table");
  TORCH_CHECK(pointer_table.scalar_type() == torch::kInt64 && pointer_table.dim() == 1,
              "pointer_table must be a CUDA int64 vector.");
  TORCH_CHECK(pointer_table.numel() >= static_cast<int64_t>(sources.size()),
              "pointer_table is smaller than the source batch.");
  std::vector<std::uint64_t> pointers(sources.size());
  for (size_t index = 0; index < sources.size(); ++index) {
    const auto& source = sources[index];
    TORCH_CHECK(source.device().is_cpu() && source.is_contiguous() && source.is_pinned(),
                "ShadowKV CUTLASS sources must be contiguous pinned CPU tensors.");
    TORCH_CHECK(source.scalar_type() == torch::kBFloat16 || source.scalar_type() == torch::kHalf,
                "ShadowKV CUTLASS sources must be BF16 or FP16.");
    void* device_pointer = nullptr;
    const auto status = cudaHostGetDevicePointer(&device_pointer, source.data_ptr(), 0);
    TORCH_CHECK(status == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(status));
    pointers[index] = reinterpret_cast<std::uint64_t>(device_pointer);
  }
  auto stream = at::cuda::getCurrentCUDAStream(pointer_table.get_device());
  const auto status = cudaMemcpyAsync(
      pointer_table.data_ptr<int64_t>(), pointers.data(),
      pointers.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream.stream());
  TORCH_CHECK(status == cudaSuccess, "ShadowKV CUTLASS pointer upload failed: ", cudaGetErrorString(status));
}

void gather_host(
    torch::Tensor pointer_table,
    torch::Tensor positions,
    torch::Tensor lengths,
    torch::Tensor output) {
  check_cuda(pointer_table, "pointer_table");
  check_cuda(positions, "positions");
  check_cuda(lengths, "lengths");
  check_cuda(output, "output");
  TORCH_CHECK(pointer_table.scalar_type() == torch::kInt64 && positions.scalar_type() == torch::kInt32 &&
                  lengths.scalar_type() == torch::kInt32 && output.scalar_type() == torch::kBFloat16,
              "ShadowKV CUTLASS gather expects int64/int32/int32/BF16 metadata and output.");
  TORCH_CHECK(positions.dim() == 2 && lengths.dim() == 1 && output.dim() == 4,
              "gather_host expects positions[B,W], lengths[B], output[B,W,H,D].");
  const int batch = static_cast<int>(positions.size(0));
  const int width = static_cast<int>(positions.size(1));
  const int elements = static_cast<int>(output.size(2) * output.size(3));
  const int64_t work = static_cast<int64_t>(batch) * width * elements;
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  const int blocks = static_cast<int>(std::min<int64_t>((work + 255) / 256, 65535));
  if (elements % 8 == 0) {
    gather_host_vector_kernel<<<blocks, 256, 0, stream.stream()>>>(
        reinterpret_cast<const std::uint64_t*>(pointer_table.data_ptr<int64_t>()),
        positions.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
        reinterpret_cast<std::uint16_t*>(output.data_ptr()), batch, width, elements / 8);
  } else {
    gather_host_kernel<__nv_bfloat16><<<blocks, 256, 0, stream.stream()>>>(
        reinterpret_cast<const std::uint64_t*>(pointer_table.data_ptr<int64_t>()),
        positions.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), batch, width, elements);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_host_per_head(
    torch::Tensor pointer_table,
    torch::Tensor positions,
    torch::Tensor lengths,
    torch::Tensor output) {
  check_cuda(pointer_table, "pointer_table");
  check_cuda(positions, "positions");
  check_cuda(lengths, "lengths");
  check_cuda(output, "output");
  TORCH_CHECK(pointer_table.scalar_type() == torch::kInt64 &&
                  positions.scalar_type() == torch::kInt32 &&
                  lengths.scalar_type() == torch::kInt32 &&
                  output.scalar_type() == torch::kBFloat16,
              "CUTLASS ShadowKV per-head gather requires BF16 output and integer metadata.");
  TORCH_CHECK(pointer_table.dim() == 1 && positions.dim() == 3 &&
                  lengths.dim() == 2 && output.dim() == 4,
              "gather_host_per_head expects pointer[B], positions[B,H,W], "
              "lengths[B,H], output[B,H,W,D].");
  const int batch = static_cast<int>(positions.size(0));
  const int heads = static_cast<int>(positions.size(1));
  const int width = static_cast<int>(positions.size(2));
  const int head_dim = static_cast<int>(output.size(3));
  TORCH_CHECK(pointer_table.numel() >= batch &&
                  lengths.sizes() == torch::IntArrayRef({batch, heads}) &&
                  output.sizes() == torch::IntArrayRef({batch, heads, width, head_dim}),
              "CUTLASS ShadowKV per-head gather dimensions disagree.");
  const int64_t work = static_cast<int64_t>(batch) * heads * width * head_dim;
  const int blocks = static_cast<int>(std::min<int64_t>((work + 255) / 256, 65535));
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  gather_host_per_head_kernel<__nv_bfloat16><<<blocks, 256, 0, stream.stream()>>>(
      reinterpret_cast<const std::uint64_t*>(pointer_table.data_ptr<int64_t>()),
      positions.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      batch, heads, width, head_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

using Bf16Epilogue = cutlass::epilogue::thread::LinearCombination<
    cutlass::bfloat16_t, 8, float, float>;
using ScoreGemm = cutlass::gemm::device::GemmBatched<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>, Bf16Epilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2, 8, 8>;
using ScoreSoftmaxEpilogue = cutlass::epilogue::thread::LinearCombination<
    cutlass::bfloat16_t,
    8,
    float,
    float,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
using ExactBatchGemmSoftmax = cutlass::BatchGemmSoftmax<
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::bfloat16_t,
    cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 256, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    ScoreSoftmaxEpilogue,
    4,
    cutlass::MatrixShape<1, 1024>,
    8,
    8,
    8,
    float,
    float,
    cutlass::bfloat16_t,
    float>;
using ReconstructionGemm = cutlass::gemm::device::GemmBatched<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>, Bf16Epilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2, 8, 8>;

// This is the actual ShadowKV gather-GEMM operator from the reference
// implementation.  It gathers complete chunk rows of U inside the CUTLASS
// input iterator, so the decode path does not materialize U_selected or run a
// separate host-pointer gather before reconstruction.
using FusedGatherGemm =
    cutlass::gemm::device::GemmUniversalBatchGatherIndices<
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        cutlass::bfloat16_t,
        cutlass::layout::ColumnMajor,
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        Bf16Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        5,
        8,
        8,
        cutlass::arch::OpMultiplyAdd,
        cutlass::ComplexTransform::kNone,
        cutlass::ComplexTransform::kNone,
        true,
        false,
        false>;

void batch_gemm_softmax(
    torch::Tensor query,
    torch::Tensor landmarks,
    torch::Tensor output,
    float scale) {
  check_cuda(query, "query");
  check_cuda(landmarks, "landmarks");
  check_cuda(output, "output");
  TORCH_CHECK(query.scalar_type() == torch::kBFloat16 && landmarks.scalar_type() == torch::kBFloat16 &&
                  output.scalar_type() == torch::kBFloat16,
              "CUTLASS ShadowKV score GEMM requires BF16 inputs and output.");
  TORCH_CHECK(query.dim() == 4 && landmarks.dim() == 4 && output.dim() == 4,
              "batch_gemm_softmax expects query[B,H,G,D], landmarks[B,H,N,D], output[B,H,G,N].");
  const int batch = static_cast<int>(query.size(0));
  const int heads = static_cast<int>(query.size(1));
  const int groups = static_cast<int>(query.size(2));
  const int head_dim = static_cast<int>(query.size(3));
  const int columns = static_cast<int>(landmarks.size(2));
  TORCH_CHECK(landmarks.size(0) == batch && landmarks.size(1) == heads && landmarks.size(3) == head_dim &&
                  output.size(0) == batch && output.size(1) == heads && output.size(2) == groups &&
                  output.size(3) == columns,
              "batch_gemm_softmax dimensions disagree.");
  TORCH_CHECK((groups * columns) % 8 == 0 && (columns * head_dim) % 8 == 0,
              "CUTLASS ShadowKV score GEMM requires padded candidate width divisible by 8.");
  ScoreGemm gemm;
  typename ScoreGemm::Arguments args(
      {groups, columns, head_dim},
      {reinterpret_cast<const cutlass::bfloat16_t*>(query.data_ptr()), head_dim},
      static_cast<int64_t>(groups) * head_dim,
      {reinterpret_cast<const cutlass::bfloat16_t*>(landmarks.data_ptr()), head_dim},
      static_cast<int64_t>(columns) * head_dim,
      {reinterpret_cast<cutlass::bfloat16_t*>(output.data_ptr()), columns},
      static_cast<int64_t>(groups) * columns,
      {reinterpret_cast<cutlass::bfloat16_t*>(output.data_ptr()), columns},
      static_cast<int64_t>(groups) * columns,
      {scale, 0.0f}, batch * heads);
  auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
  const auto can_implement = ScoreGemm::can_implement(args);
  TORCH_CHECK(can_implement == cutlass::Status::kSuccess,
              "CUTLASS ShadowKV score GEMM cannot implement problem, status ",
              static_cast<int>(can_implement));
  const auto status = gemm(args, nullptr, stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS ShadowKV score GEMM failed with status ", static_cast<int>(status));
  const int rows = batch * heads * groups;
  softmax_rows_kernel<__nv_bfloat16><<<rows, 256, 0, stream.stream()>>>(
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows, columns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void configure_shadowkv_cutlass() {
  const auto status = ExactBatchGemmSoftmax::configure();
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS ShadowKV score-kernel configuration failed with status ",
      static_cast<int>(status));
}

void batch_gemm_softmax_exact(
    torch::Tensor query,
    torch::Tensor landmarks,
    torch::Tensor logits,
    torch::Tensor probabilities,
    torch::Tensor norm,
    torch::Tensor sum,
    float scale) {
  check_cuda(query, "query");
  check_cuda(landmarks, "landmarks");
  check_cuda(logits, "logits");
  check_cuda(probabilities, "probabilities");
  check_cuda(norm, "norm");
  check_cuda(sum, "sum");
  TORCH_CHECK(
      query.scalar_type() == torch::kBFloat16 &&
          landmarks.scalar_type() == torch::kBFloat16 &&
          logits.scalar_type() == torch::kBFloat16 &&
          probabilities.scalar_type() == torch::kBFloat16 &&
          norm.scalar_type() == torch::kFloat &&
          sum.scalar_type() == torch::kFloat,
      "ShadowKV fused score/softmax requires BF16 matrices and FP32 reductions.");
  TORCH_CHECK(
      query.dim() == 4 && landmarks.dim() == 4 && logits.dim() == 4 &&
          probabilities.dim() == 4 && norm.dim() == 3 && sum.dim() == 3,
      "ShadowKV fused score/softmax received invalid tensor ranks.");
  const int batch = static_cast<int>(query.size(0));
  const int heads = static_cast<int>(query.size(1));
  const int groups = static_cast<int>(query.size(2));
  const int head_dim = static_cast<int>(query.size(3));
  const int columns = static_cast<int>(landmarks.size(2));
  TORCH_CHECK(
      landmarks.sizes() == torch::IntArrayRef({batch, heads, columns, head_dim}) &&
          logits.sizes() == torch::IntArrayRef({batch, heads, groups, columns}) &&
          probabilities.sizes() == logits.sizes() &&
          norm.size(0) == batch && norm.size(1) == heads &&
          sum.sizes() == norm.sizes(),
      "ShadowKV fused score/softmax dimensions disagree.");
  TORCH_CHECK(
      groups > 0 && columns > 0 && head_dim > 0 &&
          (groups * columns) % 8 == 0 && (columns * head_dim) % 8 == 0,
      "ShadowKV fused score/softmax requires aligned GEMM dimensions.");
  const int block_num = (columns + 256 - 1) / 256;
  TORCH_CHECK(
      norm.size(2) >= static_cast<int64_t>(block_num) * groups,
      "ShadowKV fused score/softmax norm workspace is too small.");
  const auto problem = cutlass::gemm::GemmCoord{groups, columns, head_dim};
  const int64_t batch_stride_a = static_cast<int64_t>(groups) * head_dim;
  const int64_t batch_stride_b = static_cast<int64_t>(columns) * head_dim;
  const int64_t batch_stride_matrix = static_cast<int64_t>(groups) * columns;
  const int64_t batch_stride_reduce = static_cast<int64_t>(block_num) * groups;
  typename ExactBatchGemmSoftmax::TensorRefA ref_a(
      reinterpret_cast<cutlass::bfloat16_t*>(query.data_ptr()), head_dim);
  typename ExactBatchGemmSoftmax::TensorRefB ref_b(
      reinterpret_cast<cutlass::bfloat16_t*>(landmarks.data_ptr()), head_dim);
  typename ExactBatchGemmSoftmax::TensorRefC ref_c(nullptr, columns);
  typename ExactBatchGemmSoftmax::TensorRefC ref_logits(
      reinterpret_cast<cutlass::bfloat16_t*>(logits.data_ptr()), columns);
  typename ExactBatchGemmSoftmax::TensorRefN ref_norm(norm.data_ptr<float>(), groups);
  typename ExactBatchGemmSoftmax::TensorRefSum ref_sum(sum.data_ptr<float>(), groups);
  typename ExactBatchGemmSoftmax::TensorRefSoft ref_probabilities(
      reinterpret_cast<cutlass::bfloat16_t*>(probabilities.data_ptr()), columns);
  typename ExactBatchGemmSoftmax::Arguments args(
      problem,
      batch * heads,
      ref_a,
      ref_b,
      ref_c,
      ref_logits,
      {scale, 0.0f},
      ref_norm,
      ref_sum,
      ref_probabilities,
      batch_stride_a,
      batch_stride_b,
      batch_stride_matrix,
      batch_stride_matrix,
      batch_stride_reduce,
      batch_stride_reduce,
      batch_stride_matrix);
  auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
  ExactBatchGemmSoftmax op;
  TORCH_CHECK(
      op.initialize(args) == cutlass::Status::kSuccess,
      "ShadowKV fused score/softmax initialization failed.");
  const auto status = op(stream.stream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "ShadowKV fused score/softmax failed with status ",
      static_cast<int>(status));
}

void batch_gather_gemm_rope(
    torch::Tensor pointer_table,
    torch::Tensor positions,
    torch::Tensor lengths,
    torch::Tensor sv_bmm,
    torch::Tensor cos_sin,
    torch::Tensor u_workspace,
    torch::Tensor output) {
  check_cuda(pointer_table, "pointer_table");
  check_cuda(positions, "positions");
  check_cuda(lengths, "lengths");
  check_cuda(sv_bmm, "sv_bmm");
  check_cuda(cos_sin, "cos_sin");
  check_cuda(u_workspace, "u_workspace");
  check_cuda(output, "output");
  TORCH_CHECK(pointer_table.scalar_type() == torch::kInt64 && positions.scalar_type() == torch::kInt32 &&
                  lengths.scalar_type() == torch::kInt32 && sv_bmm.scalar_type() == torch::kBFloat16 &&
                  u_workspace.scalar_type() == torch::kBFloat16 && output.scalar_type() == torch::kBFloat16 &&
                  cos_sin.scalar_type() == torch::kFloat,
              "CUTLASS ShadowKV reconstruction currently requires BF16 payloads and FP32 RoPE.");
  const int batch = static_cast<int>(positions.size(0));
  const int width = static_cast<int>(positions.size(1));
  const int rank = static_cast<int>(sv_bmm.size(1));
  const int heads = static_cast<int>(sv_bmm.size(2));
  const int head_dim = static_cast<int>(sv_bmm.size(3));
  TORCH_CHECK(positions.dim() == 2 && lengths.numel() == batch && pointer_table.numel() >= batch &&
                  u_workspace.sizes() == torch::IntArrayRef({batch, width, 1, rank}) &&
                  output.sizes() == torch::IntArrayRef({batch, width, heads, head_dim}),
              "CUTLASS ShadowKV reconstruction dimensions disagree.");
  const int output_width = heads * head_dim;
  TORCH_CHECK(output_width % 8 == 0 && (width * output_width) % 8 == 0,
              "CUTLASS ShadowKV reconstruction requires an output stride divisible by 8.");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  const int64_t work = static_cast<int64_t>(batch) * width * rank;
  gather_host_kernel<__nv_bfloat16><<<static_cast<int>(std::min<int64_t>((work + 255) / 256, 65535)), 256, 0, stream.stream()>>>(
      reinterpret_cast<const std::uint64_t*>(pointer_table.data_ptr<int64_t>()),
      positions.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(u_workspace.data_ptr()), batch, width, rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ReconstructionGemm gemm;
  typename ReconstructionGemm::Arguments args(
      {width, output_width, rank},
      {reinterpret_cast<const cutlass::bfloat16_t*>(u_workspace.data_ptr()), rank},
      static_cast<int64_t>(width) * rank,
      {reinterpret_cast<const cutlass::bfloat16_t*>(sv_bmm.data_ptr()), output_width},
      static_cast<int64_t>(rank) * output_width,
      {reinterpret_cast<const cutlass::bfloat16_t*>(output.data_ptr()), output_width},
      static_cast<int64_t>(width) * output_width,
      {reinterpret_cast<cutlass::bfloat16_t*>(output.data_ptr()), output_width},
      static_cast<int64_t>(width) * output_width, {1.0f, 0.0f}, batch);
  const auto status = gemm(args, nullptr, stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS ShadowKV reconstruction GEMM failed with status ", static_cast<int>(status));
  if (cos_sin.numel() == 0) return;
  TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == head_dim,
              "ShadowKV RoPE cache must have shape [positions, head_dim] with packed cos/sin halves.");
  const int threads = std::min(1024, std::max(1, head_dim / 2));
  apply_rope_kernel<__nv_bfloat16><<<static_cast<unsigned int>(batch * width), threads, 0, stream.stream()>>>(
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), cos_sin.data_ptr<float>(),
      positions.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(), batch, width, heads, head_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void batch_gather_gemm_fused_rope(
    torch::Tensor u_device,
    torch::Tensor sv_column_major,
    torch::Tensor chunk_indices,
    torch::Tensor positions,
    torch::Tensor lengths,
    torch::Tensor cos_sin,
    torch::Tensor output,
    int prompt_capacity,
    int chunk_size) {
  check_cuda(u_device, "u_device");
  check_cuda(sv_column_major, "sv_column_major");
  check_cuda(chunk_indices, "chunk_indices");
  check_cuda(positions, "positions");
  check_cuda(lengths, "lengths");
  check_cuda(cos_sin, "cos_sin");
  check_cuda(output, "output");
  TORCH_CHECK(
      u_device.scalar_type() == torch::kBFloat16 &&
          sv_column_major.scalar_type() == torch::kBFloat16 &&
          chunk_indices.scalar_type() == torch::kInt32 &&
          positions.scalar_type() == torch::kInt32 &&
          lengths.scalar_type() == torch::kInt32 &&
          cos_sin.scalar_type() == torch::kFloat &&
          output.scalar_type() == torch::kBFloat16,
      "Fused ShadowKV gather GEMM requires BF16 payloads, INT32 indices, "
      "and FP32 RoPE.");
  TORCH_CHECK(
      u_device.dim() == 3 && sv_column_major.dim() == 4 &&
          chunk_indices.dim() == 3 && positions.dim() == 3 &&
          lengths.dim() == 2 && output.dim() == 4,
      "Fused ShadowKV gather GEMM received an invalid tensor rank.");

  const int batch = static_cast<int>(u_device.size(0));
  const int capacity = static_cast<int>(u_device.size(1));
  const int rank = static_cast<int>(u_device.size(2));
  const int heads = static_cast<int>(sv_column_major.size(1));
  const int head_dim = static_cast<int>(sv_column_major.size(2));
  const int chunks = static_cast<int>(chunk_indices.size(2));
  const int width = chunks * chunk_size;
  TORCH_CHECK(
      capacity == prompt_capacity && sv_column_major.size(0) == batch &&
          sv_column_major.size(3) == rank && chunk_indices.size(0) == batch &&
          chunk_indices.size(1) == heads && positions.size(0) == batch &&
          positions.size(1) == heads && positions.size(2) == width &&
          lengths.size(0) == batch && lengths.size(1) == heads &&
          output.sizes() == torch::IntArrayRef({batch * heads, width, 1, head_dim}),
      "Fused ShadowKV gather GEMM dimensions disagree.");
  TORCH_CHECK(
      cos_sin.dim() == 2 && cos_sin.size(1) == head_dim &&
          rank % 8 == 0 && head_dim % 8 == 0 && width % 128 == 0,
      "Fused ShadowKV gather GEMM requires rank/head_dim aligned to 8 and "
      "a sparse width aligned to the CUTLASS tile.");

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  const auto problem_size = cutlass::gemm::GemmCoord{width, head_dim, rank};
  const auto stride_a = LayoutA::packed({prompt_capacity, rank}).stride();
  const auto stride_b = LayoutB::packed({rank, head_dim}).stride();
  const auto stride_output = LayoutC::packed({width, head_dim}).stride();
  typename FusedGatherGemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kBatched,
      problem_size,
      batch * heads,
      {1.0f, 0.0f},
      reinterpret_cast<const cutlass::bfloat16_t*>(u_device.data_ptr()),
      reinterpret_cast<const cutlass::bfloat16_t*>(sv_column_major.data_ptr()),
      reinterpret_cast<const cutlass::bfloat16_t*>(output.data_ptr()),
      reinterpret_cast<cutlass::bfloat16_t*>(output.data_ptr()),
      nullptr,
      nullptr,
      static_cast<int64_t>(prompt_capacity) * rank,
      static_cast<int64_t>(head_dim) * rank,
      static_cast<int64_t>(width) * head_dim,
      static_cast<int64_t>(width) * head_dim,
      stride_a,
      stride_b,
      stride_output,
      stride_output,
      stride_output,
      stride_output,
      chunk_indices.data_ptr<int32_t>(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      chunks,
      0,
      0,
      0,
      0,
      prompt_capacity,
      chunk_size,
      heads,
      nullptr);
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  const auto can_implement = FusedGatherGemm::can_implement(args);
  TORCH_CHECK(
      can_implement == cutlass::Status::kSuccess,
      "CUTLASS ShadowKV fused gather GEMM cannot implement problem, status ",
      static_cast<int>(can_implement));
  const auto status = FusedGatherGemm{}(args, nullptr, stream.stream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS ShadowKV fused gather GEMM failed with status ",
      static_cast<int>(status));

  // The vendored reference epilogue intentionally leaves RoPE as a separate
  // operation.  Reuse the existing graph-safe kernel on flattened
  // (request, KV-head) rows after the fused gather/GEMM.
  const int flat_batch = batch * heads;
  const int threads = std::min(1024, std::max(1, head_dim / 2));
  apply_rope_kernel<__nv_bfloat16><<<
      static_cast<unsigned int>(flat_batch * width), threads, 0, stream.stream()>>>(
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      cos_sin.data_ptr<float>(), positions.reshape({flat_batch, width}).data_ptr<int32_t>(),
      lengths.reshape({flat_batch}).data_ptr<int32_t>(), flat_batch, width, 1, head_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("set_host_pointers", &set_host_pointers);
  module.def("configure_shadowkv_cutlass", &configure_shadowkv_cutlass);
  module.def("gather_host", &gather_host);
  module.def("gather_host_per_head", &gather_host_per_head);
  module.def("batch_gemm_softmax", &batch_gemm_softmax);
  module.def("batch_gemm_softmax_exact", &batch_gemm_softmax_exact);
  module.def("batch_gather_gemm", &batch_gather_gemm_rope);
  module.def("batch_gather_gemm_rope", &batch_gather_gemm_rope);
  module.def("batch_gather_gemm_fused_rope", &batch_gather_gemm_fused_rope);
}
