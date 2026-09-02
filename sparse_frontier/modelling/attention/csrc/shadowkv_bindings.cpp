#include <torch/extension.h>

torch::Tensor retrieval_softmax_cuda(
    torch::Tensor query,
    torch::Tensor landmarks);

torch::Tensor gather_gemm_rope_cuda(
    torch::Tensor u,
    torch::Tensor sv,
    torch::Tensor position_ids,
    torch::Tensor cos_sin_cache);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "retrieval_softmax",
        &retrieval_softmax_cuda,
        "Fused landmark GEMM and per-GQA-group softmax (CUDA)");
    module.def(
        "gather_gemm_rope",
        &gather_gemm_rope_cuda,
        "Fused gathered U*SV reconstruction and NeoX RoPE (CUDA)");
}
