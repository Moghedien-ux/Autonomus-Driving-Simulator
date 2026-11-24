#pragma once
// Tensor and autograd operations (im2col/col2im conv2d, pooling, activations).
#include "Tensor.hpp"
#include "Autograd.hpp"

namespace EngineOps {

    Tensor transpose(const Tensor& A);
    Tensor matmul(const Tensor& A, const Tensor& B);
    Tensor add(const Tensor& A, const Tensor& B);
    Tensor sub(const Tensor& A, const Tensor& B);
    Tensor relu(const Tensor& A);
    Tensor leaky_relu(const Tensor& A, float slope = 0.1f);
    Tensor tanh_op(const Tensor& A);
    Tensor mse_loss(const Tensor& pred, const Tensor& target);
    Tensor gelu(const Tensor& A);
    Tensor softmax(const Tensor& A, int dim = -1);
    Tensor scale(const Tensor& A, float factor);

    ValuePtr transpose(const ValuePtr& A);
    ValuePtr matmul(const ValuePtr& A, const ValuePtr& B);
    ValuePtr add(const ValuePtr& A, const ValuePtr& B);
    ValuePtr sub(const ValuePtr& A, const ValuePtr& B);
    ValuePtr relu(const ValuePtr& A);
    ValuePtr leaky_relu(const ValuePtr& A, float slope = 0.1f);
    ValuePtr tanh_op(const ValuePtr& A);
    ValuePtr mse_loss(const ValuePtr& pred, const ValuePtr& target);
    ValuePtr gelu(const ValuePtr& A);
    ValuePtr softmax(const ValuePtr& A, int dim = -1);
    ValuePtr scale(const ValuePtr& A, float factor);

    // im2col + col2im GEMM-based 2D Convolution
    Tensor im2col(const Tensor& input, size_t kernel_h, size_t kernel_w, size_t stride = 1, size_t padding = 0);
    Tensor col2im(const Tensor& col, size_t C, size_t H, size_t W,
                  size_t kernel_h, size_t kernel_w, size_t stride = 1, size_t padding = 0);

    Tensor conv2d(const Tensor& input, const Tensor& kernel, size_t stride = 1, size_t padding = 0);
    ValuePtr conv2d(const ValuePtr& input, const ValuePtr& kernel, const ValuePtr& bias = nullptr,
                    size_t stride = 1, size_t padding = 0, bool compute_input_grad = true);

    // Spatial pooling & tensor shaping ops for CNN + HydraNet fusion
    ValuePtr max_pool2d(const ValuePtr& input, size_t pool_size = 2);
    ValuePtr flatten_to_row(const ValuePtr& input);
    ValuePtr concat_cols(const ValuePtr& A, const ValuePtr& B);

} // namespace EngineOps
