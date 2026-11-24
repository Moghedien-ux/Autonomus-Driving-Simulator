#pragma once
#include "Module.hpp"
#include "Autograd.hpp"
#include "Ops.hpp"
#include <cmath>
#include <cstdint>

class Conv2D : public Module {
public:
    ValuePtr kernel;
    ValuePtr bias;
    size_t in_channels = 0;
    size_t out_channels = 0;
    size_t kernel_size = 3;
    size_t stride = 1;
    size_t padding = 1;

    Conv2D() = default;

    Conv2D(size_t in_c, size_t out_c, size_t k_size = 3, size_t str = 1, size_t pad = 1, uint32_t seed = 42) {
        init(in_c, out_c, k_size, str, pad, seed);
    }

    void init(size_t in_c, size_t out_c, size_t k_size = 3, size_t str = 1, size_t pad = 1, uint32_t seed = 42) {
        in_channels = in_c;
        out_channels = out_c;
        kernel_size = k_size;
        stride = str;
        padding = pad;

        size_t k_flat_dim = in_channels * kernel_size * kernel_size;
        float scale = std::sqrt(2.0f / static_cast<float>(k_flat_dim));

        Tensor k_tensor({out_channels, k_flat_dim});
        uint32_t st = seed ? seed : 0x1337BEEFu;
        for (size_t i = 0; i < k_tensor.size(); ++i) {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            float u = ((st & 0x00FFFFFFu) / 8388608.0f) - 1.0f; // [-1, 1)
            k_tensor.at_flat(i) = u * scale;
        }

        Tensor b_tensor({1, out_channels}, 0.0f);
        kernel = ValueNode::make(std::move(k_tensor), {}, "conv_kernel");
        bias   = ValueNode::make(std::move(b_tensor), {}, "conv_bias");
    }

    ValuePtr forward(const ValuePtr& x, bool compute_input_grad = true) {
        return EngineOps::conv2d(x, kernel, bias, stride, padding, compute_input_grad);
    }

    std::vector<ValuePtr> parameters() const override {
        return {kernel, bias};
    }
};
