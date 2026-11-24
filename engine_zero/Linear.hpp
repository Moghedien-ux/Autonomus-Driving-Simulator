#pragma once
#include "Module.hpp"
#include "Autograd.hpp"
#include "Ops.hpp"
#include <cmath>
#include <cstdint>

class Linear : public Module {
public:
    ValuePtr W;
    ValuePtr b;
    size_t in_features = 0;
    size_t out_features = 0;

    Linear() = default;

    Linear(size_t in_f, size_t out_f, uint32_t seed = 42) {
        init(in_f, out_f, seed);
    }

    void init(size_t in_f, size_t out_f, uint32_t seed = 42) {
        in_features = in_f;
        out_features = out_f;
        float scale = std::sqrt(2.0f / static_cast<float>(in_features));

        Tensor w_tensor({in_features, out_features});
        uint32_t st = seed ? seed : 0xCAFEBABEu;
        for (size_t i = 0; i < w_tensor.size(); ++i) {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            float u = ((st & 0x00FFFFFFu) / 8388608.0f) - 1.0f;
            w_tensor.at_flat(i) = u * scale;
        }

        Tensor b_tensor({1, out_features}, 0.0f);
        W = ValueNode::make(std::move(w_tensor), {}, "W");
        b = ValueNode::make(std::move(b_tensor), {}, "b");
    }

    ValuePtr forward(const ValuePtr& x) {
        auto xw = EngineOps::matmul(x, W);
        return EngineOps::add(xw, b);
    }

    std::vector<ValuePtr> parameters() const override {
        return {W, b};
    }
};
