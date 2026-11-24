#pragma once
// SGD and AdamW optimizers for ValueNode parameters.
#include "Autograd.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

class SGD {
private:
    std::vector<ValuePtr> m_params;
    float m_lr;

public:
    SGD() = default;
    SGD(std::vector<ValuePtr> params, float lr = 0.01f)
        : m_params(std::move(params)), m_lr(lr) {}

    void step() {
        for (auto& param : m_params) {
            size_t total = param->data.size();
            float* data_ptr = param->data.data();
            const float* grad_ptr = param->grad.data();
            for (size_t i = 0; i < total; ++i) {
                data_ptr[i] -= m_lr * grad_ptr[i];
            }
        }
    }

    void zero_grad() {
        for (auto& param : m_params) {
            param->grad.fill(0.0f);
        }
    }
};

class AdamW {
private:
    std::vector<ValuePtr> m_params;
    std::vector<Tensor> m_m;
    std::vector<Tensor> m_v;
    float m_lr = 0.0025f;
    float m_beta1 = 0.9f;
    float m_beta2 = 0.999f;
    float m_eps = 1e-8f;
    float m_weight_decay = 1e-4f;

public:
    uint64_t step_count = 0;

    AdamW() = default;

    void configure(std::vector<ValuePtr> params, float lr = 0.0028f, float weight_decay = 1e-4f) {
        m_params = std::move(params);
        m_lr = lr;
        m_weight_decay = weight_decay;
        step_count = 0;
        m_m.clear();
        m_v.clear();
        for (const auto& p : m_params) {
            m_m.emplace_back(p->data.shape(), 0.0f);
            m_v.emplace_back(p->data.shape(), 0.0f);
        }
    }

    void zero_grad() {
        for (auto& param : m_params) {
            param->grad.fill(0.0f);
        }
    }

    void step(int batch_size = 1) {
        if (m_params.empty()) return;
        ++step_count;
        float inv_bc1 = 1.0f / (1.0f - std::pow(m_beta1, static_cast<float>(step_count)));
        float inv_bc2 = 1.0f / (1.0f - std::pow(m_beta2, static_cast<float>(step_count)));
        float b_scale = 1.0f / static_cast<float>(std::max(1, batch_size));

        for (size_t p_idx = 0; p_idx < m_params.size(); ++p_idx) {
            ValuePtr& param = m_params[p_idx];
            size_t n = param->data.size();
            float* w = param->data.data();
            const float* g = param->grad.data();
            float* m = m_m[p_idx].data();
            float* v = m_v[p_idx].data();
            // Do not apply weight decay to 1D bias vectors
            float wd = (param->data.shape()[0] == 1) ? 0.0f : m_weight_decay;

            for (size_t i = 0; i < n; ++i) {
                float grad = std::max(-2.0f, std::min(2.0f, g[i] * b_scale));
                m[i] = m_beta1 * m[i] + (1.0f - m_beta1) * grad;
                v[i] = m_beta2 * v[i] + (1.0f - m_beta2) * (grad * grad);
                float m_hat = m[i] * inv_bc1;
                float v_hat = v[i] * inv_bc2;
                w[i] -= m_lr * (m_hat / (std::sqrt(v_hat) + m_eps) + wd * w[i]);
            }
        }
    }
};
