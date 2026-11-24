#include "Ops.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>

namespace EngineOps {

static void accumulate_inplace(Tensor& dst, const Tensor& delta) {
    const size_t n = dst.size();
    float* d = dst.data();
    const float* s = delta.data();
    for (size_t i = 0; i < n; ++i) {
        d[i] += s[i];
    }
}

Tensor transpose(const Tensor& A) {
    if (A.ndim() != 2) {
        throw std::invalid_argument("Mat not 2D.");
    }
    size_t M = A.shape()[0];
    size_t N = A.shape()[1];

    Tensor A_T({N, M});
    const float* src = A.data();
    float* dst = A_T.data();

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            dst[j * M + i] = src[i * N + j];
        }
    }
    return A_T;
}

Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("Not 2D.");
    }
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t K_B = B.shape()[0];
    size_t N = B.shape()[1];

    if (K != K_B) {
        throw std::invalid_argument("Not fitting!");
    }

    Tensor C({M, N}, 0.0f);
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();

    constexpr size_t BLOCK_SIZE = 64;
    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            size_t i_max = std::min(i0 + BLOCK_SIZE, M);
            size_t j_max = std::min(j0 + BLOCK_SIZE, N);

            for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                size_t k_max = std::min(k0 + BLOCK_SIZE, K);

                for (size_t i = i0; i < i_max; ++i) {
                    for (size_t k = k0; k < k_max; ++k) {
                        float a_val = a_ptr[i * K + k];
                        float* c_row = &c_ptr[i * N];
                        const float* b_row = &b_ptr[k * N];
                        for (size_t j = j0; j < j_max; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }
    return C;
}

Tensor add(const Tensor& A, const Tensor& B) {
    if (A.shape() == B.shape()) {
        Tensor C(A.shape());
        size_t total_elements = A.size();
        const float* a_ptr = A.data();
        const float* b_ptr = B.data();
        float* c_ptr = C.data();
        for (size_t i = 0; i < total_elements; ++i) {
            c_ptr[i] = a_ptr[i] + b_ptr[i];
        }
        return C;
    }

    if (A.ndim() == 2 && B.ndim() == 2 && B.shape()[0] == 1 && A.shape()[1] == B.shape()[1]) {
        size_t M = A.shape()[0];
        size_t N = A.shape()[1];
        Tensor C(A.shape());
        const float* a_ptr = A.data();
        const float* b_ptr = B.data();
        float* c_ptr = C.data();
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                c_ptr[i * N + j] = a_ptr[i * N + j] + b_ptr[j];
            }
        }
        return C;
    }
    throw std::invalid_argument("TENSOR ADD: Incompatible shapes!");
}

ValuePtr add(const ValuePtr& A, const ValuePtr& B) {
    Tensor c_data = add(A->data, B->data);
    auto out = ValueNode::make(std::move(c_data), {A, B}, "add");
    ValueNode* out_raw = out.get();

    out->backward_fn = [A, B, out_raw]() {
        if (A->data.shape() != B->data.shape() && B->data.shape()[0] == 1) {
            accumulate_inplace(A->grad, out_raw->grad);
            size_t M = out_raw->grad.shape()[0];
            size_t N = out_raw->grad.shape()[1];
            const float* out_g = out_raw->grad.data();
            float* b_g = B->grad.data();
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    b_g[j] += out_g[i * N + j];
                }
            }
        } else {
            accumulate_inplace(A->grad, out_raw->grad);
            accumulate_inplace(B->grad, out_raw->grad);
        }
    };
    return out;
}

ValuePtr matmul(const ValuePtr& A, const ValuePtr& B) {
    Tensor c_data = matmul(A->data, B->data);
    auto out = ValueNode::make(std::move(c_data), {A, B}, "matmul");
    ValueNode* out_raw = out.get();

    out->backward_fn = [A, B, out_raw]() {
        Tensor grad_A = matmul(out_raw->grad, transpose(B->data));
        accumulate_inplace(A->grad, grad_A);

        Tensor grad_B = matmul(transpose(A->data), out_raw->grad);
        accumulate_inplace(B->grad, grad_B);
    };
    return out;
}

ValuePtr transpose(const ValuePtr& A) {
    Tensor t_data = transpose(A->data);
    auto out = ValueNode::make(std::move(t_data), {A}, "transpose");
    ValueNode* out_raw = out.get();

    out->backward_fn = [A, out_raw]() {
        accumulate_inplace(A->grad, transpose(out_raw->grad));
    };
    return out;
}

Tensor relu(const Tensor& A) {
    Tensor C(A.shape());
    size_t total = A.size();
    const float* src = A.data();
    float* dst = C.data();
    for (size_t i = 0; i < total; ++i) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
    return C;
}

ValuePtr relu(const ValuePtr& A) {
    Tensor c_data = relu(A->data);
    auto out = ValueNode::make(std::move(c_data), {A}, "relu");
    ValueNode* out_raw = out.get();

    out->backward_fn = [A, out_raw]() {
        size_t total = A->data.size();
        const float* a_data = A->data.data();
        const float* out_grad = out_raw->grad.data();
        float* a_grad = A->grad.data();
        for (size_t i = 0; i < total; ++i) {
            if (a_data[i] > 0.0f) {
                a_grad[i] += out_grad[i];
            }
        }
    };
    return out;
}

Tensor leaky_relu(const Tensor& A, float slope) {
    Tensor C(A.shape());
    size_t total = A.size();
    const float* src = A.data();
    float* dst = C.data();
    for (size_t i = 0; i < total; ++i) {
        dst[i] = src[i] > 0.0f ? src[i] : (slope * src[i]);
    }
    return C;
}

ValuePtr leaky_relu(const ValuePtr& A, float slope) {
    Tensor c_data = leaky_relu(A->data, slope);
    auto out = ValueNode::make(std::move(c_data), {A}, "leaky_relu");
    ValueNode* out_raw = out.get();

    out->backward_fn = [A, slope, out_raw]() {
        size_t total = A->data.size();
        const float* a_data = A->data.data();
        const float* out_grad = out_raw->grad.data();
        float* a_grad = A->grad.data();
        for (size_t i = 0; i < total; ++i) {
            a_grad[i] += out_grad[i] * ((a_data[i] > 0.0f) ? 1.0f : slope);
        }
    };
    return out;
}

Tensor tanh_op(const Tensor& A) {
    Tensor C(A.shape());
    size_t total = A.size();
    const float* src = A.data();
    float* dst = C.data();
    for (size_t i = 0; i < total; ++i) {
        dst[i] = std::tanh(src[i]);
    }
    return C;
}

ValuePtr tanh_op(const ValuePtr& A) {
    Tensor c_data = tanh_op(A->data);
    auto out = ValueNode::make(std::move(c_data), {A}, "tanh");
    ValueNode* out_raw = out.get();

    out->backward_fn = [A, out_raw]() {
        size_t total = A->data.size();
        const float* t_data = out_raw->data.data();
        const float* out_grad = out_raw->grad.data();
        float* a_grad = A->grad.data();
        for (size_t i = 0; i < total; ++i) {
            float t = t_data[i];
            a_grad[i] += out_grad[i] * (1.0f - t * t);
        }
    };
    return out;
}

Tensor sub(const Tensor& A, const Tensor& B) {
    if (A.shape() != B.shape()) {
        throw std::invalid_argument("TENSOR SUB: Shapes must match!");
    }
    Tensor C(A.shape());
    size_t total = A.size();
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();
    for (size_t i = 0; i < total; ++i) {
        c_ptr[i] = a_ptr[i] - b_ptr[i];
    }
    return C;
}

ValuePtr sub(const ValuePtr& A, const ValuePtr& B) {
    Tensor c_data = sub(A->data, B->data);
    auto out = ValueNode::make(std::move(c_data), {A, B}, "sub");
    ValueNode* out_raw = out.get();

    out->backward_fn = [A, B, out_raw]() {
        accumulate_inplace(A->grad, out_raw->grad);
        size_t total = out_raw->grad.size();
        const float* og = out_raw->grad.data();
        float* bg = B->grad.data();
        for (size_t i = 0; i < total; ++i) {
            bg[i] -= og[i];
        }
    };
    return out;
}

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape()) {
        throw std::invalid_argument("MSE LOSS: Shape mismatch!");
    }
    float sum_sq_err = 0.0f;
    size_t total = pred.size();
    const float* p_ptr = pred.data();
    const float* t_ptr = target.data();
    for (size_t i = 0; i < total; ++i) {
        float diff = p_ptr[i] - t_ptr[i];
        sum_sq_err += diff * diff;
    }
    return Tensor({1, 1}, sum_sq_err / static_cast<float>(total));
}

ValuePtr mse_loss(const ValuePtr& pred, const ValuePtr& target) {
    Tensor loss_data = mse_loss(pred->data, target->data);
    auto out = ValueNode::make(std::move(loss_data), {pred, target}, "mse_loss");
    ValueNode* out_raw = out.get();

    out->backward_fn = [pred, target, out_raw]() {
        size_t total = pred->data.size();
        float scale_f = 2.0f / static_cast<float>(total);
        const float* p_ptr = pred->data.data();
        const float* t_ptr = target->data.data();
        float* g_ptr = pred->grad.data();
        float out_g = out_raw->grad.at_flat(0);
        for (size_t i = 0; i < total; ++i) {
            g_ptr[i] += scale_f * (p_ptr[i] - t_ptr[i]) * out_g;
        }
    };
    return out;
}

Tensor gelu(const Tensor& A) {
    Tensor C(A.shape());
    size_t total = A.size();
    const float* src = A.data();
    float* dst = C.data();
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    for (size_t i = 0; i < total; ++i) {
        float x = src[i];
        float inner = SQRT_2_OVER_PI * (x + 0.044715f * x * x * x);
        dst[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
    return C;
}

ValuePtr gelu(const ValuePtr& A) {
    Tensor c_data = gelu(A->data);
    auto out = ValueNode::make(std::move(c_data), {A}, "gelu");
    ValueNode* out_raw = out.get();

    out->backward_fn = [A, out_raw]() {
        size_t total = A->data.size();
        const float* x_ptr = A->data.data();
        const float* out_grad = out_raw->grad.data();
        float* a_grad = A->grad.data();
        constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
        for (size_t i = 0; i < total; ++i) {
            float x = x_ptr[i];
            float x2 = x * x;
            float inner = SQRT_2_OVER_PI * (x + 0.044715f * x2 * x);
            float tanh_inner = std::tanh(inner);
            float sech2 = 1.0f - tanh_inner * tanh_inner;
            float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * 0.044715f * x2);
            float cdf = 0.5f * (1.0f + tanh_inner);
            float pdf = 0.5f * x * sech2 * d_inner;
            a_grad[i] += (cdf + pdf) * out_grad[i];
        }
    };
    return out;
}

Tensor softmax(const Tensor& A, int /*dim*/) {
    if (A.ndim() != 2) {
        throw std::invalid_argument("Softmax requires 2D Tensor [Batch, Features].");
    }
    size_t rows = A.shape()[0];
    size_t cols = A.shape()[1];
    Tensor C(A.shape());
    const float* src = A.data();
    float* dst = C.data();

    for (size_t r = 0; r < rows; ++r) {
        size_t row_offset = r * cols;
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t c = 0; c < cols; ++c) {
            if (src[row_offset + c] > max_val) max_val = src[row_offset + c];
        }
        float sum_exp = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float e = std::exp(src[row_offset + c] - max_val);
            dst[row_offset + c] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / sum_exp;
        for (size_t c = 0; c < cols; ++c) {
            dst[row_offset + c] *= inv_sum;
        }
    }
    return C;
}

ValuePtr softmax(const ValuePtr& A, int dim) {
    Tensor c_data = softmax(A->data, dim);
    auto out = ValueNode::make(std::move(c_data), {A}, "softmax");
    ValueNode* out_raw = out.get();

    out->backward_fn = [A, out_raw]() {
        size_t rows = A->data.shape()[0];
        size_t cols = A->data.shape()[1];
        const float* s_ptr = out_raw->data.data();
        const float* dL_dS = out_raw->grad.data();
        float* dL_dX = A->grad.data();
        for (size_t r = 0; r < rows; ++r) {
            size_t row_offset = r * cols;
            float sum_grad_s = 0.0f;
            for (size_t c = 0; c < cols; ++c) {
                size_t idx = row_offset + c;
                sum_grad_s += dL_dS[idx] * s_ptr[idx];
            }
            for (size_t c = 0; c < cols; ++c) {
                size_t idx = row_offset + c;
                dL_dX[idx] += s_ptr[idx] * (dL_dS[idx] - sum_grad_s);
            }
        }
    };
    return out;
}

Tensor scale(const Tensor& A, float factor) {
    Tensor C(A.shape());
    size_t total = A.size();
    const float* src = A.data();
    float* dst = C.data();
    for (size_t i = 0; i < total; ++i) {
        dst[i] = src[i] * factor;
    }
    return C;
}

ValuePtr scale(const ValuePtr& A, float factor) {
    Tensor c_data = scale(A->data, factor);
    auto out = ValueNode::make(std::move(c_data), {A}, "scale");
    ValueNode* out_raw = out.get();

    out->backward_fn = [A, factor, out_raw]() {
        size_t total = A->data.size();
        const float* out_g = out_raw->grad.data();
        float* a_g = A->grad.data();
        for (size_t i = 0; i < total; ++i) {
            a_g[i] += out_g[i] * factor;
        }
    };
    return out;
}

// im2col + col2im GEMM-based 2D Convolution (Full Forward & Input/Kernel Grad)
Tensor im2col(const Tensor& input, size_t kernel_h, size_t kernel_w, size_t stride, size_t padding) {
    size_t C = input.shape()[0];
    size_t H = input.shape()[1];
    size_t W = input.shape()[2];

    size_t out_h = (H + 2 * padding - kernel_h) / stride + 1;
    size_t out_w = (W + 2 * padding - kernel_w) / stride + 1;
    size_t spatial_out = out_h * out_w;

    Tensor col({C * kernel_h * kernel_w, spatial_out}, 0.0f);
    float* col_data = col.data();
    const float* in_data = input.data();

    for (size_t c = 0; c < C; ++c) {
        const float* in_plane = &in_data[c * H * W];
        for (size_t kh = 0; kh < kernel_h; ++kh) {
            for (size_t kw = 0; kw < kernel_w; ++kw) {
                size_t row_idx = c * (kernel_h * kernel_w) + kh * kernel_w + kw;
                float* col_row = &col_data[row_idx * spatial_out];

                for (size_t oh = 0; oh < out_h; ++oh) {
                    int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                    if (ih < 0 || ih >= static_cast<int>(H)) continue;
                    const float* in_row = &in_plane[ih * W];

                    for (size_t ow = 0; ow < out_w; ++ow) {
                        int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);
                        if (iw >= 0 && iw < static_cast<int>(W)) {
                            col_row[oh * out_w + ow] = in_row[iw];
                        }
                    }
                }
            }
        }
    }
    return col;
}

Tensor col2im(const Tensor& col, size_t C, size_t H, size_t W,
              size_t kernel_h, size_t kernel_w, size_t stride, size_t padding) {
    size_t out_h = (H + 2 * padding - kernel_h) / stride + 1;
    size_t out_w = (W + 2 * padding - kernel_w) / stride + 1;
    size_t spatial_out = out_h * out_w;

    Tensor dx({C, H, W}, 0.0f);
    float* dx_data = dx.data();
    const float* col_data = col.data();

    for (size_t c = 0; c < C; ++c) {
        float* dx_plane = &dx_data[c * H * W];
        for (size_t kh = 0; kh < kernel_h; ++kh) {
            for (size_t kw = 0; kw < kernel_w; ++kw) {
                size_t row_idx = c * (kernel_h * kernel_w) + kh * kernel_w + kw;
                const float* col_row = &col_data[row_idx * spatial_out];

                for (size_t oh = 0; oh < out_h; ++oh) {
                    int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                    if (ih < 0 || ih >= static_cast<int>(H)) continue;
                    float* dx_row = &dx_plane[ih * W];

                    for (size_t ow = 0; ow < out_w; ++ow) {
                        int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);
                        if (iw >= 0 && iw < static_cast<int>(W)) {
                            dx_row[iw] += col_row[oh * out_w + ow];
                        }
                    }
                }
            }
        }
    }
    return dx;
}

Tensor conv2d(const Tensor& input, const Tensor& kernel, size_t stride, size_t padding) {
    size_t H = input.shape()[1];
    size_t W = input.shape()[2];
    size_t out_c = kernel.shape()[0];
    size_t out_h = (H + 2 * padding - 3) / stride + 1;
    size_t out_w = (W + 2 * padding - 3) / stride + 1;

    Tensor col = im2col(input, 3, 3, stride, padding);
    Tensor out = matmul(kernel, col);
    out.reshape_inplace({out_c, out_h, out_w});
    return out;
}

ValuePtr conv2d(const ValuePtr& input, const ValuePtr& kernel, const ValuePtr& bias,
                size_t stride, size_t padding, bool compute_input_grad) {
    size_t C_in = input->data.shape()[0];
    size_t H_in = input->data.shape()[1];
    size_t W_in = input->data.shape()[2];
    size_t C_out = kernel->data.shape()[0];
    size_t H_out = (H_in + 2 * padding - 3) / stride + 1;
    size_t W_out = (W_in + 2 * padding - 3) / stride + 1;
    size_t spatial = H_out * W_out;

    auto col_cache = std::make_shared<Tensor>(im2col(input->data, 3, 3, stride, padding));
    Tensor out_data = matmul(kernel->data, *col_cache);

    if (bias) {
        float* od = out_data.data();
        const float* bd = bias->data.data();
        for (size_t o = 0; o < C_out; ++o) {
            float b_val = bd[o];
            float* plane = &od[o * spatial];
            for (size_t i = 0; i < spatial; ++i) {
                plane[i] += b_val;
            }
        }
    }
    out_data.reshape_inplace({C_out, H_out, W_out});

    std::vector<ValuePtr> parents = bias ? std::vector<ValuePtr>{input, kernel, bias}
                                         : std::vector<ValuePtr>{input, kernel};
    auto out = ValueNode::make(std::move(out_data), std::move(parents), "conv2d");
    ValueNode* out_raw = out.get();

    out->backward_fn = [input, kernel, bias, col_cache, C_in, H_in, W_in, C_out, spatial, stride, padding, compute_input_grad, out_raw]() {
        // View 3D output gradient {C_out, H_out, W_out} as 2D {C_out, H_out * W_out}
        Tensor out_grad_2d = out_raw->grad;
        out_grad_2d.reshape_inplace({C_out, spatial});

        // 1. Kernel weight gradient: dK = dOut_2d * col^T
        Tensor grad_k = matmul(out_grad_2d, transpose(*col_cache));
        accumulate_inplace(kernel->grad, grad_k);

        // 2. Bias gradient: db[o] = sum_i dOut_2d[o, i]
        if (bias) {
            const float* og = out_grad_2d.data();
            float* bg = bias->grad.data();
            for (size_t o = 0; o < C_out; ++o) {
                float sum = 0.0f;
                const float* plane = &og[o * spatial];
                for (size_t i = 0; i < spatial; ++i) sum += plane[i];
                bg[o] += sum;
            }
        }

        // 3. Input spatial gradient: dCol = K^T * dOut_2d -> dInput = col2im(dCol)
        if (compute_input_grad) {
            Tensor d_col = matmul(transpose(kernel->data), out_grad_2d);
            Tensor d_in = col2im(d_col, C_in, H_in, W_in, 3, 3, stride, padding);
            accumulate_inplace(input->grad, d_in);
        }
    };

    return out;
}

// 2x2 MaxPool2D with Argmax Switch Routing
ValuePtr max_pool2d(const ValuePtr& input, size_t pool_size) {
    size_t C = input->data.shape()[0];
    size_t H = input->data.shape()[1];
    size_t W = input->data.shape()[2];
    size_t H_out = H / pool_size;
    size_t W_out = W / pool_size;

    Tensor out_data({C, H_out, W_out});
    auto argmax = std::make_shared<std::vector<size_t>>(C * H_out * W_out, 0);

    const float* in_ptr = input->data.data();
    float* out_ptr = out_data.data();
    size_t* arg_ptr = argmax->data();

    for (size_t c = 0; c < C; ++c) {
        const float* in_plane = &in_ptr[c * H * W];
        float* out_plane = &out_ptr[c * H_out * W_out];
        size_t* arg_plane = &arg_ptr[c * H_out * W_out];

        for (size_t py = 0; py < H_out; ++py) {
            for (size_t px = 0; px < W_out; ++px) {
                size_t y0 = py * pool_size;
                size_t x0 = px * pool_size;
                size_t best_idx = y0 * W + x0;
                float best_val = in_plane[best_idx];

                for (size_t dy = 0; dy < pool_size; ++dy) {
                    for (size_t dx = 0; dx < pool_size; ++dx) {
                        size_t idx = (y0 + dy) * W + (x0 + dx);
                        if (in_plane[idx] > best_val) {
                            best_val = in_plane[idx];
                            best_idx = idx;
                        }
                    }
                }
                out_plane[py * W_out + px] = best_val;
                arg_plane[py * W_out + px] = c * H * W + best_idx;
            }
        }
    }

    auto out = ValueNode::make(std::move(out_data), {input}, "max_pool2d");
    ValueNode* out_raw = out.get();

    out->backward_fn = [input, argmax, out_raw]() {
        const float* og = out_raw->grad.data();
        float* ig = input->grad.data();
        const size_t* idx_map = argmax->data();
        const size_t n = out_raw->grad.size();
        for (size_t i = 0; i < n; ++i) {
            ig[idx_map[i]] += og[i];
        }
    };
    return out;
}

ValuePtr flatten_to_row(const ValuePtr& input) {
    Tensor flat = input->data;
    flat.reshape_inplace({1, input->data.size()});
    auto out = ValueNode::make(std::move(flat), {input}, "flatten");
    ValueNode* out_raw = out.get();

    out->backward_fn = [input, out_raw]() {
        const float* og = out_raw->grad.data();
        float* ig = input->grad.data();
        const size_t n = input->grad.size();
        for (size_t i = 0; i < n; ++i) {
            ig[i] += og[i];
        }
    };
    return out;
}

ValuePtr concat_cols(const ValuePtr& A, const ValuePtr& B) {
    size_t na = A->data.size();
    size_t nb = B->data.size();
    Tensor cat({1, na + nb});
    std::memcpy(cat.data(), A->data.data(), sizeof(float) * na);
    std::memcpy(cat.data() + na, B->data.data(), sizeof(float) * nb);

    auto out = ValueNode::make(std::move(cat), {A, B}, "concat_cols");
    ValueNode* out_raw = out.get();

    out->backward_fn = [A, B, na, nb, out_raw]() {
        const float* og = out_raw->grad.data();
        float* ag = A->grad.data();
        float* bg = B->grad.data();
        for (size_t i = 0; i < na; ++i) ag[i] += og[i];
        for (size_t i = 0; i < nb; ++i) bg[i] += og[na + i];
    };
    return out;
}

} // namespace EngineOps
