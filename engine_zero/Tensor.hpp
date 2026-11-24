#pragma once
// 64-byte aligned multi-dimensional tensor container.
#include <vector>
#include <initializer_list>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <new>
#include <stdexcept>

constexpr size_t ALIGNMENT_BYTES = 64;

class Tensor {
    float* m_data{nullptr};
    size_t m_size{0};
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;

    void allocate_memory();
    void compute_strides();

public:
    Tensor() = default;
    explicit Tensor(std::vector<size_t> shape);
    Tensor(std::vector<size_t> shape, float fill_value);
    Tensor(std::initializer_list<size_t> shape);

    ~Tensor();
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    [[nodiscard]] inline float* data() noexcept { return m_data; }
    [[nodiscard]] inline const float* data() const noexcept { return m_data; }
    [[nodiscard]] inline size_t size() const noexcept { return m_size; }
    [[nodiscard]] inline const std::vector<size_t>& shape() const noexcept { return m_shape; }
    [[nodiscard]] inline const std::vector<size_t>& strides() const noexcept { return m_strides; }
    [[nodiscard]] inline size_t ndim() const noexcept { return m_shape.size(); }

    inline float& at_flat(size_t index) noexcept { return m_data[index]; }
    [[nodiscard]] inline const float& at_flat(size_t index) const noexcept { return m_data[index]; }

    void fill(float v) noexcept {
        if (!m_data) return;
        if (v == 0.0f) {
            std::memset(m_data, 0, m_size * sizeof(float));
        } else {
            for (size_t i = 0; i < m_size; ++i) m_data[i] = v;
        }
    }

    void reshape_inplace(std::vector<size_t> new_shape) {
        size_t new_total = 1;
        for (size_t d : new_shape) new_total *= d;
        if (new_total != m_size) {
            throw std::invalid_argument("Tensor::reshape_inplace size mismatch");
        }
        m_shape = std::move(new_shape);
        compute_strides();
    }

    template<typename... Args>
    [[nodiscard]] inline float& operator()(Args... indices) noexcept {
        size_t flat_idx = 0;
        size_t i = 0;
        ((flat_idx += static_cast<size_t>(indices) * m_strides[i++]), ...);
        return m_data[flat_idx];
    }

    template<typename... Args>
    [[nodiscard]] inline const float& operator()(Args... indices) const noexcept {
        size_t flat_idx = 0;
        size_t i = 0;
        ((flat_idx += static_cast<size_t>(indices) * m_strides[i++]), ...);
        return m_data[flat_idx];
    }
};
