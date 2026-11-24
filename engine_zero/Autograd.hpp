#pragma once
// Reverse-mode autograd graph node. Supports custom seed gradients and
// breaks parent cycles after backward() to avoid shared_ptr leaks.
#include "Tensor.hpp"
#include <memory>
#include <functional>
#include <vector>
#include <string>

class ValueNode;
using ValuePtr = std::shared_ptr<ValueNode>;

class ValueNode : public std::enable_shared_from_this<ValueNode> {
public:
    Tensor data;
    Tensor grad;

    std::vector<ValuePtr> parents;
    std::function<void()> backward_fn;
    std::string op_name;

    explicit ValueNode(Tensor data_tensor, std::vector<ValuePtr> parents = {}, std::string op = "");

    static ValuePtr make(Tensor data, std::vector<ValuePtr> parents = {}, std::string op = "");

    // Standard scalar backward (seeds grad with 1.0f)
    void backward();

    // Seeded backward (seeds grad with custom dL/dOut tensor, e.g. for multi-output policy & saliency)
    void backward(const Tensor& seed_grad, bool release_graph = true);

    void zero_grad();
};
