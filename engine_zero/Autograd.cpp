#include "Autograd.hpp"
#include <algorithm>
#include <unordered_set>
#include <queue>

ValueNode::ValueNode(Tensor data_tensor, std::vector<ValuePtr> parents, std::string op)
    : data(std::move(data_tensor)),
      grad(data.shape(), 0.0f),
      parents(std::move(parents)),
      op_name(std::move(op)) {}

ValuePtr ValueNode::make(Tensor data, std::vector<ValuePtr> parents, std::string op) {
    return std::make_shared<ValueNode>(std::move(data), std::move(parents), std::move(op));
}

void ValueNode::zero_grad() {
    std::unordered_set<ValueNode*> visited;
    std::queue<ValuePtr> queue;
    queue.push(shared_from_this());

    while (!queue.empty()) {
        auto curr = queue.front();
        queue.pop();
        if (visited.find(curr.get()) != visited.end()) continue;
        visited.insert(curr.get());

        curr->grad.fill(0.0f);
        for (const auto& parent : curr->parents) {
            queue.push(parent);
        }
    }
}

void ValueNode::backward() {
    Tensor ones(this->data.shape(), 1.0f);
    backward(ones, true);
}

void ValueNode::backward(const Tensor& seed_grad, bool release_graph) {
    std::vector<ValuePtr> topo;
    std::unordered_set<ValueNode*> visited;

    std::function<void(const ValuePtr&)> build_topo = [&](const ValuePtr& node) {
        if (visited.find(node.get()) == visited.end()) {
            visited.insert(node.get());
            for (const auto& parent : node->parents) {
                build_topo(parent);
            }
            topo.push_back(node);
        }
    };

    build_topo(shared_from_this());
    this->grad = seed_grad;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn();
            if (release_graph) {
                (*it)->backward_fn = nullptr;
                (*it)->parents.clear();
            }
        }
    }
}
