#pragma once
#include "Autograd.hpp"
#include <vector>

class Module {
public:
    virtual ~Module() = default;

    virtual std::vector<ValuePtr> parameters() const = 0;

    void zero_grad() {
        for (auto& param : parameters()) {
            param->grad.fill(0.0f);
        }
    }
};
