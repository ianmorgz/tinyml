#pragma once

#include "tinyml/dataset/batch.hpp"
#include <memory>

namespace tinyml::dataset {
class Loader {
public:
    virtual ~Loader() = default;
    virtual void load(Batch& out) const = 0;
};
using loaderPtr = std::unique_ptr<Loader>;
}