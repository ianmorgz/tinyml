#pragma once
#include "tinyml/model/layer.hpp"

namespace tinyml::train {

void SGD_step(std::span<const model::ParamRef> params, float lr);


}