#pragma once

#include "tinyml/quant/qsequential.hpp"
#include "tinyml/dataset/dataset.hpp"

namespace tinyml::quant {

void qtest(QSequential& qnet, dataset::Dataset& dataset);

}