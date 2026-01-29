#include "tinyml/quant/qsequential.hpp"
#include "tinyml/dataset/dataset.hpp"
#include "tinyml/model/dense.hpp"
#include "tinyml/model/layer.hpp"
#include "tinyml/quant/qdense.hpp"
#include "tinyml/quant/qrelu.hpp"
#include "tinyml/quant/qtanh.hpp"
#include "../../include/tinyml/core/context.hpp"
#include "tinyml/model/sequential.hpp"

namespace tinyml::quant {
QSequential::QSequential(const model::Sequential& net, const dataset::Dataset& ds, const std::size_t callibration_size) {
    // step 1 create the layers and quantize dense weights
    make_layers(net);

    // step 2 callibrate pass on the float network
    callibrate(net, ds, callibration_size);
    // step 3 finalize model qparams + build LUTS

    // step 4 quantize dense biases
}
void QSequential::make_layers(const model::Sequential &net) {
    layers_.reserve(net.num_layers());
    for (size_t i = 0; i < net.num_layers(); ++i) {
        const auto& L = net.layer(i);
        switch (L.type()) {
            case model::LayerType::Dense: {
                const auto& d = static_cast<const model::Dense&>(L);
                layers_.push_back(std::make_unique<QDense>(d));
                break;
            }

            case model::LayerType::Tanh: {
                layers_.push_back(std::make_unique<QTanh>());
                break;
            }

            case model::LayerType::ReLu: {
                layers_.push_back(std::make_unique<QRelu>());
                break;
            }

            default:
                TINYML_EXCEPTION("Quantized Netowork make layers undefined layer type used");
        }
    }
}

void QSequential::callibrate(model::Sequential &net, dataset::Dataset &ds, const std::size_t callibration_size) {
    const std::size_t max_batch_size = net.max_batch();
    dataset::BatchView callibration_batch;
    ds.shuffle_training(567);
    const std::size_t num_layers = net.num_layers();

    tensor::Tensor<float> act_a({max_batch_size, net.max_features()});
    tensor::Tensor<float> act_b({max_batch_size, net.max_features()});

    for (std::size_t i = 0; i < callibration_size; i += max_batch_size) {
        const std::size_t remaining = callibration_size - i;
        const std::size_t batch_size = (remaining < max_batch_size) ? remaining : max_batch_size;
        ds.next_training_batch(batch_size, callibration_batch);

        tensor::TensorView<const float> in_view = callibration_batch.input;
        bool write_to_a = true;

        for (std::size_t j = 0; j < num_layers; ++j) {
            tensor::TensorView out_view(write_to_a ? act_a.data() : act_b.data(), {batch_size, net.max_features(), net.layer_features(j+1)});

            net.layer(j).forward(in_view, out_view);
            layers_[j]->observe_fp32_output(out_view.as_const());

            in_view = out_view.as_const();
            write_to_a = !write_to_a;
        }
    }
}


}
