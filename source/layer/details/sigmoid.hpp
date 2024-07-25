#pragma once
#include "layer/abstract/non_param_layer.hpp"
namespace kuiper_infer {
class SigmoidLayer : public NonParamLayer {
 public:
  SigmoidLayer() : NonParamLayer("Sigmoid") {}
  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& sigmoid_layer);
};
}  // namespace kuiper_infer
