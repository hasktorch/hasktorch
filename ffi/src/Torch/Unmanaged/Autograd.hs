{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Unmanaged.Autograd where

import Foreign.Ptr
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C

import ATen.Type

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<vector>"
C.include "<torch/torch.h>"
C.include "<torch/csrc/autograd/variable.h>"
C.include "<torch/csrc/autograd/engine.h>"
C.include "<ATen/core/functional.h>"

grad :: Ptr Tensor -> Ptr TensorList -> IO (Ptr TensorList)
grad y inputs = [C.throwBlock| std::vector<at::Tensor>* {
    torch::autograd::Variable y = *$(at::Tensor* y);
    const auto & inputs = *$(std::vector<at::Tensor>* inputs);

    torch::autograd::edge_list roots { y.gradient_edge() };
    if (!roots[0].function) {
      throw std::runtime_error("Differentiated tensor not require grad");
    }

    if (y.numel() != 1) {
      throw std::runtime_error("Differentiated tensor has more than a single element");
    }
    torch::autograd::variable_list grads { torch::ones_like(y) };

    torch::autograd::edge_list output_edges;
    output_edges.reserve(inputs.size());
    for (torch::autograd::Variable input : inputs) {
      const auto output_nr = input.output_nr();
      auto grad_fn = input.grad_fn();
      if (!grad_fn) {
        grad_fn = input.try_get_grad_accumulator();
      }
      if (!input.requires_grad()) {
        throw std::runtime_error("One of the differentiated Tensors does not require grad");
      }
      if (!grad_fn) {
        output_edges.emplace_back();
      } else {
        output_edges.emplace_back(grad_fn, output_nr);
      }
    }

    auto & engine = torch::autograd::Engine::get_default_engine();
    auto outputs = engine.execute(roots, grads,
                                  /*keep_graph=*/true,
                                  /*create_graph=*/false,
                                  output_edges);

    return new std::vector<at::Tensor>(at::fmap<at::Tensor>(outputs));
  }|]

makeIndependent :: Ptr Tensor -> IO (Ptr Tensor)
makeIndependent t = [C.throwBlock| at::Tensor* {
    return new at::Tensor($(at::Tensor* t)->detach().set_requires_grad(true));
  }|]
