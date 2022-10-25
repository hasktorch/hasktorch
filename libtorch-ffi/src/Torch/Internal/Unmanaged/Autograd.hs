{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Internal.Unmanaged.Autograd where

import Foreign.Ptr
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import Foreign.C.Types (CBool)

import Torch.Internal.Type

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<vector>"
C.include "<torch/types.h>"
C.include "<torch/csrc/autograd/variable.h>"
C.include "<torch/csrc/autograd/engine.h>"
C.include "<ATen/core/functional.h>"

grad :: Ptr Tensor -> Ptr TensorList -> IO (Ptr TensorList)
grad y inputs = [C.throwBlock| std::vector<at::Tensor>* {
    torch::autograd::Variable y = *$(at::Tensor* y);
    const auto & inputs = *$(std::vector<at::Tensor>* inputs);

    torch::autograd::edge_list roots { torch::autograd::impl::gradient_edge(y) };
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
        grad_fn = torch::autograd::impl::try_get_grad_accumulator(input);
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
                                  /*accumulate_grad=*/false, // https://github.com/pytorch/pytorch/pull/46855
                                                             // https://github.com/pytorch/pytorch/issues/46373
                                  output_edges);

    return new std::vector<at::Tensor>(at::fmap<at::Tensor>(outputs));
  }|]

makeIndependent :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
makeIndependent tensor requires_grad = [C.throwBlock| at::Tensor* {
    return new at::Tensor($(at::Tensor* tensor)->detach().set_requires_grad($(bool requires_grad)));
  }|]

dropVariable :: Ptr Tensor -> IO (Ptr Tensor)
dropVariable t = [C.throwBlock| at::Tensor* {
    auto ret = $(at::Tensor* t)->detach();
    ret.unsafeGetTensorImpl()->set_autograd_meta(nullptr);
    return new at::Tensor(ret);
  }|]
