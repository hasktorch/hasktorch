{-# LANGUAGE CPP #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -Wno-deprecations #-}

module Main where

import Torch
import qualified Torch.Internal.Type as I
import qualified Torch.Internal.Cast as I

import qualified Language.C.Inline.Context as C
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Types as C

import Foreign.C.String
import Foreign.C.Types
import Foreign

import System.IO.Unsafe

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = I.typeTable }

C.include "<torch/all.h>"

[C.emitBlock|
struct MySquareFunction : public torch::autograd::Function<MySquareFunction> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor input) {
    ctx->save_for_backward({input});
    return input * input;
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                               torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto grad_output = grad_outputs[0];

    auto grad_input = grad_output * 2 * input;
    return {grad_input};
  }
};

// Convenience wrapper
torch::Tensor mysquare(torch::Tensor x) {
  return MySquareFunction::apply(x);
}
|]

mysquarePtr
  :: Ptr I.Tensor
  -> IO (Ptr I.Tensor)
mysquarePtr _in =
  [C.throwBlock| at::Tensor* { return new at::Tensor(mysquare(*$(at::Tensor* _in)));
  }|]

mysquareFPtr
  :: ForeignPtr I.Tensor
  -> IO (ForeignPtr I.Tensor)
mysquareFPtr = I._cast1 mysquarePtr

mysquare
  :: Tensor
  -> Tensor
mysquare _in = unsafePerformIO $ I.cast1 mysquareFPtr _in

diff :: (Tensor -> Tensor) -> Tensor -> Tensor
diff func x = unsafePerformIO $ do
  tx <- makeIndependent x
  return $ head $ grad (Torch.sumAll $ func (toDependent tx)) [tx]
  
main = do
  let i = asTensor (3::Float)
  print i
  print $ mysquare i
  print $ diff mysquare i
  let i = asTensor [(3::Float),(2::Float)]
  print i
  print $ mysquare i
  print $ diff mysquare i
  
