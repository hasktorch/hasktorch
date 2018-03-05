{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
{- LANGUAGE KindSignatures #-}
module Torch.Core.Tensor.Static.Math.Signed
  ( Torch.Core.Tensor.Static.Math.Signed.abs
  , Torch.Core.Tensor.Static.Math.Signed.neg
  , Class.TensorMathSigned(..)
  ) where

import qualified Torch.Class.IsTensor as Class
import qualified Torch.Class.Tensor.Math as Class (abs_, neg_, TensorMathSigned)
import Torch.Dimensions
import Torch.Core.Tensor.Static (inplace)

abs :: forall t d . (Dimensions d, Class.IsTensor (t d), Class.TensorMathSigned (t d)) => t d -> IO (t d)
abs t = inplace (`Class.abs_` t) (dim :: Dim d)

neg :: forall t d . (Dimensions d, Class.IsTensor (t d), Class.TensorMathSigned (t d)) => t d -> IO (t d)
neg t = inplace (`Class.neg_` t) (dim :: Dim d)


