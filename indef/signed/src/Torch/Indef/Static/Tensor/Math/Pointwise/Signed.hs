module Torch.Indef.Static.Tensor.Math.Pointwise.Signed where

import Torch.Indef.Types
import qualified Torch.Class.Tensor.Math.Pointwise as Dynamic
import qualified Torch.Class.Tensor.Math.Pointwise.Static as Class

import Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed ()
import Torch.Indef.Static.Tensor ()

instance Class.TensorMathPointwiseSigned Tensor where
  _abs :: Tensor d -> Tensor d -> IO ()
  _abs r t = Dynamic._abs (asDynamic r) (asDynamic t)

  _neg :: Tensor d -> Tensor d -> IO ()
  _neg r t = Dynamic._neg (asDynamic r) (asDynamic t)



