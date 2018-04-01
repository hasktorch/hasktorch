module Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed where

import Torch.Indef.Types
import qualified Torch.Sig.Tensor.Math.Pointwise.Signed as Sig
import qualified Torch.Class.Tensor.Math.Pointwise as Class

instance Class.TensorMathPointwiseSigned Dynamic where
  abs_ r t = with2DynamicState r t Sig.c_abs
  neg_ r t = with2DynamicState r t Sig.c_neg

