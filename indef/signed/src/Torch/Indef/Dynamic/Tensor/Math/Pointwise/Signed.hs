module Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed where

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor ()
import qualified Torch.Sig.Tensor.Math.Pointwise.Signed as Sig
import qualified Torch.Class.Tensor.Math.Pointwise as Class

instance Class.TensorMathPointwiseSigned Dynamic where
  _abs r t = with2DynamicState r t Sig.c_abs
  _neg r t = with2DynamicState r t Sig.c_neg

