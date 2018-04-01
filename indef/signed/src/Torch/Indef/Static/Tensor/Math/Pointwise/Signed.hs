module Torch.Indef.Static.Tensor.Math.Pointwise.Signed where

import Torch.Indef.Types
import qualified Torch.Class.Tensor.Math.Pointwise as Dynamic
import qualified Torch.Class.Tensor.Math.Pointwise.Static as Class

import Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed ()

instance Class.TensorMathPointwiseSigned Tensor where
  abs_ :: Tensor d -> Tensor d -> IO ()
  abs_ r t = Dynamic.abs_ (asDynamic r) (asDynamic t)

  neg_ :: Tensor d -> Tensor d -> IO ()
  neg_ r t = Dynamic.neg_ (asDynamic r) (asDynamic t)



