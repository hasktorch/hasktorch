module Torch.Indef.Tensor.Static.Math.Signed where

import qualified Torch.Class.Tensor.Math as Class
import Torch.Signature.Types (dynamic, Tensor)
import Torch.Indef.Tensor.Dynamic.Math.Signed ()
import Torch.Indef.Tensor.Static ()

instance Class.TensorMathSigned (Tensor d) where
  abs_ r t = Class.abs_ (dynamic r) (dynamic t)
  neg_ r t = Class.neg_ (dynamic r) (dynamic t)



