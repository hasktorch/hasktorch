module Torch.Core.Tensor.Static.Math.Signed where

import qualified Torch.Class.C.Tensor.Math as Class
import SigTypes (dynamic, Tensor)
import Torch.Core.Tensor.Dynamic.Math.Signed ()
import Torch.Core.Tensor.Static ()

instance Class.TensorMathSigned (Tensor d) where
  abs_ r t = Class.abs_ (dynamic r) (dynamic t)
  neg_ r t = Class.neg_ (dynamic r) (dynamic t)



