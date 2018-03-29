module Torch.Core.Tensor.Static.Random where

import qualified Torch.Class.C.Tensor.Random as Class
import SigTypes (dynamic, Tensor)
import Torch.Core.Tensor.Dynamic.Random ()

instance Class.TensorRandom (Tensor d) where
  random_ = Class.random_ . dynamic
  clampedRandom_ = Class.clampedRandom_ . dynamic
  cappedRandom_ = Class.cappedRandom_ . dynamic
  geometric_ = Class.geometric_ . dynamic
  bernoulli_ = Class.bernoulli_ . dynamic
  bernoulli_FloatTensor_ = Class.bernoulli_FloatTensor_ . dynamic
  bernoulli_DoubleTensor_ = Class.bernoulli_DoubleTensor_ . dynamic


