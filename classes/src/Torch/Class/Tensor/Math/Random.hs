module Torch.Class.Tensor.Math.Random where

import Foreign
import Torch.Class.Types

class TensorMathRandom t where
  rand_  :: t -> Generator t -> IndexStorage t -> io ()
  randn_ :: t -> Generator t -> IndexStorage t -> io ()



