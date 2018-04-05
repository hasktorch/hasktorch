module Torch.Class.TH.Tensor.Math.Random where

import Torch.Class.Types
import qualified Torch.Types.TH as TH

class THTensorMathRandom t where
  rand_ :: t -> Generator t -> TH.LongStorage -> IO ()
  randn_ :: t -> Generator t -> TH.LongStorage -> IO ()
  randperm_ :: t -> Generator t -> Integer -> IO ()


