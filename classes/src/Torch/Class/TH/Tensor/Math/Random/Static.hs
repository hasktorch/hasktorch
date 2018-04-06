module Torch.Class.TH.Tensor.Math.Random.Static where

import Torch.Class.Types
import Torch.Dimensions
import qualified Torch.Types.TH as TH

class THTensorMathRandom t where
  rand_     :: Dimensions d => t d -> Generator (t d) -> TH.LongStorage -> IO ()
  randn_    :: Dimensions d => t d -> Generator (t d) -> TH.LongStorage -> IO ()
  randperm_ :: Dimensions d => t d -> Generator (t d) -> Integer -> IO ()


