module Torch.Class.TH.Tensor.Math.Random.Static where

import Torch.Class.Types
import Torch.Dimensions
import qualified Torch.Types.TH as TH

class THTensorMathRandom t where
  _rand     :: Dimensions d => t d -> Generator (t d) -> TH.LongStorage -> IO ()
  _randn    :: Dimensions d => t d -> Generator (t d) -> TH.LongStorage -> IO ()
  _randperm :: Dimensions d => t d -> Generator (t d) -> Integer -> IO ()


