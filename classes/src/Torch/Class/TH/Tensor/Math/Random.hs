module Torch.Class.TH.Tensor.Math.Random where

import Torch.Class.Types
import qualified Torch.Types.TH as TH

class THTensorMathRandom t where
  _rand :: t -> Generator t -> TH.LongStorage -> IO ()
  _randn :: t -> Generator t -> TH.LongStorage -> IO ()
  _randperm :: t -> Generator t -> Integer -> IO ()


