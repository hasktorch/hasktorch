module Torch.Indef.TH.Static.Tensor.Math.Random where

import Torch.Indef.Types
import Torch.Dimensions
import Torch.Indef.TH.Dynamic.Tensor.Math.Random ()
import qualified Torch.Class.TH.Tensor.Math.Random as Dynamic
import qualified Torch.Class.TH.Tensor.Math.Random.Static as Class
import qualified Torch.Types.TH as TH

instance Class.THTensorMathRandom Tensor where
  _rand :: Dimensions d => Tensor d -> Generator -> TH.LongStorage -> IO ()
  _rand t = Dynamic._rand (asDynamic t)

  _randn :: Dimensions d => Tensor d -> Generator -> TH.LongStorage -> IO ()
  _randn t = Dynamic._randn (asDynamic t)

  _randperm :: Dimensions d => Tensor d -> Generator -> Integer -> IO ()
  _randperm t = Dynamic._randperm (asDynamic t)


