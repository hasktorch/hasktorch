module Torch.Indef.Static.Tensor.Math.Random.TH where

import Torch.Indef.Types
import Torch.Dimensions
import qualified Torch.Indef.Dynamic.Tensor.Math.Random.TH as Dynamic
import qualified Torch.Types.TH as TH

_rand :: Dimensions d => Tensor d -> Generator -> TH.LongStorage -> IO ()
_rand t = Dynamic._rand (asDynamic t)

_randn :: Dimensions d => Tensor d -> Generator -> TH.LongStorage -> IO ()
_randn t = Dynamic._randn (asDynamic t)

_randperm :: Dimensions d => Tensor d -> Generator -> Integer -> IO ()
_randperm t = Dynamic._randperm (asDynamic t)


