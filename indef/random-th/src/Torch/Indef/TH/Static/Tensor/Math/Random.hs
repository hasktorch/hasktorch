module Torch.Indef.TH.Static.Tensor.Math.Random where

import Torch.Indef.Types
import Torch.Dimensions
import Torch.Indef.TH.Dynamic.Tensor.Math.Random ()
import qualified Torch.Class.TH.Tensor.Math.Random as Dynamic
import qualified Torch.Class.TH.Tensor.Math.Random.Static as Class
import qualified Torch.Types.TH as TH

instance Class.THTensorMathRandom Tensor where
  rand_ :: Dimensions d => Tensor d -> Generator -> TH.LongStorage -> IO ()
  rand_ t = Dynamic.rand_ (asDynamic t)

  randn_ :: Dimensions d => Tensor d -> Generator -> TH.LongStorage -> IO ()
  randn_ t = Dynamic.randn_ (asDynamic t)

  randperm_ :: Dimensions d => Tensor d -> Generator -> Integer -> IO ()
  randperm_ t = Dynamic.randperm_ (asDynamic t)


