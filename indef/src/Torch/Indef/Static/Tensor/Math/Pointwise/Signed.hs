module Torch.Indef.Static.Tensor.Math.Pointwise.Signed where

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed as Dynamic

_abs :: Tensor d -> Tensor d -> IO ()
_abs r t = Dynamic._abs (asDynamic r) (asDynamic t)

_neg :: Tensor d -> Tensor d -> IO ()
_neg r t = Dynamic._neg (asDynamic r) (asDynamic t)

neg, abs :: (Dimensions d) => Tensor d -> IO (Tensor d)
neg t = withEmpty (`_neg` t)
abs t = withEmpty (`_abs` t)


