module Torch.Indef.Static.Tensor.Math.Pointwise.Floating where

import GHC.Int
import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.Pointwise as Dynamic
import qualified Torch.Class.Tensor.Math.Pointwise.Static as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating ()

instance Class.TensorMathPointwiseFloating Tensor where
  cinv_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  cinv_ a b = Dynamic.cinv_ (asDynamic a) (asDynamic b)

  sigmoid_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  sigmoid_ a b = Dynamic.sigmoid_ (asDynamic a) (asDynamic b)

  log_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  log_ a b = Dynamic.log_ (asDynamic a) (asDynamic b)

  lgamma_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  lgamma_ a b = Dynamic.lgamma_ (asDynamic a) (asDynamic b)

  log1p_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  log1p_ a b = Dynamic.log1p_ (asDynamic a) (asDynamic b)

  exp_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  exp_ a b = Dynamic.exp_ (asDynamic a) (asDynamic b)

  cos_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  cos_ a b = Dynamic.cos_ (asDynamic a) (asDynamic b)

  acos_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  acos_ a b = Dynamic.acos_ (asDynamic a) (asDynamic b)

  cosh_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  cosh_ a b = Dynamic.cosh_ (asDynamic a) (asDynamic b)

  sin_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  sin_ a b = Dynamic.sin_ (asDynamic a) (asDynamic b)

  asin_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  asin_ a b = Dynamic.asin_ (asDynamic a) (asDynamic b)

  sinh_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  sinh_ a b = Dynamic.sinh_ (asDynamic a) (asDynamic b)

  tan_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  tan_ a b = Dynamic.tan_ (asDynamic a) (asDynamic b)

  atan_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  atan_ a b = Dynamic.atan_ (asDynamic a) (asDynamic b)

  tanh_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  tanh_ a b = Dynamic.tanh_ (asDynamic a) (asDynamic b)

  erf_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  erf_ a b = Dynamic.erf_ (asDynamic a) (asDynamic b)

  erfinv_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  erfinv_ a b = Dynamic.erfinv_ (asDynamic a) (asDynamic b)

  sqrt_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  sqrt_ a b = Dynamic.sqrt_ (asDynamic a) (asDynamic b)

  rsqrt_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  rsqrt_ a b = Dynamic.rsqrt_ (asDynamic a) (asDynamic b)

  ceil_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  ceil_ a b = Dynamic.ceil_ (asDynamic a) (asDynamic b)

  floor_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  floor_ a b = Dynamic.floor_ (asDynamic a) (asDynamic b)

  round_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  round_ a b = Dynamic.round_ (asDynamic a) (asDynamic b)

  trunc_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  trunc_ a b = Dynamic.trunc_ (asDynamic a) (asDynamic b)

  frac_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  frac_ a b = Dynamic.frac_ (asDynamic a) (asDynamic b)

  pow_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> HsReal -> IO ()
  pow_ a b = Dynamic.pow_ (asDynamic a) (asDynamic b)

  tpow_ :: (Dimensions d, Dimensions d') => Tensor d -> HsReal -> Tensor d' -> IO ()
  tpow_ a v b = Dynamic.tpow_ (asDynamic a) v (asDynamic b)

  atan2_ :: (Dimensions d, Dimensions d', Dimensions d'') => Tensor d -> Tensor d' -> Tensor d'' -> IO ()
  atan2_ a b c = Dynamic.atan2_ (asDynamic a) (asDynamic b) (asDynamic c)

  lerp_ :: (Dimensions d, Dimensions d', Dimensions d'') => Tensor d -> Tensor d' -> Tensor d'' -> HsReal -> IO ()
  lerp_ a b c = Dynamic.lerp_ (asDynamic a) (asDynamic b) (asDynamic c)


