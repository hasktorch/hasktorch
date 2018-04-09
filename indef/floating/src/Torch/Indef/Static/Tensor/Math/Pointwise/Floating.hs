module Torch.Indef.Static.Tensor.Math.Pointwise.Floating where

import GHC.Int
import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.Pointwise as Dynamic
import qualified Torch.Class.Tensor.Math.Pointwise.Static as Class

import Torch.Indef.Types
import Torch.Indef.Static.Tensor.Math.Pointwise ()
import Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating ()

instance Class.TensorMathPointwiseFloating Tensor where
  _cinv a b = Dynamic._cinv (asDynamic a) (asDynamic b)
  _sigmoid a b = Dynamic._sigmoid (asDynamic a) (asDynamic b)
  _log a b = Dynamic._log (asDynamic a) (asDynamic b)
  _lgamma a b = Dynamic._lgamma (asDynamic a) (asDynamic b)
  _log1p a b = Dynamic._log1p (asDynamic a) (asDynamic b)
  _exp a b = Dynamic._exp (asDynamic a) (asDynamic b)
  _cos a b = Dynamic._cos (asDynamic a) (asDynamic b)
  _acos a b = Dynamic._acos (asDynamic a) (asDynamic b)
  _cosh a b = Dynamic._cosh (asDynamic a) (asDynamic b)
  _sin a b = Dynamic._sin (asDynamic a) (asDynamic b)
  _asin a b = Dynamic._asin (asDynamic a) (asDynamic b)
  _sinh a b = Dynamic._sinh (asDynamic a) (asDynamic b)
  _tan a b = Dynamic._tan (asDynamic a) (asDynamic b)
  _atan a b = Dynamic._atan (asDynamic a) (asDynamic b)
  _tanh a b = Dynamic._tanh (asDynamic a) (asDynamic b)
  _erf a b = Dynamic._erf (asDynamic a) (asDynamic b)
  _erfinv a b = Dynamic._erfinv (asDynamic a) (asDynamic b)
  _sqrt a b = Dynamic._sqrt (asDynamic a) (asDynamic b)
  _rsqrt a b = Dynamic._rsqrt (asDynamic a) (asDynamic b)
  _ceil a b = Dynamic._ceil (asDynamic a) (asDynamic b)
  _floor a b = Dynamic._floor (asDynamic a) (asDynamic b)
  _round a b = Dynamic._round (asDynamic a) (asDynamic b)
  _trunc a b = Dynamic._trunc (asDynamic a) (asDynamic b)
  _frac a b = Dynamic._frac (asDynamic a) (asDynamic b)
  _pow a b = Dynamic._pow (asDynamic a) (asDynamic b)
  _tpow a v b = Dynamic._tpow (asDynamic a) v (asDynamic b)
  _atan2 a b c = Dynamic._atan2 (asDynamic a) (asDynamic b) (asDynamic c)
  _lerp a b c = Dynamic._lerp (asDynamic a) (asDynamic b) (asDynamic c)


