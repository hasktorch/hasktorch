module Torch.Indef.Static.Tensor.Math.Pointwise where

import qualified Torch.Class.Tensor.Math.Pointwise        as Dynamic
import qualified Torch.Class.Tensor.Math.Pointwise.Static as Class

import Torch.Dimensions
import Torch.Indef.Types
import Torch.Indef.Static.Tensor.Math ()
import Torch.Indef.Dynamic.Tensor.Math.Pointwise ()

instance Class.TensorMathPointwise Tensor where
  _sign r t = Dynamic._sign (asDynamic r) (asDynamic t)
  _cross ret a b d = Dynamic._cross (asDynamic ret) (asDynamic a) (asDynamic b) d
  _clamp ret a mn mx = Dynamic._clamp (asDynamic ret) (asDynamic a) mn mx
  _cadd ret a v b = Dynamic._cadd (asDynamic ret) (asDynamic a) v (asDynamic b)
  _csub ret a v b = Dynamic._csub (asDynamic ret) (asDynamic a) v (asDynamic b)
  _cmul ret a b = Dynamic._cmul (asDynamic ret) (asDynamic a) (asDynamic b)
  _cpow ret a b = Dynamic._cpow (asDynamic ret) (asDynamic a) (asDynamic b)
  _cdiv ret a b = Dynamic._cdiv (asDynamic ret) (asDynamic a) (asDynamic b)
  _clshift ret a b = Dynamic._clshift (asDynamic ret) (asDynamic a) (asDynamic b)
  _crshift ret a b = Dynamic._crshift (asDynamic ret) (asDynamic a) (asDynamic b)
  _cfmod ret a b = Dynamic._cfmod (asDynamic ret) (asDynamic a) (asDynamic b)
  _cremainder ret a b = Dynamic._cremainder (asDynamic ret) (asDynamic a) (asDynamic b)
  _cmax ret a b = Dynamic._cmax (asDynamic ret) (asDynamic a) (asDynamic b)
  _cmin ret a b = Dynamic._cmin (asDynamic ret) (asDynamic a) (asDynamic b)
  _cmaxValue ret a v = Dynamic._cmaxValue (asDynamic ret) (asDynamic a) v
  _cminValue ret a v = Dynamic._cminValue (asDynamic ret) (asDynamic a) v
  _cbitand ret a b = Dynamic._cbitand (asDynamic ret) (asDynamic a) (asDynamic b)
  _cbitor ret a b = Dynamic._cbitor (asDynamic ret) (asDynamic a) (asDynamic b)
  _cbitxor ret a b = Dynamic._cbitxor (asDynamic ret) (asDynamic a) (asDynamic b)
  _addcmul ret a v b c = Dynamic._addcmul (asDynamic ret) (asDynamic a) v (asDynamic b) (asDynamic c)
  _addcdiv ret a v b c = Dynamic._addcdiv (asDynamic ret) (asDynamic a) v (asDynamic b) (asDynamic c)
