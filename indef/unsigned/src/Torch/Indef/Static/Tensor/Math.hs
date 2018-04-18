module Torch.Indef.Static.Tensor.Math where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.Static as Class
import qualified Torch.Class.Tensor.Math as Dynamic
import qualified Torch.Types.TH as TH

import Torch.Indef.Types
import Torch.Indef.Static.Tensor ()
import Torch.Indef.Dynamic.Tensor.Math ()

instance Class.TensorMath Tensor where
  _fill r = Dynamic._fill (asDynamic r)
  _zero r = Dynamic._zero (asDynamic r)
  _zeros r = Dynamic._zeros (asDynamic r)
  _zerosLike r t = Dynamic._zerosLike (asDynamic r) (asDynamic t)
  _ones r = Dynamic._ones (asDynamic r)
  _onesLike r t = Dynamic._onesLike (asDynamic r) (asDynamic t)
  numel t = Dynamic.numel (asDynamic t)
  _reshape r t = Dynamic._reshape (asDynamic r) (asDynamic t)
  _cat r a b = Dynamic._cat (asDynamic r) (asDynamic a) (asDynamic b)
  _catArray r = Dynamic._catArray (asDynamic r)
  _nonzero r t = Dynamic._nonzero (longAsDynamic r) (asDynamic t)
  _tril r t = Dynamic._tril (asDynamic r) (asDynamic t)
  _triu r t = Dynamic._triu (asDynamic r) (asDynamic t)
  _diag r t = Dynamic._diag (asDynamic r) (asDynamic t)
  _eye r = Dynamic._eye (asDynamic r)
  trace r = Dynamic.trace (asDynamic r)
  _arange r = Dynamic._arange (asDynamic r)
  _range r = Dynamic._range (asDynamic r)


