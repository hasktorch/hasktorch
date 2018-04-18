module Torch.Indef.Static.Tensor.Math.Pairwise where

import qualified Torch.Class.Tensor.Math.Pairwise        as Dynamic
import qualified Torch.Class.Tensor.Math.Pairwise.Static as Class

import Torch.Indef.Types
import Torch.Indef.Static.Tensor ()
import Torch.Indef.Dynamic.Tensor.Math.Pairwise ()

instance Class.TensorMathPairwise Tensor where
  _add r t v = Dynamic._add (asDynamic r) (asDynamic t) v
  _sub r t v = Dynamic._sub (asDynamic r) (asDynamic t) v
  _add_scaled r t v0 v1 = Dynamic._add_scaled (asDynamic r) (asDynamic t) v0 v1
  _sub_scaled r t v0 v1 = Dynamic._sub_scaled (asDynamic r) (asDynamic t) v0 v1
  _mul r t v = Dynamic._mul (asDynamic r) (asDynamic t) v
  _div r t v = Dynamic._div (asDynamic r) (asDynamic t) v
  _lshift r t v = Dynamic._lshift (asDynamic r) (asDynamic t) v
  _rshift r t v = Dynamic._rshift (asDynamic r) (asDynamic t) v
  _fmod r t v = Dynamic._fmod (asDynamic r) (asDynamic t) v
  _remainder r t v = Dynamic._remainder (asDynamic r) (asDynamic t) v
  _bitand r t v = Dynamic._bitand (asDynamic r) (asDynamic t) v
  _bitor r t v = Dynamic._bitor (asDynamic r) (asDynamic t) v
  _bitxor r t v = Dynamic._bitxor (asDynamic r) (asDynamic t) v
  equal r t = Dynamic.equal (asDynamic r) (asDynamic t)



