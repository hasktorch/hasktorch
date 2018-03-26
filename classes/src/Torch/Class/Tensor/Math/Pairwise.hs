module Torch.Class.Tensor.Math.Pairwise where

import Torch.Class.Types

class TensorMathPairwise t where
 add_ :: t -> t -> HsReal t -> io ()
 sub_ :: t -> t -> HsReal t -> io ()
 add_scaled_ :: t -> t -> HsReal t -> HsReal t -> io ()
 sub_scaled_ :: t -> t -> HsReal t -> HsReal t -> io ()
 mul_ :: t -> t -> HsReal t -> io ()
 div_ :: t -> t -> HsReal t -> io ()
 lshift_ :: t -> t -> HsReal t -> io ()
 rshift_ :: t -> t -> HsReal t -> io ()
 fmod_ :: t -> t -> HsReal t -> io ()
 remainder_ :: t -> t -> HsReal t -> io ()
 bitand_ :: t -> t -> HsReal t -> io ()
 bitor_ :: t -> t -> HsReal t -> io ()
 bitxor_ :: t -> t -> HsReal t -> io ()
 equal_ :: t -> t -> io Bool

