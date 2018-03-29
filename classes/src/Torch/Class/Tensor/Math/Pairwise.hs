module Torch.Class.Tensor.Math.Pairwise where

import Torch.Class.Types

class TensorMathPairwise t where
  add_ :: t -> t -> HsReal t -> IO ()
  sub_ :: t -> t -> HsReal t -> IO ()
  add_scaled_ :: t -> t -> HsReal t -> HsReal t -> IO ()
  sub_scaled_ :: t -> t -> HsReal t -> HsReal t -> IO ()
  mul_ :: t -> t -> HsReal t -> IO ()
  div_ :: t -> t -> HsReal t -> IO ()
  lshift_ :: t -> t -> HsReal t -> IO ()
  rshift_ :: t -> t -> HsReal t -> IO ()
  fmod_ :: t -> t -> HsReal t -> IO ()
  remainder_ :: t -> t -> HsReal t -> IO ()
  bitand_ :: t -> t -> HsReal t -> IO ()
  bitor_ :: t -> t -> HsReal t -> IO ()
  bitxor_ :: t -> t -> HsReal t -> IO ()
  equal :: t -> t -> IO Bool

