module Torch.Indef.Static.Tensor.Math.Pairwise where

import qualified Torch.Class.Tensor.Math.Pairwise        as Dynamic
import qualified Torch.Class.Tensor.Math.Pairwise.Static as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Pairwise ()

instance Class.TensorMathPairwise (Tensor d) where
  add_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  add_ r t v = Dynamic.add_ (asDynamic r) (asDynamic t) v

  sub_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  sub_ r t v = Dynamic.sub_ (asDynamic r) (asDynamic t) v

  add_scaled_ :: Tensor d -> Tensor d -> HsReal -> HsReal -> IO ()
  add_scaled_ r t v0 v1 = Dynamic.add_scaled_ (asDynamic r) (asDynamic t) v0 v1

  sub_scaled_ :: Tensor d -> Tensor d -> HsReal -> HsReal -> IO ()
  sub_scaled_ r t v0 v1 = Dynamic.sub_scaled_ (asDynamic r) (asDynamic t) v0 v1

  mul_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  mul_ r t v = Dynamic.mul_ (asDynamic r) (asDynamic t) v

  div_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  div_ r t v = Dynamic.div_ (asDynamic r) (asDynamic t) v

  lshift_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  lshift_ r t v = Dynamic.lshift_ (asDynamic r) (asDynamic t) v

  rshift_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  rshift_ r t v = Dynamic.rshift_ (asDynamic r) (asDynamic t) v

  fmod_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  fmod_ r t v = Dynamic.fmod_ (asDynamic r) (asDynamic t) v

  remainder_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  remainder_ r t v = Dynamic.remainder_ (asDynamic r) (asDynamic t) v

  bitand_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  bitand_ r t v = Dynamic.bitand_ (asDynamic r) (asDynamic t) v

  bitor_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  bitor_ r t v = Dynamic.bitor_ (asDynamic r) (asDynamic t) v

  bitxor_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  bitxor_ r t v = Dynamic.bitxor_ (asDynamic r) (asDynamic t) v

  equal :: Tensor d -> Tensor d -> IO Bool
  equal r t = Dynamic.equal (asDynamic r) (asDynamic t)



