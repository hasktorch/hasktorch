module Torch.Indef.Static.Tensor.Math.Compare where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.Compare.Static as Class
import qualified Torch.Class.Tensor.Math.Compare as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Compare ()

instance Class.TensorMathCompare Tensor d d where
  ltValue_ :: ByteTensor d -> Tensor d -> HsReal -> IO ()
  ltValue_ m t v = Dynamic.ltValue_ (byteAsDynamic m) (asDynamic t) v

  leValue_ :: ByteTensor d -> Tensor d -> HsReal -> IO ()
  leValue_ m t v = Dynamic.leValue_ (byteAsDynamic m) (asDynamic t) v

  gtValue_ :: ByteTensor d -> Tensor d -> HsReal -> IO ()
  gtValue_ m t v = Dynamic.gtValue_ (byteAsDynamic m) (asDynamic t) v

  geValue_ :: ByteTensor d -> Tensor d -> HsReal -> IO ()
  geValue_ m t v = Dynamic.geValue_ (byteAsDynamic m) (asDynamic t) v

  neValue_ :: ByteTensor d -> Tensor d -> HsReal -> IO ()
  neValue_ m t v = Dynamic.neValue_ (byteAsDynamic m) (asDynamic t) v

  eqValue_ :: ByteTensor d -> Tensor d -> HsReal -> IO ()
  eqValue_ m t v = Dynamic.eqValue_ (byteAsDynamic m) (asDynamic t) v

  ltValueT_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  ltValueT_ r t v = Dynamic.ltValueT_ (asDynamic r) (asDynamic t) v

  leValueT_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  leValueT_ r t v = Dynamic.leValueT_ (asDynamic r) (asDynamic t) v

  gtValueT_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  gtValueT_ r t v = Dynamic.gtValueT_ (asDynamic r) (asDynamic t) v

  geValueT_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  geValueT_ r t v = Dynamic.geValueT_ (asDynamic r) (asDynamic t) v

  neValueT_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  neValueT_ r t v = Dynamic.neValueT_ (asDynamic r) (asDynamic t) v

  eqValueT_ :: Tensor d -> Tensor d -> HsReal -> IO ()
  eqValueT_ r t v = Dynamic.eqValueT_ (asDynamic r) (asDynamic t) v

