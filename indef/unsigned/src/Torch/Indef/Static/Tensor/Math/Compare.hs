module Torch.Indef.Static.Tensor.Math.Compare where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.Compare.Static as Class
import qualified Torch.Class.Tensor.Math.Compare as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Compare ()
import Torch.Indef.Static.Tensor ()

instance Class.TensorMathCompare Tensor where
  _ltValue :: ByteTensor n -> Tensor d -> HsReal -> IO ()
  _ltValue m t v = Dynamic._ltValue (byteAsDynamic m) (asDynamic t) v

  _leValue :: ByteTensor n -> Tensor d -> HsReal -> IO ()
  _leValue m t v = Dynamic._leValue (byteAsDynamic m) (asDynamic t) v

  _gtValue :: ByteTensor n -> Tensor d -> HsReal -> IO ()
  _gtValue m t v = Dynamic._gtValue (byteAsDynamic m) (asDynamic t) v

  _geValue :: ByteTensor n -> Tensor d -> HsReal -> IO ()
  _geValue m t v = Dynamic._geValue (byteAsDynamic m) (asDynamic t) v

  _neValue :: ByteTensor n -> Tensor d -> HsReal -> IO ()
  _neValue m t v = Dynamic._neValue (byteAsDynamic m) (asDynamic t) v

  _eqValue :: ByteTensor n -> Tensor d -> HsReal -> IO ()
  _eqValue m t v = Dynamic._eqValue (byteAsDynamic m) (asDynamic t) v

  _ltValueT :: Tensor d -> Tensor d -> HsReal -> IO ()
  _ltValueT r t v = Dynamic._ltValueT (asDynamic r) (asDynamic t) v

  _leValueT :: Tensor d -> Tensor d -> HsReal -> IO ()
  _leValueT r t v = Dynamic._leValueT (asDynamic r) (asDynamic t) v

  _gtValueT :: Tensor d -> Tensor d -> HsReal -> IO ()
  _gtValueT r t v = Dynamic._gtValueT (asDynamic r) (asDynamic t) v

  _geValueT :: Tensor d -> Tensor d -> HsReal -> IO ()
  _geValueT r t v = Dynamic._geValueT (asDynamic r) (asDynamic t) v

  _neValueT :: Tensor d -> Tensor d -> HsReal -> IO ()
  _neValueT r t v = Dynamic._neValueT (asDynamic r) (asDynamic t) v

  _eqValueT :: Tensor d -> Tensor d -> HsReal -> IO ()
  _eqValueT r t v = Dynamic._eqValueT (asDynamic r) (asDynamic t) v

