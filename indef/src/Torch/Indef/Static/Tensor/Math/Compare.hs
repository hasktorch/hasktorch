module Torch.Indef.Static.Tensor.Math.Compare where

import Torch.Dimensions

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.Math.Compare as Dynamic

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

-- ltValue, leValue, gtValue, geValue, neValue, eqValue
--   :: (Dimensions d)
--   => Tensor d -> HsReal -> IO (MaskTensor Tensor d)
-- ltValue a b = withEmpty $ \r -> _ltValue r a b
-- leValue a b = withEmpty $ \r -> _leValue r a b
-- gtValue a b = withEmpty $ \r -> _gtValue r a b
-- geValue a b = withEmpty $ \r -> _geValue r a b
-- neValue a b = withEmpty $ \r -> _neValue r a b
-- eqValue a b = withEmpty $ \r -> _eqValue r a b

ltValueT, leValueT, gtValueT, geValueT, neValueT, eqValueT
  :: (Dimensions d)
  => Tensor d -> HsReal -> IO (Tensor d)
ltValueT a b = withEmpty $ \r -> _ltValueT r a b
leValueT a b = withEmpty $ \r -> _leValueT r a b
gtValueT a b = withEmpty $ \r -> _gtValueT r a b
geValueT a b = withEmpty $ \r -> _geValueT r a b
neValueT a b = withEmpty $ \r -> _neValueT r a b
eqValueT a b = withEmpty $ \r -> _eqValueT r a b

ltValueT_, leValueT_, gtValueT_, geValueT_, neValueT_, eqValueT_
  :: (Dimensions d)
  => Tensor d -> HsReal -> IO (Tensor d)
ltValueT_ a b = _ltValueT a a b >> pure a
leValueT_ a b = _leValueT a a b >> pure a
gtValueT_ a b = _gtValueT a a b >> pure a
geValueT_ a b = _geValueT a a b >> pure a
neValueT_ a b = _neValueT a a b >> pure a
eqValueT_ a b = _eqValueT a a b >> pure a

