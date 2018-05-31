module Torch.Indef.Dynamic.Tensor.Math.Compare where

import qualified Torch.Sig.Tensor.Math.Compare as Sig

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor

_ltValue :: MaskDynamic -> Dynamic -> HsReal -> IO ()
_ltValue bt t0 v = withDynamicState t0 $ \s' t0' -> withMask bt $ \bt' ->  Sig.c_ltValue s' bt' t0' (hs2cReal v)

_leValue :: MaskDynamic -> Dynamic -> HsReal -> IO ()
_leValue bt t0 v = withDynamicState t0 $ \s' t0' -> withMask bt $ \bt' ->  Sig.c_leValue s' bt' t0' (hs2cReal v)

_gtValue :: MaskDynamic -> Dynamic -> HsReal -> IO ()
_gtValue bt t0 v = withDynamicState t0 $ \s' t0' -> withMask bt $ \bt' ->  Sig.c_gtValue s' bt' t0' (hs2cReal v)

_geValue :: MaskDynamic -> Dynamic -> HsReal -> IO ()
_geValue bt t0 v = withDynamicState t0 $ \s' t0' -> withMask bt $ \bt' ->  Sig.c_geValue s' bt' t0' (hs2cReal v)

_neValue :: MaskDynamic -> Dynamic -> HsReal -> IO ()
_neValue bt t0 v = withDynamicState t0 $ \s' t0' -> withMask bt $ \bt' ->  Sig.c_neValue s' bt' t0' (hs2cReal v)

_eqValue :: MaskDynamic -> Dynamic -> HsReal -> IO ()
_eqValue bt t0 v = withDynamicState t0 $ \s' t0' -> withMask bt $ \bt' ->  Sig.c_eqValue s' bt' t0' (hs2cReal v)

_ltValueT :: Dynamic -> Dynamic -> HsReal -> IO ()
_ltValueT t0 t1 v = with2DynamicState t0 t1 (shuffle3 Sig.c_ltValueT (hs2cReal v))

_leValueT :: Dynamic -> Dynamic -> HsReal -> IO ()
_leValueT t0 t1 v = with2DynamicState t0 t1 (shuffle3 Sig.c_leValueT (hs2cReal v))

_gtValueT :: Dynamic -> Dynamic -> HsReal -> IO ()
_gtValueT t0 t1 v = with2DynamicState t0 t1 (shuffle3 Sig.c_gtValueT (hs2cReal v))

_geValueT :: Dynamic -> Dynamic -> HsReal -> IO ()
_geValueT t0 t1 v = with2DynamicState t0 t1 (shuffle3 Sig.c_geValueT (hs2cReal v))

_neValueT :: Dynamic -> Dynamic -> HsReal -> IO ()
_neValueT t0 t1 v = with2DynamicState t0 t1 (shuffle3 Sig.c_neValueT (hs2cReal v))

_eqValueT :: Dynamic -> Dynamic -> HsReal -> IO ()
_eqValueT t0 t1 v = with2DynamicState t0 t1 (shuffle3 Sig.c_eqValueT (hs2cReal v))

-- ltValue, leValue, gtValue, geValue, neValue, eqValue
--   :: Dynamic -> HsReal -> IO MaskDynamic
-- ltValue a b = getDims a >>= new' >>= \r -> _ltValue r a b >> pure r
-- leValue a b = getDims a >>= new' >>= \r -> _leValue r a b >> pure r
-- gtValue a b = getDims a >>= new' >>= \r -> _gtValue r a b >> pure r
-- geValue a b = getDims a >>= new' >>= \r -> _geValue r a b >> pure r
-- neValue a b = getDims a >>= new' >>= \r -> _neValue r a b >> pure r
-- eqValue a b = getDims a >>= new' >>= \r -> _eqValue r a b >> pure r

ltValueT, leValueT, gtValueT, geValueT, neValueT, eqValueT
  :: Dynamic -> HsReal -> IO Dynamic
ltValueT  a b = withEmpty a $ \r -> _ltValueT r a b
leValueT  a b = withEmpty a $ \r -> _leValueT r a b
gtValueT  a b = withEmpty a $ \r -> _gtValueT r a b
geValueT  a b = withEmpty a $ \r -> _geValueT r a b
neValueT  a b = withEmpty a $ \r -> _neValueT r a b
eqValueT  a b = withEmpty a $ \r -> _eqValueT r a b

ltValueT_, leValueT_, gtValueT_, geValueT_, neValueT_, eqValueT_
  :: Dynamic -> HsReal -> IO Dynamic
ltValueT_ a b = _ltValueT a a b >> pure a
leValueT_ a b = _leValueT a a b >> pure a
gtValueT_ a b = _gtValueT a a b >> pure a
geValueT_ a b = _geValueT a a b >> pure a
neValueT_ a b = _neValueT a a b >> pure a
eqValueT_ a b = _eqValueT a a b >> pure a

