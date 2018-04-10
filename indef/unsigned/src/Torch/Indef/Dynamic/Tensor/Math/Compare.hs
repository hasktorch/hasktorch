module Torch.Indef.Dynamic.Tensor.Math.Compare where

import Torch.Class.Tensor.Math.Compare
import qualified Torch.Sig.Tensor.Math.Compare as Sig

import Torch.Indef.Types

instance TensorMathCompare Dynamic where
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


