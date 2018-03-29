module Torch.Indef.Dynamic.Tensor.Math.Compare where

import Torch.Class.Tensor.Math.Compare
import qualified Torch.Sig.Tensor.Math.Compare as Sig

import Torch.Indef.Types

instance TensorMathCompare Dynamic where
  ltValue_ :: MaskDynamic -> Dynamic -> HsReal -> IO ()
  ltValue_ bt t0 v = withDynamicState t0 $ \s' t0' -> withMask bt $ \bt' ->  Sig.c_ltValue s' bt' t0' (hs2cReal v)

  leValue_ :: MaskDynamic -> Dynamic -> HsReal -> IO ()
  leValue_ bt t0 v = withDynamicState t0 $ \s' t0' -> withMask bt $ \bt' ->  Sig.c_leValue s' bt' t0' (hs2cReal v)

  gtValue_ :: MaskDynamic -> Dynamic -> HsReal -> IO ()
  gtValue_ bt t0 v = withDynamicState t0 $ \s' t0' -> withMask bt $ \bt' ->  Sig.c_gtValue s' bt' t0' (hs2cReal v)

  geValue_ :: MaskDynamic -> Dynamic -> HsReal -> IO ()
  geValue_ bt t0 v = withDynamicState t0 $ \s' t0' -> withMask bt $ \bt' ->  Sig.c_geValue s' bt' t0' (hs2cReal v)

  neValue_ :: MaskDynamic -> Dynamic -> HsReal -> IO ()
  neValue_ bt t0 v = withDynamicState t0 $ \s' t0' -> withMask bt $ \bt' ->  Sig.c_neValue s' bt' t0' (hs2cReal v)

  eqValue_ :: MaskDynamic -> Dynamic -> HsReal -> IO ()
  eqValue_ bt t0 v = withDynamicState t0 $ \s' t0' -> withMask bt $ \bt' ->  Sig.c_eqValue s' bt' t0' (hs2cReal v)

  ltValueT_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  ltValueT_ t0 t1 v = with2DynamicState t0 t1 (shuffle3 Sig.c_ltValueT (hs2cReal v))

  leValueT_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  leValueT_ t0 t1 v = with2DynamicState t0 t1 (shuffle3 Sig.c_leValueT (hs2cReal v))

  gtValueT_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  gtValueT_ t0 t1 v = with2DynamicState t0 t1 (shuffle3 Sig.c_gtValueT (hs2cReal v))

  geValueT_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  geValueT_ t0 t1 v = with2DynamicState t0 t1 (shuffle3 Sig.c_geValueT (hs2cReal v))

  neValueT_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  neValueT_ t0 t1 v = with2DynamicState t0 t1 (shuffle3 Sig.c_neValueT (hs2cReal v))

  eqValueT_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  eqValueT_ t0 t1 v = with2DynamicState t0 t1 (shuffle3 Sig.c_eqValueT (hs2cReal v))


