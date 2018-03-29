module Torch.Indef.Dynamic.Tensor.Math.Pointwise where

import Torch.Class.Tensor.Math.Pointwise
import Torch.Indef.Types

import qualified Torch.Sig.Tensor.Math.Pointwise as Sig

instance TensorMathPointwise Dynamic where
  sign_ :: Dynamic -> Dynamic -> IO ()
  sign_ r t = with2DynamicState r t Sig.c_sign

  cross_ :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
  cross_ t0 t1 t2 i0 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' -> Sig.c_cross s' t0' t1' t2' (fromIntegral i0)

  clamp_ :: Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
  clamp_ r t v0 v1 = with2DynamicState r t $ shuffle3'2 Sig.c_clamp (hs2cReal v0) (hs2cReal v1)

  cmax_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cmax_ r t0 t1 = with3DynamicState r t0 t1 $ \s' r' t0' t1' ->  Sig.c_cmax s' r' t0' t1'

  cmin_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cmin_ r t0 t1 = with3DynamicState r t0 t1 $ \s' r' t0' t1' ->  Sig.c_cmin s' r' t0' t1'

  cmaxValue_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  cmaxValue_ r t v = with2DynamicState r t $ shuffle3 Sig.c_cmaxValue (hs2cReal v)

  cminValue_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  cminValue_ r t v = with2DynamicState r t $ shuffle3 Sig.c_cminValue (hs2cReal v)

  addcmul_ :: Dynamic -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  addcmul_ t0 t1 v t2 t3 =
    with2DynamicState t0 t1 $ \s' t0' t1' ->
      with2DynamicState t2 t3 $ \_ t2' t3' ->
        Sig.c_addcmul s' t0' t1' (hs2cReal v) t2' t3'

  addcdiv_ :: Dynamic -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  addcdiv_ t0 t1 v t2 t3 =
    with2DynamicState t0 t1 $ \s' t0' t1' ->
      with2DynamicState t2 t3 $ \_ t2' t3' ->
        Sig.c_addcdiv s' t0' t1' (hs2cReal v) t2' t3'

  cadd_ :: Dynamic -> Dynamic -> HsReal -> Dynamic -> IO ()
  cadd_ t0 t1 v t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' -> Sig.c_cadd s' t0' t1' (hs2cReal v) t2'

  csub_ :: Dynamic -> Dynamic -> HsReal -> Dynamic -> IO ()
  csub_ t0 t1 v t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' -> Sig.c_csub s' t0' t1' (hs2cReal v) t2'

  cmul_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cmul_ r t0 t1 = with3DynamicState r t0 t1 Sig.c_cmul

  cpow_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cpow_ r t0 t1 = with3DynamicState r t0 t1 Sig.c_cpow

  cdiv_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cdiv_ r t0 t1 = with3DynamicState r t0 t1 Sig.c_cdiv

  clshift_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  clshift_ r t0 t1 = with3DynamicState r t0 t1 Sig.c_clshift

  crshift_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  crshift_ r t0 t1 = with3DynamicState r t0 t1 Sig.c_crshift

  cfmod_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cfmod_ r t0 t1 = with3DynamicState r t0 t1 Sig.c_cfmod

  cremainder_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cremainder_ r t0 t1 = with3DynamicState r t0 t1 Sig.c_cremainder

  cbitand_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cbitand_ r t0 t1 = with3DynamicState r t0 t1 Sig.c_cbitand

  cbitor_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cbitor_ r t0 t1 = with3DynamicState r t0 t1 Sig.c_cbitor

  cbitxor_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cbitxor_ r t0 t1 = with3DynamicState r t0 t1 Sig.c_cbitxor


