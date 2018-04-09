module Torch.Indef.Dynamic.Tensor.Math.Pointwise where

import Torch.Class.Tensor.Math.Pointwise
import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor ()
import Torch.Dimensions

import qualified Torch.Sig.Tensor.Math.Pointwise as Sig

instance TensorMathPointwise Dynamic where
  _sign :: Dynamic -> Dynamic -> IO ()
  _sign r t = with2DynamicState r t Sig.c_sign

  _cross :: Dynamic -> Dynamic -> Dynamic -> DimVal -> IO ()
  _cross t0 t1 t2 i0 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' -> Sig.c_cross s' t0' t1' t2' (fromIntegral i0)

  _clamp :: Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
  _clamp r t v0 v1 = with2DynamicState r t $ shuffle3'2 Sig.c_clamp (hs2cReal v0) (hs2cReal v1)

  _cmax :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _cmax r t0 t1 = with3DynamicState r t0 t1 $ \s' r' t0' t1' ->  Sig.c_cmax s' r' t0' t1'

  _cmin :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _cmin r t0 t1 = with3DynamicState r t0 t1 $ \s' r' t0' t1' ->  Sig.c_cmin s' r' t0' t1'

  _cmaxValue :: Dynamic -> Dynamic -> HsReal -> IO ()
  _cmaxValue r t v = with2DynamicState r t $ shuffle3 Sig.c_cmaxValue (hs2cReal v)

  _cminValue :: Dynamic -> Dynamic -> HsReal -> IO ()
  _cminValue r t v = with2DynamicState r t $ shuffle3 Sig.c_cminValue (hs2cReal v)

  _addcmul :: Dynamic -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  _addcmul t0 t1 v t2 t3 =
    with2DynamicState t0 t1 $ \s' t0' t1' ->
      with2DynamicState t2 t3 $ \_ t2' t3' ->
        Sig.c_addcmul s' t0' t1' (hs2cReal v) t2' t3'

  _addcdiv :: Dynamic -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  _addcdiv t0 t1 v t2 t3 =
    with2DynamicState t0 t1 $ \s' t0' t1' ->
      with2DynamicState t2 t3 $ \_ t2' t3' ->
        Sig.c_addcdiv s' t0' t1' (hs2cReal v) t2' t3'

  _cadd :: Dynamic -> Dynamic -> HsReal -> Dynamic -> IO ()
  _cadd t0 t1 v t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' -> Sig.c_cadd s' t0' t1' (hs2cReal v) t2'

  _csub :: Dynamic -> Dynamic -> HsReal -> Dynamic -> IO ()
  _csub t0 t1 v t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' -> Sig.c_csub s' t0' t1' (hs2cReal v) t2'

  _cmul :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _cmul r t0 t1 = with3DynamicState r t0 t1 Sig.c_cmul

  _cpow :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _cpow r t0 t1 = with3DynamicState r t0 t1 Sig.c_cpow

  _cdiv :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _cdiv r t0 t1 = with3DynamicState r t0 t1 Sig.c_cdiv

  _clshift :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _clshift r t0 t1 = with3DynamicState r t0 t1 Sig.c_clshift

  _crshift :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _crshift r t0 t1 = with3DynamicState r t0 t1 Sig.c_crshift

  _cfmod :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _cfmod r t0 t1 = with3DynamicState r t0 t1 Sig.c_cfmod

  _cremainder :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _cremainder r t0 t1 = with3DynamicState r t0 t1 Sig.c_cremainder

  _cbitand :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _cbitand r t0 t1 = with3DynamicState r t0 t1 Sig.c_cbitand

  _cbitor :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _cbitor r t0 t1 = with3DynamicState r t0 t1 Sig.c_cbitor

  _cbitxor :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _cbitxor r t0 t1 = with3DynamicState r t0 t1 Sig.c_cbitxor


