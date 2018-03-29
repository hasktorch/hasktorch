{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Indef.Dynamic.Tensor.Math where

import Torch.Types.TH
import Foreign
import Foreign.C.Types
import GHC.Int
import qualified Torch.Sig.IsTensor    as Sig
import qualified Torch.Sig.Tensor.Math as Sig
import qualified Torch.Class.Tensor.Math as Class
import qualified Torch.Types.TH.Byte   as B
import qualified Torch.Types.TH.Long   as L
import Torch.Types.TH.Random (Generator(..))

import Torch.Indef.Types

type ByteTensor = B.Dynamic
type LongTensor = L.Dynamic
type LongStorage = L.Storage

twoTensorsAndReal :: (Ptr CTensor -> Ptr CTensor -> CReal -> IO ()) -> Dynamic -> Dynamic -> HsReal -> IO ()
twoTensorsAndReal fn t0 t1 v0 = _with2Tensors t0 t1 $ \t0' t1' -> fn t0' t1' (hs2cReal v0)

twoTensorsAndTwoReals :: (Ptr CTensor -> Ptr CTensor -> CReal -> CReal -> IO ()) -> Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
twoTensorsAndTwoReals fn t0 t1 v0 v1 = _with2Tensors t0 t1  $ \t0' t1' -> fn t0' t1' (hs2cReal v0) (hs2cReal v1)


instance Class.TensorMath Dynamic where
  fill_ :: Dynamic -> HsReal -> IO ()
  fill_ t v = _withTensor t $ \t' -> Sig.c_fill t' (hs2cReal v)

  zero_ :: Dynamic -> IO ()
  zero_ = withTensor Sig.c_zero

  maskedFill_ :: Dynamic -> ByteTensor -> HsReal -> IO ()
  maskedFill_ res b v = _withTensor res $ \res' -> withForeignPtr (B.tensor b) $ \b' -> Sig.c_maskedFill res' b' (hs2cReal v)

  maskedCopy_ :: Dynamic -> ByteTensor -> Dynamic -> IO ()
  maskedCopy_ res b t = _with2Tensors res t $ \res' t' -> withForeignPtr (B.tensor b) $ \b' -> Sig.c_maskedCopy res' b' t'

  maskedSelect_ :: Dynamic -> Dynamic -> ByteTensor -> IO ()
  maskedSelect_ res t b = _with2Tensors res t $ \res' t' -> withForeignPtr (B.tensor b) $ \b' -> Sig.c_maskedSelect res' t' b'

  nonzero_ :: LongTensor -> Dynamic -> IO ()
  nonzero_ l t = _withTensor t $ \t' -> withForeignPtr (L.tensor l) $ \l' -> Sig.c_nonzero l' t'

  dot :: Dynamic -> Dynamic -> IO HsAccReal
  dot = with2Tensors (\t0' t1' -> c2hsAccReal <$> Sig.c_dot t0' t1' )

  minall :: Dynamic -> IO HsReal
  minall = withTensor (fmap c2hsReal . Sig.c_minall)

  maxall :: Dynamic -> IO HsReal
  maxall = withTensor (fmap c2hsReal . Sig.c_maxall)

  medianall :: Dynamic -> IO HsReal
  medianall = withTensor (fmap c2hsReal . Sig.c_medianall)

  sumall :: Dynamic -> IO HsAccReal
  sumall = withTensor (fmap c2hsAccReal . Sig.c_sumall)

  prodall :: Dynamic -> IO HsAccReal
  prodall = withTensor (fmap c2hsAccReal . Sig.c_prodall)

  add_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  add_ = twoTensorsAndReal Sig.c_add

  sub_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  sub_ = twoTensorsAndReal Sig.c_sub

  add_scaled_ :: Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
  add_scaled_ = twoTensorsAndTwoReals Sig.c_add_scaled

  sub_scaled_ :: Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
  sub_scaled_ = twoTensorsAndTwoReals Sig.c_sub_scaled

  mul_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  mul_ = twoTensorsAndReal Sig.c_mul

  div_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  div_ = twoTensorsAndReal Sig.c_div

  lshift_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  lshift_ = twoTensorsAndReal Sig.c_lshift

  rshift_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  rshift_ = twoTensorsAndReal Sig.c_rshift

  fmod_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  fmod_ = twoTensorsAndReal Sig.c_fmod

  remainder_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  remainder_ = twoTensorsAndReal Sig.c_remainder

  clamp_ :: Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
  clamp_ = twoTensorsAndTwoReals Sig.c_clamp

  bitand_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  bitand_ = twoTensorsAndReal Sig.c_bitand

  bitor_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  bitor_ = twoTensorsAndReal Sig.c_bitor

  bitxor_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  bitxor_ = twoTensorsAndReal Sig.c_bitxor

  cadd_ :: Dynamic -> Dynamic -> HsReal -> Dynamic -> IO ()
  cadd_ t0 t1 v t2 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_cadd t0' t1' (hs2cReal v) t2'

  csub_ :: Dynamic -> Dynamic -> HsReal -> Dynamic -> IO ()
  csub_ t0 t1 v t2 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_csub t0' t1' (hs2cReal v) t2'

  cmul_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cmul_ = with3Tensors Sig.c_cmul

  cpow_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cpow_ = with3Tensors Sig.c_cpow

  cdiv_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cdiv_ = with3Tensors Sig.c_cdiv

  clshift_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  clshift_ = with3Tensors Sig.c_clshift

  crshift_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  crshift_ = with3Tensors Sig.c_crshift

  cfmod_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cfmod_ = with3Tensors Sig.c_cfmod

  cremainder_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cremainder_ = with3Tensors Sig.c_cremainder

  cbitand_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cbitand_ = with3Tensors Sig.c_cbitand

  cbitor_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cbitor_ = with3Tensors Sig.c_cbitor

  cbitxor_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cbitxor_ = with3Tensors Sig.c_cbitxor

  addcmul_ :: Dynamic -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  addcmul_ t0 t1 v t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addcmul t0' t1' (hs2cReal v) t2' t3'

  addcdiv_ :: Dynamic -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  addcdiv_ t0 t1 v t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addcdiv t0' t1' (hs2cReal v) t2' t3'

  addmv_ :: Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  addmv_ t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addmv t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  addmm_ :: Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  addmm_ t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addmm t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  addr_ :: Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  addr_ t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addr t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  addbmm_ :: Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  addbmm_ t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addbmm t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  baddbmm_ :: Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  baddbmm_ t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_baddbmm t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  match_ :: Dynamic -> Dynamic -> Dynamic -> HsReal -> IO ()
  match_ t0 t1 t2 v0  = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_match t0' t1' t2' (hs2cReal v0)

  numel :: Dynamic -> IO Int64
  numel = withTensor (fmap fromIntegral . Sig.c_numel)

  max_ :: (Tensor, LongTensor) -> Dynamic -> Int32 -> Int32 -> IO ()
  max_ (t0, ls) t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.tensor ls) $ \ls' ->
       Sig.c_max t0' ls' t1' (CInt i0) (CInt i1)

  min_ :: (Tensor, LongTensor) -> Dynamic -> Int32 -> Int32 -> IO ()
  min_ (t0, ls) t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.tensor ls) $ \ls' ->
       Sig.c_min t0' ls' t1' (CInt i0) (CInt i1)

  kthvalue_ :: (Tensor, LongTensor) -> Dynamic -> Int64 -> Int32 -> Int32 -> IO ()
  kthvalue_ (t0, ls) t1 pd i0 i1 = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.tensor ls) $ \ls' ->
       Sig.c_kthvalue t0' ls' t1' (CLLong pd) (CInt i0) (CInt i1)

  median_ :: (Tensor, LongTensor) -> Dynamic -> Int32 -> Int32 -> IO ()
  median_ (t0, ls) t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.tensor ls) $ \ls' ->
      Sig.c_median t0' ls' t1' (CInt i0) (CInt i1)

  sum_ :: Dynamic -> Dynamic -> Int32 -> Int32 -> IO ()
  sum_ t0 t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_sum t0' t1' (CInt i0) (CInt i1)

  prod_ :: Dynamic -> Dynamic -> Int32 -> Int32 -> IO ()
  prod_ t0 t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_prod t0' t1' (CInt i0) (CInt i1)

  cumsum_ :: Dynamic -> Dynamic -> Int32 -> IO ()
  cumsum_ t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_cumsum t0' t1' (CInt i0)

  cumprod_ :: Dynamic -> Dynamic -> Int32 -> IO ()
  cumprod_ t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_cumprod t0' t1' (CInt i0)

  sign_ :: Dynamic -> Dynamic -> IO ()
  sign_ = with2Tensors Sig.c_sign

  trace :: Dynamic -> IO HsAccReal
  trace = withTensor (fmap c2hsAccReal . Sig.c_trace)

  cross_ :: Dynamic -> Dynamic -> Dynamic -> Int32 -> IO ()
  cross_ t0 t1 t2 i0 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_cross t0' t1' t2' (CInt i0)

  cmax_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cmax_ = with3Tensors Sig.c_cmax

  cmin_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  cmin_ = with3Tensors Sig.c_cmin

  cmaxValue_    :: Dynamic -> Dynamic -> HsReal -> IO ()
  cmaxValue_ = twoTensorsAndReal Sig.c_cmaxValue

  cminValue_    :: Dynamic -> Dynamic -> HsReal -> IO ()
  cminValue_ = twoTensorsAndReal Sig.c_cminValue

  zeros_ :: Dynamic -> LongStorage -> IO ()
  zeros_ t0 ls = _withTensor t0 $ \t0' ->
    withForeignPtr (L.storage ls) $ \ls' ->
      Sig.c_zeros t0' ls'

  zerosLike_ :: Dynamic -> Dynamic -> IO ()
  zerosLike_ = with2Tensors Sig.c_zerosLike

  ones_ :: Dynamic -> LongStorage -> IO ()
  ones_ t0 ls = _withTensor t0 $ \t0' ->
    withForeignPtr (L.storage ls) $ \ls' ->
      Sig.c_ones t0' ls'

  onesLike_ :: Dynamic -> Dynamic -> IO ()
  onesLike_ = with2Tensors Sig.c_onesLike

  diag_ :: Dynamic -> Dynamic -> Int32 -> IO ()
  diag_ t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_diag t0' t1' (CInt i0)

  eye_ :: Dynamic -> Int64 -> Int64 -> IO ()
  eye_ t0 l0 l1 = _withTensor t0 $ \t0' -> Sig.c_eye t0' (CLLong l0) (CLLong l1)

  arange_ :: Dynamic -> HsAccReal -> HsAccReal -> HsAccReal -> IO ()
  arange_ t0 a0 a1 a2 = _withTensor t0 $ \t0' -> Sig.c_arange t0' (hs2cAccReal a0) (hs2cAccReal a1) (hs2cAccReal a2)

  range_ :: Dynamic -> HsAccReal-> HsAccReal-> HsAccReal-> IO ()
  range_ t0 a0 a1 a2 = _withTensor t0 $ \t0' -> Sig.c_range t0' (hs2cAccReal a0) (hs2cAccReal a1) (hs2cAccReal a2)

  randperm_ :: Dynamic -> Generator -> Int64 -> IO ()
  randperm_ t0 pg i0 = _withTensor t0 $ \t0' ->
    withForeignPtr (rng pg) $ \pg' ->
      Sig.c_randperm t0' pg' (CLLong i0)

  reshape_ :: Dynamic -> Dynamic -> LongStorage -> IO ()
  reshape_ t0 t1 ls = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.storage ls) $ \ls' ->
      Sig.c_reshape t0' t1' ls'

  tril_ :: Dynamic -> Dynamic -> Int64 -> IO ()
  tril_ t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_tril t0' t1' (CLLong i0)

  triu_ :: Dynamic -> Dynamic -> Int64 -> IO ()
  triu_ t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_triu t0' t1' (CLLong i0)

  cat_ :: Dynamic -> Dynamic -> Dynamic -> Int32 -> IO ()
  cat_ t0 t1 t2 i0 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_cat t0' t1' t2' (CInt i0)

  -- catArray     :: Dynamic -> [Tensor] -> Int32 -> Int32 -> IO ()

  equal :: Dynamic -> Dynamic -> IO Int32
  equal = with2Tensors (\t0' t1' -> fmap fromIntegral $ Sig.c_equal t0' t1' )

