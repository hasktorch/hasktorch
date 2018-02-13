{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Dynamic.Math where

import THTypes
import Foreign
import Foreign.C.Types
import GHC.Int
import qualified Tensor     as Sig
import qualified TensorMath as Sig
import qualified Torch.Class.C.Tensor.Math as Class

import Torch.Core.Types

twoTensorsAndReal :: (Ptr CTensor -> Ptr CTensor -> CReal -> IO ()) -> Tensor -> Tensor -> HsReal -> IO ()
twoTensorsAndReal fn t0 t1 v0 = _with2Tensors t0 t1 $ \t0' t1' -> fn t0' t1' (hs2cReal v0)

twoTensorsAndTwoReals :: (Ptr CTensor -> Ptr CTensor -> CReal -> CReal -> IO ()) -> Tensor -> Tensor -> HsReal -> HsReal -> IO ()
twoTensorsAndTwoReals fn t0 t1 v0 v1 = _with2Tensors t0 t1  $ \t0' t1' -> fn t0' t1' (hs2cReal v0) (hs2cReal v1)

instance Class.TensorMath Tensor where
  fill :: Tensor -> HsReal -> IO ()
  fill t v = _withTensor t $ \t' -> Sig.c_fill t' (hs2cReal v)

  zero :: Tensor -> IO ()
  zero = withTensor Sig.c_zero

  maskedFill :: Tensor -> Ptr CTHByteTensor -> HsReal -> IO ()
  maskedFill res bt v = _withTensor res $ \res' -> Sig.c_maskedFill res' bt (hs2cReal v)

  maskedCopy :: Tensor -> Ptr CTHByteTensor -> Tensor -> IO ()
  maskedCopy res bt t = _with2Tensors res t $ \res' t' -> Sig.c_maskedCopy res' bt t'

  maskedSelect :: Tensor -> Tensor -> Ptr CTHByteTensor -> IO ()
  maskedSelect res t bts = _with2Tensors res t $ \res' t' -> Sig.c_maskedSelect res' t' bts

  nonzero :: Ptr CTHLongTensor -> Tensor -> IO ()
  nonzero ix t = _withTensor t $ \t' -> Sig.c_nonzero ix t'

  indexSelect  :: Tensor -> Tensor -> Int32 -> Ptr CTHLongTensor -> IO ()
  indexSelect t0 t1 i ls = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_indexSelect t0' t1' (CInt i) ls

  indexCopy :: Tensor -> Int32 -> Ptr CTHLongTensor -> Tensor -> IO ()
  indexCopy t0 i ls t1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_indexCopy t0' (CInt i) ls t1'

  indexAdd :: Tensor -> Int32 -> Ptr CTHLongTensor -> Tensor -> IO ()
  indexAdd t0 i ls t1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_indexAdd t0' (CInt i) ls t1'

  indexFill    :: Tensor -> Int32 -> Ptr CTHLongTensor -> HsReal -> IO ()
  indexFill t0 i ls v = _withTensor t0 $ \t0' -> Sig.c_indexFill t0' (CInt i) ls (hs2cReal v)

  take :: Tensor -> Tensor -> Ptr CTHLongTensor -> IO ()
  take t0 t1 ls = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_take t0' t1' ls

  put :: Tensor -> Ptr CTHLongTensor -> Tensor -> Int32 -> IO ()
  put t0 ls t1 i = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_put t0' ls t1' (CInt i)

  gather :: Tensor -> Tensor -> Int32 -> Ptr CTHLongTensor -> IO ()
  gather t0 t1 i ls = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_gather t0' t1' (CInt i) ls

  scatter      :: Tensor -> Int32 -> Ptr CTHLongTensor -> Tensor -> IO ()
  scatter t0 i ls t1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_scatter t0' (CInt i) ls t1'

  scatterAdd   :: Tensor -> Int32 -> Ptr CTHLongTensor -> Tensor -> IO ()
  scatterAdd t0 i ls t1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_scatterAdd t0' (CInt i) ls t1'

  scatterFill  :: Tensor -> Int32 -> Ptr CTHLongTensor -> HsReal -> IO ()
  scatterFill t0 i ls v = _withTensor t0 $ \t0' -> Sig.c_scatterFill t0' (CInt i) ls (hs2cReal v)

  dot :: Tensor -> Tensor -> IO HsAccReal
  dot = with2Tensors (\t0' t1' -> c2hsAccReal <$> Sig.c_dot t0' t1' )

  minall :: Tensor -> IO HsReal
  minall = withTensor (fmap c2hsReal . Sig.c_minall)

  maxall :: Tensor -> IO HsReal
  maxall = withTensor (fmap c2hsReal . Sig.c_maxall)

  medianall :: Tensor -> IO HsReal
  medianall = withTensor (fmap c2hsReal . Sig.c_medianall)

  sumall :: Tensor -> IO HsAccReal
  sumall = withTensor (fmap c2hsAccReal . Sig.c_sumall)

  prodall :: Tensor -> IO HsAccReal
  prodall = withTensor (fmap c2hsAccReal . Sig.c_prodall)

  add :: Tensor -> Tensor -> HsReal -> IO ()
  add = twoTensorsAndReal Sig.c_add

  sub :: Tensor -> Tensor -> HsReal -> IO ()
  sub = twoTensorsAndReal Sig.c_sub

  add_scaled :: Tensor -> Tensor -> HsReal -> HsReal -> IO ()
  add_scaled = twoTensorsAndTwoReals Sig.c_add_scaled

  sub_scaled :: Tensor -> Tensor -> HsReal -> HsReal -> IO ()
  sub_scaled = twoTensorsAndTwoReals Sig.c_sub_scaled

  mul :: Tensor -> Tensor -> HsReal -> IO ()
  mul = twoTensorsAndReal Sig.c_mul

  div :: Tensor -> Tensor -> HsReal -> IO ()
  div = twoTensorsAndReal Sig.c_div

  lshift :: Tensor -> Tensor -> HsReal -> IO ()
  lshift = twoTensorsAndReal Sig.c_lshift

  rshift :: Tensor -> Tensor -> HsReal -> IO ()
  rshift = twoTensorsAndReal Sig.c_rshift

  fmod :: Tensor -> Tensor -> HsReal -> IO ()
  fmod = twoTensorsAndReal Sig.c_fmod

  remainder :: Tensor -> Tensor -> HsReal -> IO ()
  remainder = twoTensorsAndReal Sig.c_remainder

  clamp :: Tensor -> Tensor -> HsReal -> HsReal -> IO ()
  clamp = twoTensorsAndTwoReals Sig.c_clamp

  bitand :: Tensor -> Tensor -> HsReal -> IO ()
  bitand = twoTensorsAndReal Sig.c_bitand

  bitor :: Tensor -> Tensor -> HsReal -> IO ()
  bitor = twoTensorsAndReal Sig.c_bitor

  bitxor :: Tensor -> Tensor -> HsReal -> IO ()
  bitxor = twoTensorsAndReal Sig.c_bitxor

  cadd :: Tensor -> Tensor -> HsReal -> Tensor -> IO ()
  cadd t0 t1 v t2 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_cadd t0' t1' (hs2cReal v) t2'

  csub :: Tensor -> Tensor -> HsReal -> Tensor -> IO ()
  csub t0 t1 v t2 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_csub t0' t1' (hs2cReal v) t2'

  cmul :: Tensor -> Tensor -> Tensor -> IO ()
  cmul = with3Tensors Sig.c_cmul

  cpow :: Tensor -> Tensor -> Tensor -> IO ()
  cpow = with3Tensors Sig.c_cpow

  cdiv :: Tensor -> Tensor -> Tensor -> IO ()
  cdiv = with3Tensors Sig.c_cdiv

  clshift :: Tensor -> Tensor -> Tensor -> IO ()
  clshift = with3Tensors Sig.c_clshift

  crshift :: Tensor -> Tensor -> Tensor -> IO ()
  crshift = with3Tensors Sig.c_crshift

  cfmod :: Tensor -> Tensor -> Tensor -> IO ()
  cfmod = with3Tensors Sig.c_cfmod

  cremainder :: Tensor -> Tensor -> Tensor -> IO ()
  cremainder = with3Tensors Sig.c_cremainder

  cbitand :: Tensor -> Tensor -> Tensor -> IO ()
  cbitand = with3Tensors Sig.c_cbitand

  cbitor :: Tensor -> Tensor -> Tensor -> IO ()
  cbitor = with3Tensors Sig.c_cbitor

  cbitxor :: Tensor -> Tensor -> Tensor -> IO ()
  cbitxor = with3Tensors Sig.c_cbitxor

  addcmul :: Tensor -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  addcmul t0 t1 v t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addcmul t0' t1' (hs2cReal v) t2' t3'

  addcdiv :: Tensor -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  addcdiv t0 t1 v t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addcdiv t0' t1' (hs2cReal v) t2' t3'

  addmv :: Tensor -> HsReal -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  addmv t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addmv t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  addmm :: Tensor -> HsReal -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  addmm t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addmm t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  addr :: Tensor -> HsReal -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  addr t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addr t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  addbmm :: Tensor -> HsReal -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  addbmm t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addbmm t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  baddbmm :: Tensor -> HsReal -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  baddbmm t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_baddbmm t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  match :: Tensor -> Tensor -> Tensor -> HsReal -> IO ()
  match t0 t1 t2 v0  = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_match t0' t1' t2' (hs2cReal v0)

  numel :: Tensor -> IO Int64
  numel = withTensor (fmap fromIntegral . Sig.c_numel)

  max :: Tensor -> Ptr CTHLongTensor -> Tensor -> Int32 -> Int32 -> IO ()
  max t0 ls t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_max t0' ls t1' (CInt i0) (CInt i1)

  min :: Tensor -> Ptr CTHLongTensor -> Tensor -> Int32 -> Int32 -> IO ()
  min t0 ls t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_min t0' ls t1' (CInt i0) (CInt i1)

  kthvalue :: Tensor -> Ptr CTHLongTensor -> Tensor -> Int64 -> Int32 -> Int32 -> IO ()
  kthvalue t0 ls t1 pd i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_kthvalue t0' ls t1' (CLLong pd) (CInt i0) (CInt i1)

  mode :: Tensor -> Ptr CTHLongTensor -> Tensor -> Int32 -> Int32 -> IO ()
  mode t0 ls t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_mode t0' ls t1' (CInt i0) (CInt i1)

  median :: Tensor -> Ptr CTHLongTensor -> Tensor -> Int32 -> Int32 -> IO ()
  median t0 ls t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_median t0' ls t1' (CInt i0) (CInt i1)

  sum :: Tensor -> Tensor -> Int32 -> Int32 -> IO ()
  sum t0 t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_sum t0' t1' (CInt i0) (CInt i1)

  prod :: Tensor -> Tensor -> Int32 -> Int32 -> IO ()
  prod t0 t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_prod t0' t1' (CInt i0) (CInt i1)

  cumsum :: Tensor -> Tensor -> Int32 -> IO ()
  cumsum t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_cumsum t0' t1' (CInt i0)

  cumprod :: Tensor -> Tensor -> Int32 -> IO ()
  cumprod t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_cumprod t0' t1' (CInt i0)

  sign :: Tensor -> Tensor -> IO ()
  sign = with2Tensors Sig.c_sign

  trace :: Tensor -> IO HsAccReal
  trace = withTensor (fmap c2hsAccReal . Sig.c_trace)

  cross :: Tensor -> Tensor -> Tensor -> Int32 -> IO ()
  cross t0 t1 t2 i0 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_cross t0' t1' t2' (CInt i0)

  cmax :: Tensor -> Tensor -> Tensor -> IO ()
  cmax = with3Tensors Sig.c_cmax

  cmin :: Tensor -> Tensor -> Tensor -> IO ()
  cmin = with3Tensors Sig.c_cmin

  cmaxValue    :: Tensor -> Tensor -> HsReal -> IO ()
  cmaxValue = twoTensorsAndReal Sig.c_cmaxValue

  cminValue    :: Tensor -> Tensor -> HsReal -> IO ()
  cminValue = twoTensorsAndReal Sig.c_cminValue

  zeros :: Tensor -> Ptr CTHLongStorage -> IO ()
  zeros t0 ls = _withTensor t0 $ \t0' -> Sig.c_zeros t0' ls

  zerosLike :: Tensor -> Tensor -> IO ()
  zerosLike = with2Tensors Sig.c_zerosLike

  ones :: Tensor -> Ptr CTHLongStorage -> IO ()
  ones t0 ls = _withTensor t0 $ \t0' -> Sig.c_ones t0' ls

  onesLike :: Tensor -> Tensor -> IO ()
  onesLike = with2Tensors Sig.c_onesLike

  diag :: Tensor -> Tensor -> Int32 -> IO ()
  diag t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_diag t0' t1' (CInt i0)

  eye :: Tensor -> Int64 -> Int64 -> IO ()
  eye t0 l0 l1 = _withTensor t0 $ \t0' -> Sig.c_eye t0' (CLLong l0) (CLLong l1)

  arange :: Tensor -> HsAccReal -> HsAccReal -> HsAccReal -> IO ()
  arange t0 a0 a1 a2 = _withTensor t0 $ \t0' -> Sig.c_arange t0' (hs2cAccReal a0) (hs2cAccReal a1) (hs2cAccReal a2)

  range :: Tensor -> HsAccReal-> HsAccReal-> HsAccReal-> IO ()
  range t0 a0 a1 a2 = _withTensor t0 $ \t0' -> Sig.c_range t0' (hs2cAccReal a0) (hs2cAccReal a1) (hs2cAccReal a2)

  randperm :: Tensor -> Ptr CTHGenerator -> Int64 -> IO ()
  randperm t0 pg i0 = _withTensor t0 $ \t0' -> Sig.c_randperm t0' pg (CLLong i0)

  reshape :: Tensor -> Tensor -> Ptr CTHLongStorage -> IO ()
  reshape t0 t1 ls = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_reshape t0' t1' ls

  sort :: Tensor -> Ptr CTHLongTensor -> Tensor -> Int32 -> Int32 -> IO ()
  sort t0 ls t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_sort t0' ls t1' (CInt i0) (CInt i1)

  topk :: Tensor -> Ptr CTHLongTensor -> Tensor -> Int64 -> Int32 -> Int32 -> Int32 -> IO ()
  topk t0 ls t1 l i0 i1 i2 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_topk t0' ls t1' (CLLong l) (CInt i0) (CInt i1) (CInt i2)

  tril :: Tensor -> Tensor -> Int64 -> IO ()
  tril t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_tril t0' t1' (CLLong i0)

  triu :: Tensor -> Tensor -> Int64 -> IO ()
  triu t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_triu t0' t1' (CLLong i0)

  cat :: Tensor -> Tensor -> Tensor -> Int32 -> IO ()
  cat t0 t1 t2 i0 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_cat t0' t1' t2' (CInt i0)

  -- catArray     :: Tensor -> [Tensor] -> Int32 -> Int32 -> IO ()

  equal :: Tensor -> Tensor -> IO Int32
  equal = with2Tensors (\t0' t1' -> fmap fromIntegral $ Sig.c_equal t0' t1' )

  ltValue :: Ptr CTHByteTensor -> Tensor -> HsReal -> IO ()
  ltValue bt t0 v = _withTensor t0 $ \t0' -> Sig.c_ltValue bt t0' (hs2cReal v)

  leValue :: Ptr CTHByteTensor -> Tensor -> HsReal -> IO ()
  leValue bt t0 v = _withTensor t0 $ \t0' -> Sig.c_leValue bt t0' (hs2cReal v)

  gtValue :: Ptr CTHByteTensor -> Tensor -> HsReal -> IO ()
  gtValue bt t0 v = _withTensor t0 $ \t0' -> Sig.c_gtValue bt t0' (hs2cReal v)

  geValue :: Ptr CTHByteTensor -> Tensor -> HsReal -> IO ()
  geValue bt t0 v = _withTensor t0 $ \t0' -> Sig.c_geValue bt t0' (hs2cReal v)

  neValue :: Ptr CTHByteTensor -> Tensor -> HsReal -> IO ()
  neValue bt t0 v = _withTensor t0 $ \t0' -> Sig.c_neValue bt t0' (hs2cReal v)

  eqValue :: Ptr CTHByteTensor -> Tensor -> HsReal -> IO ()
  eqValue bt t0 v = _withTensor t0 $ \t0' -> Sig.c_eqValue bt t0' (hs2cReal v)

  ltValueT :: Tensor -> Tensor -> HsReal -> IO ()
  ltValueT = twoTensorsAndReal Sig.c_ltValueT

  leValueT :: Tensor -> Tensor -> HsReal -> IO ()
  leValueT = twoTensorsAndReal Sig.c_leValueT

  gtValueT :: Tensor -> Tensor -> HsReal -> IO ()
  gtValueT = twoTensorsAndReal Sig.c_gtValueT

  geValueT :: Tensor -> Tensor -> HsReal -> IO ()
  geValueT = twoTensorsAndReal Sig.c_geValueT

  neValueT :: Tensor -> Tensor -> HsReal -> IO ()
  neValueT = twoTensorsAndReal Sig.c_neValueT

  eqValueT :: Tensor -> Tensor -> HsReal -> IO ()
  eqValueT = twoTensorsAndReal Sig.c_eqValueT

  ltTensor :: Ptr CTHByteTensor -> Tensor -> Tensor -> IO ()
  ltTensor bt t0 t1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_ltTensor bt t0' t1'

  leTensor :: Ptr CTHByteTensor -> Tensor -> Tensor -> IO ()
  leTensor bt t0 t1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_leTensor bt t0' t1'

  gtTensor :: Ptr CTHByteTensor -> Tensor -> Tensor -> IO ()
  gtTensor bt t0 t1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_gtTensor bt t0' t1'

  geTensor :: Ptr CTHByteTensor -> Tensor -> Tensor -> IO ()
  geTensor bt t0 t1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_geTensor bt t0' t1'

  neTensor :: Ptr CTHByteTensor -> Tensor -> Tensor -> IO ()
  neTensor bt t0 t1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_neTensor bt t0' t1'

  eqTensor :: Ptr CTHByteTensor -> Tensor -> Tensor -> IO ()
  eqTensor bt t0 t1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_eqTensor bt t0' t1'

  ltTensorT :: Tensor -> Tensor -> Tensor -> IO ()
  ltTensorT = with3Tensors Sig.c_ltTensorT

  leTensorT :: Tensor -> Tensor -> Tensor -> IO ()
  leTensorT = with3Tensors Sig.c_leTensorT

  gtTensorT :: Tensor -> Tensor -> Tensor -> IO ()
  gtTensorT = with3Tensors Sig.c_gtTensorT

  geTensorT :: Tensor -> Tensor -> Tensor -> IO ()
  geTensorT = with3Tensors Sig.c_geTensorT

  neTensorT :: Tensor -> Tensor -> Tensor -> IO ()
  neTensorT = with3Tensors Sig.c_neTensorT

  eqTensorT :: Tensor -> Tensor -> Tensor -> IO ()
  eqTensorT = with3Tensors Sig.c_eqTensorT
