{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Indef.Tensor.Dynamic.Math where

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

type ByteTensor = B.DynTensor
type LongTensor = L.DynTensor
type LongStorage = L.Storage

twoTensorsAndReal :: (Ptr CTensor -> Ptr CTensor -> CReal -> IO ()) -> Tensor -> Tensor -> HsReal -> IO ()
twoTensorsAndReal fn t0 t1 v0 = _with2Tensors t0 t1 $ \t0' t1' -> fn t0' t1' (hs2cReal v0)

twoTensorsAndTwoReals :: (Ptr CTensor -> Ptr CTensor -> CReal -> CReal -> IO ()) -> Tensor -> Tensor -> HsReal -> HsReal -> IO ()
twoTensorsAndTwoReals fn t0 t1 v0 v1 = _with2Tensors t0 t1  $ \t0' t1' -> fn t0' t1' (hs2cReal v0) (hs2cReal v1)


instance Class.TensorMath Tensor where
  fill_ :: Tensor -> HsReal -> IO ()
  fill_ t v = _withTensor t $ \t' -> Sig.c_fill t' (hs2cReal v)

  zero_ :: Tensor -> IO ()
  zero_ = withTensor Sig.c_zero

  maskedFill_ :: Tensor -> ByteTensor -> HsReal -> IO ()
  maskedFill_ res b v = _withTensor res $ \res' -> withForeignPtr (B.tensor b) $ \b' -> Sig.c_maskedFill res' b' (hs2cReal v)

  maskedCopy_ :: Tensor -> ByteTensor -> Tensor -> IO ()
  maskedCopy_ res b t = _with2Tensors res t $ \res' t' -> withForeignPtr (B.tensor b) $ \b' -> Sig.c_maskedCopy res' b' t'

  maskedSelect_ :: Tensor -> Tensor -> ByteTensor -> IO ()
  maskedSelect_ res t b = _with2Tensors res t $ \res' t' -> withForeignPtr (B.tensor b) $ \b' -> Sig.c_maskedSelect res' t' b'

  nonzero_ :: LongTensor -> Tensor -> IO ()
  nonzero_ l t = _withTensor t $ \t' -> withForeignPtr (L.tensor l) $ \l' -> Sig.c_nonzero l' t'

  indexSelect_  :: Tensor -> Tensor -> Int32 -> LongTensor -> IO ()
  indexSelect_ t0 t1 i l = _with2Tensors t0 t1 $ \t0' t1' -> withForeignPtr (L.tensor l) $ \l' -> Sig.c_indexSelect t0' t1' (CInt i) l'

  indexCopy_ :: Tensor -> Int32 -> LongTensor -> Tensor -> IO ()
  indexCopy_ t0 i l t1 = _with2Tensors t0 t1 $ \t0' t1' -> withForeignPtr (L.tensor l) $ \l' -> Sig.c_indexCopy t0' (CInt i) l' t1'

  indexAdd_ :: Tensor -> Int32 -> LongTensor -> Tensor -> IO ()
  indexAdd_ t0 i l t1 = _with2Tensors t0 t1 $ \t0' t1' -> withForeignPtr (L.tensor l) $ \l' -> Sig.c_indexAdd t0' (CInt i) l' t1'

  indexFill_    :: Tensor -> Int32 -> LongTensor -> HsReal -> IO ()
  indexFill_ t0 i ls v = _withTensor t0 $ \t0' -> withForeignPtr (L.tensor ls) $ \l' -> Sig.c_indexFill t0' (CInt i) l' (hs2cReal v)

  take_ :: Tensor -> Tensor -> LongTensor -> IO ()
  take_ t0 t1 ls = _with2Tensors t0 t1 $ \t0' t1' -> withForeignPtr (L.tensor ls) $ \ls' -> Sig.c_take t0' t1' ls'

  put_ :: Tensor -> LongTensor -> Tensor -> Int32 -> IO ()
  put_ t0 ls t1 i = _with2Tensors t0 t1 $ \t0' t1' -> withForeignPtr (L.tensor ls) $ \ls' -> Sig.c_put t0' ls' t1' (CInt i)

  gather_ :: Tensor -> Tensor -> Int32 -> LongTensor -> IO ()
  gather_ t0 t1 i ls = _with2Tensors t0 t1 $ \t0' t1' -> withForeignPtr (L.tensor ls) $ \ls' -> Sig.c_gather t0' t1' (CInt i) ls'

  scatter_      :: Tensor -> Int32 -> LongTensor -> Tensor -> IO ()
  scatter_ t0 i ls t1 = _with2Tensors t0 t1 $ \t0' t1' -> withForeignPtr (L.tensor ls) $ \ls' -> Sig.c_scatter t0' (CInt i) ls' t1'

  scatterAdd_   :: Tensor -> Int32 -> LongTensor -> Tensor -> IO ()
  scatterAdd_ t0 i ls t1 = _with2Tensors t0 t1 $ \t0' t1' -> withForeignPtr (L.tensor ls) $ \ls' -> Sig.c_scatterAdd t0' (CInt i) ls' t1'

  scatterFill_  :: Tensor -> Int32 -> LongTensor -> HsReal -> IO ()
  scatterFill_ t0 i ls v = _withTensor t0 $ \t0' -> withForeignPtr (L.tensor ls) $ \ls' -> Sig.c_scatterFill t0' (CInt i) ls' (hs2cReal v)

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

  add_ :: Tensor -> Tensor -> HsReal -> IO ()
  add_ = twoTensorsAndReal Sig.c_add

  sub_ :: Tensor -> Tensor -> HsReal -> IO ()
  sub_ = twoTensorsAndReal Sig.c_sub

  add_scaled_ :: Tensor -> Tensor -> HsReal -> HsReal -> IO ()
  add_scaled_ = twoTensorsAndTwoReals Sig.c_add_scaled

  sub_scaled_ :: Tensor -> Tensor -> HsReal -> HsReal -> IO ()
  sub_scaled_ = twoTensorsAndTwoReals Sig.c_sub_scaled

  mul_ :: Tensor -> Tensor -> HsReal -> IO ()
  mul_ = twoTensorsAndReal Sig.c_mul

  div_ :: Tensor -> Tensor -> HsReal -> IO ()
  div_ = twoTensorsAndReal Sig.c_div

  lshift_ :: Tensor -> Tensor -> HsReal -> IO ()
  lshift_ = twoTensorsAndReal Sig.c_lshift

  rshift_ :: Tensor -> Tensor -> HsReal -> IO ()
  rshift_ = twoTensorsAndReal Sig.c_rshift

  fmod_ :: Tensor -> Tensor -> HsReal -> IO ()
  fmod_ = twoTensorsAndReal Sig.c_fmod

  remainder_ :: Tensor -> Tensor -> HsReal -> IO ()
  remainder_ = twoTensorsAndReal Sig.c_remainder

  clamp_ :: Tensor -> Tensor -> HsReal -> HsReal -> IO ()
  clamp_ = twoTensorsAndTwoReals Sig.c_clamp

  bitand_ :: Tensor -> Tensor -> HsReal -> IO ()
  bitand_ = twoTensorsAndReal Sig.c_bitand

  bitor_ :: Tensor -> Tensor -> HsReal -> IO ()
  bitor_ = twoTensorsAndReal Sig.c_bitor

  bitxor_ :: Tensor -> Tensor -> HsReal -> IO ()
  bitxor_ = twoTensorsAndReal Sig.c_bitxor

  cadd_ :: Tensor -> Tensor -> HsReal -> Tensor -> IO ()
  cadd_ t0 t1 v t2 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_cadd t0' t1' (hs2cReal v) t2'

  csub_ :: Tensor -> Tensor -> HsReal -> Tensor -> IO ()
  csub_ t0 t1 v t2 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_csub t0' t1' (hs2cReal v) t2'

  cmul_ :: Tensor -> Tensor -> Tensor -> IO ()
  cmul_ = with3Tensors Sig.c_cmul

  cpow_ :: Tensor -> Tensor -> Tensor -> IO ()
  cpow_ = with3Tensors Sig.c_cpow

  cdiv_ :: Tensor -> Tensor -> Tensor -> IO ()
  cdiv_ = with3Tensors Sig.c_cdiv

  clshift_ :: Tensor -> Tensor -> Tensor -> IO ()
  clshift_ = with3Tensors Sig.c_clshift

  crshift_ :: Tensor -> Tensor -> Tensor -> IO ()
  crshift_ = with3Tensors Sig.c_crshift

  cfmod_ :: Tensor -> Tensor -> Tensor -> IO ()
  cfmod_ = with3Tensors Sig.c_cfmod

  cremainder_ :: Tensor -> Tensor -> Tensor -> IO ()
  cremainder_ = with3Tensors Sig.c_cremainder

  cbitand_ :: Tensor -> Tensor -> Tensor -> IO ()
  cbitand_ = with3Tensors Sig.c_cbitand

  cbitor_ :: Tensor -> Tensor -> Tensor -> IO ()
  cbitor_ = with3Tensors Sig.c_cbitor

  cbitxor_ :: Tensor -> Tensor -> Tensor -> IO ()
  cbitxor_ = with3Tensors Sig.c_cbitxor

  addcmul_ :: Tensor -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  addcmul_ t0 t1 v t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addcmul t0' t1' (hs2cReal v) t2' t3'

  addcdiv_ :: Tensor -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  addcdiv_ t0 t1 v t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addcdiv t0' t1' (hs2cReal v) t2' t3'

  addmv_ :: Tensor -> HsReal -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  addmv_ t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addmv t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  addmm_ :: Tensor -> HsReal -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  addmm_ t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addmm t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  addr_ :: Tensor -> HsReal -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  addr_ t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addr t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  addbmm_ :: Tensor -> HsReal -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  addbmm_ t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_addbmm t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  baddbmm_ :: Tensor -> HsReal -> Tensor -> HsReal -> Tensor -> Tensor -> IO ()
  baddbmm_ t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_baddbmm t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

  match_ :: Tensor -> Tensor -> Tensor -> HsReal -> IO ()
  match_ t0 t1 t2 v0  = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_match t0' t1' t2' (hs2cReal v0)

  numel :: Tensor -> IO Int64
  numel = withTensor (fmap fromIntegral . Sig.c_numel)

  max_ :: (Tensor, LongTensor) -> Tensor -> Int32 -> Int32 -> IO ()
  max_ (t0, ls) t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.tensor ls) $ \ls' ->
       Sig.c_max t0' ls' t1' (CInt i0) (CInt i1)

  min_ :: (Tensor, LongTensor) -> Tensor -> Int32 -> Int32 -> IO ()
  min_ (t0, ls) t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.tensor ls) $ \ls' ->
       Sig.c_min t0' ls' t1' (CInt i0) (CInt i1)

  kthvalue_ :: (Tensor, LongTensor) -> Tensor -> Int64 -> Int32 -> Int32 -> IO ()
  kthvalue_ (t0, ls) t1 pd i0 i1 = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.tensor ls) $ \ls' ->
       Sig.c_kthvalue t0' ls' t1' (CLLong pd) (CInt i0) (CInt i1)

  mode_ :: (Tensor, LongTensor) -> Tensor -> Int32 -> Int32 -> IO ()
  mode_ (t0, ls) t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.tensor ls) $ \ls' ->
       Sig.c_mode t0' ls' t1' (CInt i0) (CInt i1)

  median_ :: (Tensor, LongTensor) -> Tensor -> Int32 -> Int32 -> IO ()
  median_ (t0, ls) t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.tensor ls) $ \ls' ->
      Sig.c_median t0' ls' t1' (CInt i0) (CInt i1)

  sum_ :: Tensor -> Tensor -> Int32 -> Int32 -> IO ()
  sum_ t0 t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_sum t0' t1' (CInt i0) (CInt i1)

  prod_ :: Tensor -> Tensor -> Int32 -> Int32 -> IO ()
  prod_ t0 t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_prod t0' t1' (CInt i0) (CInt i1)

  cumsum_ :: Tensor -> Tensor -> Int32 -> IO ()
  cumsum_ t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_cumsum t0' t1' (CInt i0)

  cumprod_ :: Tensor -> Tensor -> Int32 -> IO ()
  cumprod_ t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_cumprod t0' t1' (CInt i0)

  sign_ :: Tensor -> Tensor -> IO ()
  sign_ = with2Tensors Sig.c_sign

  trace :: Tensor -> IO HsAccReal
  trace = withTensor (fmap c2hsAccReal . Sig.c_trace)

  cross_ :: Tensor -> Tensor -> Tensor -> Int32 -> IO ()
  cross_ t0 t1 t2 i0 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_cross t0' t1' t2' (CInt i0)

  cmax_ :: Tensor -> Tensor -> Tensor -> IO ()
  cmax_ = with3Tensors Sig.c_cmax

  cmin_ :: Tensor -> Tensor -> Tensor -> IO ()
  cmin_ = with3Tensors Sig.c_cmin

  cmaxValue_    :: Tensor -> Tensor -> HsReal -> IO ()
  cmaxValue_ = twoTensorsAndReal Sig.c_cmaxValue

  cminValue_    :: Tensor -> Tensor -> HsReal -> IO ()
  cminValue_ = twoTensorsAndReal Sig.c_cminValue

  zeros_ :: Tensor -> LongStorage -> IO ()
  zeros_ t0 ls = _withTensor t0 $ \t0' ->
    withForeignPtr (L.storage ls) $ \ls' ->
      Sig.c_zeros t0' ls'

  zerosLike_ :: Tensor -> Tensor -> IO ()
  zerosLike_ = with2Tensors Sig.c_zerosLike

  ones_ :: Tensor -> LongStorage -> IO ()
  ones_ t0 ls = _withTensor t0 $ \t0' ->
    withForeignPtr (L.storage ls) $ \ls' ->
      Sig.c_ones t0' ls'

  onesLike_ :: Tensor -> Tensor -> IO ()
  onesLike_ = with2Tensors Sig.c_onesLike

  diag_ :: Tensor -> Tensor -> Int32 -> IO ()
  diag_ t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_diag t0' t1' (CInt i0)

  eye_ :: Tensor -> Int64 -> Int64 -> IO ()
  eye_ t0 l0 l1 = _withTensor t0 $ \t0' -> Sig.c_eye t0' (CLLong l0) (CLLong l1)

  arange_ :: Tensor -> HsAccReal -> HsAccReal -> HsAccReal -> IO ()
  arange_ t0 a0 a1 a2 = _withTensor t0 $ \t0' -> Sig.c_arange t0' (hs2cAccReal a0) (hs2cAccReal a1) (hs2cAccReal a2)

  range_ :: Tensor -> HsAccReal-> HsAccReal-> HsAccReal-> IO ()
  range_ t0 a0 a1 a2 = _withTensor t0 $ \t0' -> Sig.c_range t0' (hs2cAccReal a0) (hs2cAccReal a1) (hs2cAccReal a2)

  randperm_ :: Tensor -> Generator -> Int64 -> IO ()
  randperm_ t0 pg i0 = _withTensor t0 $ \t0' ->
    withForeignPtr (rng pg) $ \pg' ->
      Sig.c_randperm t0' pg' (CLLong i0)

  reshape_ :: Tensor -> Tensor -> LongStorage -> IO ()
  reshape_ t0 t1 ls = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.storage ls) $ \ls' ->
      Sig.c_reshape t0' t1' ls'

  sort_ :: Tensor -> LongTensor -> Tensor -> Int32 -> Int32 -> IO ()
  sort_ t0 ls t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.tensor ls) $ \ls' ->
      Sig.c_sort t0' ls' t1' (CInt i0) (CInt i1)

  topk_ :: Tensor -> LongTensor -> Tensor -> Int64 -> Int32 -> Int32 -> Int32 -> IO ()
  topk_ t0 ls t1 l i0 i1 i2 = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.tensor ls) $ \ls' ->
      Sig.c_topk t0' ls' t1' (CLLong l) (CInt i0) (CInt i1) (CInt i2)

  tril_ :: Tensor -> Tensor -> Int64 -> IO ()
  tril_ t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_tril t0' t1' (CLLong i0)

  triu_ :: Tensor -> Tensor -> Int64 -> IO ()
  triu_ t0 t1 i0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_triu t0' t1' (CLLong i0)

  cat_ :: Tensor -> Tensor -> Tensor -> Int32 -> IO ()
  cat_ t0 t1 t2 i0 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_cat t0' t1' t2' (CInt i0)

  -- catArray     :: Tensor -> [Tensor] -> Int32 -> Int32 -> IO ()

  equal :: Tensor -> Tensor -> IO Int32
  equal = with2Tensors (\t0' t1' -> fmap fromIntegral $ Sig.c_equal t0' t1' )

  ltValue_ :: ByteTensor -> Tensor -> HsReal -> IO ()
  ltValue_ bt t0 v = _withTensor t0 $ \t0' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_ltValue bt' t0' (hs2cReal v)

  leValue_ :: ByteTensor -> Tensor -> HsReal -> IO ()
  leValue_ bt t0 v = _withTensor t0 $ \t0' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_leValue bt' t0' (hs2cReal v)

  gtValue_ :: ByteTensor -> Tensor -> HsReal -> IO ()
  gtValue_ bt t0 v = _withTensor t0 $ \t0' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_gtValue bt' t0' (hs2cReal v)

  geValue_ :: ByteTensor -> Tensor -> HsReal -> IO ()
  geValue_ bt t0 v = _withTensor t0 $ \t0' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_geValue bt' t0' (hs2cReal v)

  neValue_ :: ByteTensor -> Tensor -> HsReal -> IO ()
  neValue_ bt t0 v = _withTensor t0 $ \t0' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_neValue bt' t0' (hs2cReal v)

  eqValue_ :: ByteTensor -> Tensor -> HsReal -> IO ()
  eqValue_ bt t0 v = _withTensor t0 $ \t0' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_eqValue bt' t0' (hs2cReal v)

  ltValueT_ :: Tensor -> Tensor -> HsReal -> IO ()
  ltValueT_ = twoTensorsAndReal Sig.c_ltValueT

  leValueT_ :: Tensor -> Tensor -> HsReal -> IO ()
  leValueT_ = twoTensorsAndReal Sig.c_leValueT

  gtValueT_ :: Tensor -> Tensor -> HsReal -> IO ()
  gtValueT_ = twoTensorsAndReal Sig.c_gtValueT

  geValueT_ :: Tensor -> Tensor -> HsReal -> IO ()
  geValueT_ = twoTensorsAndReal Sig.c_geValueT

  neValueT_ :: Tensor -> Tensor -> HsReal -> IO ()
  neValueT_ = twoTensorsAndReal Sig.c_neValueT

  eqValueT_ :: Tensor -> Tensor -> HsReal -> IO ()
  eqValueT_ = twoTensorsAndReal Sig.c_eqValueT

  ltTensor_ :: ByteTensor -> Tensor -> Tensor -> IO ()
  ltTensor_ bt t0 t1 = _with2Tensors t0 t1 $ \t0' t1' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_ltTensor bt' t0' t1'

  leTensor_ :: ByteTensor -> Tensor -> Tensor -> IO ()
  leTensor_ bt t0 t1 = _with2Tensors t0 t1 $ \t0' t1' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_leTensor bt' t0' t1'

  gtTensor_ :: ByteTensor -> Tensor -> Tensor -> IO ()
  gtTensor_ bt t0 t1 = _with2Tensors t0 t1 $ \t0' t1' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_gtTensor bt' t0' t1'

  geTensor_ :: ByteTensor -> Tensor -> Tensor -> IO ()
  geTensor_ bt t0 t1 = _with2Tensors t0 t1 $ \t0' t1' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_geTensor bt' t0' t1'

  neTensor_ :: ByteTensor -> Tensor -> Tensor -> IO ()
  neTensor_ bt t0 t1 = _with2Tensors t0 t1 $ \t0' t1' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_neTensor bt' t0' t1'

  eqTensor_ :: ByteTensor -> Tensor -> Tensor -> IO ()
  eqTensor_ bt t0 t1 = _with2Tensors t0 t1 $ \t0' t1' -> withForeignPtr (B.tensor bt) $ \bt' -> Sig.c_eqTensor bt' t0' t1'

  ltTensorT_ :: Tensor -> Tensor -> Tensor -> IO ()
  ltTensorT_ = with3Tensors Sig.c_ltTensorT

  leTensorT_ :: Tensor -> Tensor -> Tensor -> IO ()
  leTensorT_ = with3Tensors Sig.c_leTensorT

  gtTensorT_ :: Tensor -> Tensor -> Tensor -> IO ()
  gtTensorT_ = with3Tensors Sig.c_gtTensorT

  geTensorT_ :: Tensor -> Tensor -> Tensor -> IO ()
  geTensorT_ = with3Tensors Sig.c_geTensorT

  neTensorT_ :: Tensor -> Tensor -> Tensor -> IO ()
  neTensorT_ = with3Tensors Sig.c_neTensorT

  eqTensorT_ :: Tensor -> Tensor -> Tensor -> IO ()
  eqTensorT_ = with3Tensors Sig.c_eqTensorT
