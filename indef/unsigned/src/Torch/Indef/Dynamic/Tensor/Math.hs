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

  clamp_ :: Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
  clamp_ = twoTensorsAndTwoReals Sig.c_clamp

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

  kthvalue_ :: (Tensor, LongTensor) -> Dynamic -> Int64 -> Int32 -> Int32 -> IO ()
  kthvalue_ (t0, ls) t1 pd i0 i1 = _with2Tensors t0 t1 $ \t0' t1' ->
    withForeignPtr (L.tensor ls) $ \ls' ->
       Sig.c_kthvalue t0' ls' t1' (CLLong pd) (CInt i0) (CInt i1)

  sign_ :: Dynamic -> Dynamic -> IO ()
  sign_ = with2Tensors Sig.c_sign

  trace :: Dynamic -> IO HsAccReal
  trace = withTensor (fmap c2hsAccReal . Sig.c_trace)

  cross_ :: Dynamic -> Dynamic -> Dynamic -> Int32 -> IO ()
  cross_ t0 t1 t2 i0 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_cross t0' t1' t2' (CInt i0)

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

