{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
module Torch.Core.Tensor.Dynamic
  ( Tensor(..)
  ) where

import Foreign (Ptr, withForeignPtr, newForeignPtr)
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import Torch.Class.Internal (HsReal, HsAccReal, HsStorage)
import THTypes
import qualified Tensor as Sig
import qualified Torch.Class.Tensor as Class

import Torch.Core.Types hiding (HsReal, HsAccReal)
import Torch.Core.Storage (asStorage)

asTensor :: Ptr CTensor -> IO Tensor
asTensor = fmap Tensor . newForeignPtr Sig.p_free

instance Class.IsTensor Tensor where
  -- tensordata :: Tensor -> IO (Ptr (HsReal Tensor))
  -- tensordata t = withForeignPtr (tensor t) Sig.c_data

  clearFlag :: Tensor -> CChar -> IO ()
  clearFlag t cc = withForeignPtr (tensor t) $ \t' -> Sig.c_clearFlag t' cc

  desc :: Tensor -> IO CTHDescBuff
  desc t = withForeignPtr (tensor t) (pure . Sig.c_desc)

  expand :: Tensor -> Tensor -> Ptr CTHLongStorage -> IO ()
  expand t0 t1 ls =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_expand t0' t1' ls

  -- expandNd :: Ptr Tensor -> Ptr Tensor -> CInt -> IO ()
  -- expandNd = undefined

  free :: Tensor -> IO ()
  free t = withForeignPtr (tensor t) Sig.c_free

  freeCopyTo :: Tensor -> Tensor -> IO ()
  freeCopyTo t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_freeCopyTo t0' t1'

  get1d :: Tensor -> CLLong -> IO (HsReal Tensor)
  get1d t d1 = withForeignPtr (tensor t) $ \t' -> pure . c2hsReal $ Sig.c_get1d t' d1

  get2d :: Tensor -> CLLong -> CLLong -> IO (HsReal Tensor)
  get2d t d1 d2 = withForeignPtr (tensor t) $ \t' -> pure . c2hsReal $ Sig.c_get2d t' d1 d2

  get3d :: Tensor -> CLLong -> CLLong -> CLLong -> IO (HsReal Tensor)
  get3d t d1 d2 d3 = withForeignPtr (tensor t) $ \t' -> pure . c2hsReal $ Sig.c_get3d t' d1 d2 d3

  get4d :: Tensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO (HsReal Tensor)
  get4d t d1 d2 d3 d4 = withForeignPtr (tensor t) $ \t' -> pure . c2hsReal $ Sig.c_get4d t' d1 d2 d3 d4

  isContiguous :: Tensor -> IO Bool
  isContiguous t =
    withForeignPtr (tensor t) $ \t' ->
      pure $ 1 == Sig.c_isContiguous t'

  isSameSizeAs :: Tensor -> Tensor -> IO Bool
  isSameSizeAs t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        pure $ 1 == Sig.c_isSetTo t0' t1'

  isSetTo :: Tensor -> Tensor -> IO Bool
  isSetTo t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        pure $ 1 == Sig.c_isSetTo t0' t1'

  isSize :: Tensor -> Ptr CTHLongStorage -> IO Bool
  isSize t ls =
    withForeignPtr (tensor t) $ \t' ->
      pure $ 1 == Sig.c_isSize t' ls

  nDimension :: Tensor -> IO CInt
  nDimension t = withForeignPtr (tensor t) (pure . Sig.c_nDimension)

  nElement :: Tensor -> IO CPtrdiff
  nElement t = withForeignPtr (tensor t) (pure . Sig.c_nElement)

  narrow :: Tensor -> Tensor -> CInt -> CLLong -> CLLong -> IO ()
  narrow t0 t1 a b c =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_narrow t0' t1' a b c

  new :: IO Tensor
  new = Sig.c_new >>= asTensor

  newClone :: Tensor -> IO Tensor
  newClone t = withForeignPtr (tensor t) Sig.c_newClone >>= asTensor

  newContiguous :: Tensor -> IO Tensor
  newContiguous t =
    withForeignPtr (tensor t) Sig.c_newContiguous >>= asTensor

  newExpand :: Tensor -> Ptr CTHLongStorage -> IO Tensor
  newExpand t ls =
    withForeignPtr (tensor t) (\t' -> Sig.c_newExpand t' ls) >>= asTensor

  newNarrow :: Tensor -> CInt -> CLLong -> CLLong -> IO Tensor
  newNarrow t a b c =
    withForeignPtr (tensor t) (\t' -> Sig.c_newNarrow t' a b c) >>= asTensor

  newSelect :: Tensor -> CInt -> CLLong -> IO Tensor
  newSelect t a b =
    withForeignPtr (tensor t) (\t' -> Sig.c_newSelect t' a b) >>= asTensor

  newSizeOf :: Tensor -> IO (Ptr CTHLongStorage)
  newSizeOf t = withForeignPtr (tensor t) Sig.c_newSizeOf

  newStrideOf :: Tensor -> IO (Ptr CTHLongStorage)
  newStrideOf t = withForeignPtr (tensor t) Sig.c_newStrideOf

  newTranspose :: Tensor -> CInt -> CInt -> IO Tensor
  newTranspose t a b =
    withForeignPtr (tensor t) (\t' -> Sig.c_newTranspose t' a b) >>= asTensor

  newUnfold :: Tensor -> CInt -> CLLong -> CLLong -> IO Tensor
  newUnfold t a b c =
    withForeignPtr (tensor t) (\t' -> Sig.c_newUnfold t' a b c) >>= asTensor

  newView :: Tensor -> Ptr CTHLongStorage -> IO Tensor
  newView t ls =
    withForeignPtr (tensor t) (\t' -> Sig.c_newView t' ls) >>= asTensor

  newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO Tensor
  newWithSize ls0 ls1 = Sig.c_newWithSize ls0 ls1 >>= asTensor

  newWithSize1d :: CLLong -> IO Tensor
  newWithSize1d a0 = Sig.c_newWithSize1d a0 >>= asTensor

  newWithSize2d :: CLLong -> CLLong -> IO Tensor
  newWithSize2d a0 a1 = Sig.c_newWithSize2d a0 a1 >>= asTensor

  newWithSize3d :: CLLong -> CLLong -> CLLong -> IO Tensor
  newWithSize3d a0 a1 a2 = Sig.c_newWithSize3d a0 a1 a2 >>= asTensor

  newWithSize4d :: CLLong -> CLLong -> CLLong -> CLLong -> IO Tensor
  newWithSize4d a0 a1 a2 a3 = Sig.c_newWithSize4d a0 a1 a2 a3 >>= asTensor

  newWithStorage :: HsStorage Tensor -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO Tensor
  newWithStorage s pd ls ls0 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage s' pd ls ls0)
      >>= asTensor

  newWithStorage1d :: HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> IO Tensor
  newWithStorage1d s pd l00 l01 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage1d s' pd l00 l01)
      >>= asTensor


  newWithStorage2d :: HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO Tensor
  newWithStorage2d s pd d00 d01 d10 d11 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage2d s' pd d00 d01 d10 d11)
      >>= asTensor


  newWithStorage3d :: HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO Tensor
  newWithStorage3d s pd d00 d01 d10 d11 d20 d21 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage3d s' pd d00 d01 d10 d11 d20 d21)
      >>= asTensor


  newWithStorage4d :: HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO Tensor
  newWithStorage4d s pd d00 d01 d10 d11 d20 d21 d30 d31 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage4d s' pd d00 d01 d10 d11 d20 d21 d30 d31)
      >>= asTensor

  newWithTensor :: Tensor -> IO Tensor
  newWithTensor t = withForeignPtr (tensor t) Sig.c_newWithTensor >>= asTensor

  resize :: Tensor -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  resize t l0 l1 = withForeignPtr (tensor t) (\t' -> Sig.c_resize t' l0 l1)

  resize1d :: Tensor -> CLLong -> IO ()
  resize1d t l0 = withForeignPtr (tensor t) (\t' -> Sig.c_resize1d t' l0)

  resize2d :: Tensor -> CLLong -> CLLong -> IO ()
  resize2d t l0 l1 = withForeignPtr (tensor t) (\t' -> Sig.c_resize2d t' l0 l1)

  resize3d :: Tensor -> CLLong -> CLLong -> CLLong -> IO ()
  resize3d t l0 l1 l2 = withForeignPtr (tensor t) (\t' -> Sig.c_resize3d t' l0 l1 l2)

  resize4d :: Tensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  resize4d t l0 l1 l2 l3 = withForeignPtr (tensor t) (\t' -> Sig.c_resize4d t' l0 l1 l2 l3)

  resize5d :: Tensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  resize5d t l0 l1 l2 l3 l4 = withForeignPtr (tensor t) (\t' -> Sig.c_resize5d t' l0 l1 l2 l3 l4)

  resizeAs :: Tensor -> Tensor -> IO ()
  resizeAs t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_resizeAs t0' t1'

  resizeNd :: Tensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
  resizeNd t i l0 l1 = withForeignPtr (tensor t) (\t' -> Sig.c_resizeNd t' i l0 l1)

  retain :: Tensor -> IO ()
  retain t = withForeignPtr (tensor t) Sig.c_retain

  select :: Tensor -> Tensor -> CInt -> CLLong -> IO ()
  select t0 t1 a b =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_select t0' t1' a b

  set :: Tensor -> Tensor -> IO ()
  set t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_set t0' t1'

  set1d :: Tensor -> CLLong -> HsReal Tensor -> IO ()
  set1d t l0 v = withForeignPtr (tensor t) (\t' -> Sig.c_set1d t' l0 (hs2cReal v))

  set2d :: Tensor -> CLLong -> CLLong -> HsReal Tensor -> IO ()
  set2d t l0 l1 v = withForeignPtr (tensor t) (\t' -> Sig.c_set2d t' l0 l1 (hs2cReal v))

  set3d :: Tensor -> CLLong -> CLLong -> CLLong -> HsReal Tensor -> IO ()
  set3d t l0 l1 l2 v = withForeignPtr (tensor t) (\t' -> Sig.c_set3d t' l0 l1 l2 (hs2cReal v))

  set4d :: Tensor -> CLLong -> CLLong -> CLLong -> CLLong -> HsReal Tensor -> IO ()
  set4d t l0 l1 l2 l3 v = withForeignPtr (tensor t) (\t' -> Sig.c_set4d t' l0 l1 l2 l3 (hs2cReal v))

  setFlag :: Tensor -> CChar -> IO ()
  setFlag t l0 = withForeignPtr (tensor t) (\t' -> Sig.c_setFlag t' l0)

  setStorage :: Tensor -> HsStorage Tensor -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  setStorage t s a b c =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage t' s' a b c


  setStorage1d :: Tensor -> HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> IO ()
  setStorage1d t s pd d00 d01 =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage1d t' s' pd d00 d01


  setStorage2d :: Tensor -> HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorage2d t s pd d00 d01 d10 d11 =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage2d t' s' pd d00 d01 d10 d11


  setStorage3d :: Tensor -> HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorage3d t s pd d00 d01 d10 d11 d20 d21 =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage3d t' s' pd d00 d01 d10 d11 d20 d21


  setStorage4d :: Tensor -> HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorage4d t s pd d00 d01 d10 d11 d20 d21 d30 d31  =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage4d t' s' pd d00 d01 d10 d11 d20 d21 d30 d31


  setStorageNd :: Tensor -> HsStorage Tensor -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
  setStorageNd t s a b c d =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorageNd t' s' a b c d


  size :: Tensor -> CInt -> IO CLLong
  size t l0 = withForeignPtr (tensor t) (\t' -> pure $ Sig.c_size t' l0)

  sizeDesc :: Tensor -> IO CTHDescBuff
  sizeDesc t = withForeignPtr (tensor t) (pure . Sig.c_sizeDesc)

  squeeze :: Tensor -> Tensor -> IO ()
  squeeze t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_squeeze t0' t1'

  squeeze1d :: Tensor -> Tensor -> CInt -> IO ()
  squeeze1d t0 t1 d =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_squeeze1d t0' t1' d

  storage :: Tensor -> IO (HsStorage Tensor)
  storage t = withForeignPtr (tensor t) Sig.c_storage >>= asStorage

  storageOffset :: Tensor -> IO CPtrdiff
  storageOffset t = withForeignPtr (tensor t) (pure . Sig.c_storageOffset)

  stride :: Tensor -> CInt -> IO CLLong
  stride t a = withForeignPtr (tensor t) (\t' -> pure $ Sig.c_stride t' a)

  transpose :: Tensor -> Tensor -> CInt -> CInt -> IO ()
  transpose t0 t1 a b =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_transpose t0' t1' a b

  unfold :: Tensor -> Tensor -> CInt -> CLLong -> CLLong -> IO ()
  unfold t0 t1 a b c =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_unfold t0' t1' a b c

  unsqueeze1d :: Tensor -> Tensor -> CInt -> IO ()
  unsqueeze1d t0 t1 d =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_unsqueeze1d t0' t1' d

