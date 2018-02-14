{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Dynamic.IsTensor
  ( asTensor
  , Class.IsTensor(..)
  ) where

import Data.Coerce (coerce)
import Foreign (Ptr, withForeignPtr, newForeignPtr, Storable(peek))
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import GHC.Int
import THTypes
import Control.Monad.Managed

import qualified THLongTypes as Long
import qualified THLongStorage as Long
import qualified Foreign.Marshal.Array as FM
import qualified Tensor as Sig
import qualified Storage as StorageSig (c_size)
import qualified Torch.Class.C.IsTensor as Class

import Torch.Core.Types
import Torch.Core.Storage (asStorageM)

asTensor :: Ptr CTensor -> IO Tensor
asTensor = fmap asDyn . newForeignPtr Sig.p_free

instance Class.IsTensor Tensor where
  clearFlag_ :: Tensor -> Int8 -> IO ()
  clearFlag_ t cc = withForeignPtr (tensor t) $ \t' -> Sig.c_clearFlag t' (CChar cc)

  tensordata :: Tensor -> IO [HsReal]
  tensordata = ptrArray2hs Sig.c_data arrayLen . tensor
   where
    arrayLen :: Ptr CTensor -> IO Int
    arrayLen p = Sig.c_storage p >>= StorageSig.c_size >>= pure . fromIntegral

  desc :: Tensor -> IO CTHDescBuff
  desc t = withForeignPtr (tensor t) Sig.c_desc

  expand_ :: Tensor -> Tensor -> Long.Storage -> IO ()
  expand_ t0 t1 ls =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        withForeignPtr (Long.storage ls) $ \ls' ->
          Sig.c_expand t0' t1' ls'

  -- | https://github.com/torch/torch7/blob/2186e414ad8fc4dfc9f2ed090c0bf8a0e1946e62/lib/TH/generic/THTensor.c#L319
  expandNd_ :: [Tensor] -> [Tensor] -> Int32 -> IO ()
  expandNd_ rets' ops' count = do
    rets <- ptrPtrTensors rets'
    ops  <- ptrPtrTensors ops'
    Sig.c_expandNd rets ops (CInt count)
   where
    ptrPtrTensors :: [Tensor] -> IO (Ptr (Ptr CTensor))
    ptrPtrTensors ts
      = mapM (`withForeignPtr` pure) (fmap tensor ts)
      >>= FM.newArray

  free_ :: Tensor -> IO ()
  free_ t = withForeignPtr (tensor t) Sig.c_free

  freeCopyTo_ :: Tensor -> Tensor -> IO ()
  freeCopyTo_ t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_freeCopyTo t0' t1'

  get1d :: Tensor -> Int64 -> IO HsReal
  get1d t d1 = withForeignPtr (tensor t) $ \t' -> c2hsReal <$> Sig.c_get1d t' (CLLong d1)

  get2d :: Tensor -> Int64 -> Int64 -> IO HsReal
  get2d t d1 d2 = withForeignPtr (tensor t) $ \t' -> c2hsReal <$> Sig.c_get2d t' (CLLong d1) (CLLong d2)

  get3d :: Tensor -> Int64 -> Int64 -> Int64 -> IO HsReal
  get3d t d1 d2 d3 = withForeignPtr (tensor t) $ \t' -> c2hsReal <$> Sig.c_get3d t' (CLLong d1) (CLLong d2) (CLLong d3)

  get4d :: Tensor -> Int64 -> Int64 -> Int64 -> Int64 -> IO HsReal
  get4d t d1 d2 d3 d4 = withForeignPtr (tensor t) $ \t' -> c2hsReal <$> Sig.c_get4d t' (CLLong d1) (CLLong d2) (CLLong d3) (CLLong d4)

  isContiguous :: Tensor -> IO Bool
  isContiguous t =
    withForeignPtr (tensor t) $ \t' ->
      (1 ==) <$> Sig.c_isContiguous t'

  isSameSizeAs :: Tensor -> Tensor -> IO Bool
  isSameSizeAs t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        (1 ==) <$> Sig.c_isSetTo t0' t1'

  isSetTo :: Tensor -> Tensor -> IO Bool
  isSetTo t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        (1 ==) <$> Sig.c_isSetTo t0' t1'

  isSize :: Tensor -> Long.Storage -> IO Bool
  isSize t ls =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (Long.storage ls) $ \ls' ->
        (1 ==) <$> Sig.c_isSize t' ls'

  nDimension :: Tensor -> IO Int32
  nDimension t = withForeignPtr (tensor t) (fmap fromIntegral . Sig.c_nDimension)

  nElement :: Tensor -> IO Int64
  nElement t = withForeignPtr (tensor t) (fmap fromIntegral . Sig.c_nElement)

  narrow_ :: Tensor -> Tensor -> Int32 -> Int64 -> Int64 -> IO ()
  narrow_ t0 t1 a b c =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_narrow t0' t1' (CInt a) (CLLong b) (CLLong c)

  empty :: IO Tensor
  empty = Sig.c_new >>= asTensor

  newClone :: Tensor -> IO Tensor
  newClone t = withForeignPtr (tensor t) Sig.c_newClone >>= asTensor

  newContiguous :: Tensor -> IO Tensor
  newContiguous t =
    withForeignPtr (tensor t) Sig.c_newContiguous >>= asTensor

  newExpand :: Tensor -> Long.Storage -> IO Tensor
  newExpand t ls =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (Long.storage ls) $ \ls' ->
        Sig.c_newExpand t' ls' >>= asTensor

  newNarrow :: Tensor -> Int32 -> Int64 -> Int64 -> IO Tensor
  newNarrow t a b c =
    withForeignPtr (tensor t) (\t' -> Sig.c_newNarrow t' (CInt a) (CLLong b) (CLLong c)) >>= asTensor

  newSelect :: Tensor -> Int32 -> Int64 -> IO Tensor
  newSelect t a b =
    withForeignPtr (tensor t) (\t' -> Sig.c_newSelect t' (CInt a) (CLLong b)) >>= asTensor

  newSizeOf :: Tensor -> IO (Long.Storage)
  newSizeOf t = withForeignPtr (tensor t) Sig.c_newSizeOf >>= fmap Long.Storage . newForeignPtr Long.p_free


  newStrideOf :: Tensor -> IO (Long.Storage)
  newStrideOf t = withForeignPtr (tensor t) Sig.c_newStrideOf >>= fmap Long.Storage . newForeignPtr Long.p_free

  newTranspose :: Tensor -> Int32 -> Int32 -> IO Tensor
  newTranspose t a b =
    withForeignPtr (tensor t) (\t' -> Sig.c_newTranspose t' (CInt a) (CInt b)) >>= asTensor

  newUnfold :: Tensor -> Int32 -> Int64 -> Int64 -> IO Tensor
  newUnfold t a b c =
    withForeignPtr (tensor t) (\t' -> Sig.c_newUnfold t' (CInt a) (CLLong b) (CLLong c)) >>= asTensor

  newView :: Tensor -> Long.Storage -> IO Tensor
  newView t ls =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (Long.storage ls) $ \ls' ->
        Sig.c_newView t' ls' >>= asTensor

  newWithSize :: Long.Storage -> Long.Storage -> IO Tensor
  newWithSize l0 l1 =
    withForeignPtr (Long.storage l0) $ \l0' ->
      withForeignPtr (Long.storage l1) $ \l1' ->
        Sig.c_newWithSize l0' l1' >>= asTensor

  newWithSize1d :: Int64 -> IO Tensor
  newWithSize1d a0 = Sig.c_newWithSize1d (CLLong a0) >>= asTensor

  newWithSize2d :: Int64 -> Int64 -> IO Tensor
  newWithSize2d a0 a1 = Sig.c_newWithSize2d (CLLong a0) (CLLong a1) >>= asTensor

  newWithSize3d :: Int64 -> Int64 -> Int64 -> IO Tensor
  newWithSize3d a0 a1 a2 = Sig.c_newWithSize3d (CLLong a0) (CLLong a1) (CLLong a2) >>= asTensor

  newWithSize4d :: Int64 -> Int64 -> Int64 -> Int64 -> IO Tensor
  newWithSize4d a0 a1 a2 a3 = Sig.c_newWithSize4d (CLLong a0) (CLLong a1) (CLLong a2) (CLLong a3) >>= asTensor

  newWithStorage :: Storage -> Int64 -> Long.Storage -> Long.Storage -> IO Tensor
  newWithStorage s pd l0 l1 =
    withForeignPtr (storage s) $ \s' ->
      withForeignPtr (Long.storage l0) $ \l0' ->
        withForeignPtr (Long.storage l1) $ \l1' ->
          Sig.c_newWithStorage s' (CPtrdiff pd) l0' l1' >>= asTensor

  newWithStorage1d :: Storage -> Int64 -> Int64 -> Int64 -> IO Tensor
  newWithStorage1d s pd d00 d01 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage1d s' (CPtrdiff pd)
      (CLLong d00) (CLLong d01)
    ) >>= asTensor


  newWithStorage2d :: Storage -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO Tensor
  newWithStorage2d s pd d00 d01 d10 d11 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage2d s' (CPtrdiff pd)
      (CLLong d00) (CLLong d01)
      (CLLong d10) (CLLong d11)
    ) >>= asTensor


  newWithStorage3d :: Storage -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO Tensor
  newWithStorage3d s pd d00 d01 d10 d11 d20 d21 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage3d s' (CPtrdiff pd)
      (CLLong d00) (CLLong d01)
      (CLLong d10) (CLLong d11)
      (CLLong d20) (CLLong d21)
    ) >>= asTensor


  newWithStorage4d :: Storage -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO Tensor
  newWithStorage4d s pd d00 d01 d10 d11 d20 d21 d30 d31 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage4d s' (CPtrdiff pd)
      (CLLong d00) (CLLong d01)
      (CLLong d10) (CLLong d11)
      (CLLong d20) (CLLong d21)
      (CLLong d30) (CLLong d31)
    ) >>= asTensor

  newWithTensor :: Tensor -> IO Tensor
  newWithTensor t = withForeignPtr (tensor t) Sig.c_newWithTensor >>= asTensor

  resize_ :: Tensor -> Long.Storage -> Long.Storage -> IO ()
  resize_ t l0 l1 = runManaged $ do
    t' <- managed $ withForeignPtr (tensor t)
    l0' <- managed $ withForeignPtr (Long.storage l0)
    l1' <- managed $ withForeignPtr (Long.storage l1)
    liftIO $ Sig.c_resize t' l0' l1'

  resize1d_ :: Tensor -> Int64 -> IO ()
  resize1d_ t l0 = withForeignPtr (tensor t) (\t' -> Sig.c_resize1d t' (CLLong l0))

  resize2d_ :: Tensor -> Int64 -> Int64 -> IO ()
  resize2d_ t l0 l1 = withForeignPtr (tensor t) $ \t' -> Sig.c_resize2d t'
      (CLLong l0) (CLLong l1)

  resize3d_ :: Tensor -> Int64 -> Int64 -> Int64 -> IO ()
  resize3d_ t l0 l1 l2 = withForeignPtr (tensor t) $ \t' -> Sig.c_resize3d t'
      (CLLong l0) (CLLong l1) (CLLong l2)

  resize4d_ :: Tensor -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize4d_ t l0 l1 l2 l3 = withForeignPtr (tensor t) $ \t' -> Sig.c_resize4d t'
      (CLLong l0) (CLLong l1) (CLLong l2) (CLLong l3)

  resize5d_ :: Tensor -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize5d_ t l0 l1 l2 l3 l4 = withForeignPtr (tensor t) $ \t' -> Sig.c_resize5d t'
      (CLLong l0) (CLLong l1) (CLLong l2) (CLLong l3) (CLLong l4)

  resizeAs_ :: Tensor -> Tensor -> IO ()
  resizeAs_ t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_resizeAs t0' t1'

  resizeNd_ :: Tensor -> Int32 -> [Int64] -> [Int64] -> IO ()
  resizeNd_ t i l0' l1' = do
    l0 <- FM.newArray (coerce l0' :: [CLLong])
    l1 <- FM.newArray (coerce l1' :: [CLLong])
    withForeignPtr (tensor t) $ \t' -> Sig.c_resizeNd t' (CInt i) l0 l1

  retain :: Tensor -> IO ()
  retain t = withForeignPtr (tensor t) Sig.c_retain

  select_ :: Tensor -> Tensor -> Int32 -> Int64 -> IO ()
  select_ t0 t1 a b =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_select t0' t1' (CInt a) (CLLong b)

  set_ :: Tensor -> Tensor -> IO ()
  set_ t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_set t0' t1'

  set1d_ :: Tensor -> Int64 -> HsReal -> IO ()
  set1d_ t l0 v = withForeignPtr (tensor t) (\t' -> Sig.c_set1d t' (CLLong l0) (hs2cReal v))

  set2d_ :: Tensor -> Int64 -> Int64 -> HsReal -> IO ()
  set2d_ t l0 l1 v = withForeignPtr (tensor t) (\t' -> Sig.c_set2d t' (CLLong l0) (CLLong l1) (hs2cReal v))

  set3d_ :: Tensor -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
  set3d_ t l0 l1 l2 v = withForeignPtr (tensor t) (\t' -> Sig.c_set3d t' (CLLong l0) (CLLong l1) (CLLong l2) (hs2cReal v))

  set4d_ :: Tensor -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
  set4d_ t l0 l1 l2 l3 v = withForeignPtr (tensor t) (\t' -> Sig.c_set4d t' (CLLong l0) (CLLong l1) (CLLong l2) (CLLong l3) (hs2cReal v))

  setFlag_ :: Tensor -> Int8 -> IO ()
  setFlag_ t l0 = withForeignPtr (tensor t) (\t' -> Sig.c_setFlag t' (CChar l0))

  setStorage_ :: Tensor -> Storage -> Int64 -> Long.Storage -> Long.Storage -> IO ()
  setStorage_ t s a b c = runManaged $ do
    t' <- managed $ withForeignPtr (tensor t)
    s' <- managed $ withForeignPtr (storage s)
    b' <- managed $ withForeignPtr (Long.storage b)
    c' <- managed $ withForeignPtr (Long.storage c)
    liftIO $ Sig.c_setStorage t' s' (CPtrdiff a) b' c'


  setStorage1d_ :: Tensor -> Storage -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage1d_ t s pd d00 d01 =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage1d t' s' (CPtrdiff pd)
          (CLLong d00) (CLLong d01)

  setStorage2d_ :: Tensor -> Storage -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage2d_ t s pd d00 d01 d10 d11 =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage2d t' s' (CPtrdiff pd)
          (CLLong d00) (CLLong d01)
          (CLLong d10) (CLLong d11)


  setStorage3d_ :: Tensor -> Storage -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage3d_ t s pd d00 d01 d10 d11 d20 d21 =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage3d t' s' (CPtrdiff pd)
          (CLLong d00) (CLLong d01)
          (CLLong d10) (CLLong d11)
          (CLLong d20) (CLLong d21)

  setStorage4d_ :: Tensor -> Storage -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage4d_ t s pd d00 d01 d10 d11 d20 d21 d30 d31  =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage4d t' s' (CPtrdiff pd)
          (CLLong d00) (CLLong d01)
          (CLLong d10) (CLLong d11)
          (CLLong d20) (CLLong d21)
          (CLLong d30) (CLLong d31)

  setStorageNd_ :: Tensor -> Storage -> Int64 -> Int32 -> [Int64] -> [Int64] -> IO ()
  setStorageNd_ t s a b hsc hsd = do
    c <- FM.newArray (coerce hsc :: [CLLong])
    d <- FM.newArray (coerce hsd :: [CLLong])
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorageNd t' s' (CPtrdiff a) (CInt b) c d


  size :: Tensor -> Int32 -> IO Int64
  size t l0 = withForeignPtr (tensor t) (\t' -> fromIntegral <$> Sig.c_size t' (CInt l0))

  sizeDesc :: Tensor -> IO CTHDescBuff
  sizeDesc t = withForeignPtr (tensor t) (Sig.c_sizeDesc)

  squeeze_ :: Tensor -> Tensor -> IO ()
  squeeze_ t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_squeeze t0' t1'

  squeeze1d_ :: Tensor -> Tensor -> Int32 -> IO ()
  squeeze1d_ t0 t1 d =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_squeeze1d t0' t1' (CInt d)

  storage :: Tensor -> IO Storage
  storage t = withForeignPtr (tensor t) Sig.c_storage >>= asStorageM

  storageOffset :: Tensor -> IO Int64
  storageOffset t = withForeignPtr (tensor t) (fmap fromIntegral . Sig.c_storageOffset)

  stride :: Tensor -> Int32 -> IO Int64
  stride t a = withForeignPtr (tensor t) (\t' -> fmap fromIntegral $ Sig.c_stride t' (CInt a))

  transpose_ :: Tensor -> Tensor -> Int32 -> Int32 -> IO ()
  transpose_ t0 t1 a b =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_transpose t0' t1' (CInt a) (CInt b)

  unfold_ :: Tensor -> Tensor -> Int32 -> Int64 -> Int64 -> IO ()
  unfold_ t0 t1 a b c =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_unfold t0' t1' (CInt a) (CLLong b) (CLLong c)

  unsqueeze1d_ :: Tensor -> Tensor -> Int32 -> IO ()
  unsqueeze1d_ t0 t1 d =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_unsqueeze1d t0' t1' (CInt d)

