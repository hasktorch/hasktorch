{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Indef.Dynamic.Tensor where

import Data.Coerce (coerce)
-- import Foreign (Ptr, withForeignPtr, newForeignPtr, Storable(peek))
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import GHC.Int
import Torch.Types.TH hiding (CState, IndexStorage)
import Control.Monad.Managed
import Torch.Dimensions
import Torch.Class.Types (Stride(..), Size(..), StorageOffset(..), Step(..), SizesStorage, StridesStorage)

import Torch.Sig.Types
import qualified Foreign.Marshal.Array     as FM
import qualified Torch.Sig.Types           as Sig
import qualified Torch.Sig.Types.Global    as Sig
import qualified Torch.Sig.Tensor          as Sig
import qualified Torch.Sig.Tensor.Memory   as Sig
import qualified Torch.Sig.Storage         as StorageSig (c_size)
import qualified Torch.Class.Tensor        as Class

import qualified Torch.Types.TH.Long       as Long
import qualified Torch.FFI.TH.Long.Storage as Long

import Torch.Indef.Types

instance Class.Tensor Dynamic where
  clearFlag_ :: Dynamic -> Int8 -> IO ()
  clearFlag_ t cc = withDynamicState t $ \s' t' ->
    Sig.c_clearFlag s' t' (CChar cc)

  tensordata :: Dynamic -> IO [HsReal]
  tensordata t = withDynamicState t $ \s' t' ->
    ptrArray2hs (Sig.c_data s') (arrayLen s') (Sig.ctensor t)
   where
    arrayLen :: Ptr CState -> Ptr CTensor -> IO Int
    arrayLen s' p = Sig.c_storage s' p >>= StorageSig.c_size s' >>= pure . fromIntegral

  free_ :: Dynamic -> IO ()
  free_ t = withDynamicState t Sig.c_free

  freeCopyTo_ :: Dynamic -> Dynamic -> IO ()
  freeCopyTo_ t0 t1 = withDynamicState t0 $ \s t0' ->
    withForeignPtr (ctensor t1) $ \t1' ->
      Sig.c_freeCopyTo s t0' t1'

{-
  get1d :: Dynamic -> Int64 -> IO HsReal
  get1d t d1 = withDynamicState t $ \s t' -> c2hsReal <$> Sig.c_get1d s t' (fromIntegral d1)

  get2d :: Dynamic -> Int64 -> Int64 -> IO HsReal
  get2d t d1 d2 = withDynamicState t $ \s t' -> c2hsReal <$> Sig.c_get2d s t' (fromIntegral d1) (fromIntegral d2)

  get3d :: Dynamic -> Int64 -> Int64 -> Int64 -> IO HsReal
  get3d t d1 d2 d3 = withDynamicState t $ \s t' -> c2hsReal <$> Sig.c_get3d s t' (fromIntegral d1) (fromIntegral d2) (fromIntegral d3)

  get4d :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> IO HsReal
  get4d t d1 d2 d3 d4 = withDynamicState t $ \s t' -> c2hsReal <$> Sig.c_get4d s t' (fromIntegral d1) (fromIntegral d2) (fromIntegral d3) (fromIntegral d4)

  isContiguous :: Dynamic -> IO Bool
  isContiguous t =
    withDynamicState t $ \s t' ->
      (1 ==) <$> Sig.c_isContiguous s t'

  isSameSizeAs :: Dynamic -> Dynamic -> IO Bool
  isSameSizeAs t0 t1 = withState $ \s ->
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        (1 ==) <$> Sig.c_isSetTo s t0' t1'

  isSetTo :: Dynamic -> Dynamic -> IO Bool
  isSetTo t0 t1 = withState $ \s ->
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        (1 ==) <$> Sig.c_isSetTo s t0' t1'

  isSize :: Dynamic -> Sig.IndexStorage -> IO Bool
  isSize t ls = withDynamicState $ \s t' ->
    withForeignPtr (Sig.cstorage ls) $ \ls' ->
      (1 ==) <$> Sig.c_isSize s t' ls'

  nDimension :: Dynamic -> IO Int32
  nDimension t = withDynamicState t (\s t' -> fromIntegral <$> Sig.c_nDimension s t')

  nElement :: Dynamic -> IO Int64
  nElement t = withDynamicState t (\s t' -> fmap fromIntegral $ Sig.c_nElement s t')

  narrow_ :: Dynamic -> Dynamic -> DimVal -> Int64 -> Size -> IO ()
  narrow_ t0 t1 a b c = withDynamicState t0 $ \s t0' ->
    withForeignPtr (ctensor t1) $ \t1' ->
      Sig.c_narrow s t0' t1' (fromIntegral a) (fromIntegral b) (fromIntegral c)

  empty :: IO Dynamic
  empty = Sig.newCState >>= \s -> Sig.c_new s >>= mkDynamic s

  newClone :: Dynamic -> IO Dynamic
  newClone t = withDynamicState t Sig.c_newClone >>= mkDynamicIO

  newContiguous :: Dynamic -> IO Dynamic
  newContiguous t =
    withDynamicState t Sig.c_newContiguous >>= mkDynamicIO

  -- newExpand :: Dynamic -> Long.Storage -> IO Dynamic
  -- newExpand t ls =
  --   withDynamicState t $ \t' ->
  --     withDynamicState (Long.storage ls) $ \ls' ->
  --       Sig.c_newExpand t' ls' >>= mkDynamicIO

  newNarrow :: Dynamic -> DimVal -> Int64 -> Size -> IO Dynamic
  newNarrow t a b c =
    withDynamicState t (\t' -> Sig.c_newNarrow t' (fromIntegral a) (fromIntegral b) (fromIntegral c)) >>= mkDynamicIO

  newSelect :: Dynamic -> DimVal -> Int64 -> IO Dynamic
  newSelect t a b =
    withDynamicState t (\t' -> Sig.c_newSelect t' (fromIntegral a) (fromIntegral b)) >>= mkDynamicIO

  newSizeOf :: Dynamic -> IO (Long.Storage)
  newSizeOf t = withDynamicState t Sig.c_newSizeOf >>= fmap Long.asStorage . newForeignPtr Long.p_free


  newStrideOf :: Dynamic -> IO (Long.Storage)
  newStrideOf t = withDynamicState t Sig.c_newStrideOf >>= fmap Long.asStorage . newForeignPtr Long.p_free

  newTranspose :: Dynamic -> DimVal -> DimVal -> IO Dynamic
  newTranspose t a b =
    withDynamicState t (\t' -> Sig.c_newTranspose t' (fromIntegral a) (fromIntegral b)) >>= mkDynamicIO

  newUnfold :: Dynamic -> DimVal -> Int64 -> Int64 -> IO Dynamic
  newUnfold t a b c =
    withDynamicState t (\t' -> Sig.c_newUnfold t' (fromIntegral a) (fromIntegral b) (fromIntegral c)) >>= mkDynamicIO

  newView :: Dynamic -> Long.Storage -> IO Dynamic
  newView t ls =
    withForeignPtr (ctensor t) $ \t' ->
      withForeignPtr (Long.storage ls) $ \ls' ->
        Sig.c_newView t' ls' >>= mkDynamicIO

  newWithSize :: Long.Storage -> Long.Storage -> IO Dynamic
  newWithSize l0 l1 =
    withForeignPtr (Long.storage l0) $ \l0' ->
      withForeignPtr (Long.storage l1) $ \l1' ->
        Sig.c_newWithSize l0' l1' >>= mkDynamicIO

  newWithSize1d :: Size -> IO Dynamic
  newWithSize1d a0 = Sig.c_newWithSize1d (fromIntegral a0) >>= mkDynamicIO

  newWithSize2d :: Size -> Size -> IO Dynamic
  newWithSize2d a0 a1 = Sig.c_newWithSize2d (fromIntegral a0) (fromIntegral a1) >>= mkDynamicIO

  newWithSize3d :: Size -> Size -> Size -> IO Dynamic
  newWithSize3d a0 a1 a2 = Sig.c_newWithSize3d (fromIntegral a0) (fromIntegral a1) (fromIntegral a2) >>= mkDynamicIO

  newWithSize4d :: Size -> Size -> Size -> Size -> IO Dynamic
  newWithSize4d a0 a1 a2 a3 = Sig.c_newWithSize4d (fromIntegral a0) (fromIntegral a1) (fromIntegral a2) (fromIntegral a3) >>= mkDynamicIO

  newWithStorage :: Storage -> StorageOffset -> Long.Storage -> Long.Storage -> IO Dynamic
  newWithStorage s pd l0 l1 =
    withForeignPtr (storage s) $ \s' ->
      withForeignPtr (Long.storage l0) $ \l0' ->
        withForeignPtr (Long.storage l1) $ \l1' ->
          Sig.c_newWithStorage s' (fromIntegral pd) l0' l1' >>= mkDynamicIO

  newWithStorage1d :: Storage -> StorageOffset -> (Size, Stride) -> IO Dynamic
  newWithStorage1d s pd (d00,d01) =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage1d s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
    ) >>= mkDynamicIO


  newWithStorage2d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO Dynamic
  newWithStorage2d s pd (d00,d01) (d10,d11) =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage2d s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)
    ) >>= mkDynamicIO


  newWithStorage3d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO Dynamic
  newWithStorage3d s pd (d00,d01) (d10,d11) (d20,d21) =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage3d s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)
      (fromIntegral d20) (fromIntegral d21)
    ) >>= mkDynamicIO


  newWithStorage4d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO Dynamic
  newWithStorage4d s pd (d00,d01) (d10,d11) (d20,d21) (d30,d31) =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage4d s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)
      (fromIntegral d20) (fromIntegral d21)
      (fromIntegral d30) (fromIntegral d31)
    ) >>= mkDynamicIO

  newWithTensor :: Dynamic -> IO Dynamic
  newWithTensor t = withForeignPtr (ctensor t) Sig.c_newWithTensor >>= mkDynamicIO

  resize_ :: Dynamic -> Long.Storage -> Long.Storage -> IO ()
  resize_ t l0 l1 = runManaged $ do
    t' <- managed $ withForeignPtr (ctensor t)
    l0' <- managed $ withForeignPtr (Long.storage l0)
    l1' <- managed $ withForeignPtr (Long.storage l1)
    liftIO $ Sig.c_resize t' l0' l1'

  resize1d_ :: Dynamic -> Int64 -> IO ()
  resize1d_ t l0 = withForeignPtr (ctensor t) (\t' -> Sig.c_resize1d t' (fromIntegral l0))

  resize2d_ :: Dynamic -> Int64 -> Int64 -> IO ()
  resize2d_ t l0 l1 = withForeignPtr (ctensor t) $ \t' -> Sig.c_resize2d t'
      (fromIntegral l0) (fromIntegral l1)

  resize3d_ :: Dynamic -> Int64 -> Int64 -> Int64 -> IO ()
  resize3d_ t l0 l1 l2 = withForeignPtr (ctensor t) $ \t' -> Sig.c_resize3d t'
      (fromIntegral l0) (fromIntegral l1) (fromIntegral l2)

  resize4d_ :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize4d_ t l0 l1 l2 l3 = withForeignPtr (ctensor t) $ \t' -> Sig.c_resize4d t'
      (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (fromIntegral l3)

  resize5d_ :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize5d_ t l0 l1 l2 l3 l4 = withForeignPtr (ctensor t) $ \t' -> Sig.c_resize5d t'
      (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (fromIntegral l3) (fromIntegral l4)

  resizeAs_ :: Dynamic -> Dynamic -> IO ()
  resizeAs_ t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_resizeAs t0' t1'

  resizeNd_ :: Dynamic -> Int32 -> [Size] -> [Stride] -> IO ()
  resizeNd_ t i l0' l1' = do
    l0 <- FM.newArray (coerce l0' :: [CLLong])
    l1 <- FM.newArray (coerce l1' :: [CLLong])
    withForeignPtr (ctensor t) $ \t' -> Sig.c_resizeNd t' (fromIntegral i) l0 l1

  retain :: Dynamic -> IO ()
  retain t = withForeignPtr (ctensor t) Sig.c_retain

  select_ :: Dynamic -> Dynamic -> DimVal -> Int64 -> IO ()
  select_ t0 t1 a b =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_select t0' t1' (fromIntegral a) (fromIntegral b)

  set_ :: Dynamic -> Dynamic -> IO ()
  set_ t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_set t0' t1'

  set1d_ :: Dynamic -> Int64 -> HsReal -> IO ()
  set1d_ t l0 v = withForeignPtr (ctensor t) (\t' -> Sig.c_set1d t' (fromIntegral l0) (hs2cReal v))

  set2d_ :: Dynamic -> Int64 -> Int64 -> HsReal -> IO ()
  set2d_ t l0 l1 v = withForeignPtr (ctensor t) (\t' -> Sig.c_set2d t' (fromIntegral l0) (fromIntegral l1) (hs2cReal v))

  set3d_ :: Dynamic -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
  set3d_ t l0 l1 l2 v = withForeignPtr (ctensor t) (\t' -> Sig.c_set3d t' (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (hs2cReal v))

  set4d_ :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
  set4d_ t l0 l1 l2 l3 v = withForeignPtr (ctensor t) (\t' -> Sig.c_set4d t' (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (fromIntegral l3) (hs2cReal v))

  setFlag_ :: Dynamic -> Int8 -> IO ()
  setFlag_ t l0 = withForeignPtr (ctensor t) (\t' -> Sig.c_setFlag t' (CChar l0))

  setStorage_ :: Dynamic -> Storage -> StorageOffset -> Long.Storage -> Long.Storage -> IO ()
  setStorage_ t s a b c = runManaged $ do
    t' <- managed $ withForeignPtr (ctensor t)
    s' <- managed $ withForeignPtr (storage s)
    b' <- managed $ withForeignPtr (Long.storage b)
    c' <- managed $ withForeignPtr (Long.storage c)
    liftIO $ Sig.c_setStorage t' s' (fromIntegral a) b' c'


  setStorage1d_ :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> IO ()
  setStorage1d_ t s pd (d00,d01) =
    withForeignPtr (ctensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage1d t' s' (fromIntegral pd)
          (fromIntegral d00) (fromIntegral d01)

  setStorage2d_ :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage2d_ t s pd (d00,d01) (d10,d11) =
    withForeignPtr (ctensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage2d t' s' (fromIntegral pd)
          (fromIntegral d00) (fromIntegral d01)
          (fromIntegral d10) (fromIntegral d11)


  setStorage3d_ :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage3d_ t s pd (d00,d01) (d10,d11) (d20,d21) =
    withForeignPtr (ctensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage3d t' s' (fromIntegral pd)
          (fromIntegral d00) (fromIntegral d01)
          (fromIntegral d10) (fromIntegral d11)
          (fromIntegral d20) (fromIntegral d21)

  setStorage4d_ :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage4d_ t s pd (d00,d01) (d10,d11) (d20,d21) (d30,d31) =
    withForeignPtr (ctensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage4d t' s' (fromIntegral pd)
          (fromIntegral d00) (fromIntegral d01)
          (fromIntegral d10) (fromIntegral d11)
          (fromIntegral d20) (fromIntegral d21)
          (fromIntegral d30) (fromIntegral d31)

  setStorageNd_ :: Dynamic -> Storage -> StorageOffset -> DimVal -> [Size] -> [Stride] -> IO ()
  setStorageNd_ t s a b hsc hsd = do
    c <- FM.newArray (coerce hsc :: [CLLong])
    d <- FM.newArray (coerce hsd :: [CLLong])
    withForeignPtr (ctensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorageNd t' s' (fromIntegral a) (fromIntegral b) c d


  size :: Dynamic -> DimVal -> IO Size
  size t l0 = withForeignPtr (ctensor t) $ \t' -> fromIntegral <$> Sig.c_size t' (fromIntegral l0)

  sizeDesc :: Dynamic -> IO CTHDescBuff
  sizeDesc t = withForeignPtr (ctensor t) (Sig.c_sizeDesc)

  squeeze_ :: Dynamic -> Dynamic -> IO ()
  squeeze_ t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_squeeze t0' t1'

  squeeze1d_ :: Dynamic -> Dynamic -> DimVal -> IO ()
  squeeze1d_ t0 t1 d =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_squeeze1d t0' t1' (fromIntegral d)

  storage :: Dynamic -> IO Storage
  storage t = withForeignPtr (ctensor t) Sig.c_storage >>= mkStorageIO

  storageOffset :: Dynamic -> IO StorageOffset
  storageOffset t = withForeignPtr (ctensor t) (fmap fromIntegral . Sig.c_storageOffset)

  stride :: Dynamic -> DimVal -> IO Stride
  stride t a = withForeignPtr (ctensor t) (\t' -> fmap fromIntegral $ Sig.c_stride t' (fromIntegral a))

  transpose_ :: Dynamic -> Dynamic -> DimVal -> DimVal -> IO ()
  transpose_ t0 t1 a b =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_transpose t0' t1' (fromIntegral a) (fromIntegral b)

  unfold_ :: Dynamic -> Dynamic -> DimVal -> Size -> Step -> IO ()
  unfold_ t0 t1 a b c =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_unfold t0' t1' (fromIntegral a) (fromIntegral b) (fromIntegral c)

  unsqueeze1d_ :: Dynamic -> Dynamic -> DimVal -> IO ()
  unsqueeze1d_ t0 t1 d =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_unsqueeze1d t0' t1' (fromIntegral d)
-}
