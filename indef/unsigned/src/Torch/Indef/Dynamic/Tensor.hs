{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Indef.Dynamic.Tensor () where

import Data.Coerce (coerce)
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import GHC.Int
import Control.Monad ((>=>))
import Control.Monad.Managed
import Data.List.NonEmpty (NonEmpty(..), toList)
import Torch.Dimensions
import Torch.Class.Types (Stride(..), Size(..), StorageOffset(..), Step(..), SizesStorage, StridesStorage)

import qualified Torch.Types.TH            as TH
import qualified Foreign.Marshal.Array     as FM
import qualified Torch.Sig.State           as Sig
import qualified Torch.Sig.Types           as Sig
import qualified Torch.Sig.Types.Global    as Sig
import qualified Torch.Sig.Tensor          as Sig
import qualified Torch.Sig.Tensor.Memory   as Sig
import qualified Torch.Sig.Storage         as StorageSig (c_size)
import qualified Torch.Class.Tensor        as Class

import Torch.Indef.Types

longCStorage = snd . longStorageState
longCStorageTH = snd . TH.longStorageState

instance Class.IsTensor Dynamic where
  clearFlag_ :: Dynamic -> Int8 -> IO ()
  clearFlag_ t cc = withDynamicState t $ shuffle2 Sig.c_clearFlag (CChar cc)

  tensordata :: Dynamic -> IO [HsReal]
  tensordata t = withDynamicState t $ \s' t' ->
    ptrArray2hs (Sig.c_data s') (arrayLen s') (Sig.ctensor t)
   where
    arrayLen :: Ptr CState -> Ptr CTensor -> IO Int
    arrayLen s' p = Sig.c_storage s' p >>= fmap fromIntegral . StorageSig.c_size s'

  free_ :: Dynamic -> IO ()
  free_ t = withDynamicState t Sig.c_free

  freeCopyTo_ :: Dynamic -> Dynamic -> IO ()
  freeCopyTo_ t0 t1 = with2DynamicState t0 t1 $ \s t0' t1' ->
      Sig.c_freeCopyTo s t0' t1'

  get1d :: Dynamic -> Int64 -> IO HsReal
  get1d t d1 = withDynamicState t $ \s t' -> c2hsReal <$> Sig.c_get1d s t' (fromIntegral d1)

  get2d :: Dynamic -> Int64 -> Int64 -> IO HsReal
  get2d t d1 d2 = withDynamicState t $ \s t' -> c2hsReal <$> Sig.c_get2d s t' (fromIntegral d1) (fromIntegral d2)

  get3d :: Dynamic -> Int64 -> Int64 -> Int64 -> IO HsReal
  get3d t d1 d2 d3 = withDynamicState t $ \s t' -> c2hsReal <$> Sig.c_get3d s t' (fromIntegral d1) (fromIntegral d2) (fromIntegral d3)

  get4d :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> IO HsReal
  get4d t d1 d2 d3 d4 = withDynamicState t $ \s t' -> c2hsReal <$> Sig.c_get4d s t' (fromIntegral d1) (fromIntegral d2) (fromIntegral d3) (fromIntegral d4)

  isContiguous :: Dynamic -> IO Bool
  isContiguous t = withDynamicState t $ \s t' ->
    (1 ==) <$> Sig.c_isContiguous s t'

  isSameSizeAs :: Dynamic -> Dynamic -> IO Bool
  isSameSizeAs t0 t1 = with2DynamicState t0 t1 $ \s t0' t1' ->
    (1 ==) <$> Sig.c_isSetTo s t0' t1'

  isSetTo :: Dynamic -> Dynamic -> IO Bool
  isSetTo t0 t1 = with2DynamicState t0 t1 $ \s t0' t1' ->
    (1 ==) <$> Sig.c_isSetTo s t0' t1'

  isSize :: Dynamic -> TH.LongStorage -> IO Bool
  isSize t ls = withDynamicState t $ \s t' ->
    withForeignPtr (snd $ TH.longStorageState ls) (fmap (1 ==) . Sig.c_isSize s t')

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

  newExpand :: Dynamic -> TH.IndexStorage -> IO Dynamic
  newExpand r ix = flip with pure $ do
    s <- manage' Sig.dynamicStateRef r
    r' <- manage' Sig.ctensor r
    ix' <- manage' (snd . TH.longStorageState) ix
    liftIO $ Sig.c_newExpand s r' ix' >>= mkDynamic s

  expand :: Dynamic -> Dynamic -> TH.IndexStorage -> IO ()
  expand r t ix = runManaged . joinIO $ Sig.c_expand
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.ctensor t
    <*> manage' (snd . TH.longStorageState) ix

  expandNd  :: NonEmpty Dynamic -> NonEmpty Dynamic -> Int -> IO ()
  expandNd (rets@(s:|_)) ops i = runManaged $ do
    st    <- manage' Sig.dynamicStateRef s
    rets' <- mngNonEmpty rets
    ops'  <- mngNonEmpty ops
    liftIO $ Sig.c_expandNd st rets' ops' (fromIntegral i)
   where
    mngNonEmpty :: NonEmpty Dynamic -> Managed (Ptr (Ptr CTensor))
    mngNonEmpty = mapM toMPtr . toList >=> mWithArray

    mWithArray :: [Ptr a] -> Managed (Ptr (Ptr a))
    mWithArray as = managed (FM.withArray as)

    toMPtr :: Dynamic -> Managed (Ptr CTensor)
    toMPtr d = managed (withForeignPtr (Sig.ctensor d))


  newClone :: Dynamic -> IO Dynamic
  newClone t = withDynamicState t $ \s' t' -> Sig.c_newClone s' t' >>= mkDynamic s'

  newContiguous :: Dynamic -> IO Dynamic
  newContiguous t = withDynamicState t $ \s' t' -> Sig.c_newContiguous s' t' >>= mkDynamic s'

  newNarrow :: Dynamic -> DimVal -> Int64 -> Size -> IO Dynamic
  newNarrow t a b c = withDynamicState t $ \s' t' -> Sig.c_newNarrow s' t' (fromIntegral a) (fromIntegral b) (fromIntegral c) >>= mkDynamic s'

  newSelect :: Dynamic -> DimVal -> Int64 -> IO Dynamic
  newSelect t a b = withDynamicState t $ \s' t' -> Sig.c_newSelect s' t' (fromIntegral a) (fromIntegral b) >>= mkDynamic s'

  newSizeOf :: Dynamic -> IO (TH.IndexStorage)
  newSizeOf t = withDynamicState t $ \s' t' -> Sig.c_newSizeOf s' t' >>= mkCPUIxStorage

  newStrideOf :: Dynamic -> IO (TH.IndexStorage)
  newStrideOf t = withDynamicState t $ \s' t' -> Sig.c_newStrideOf s' t' >>= mkCPUIxStorage

  newTranspose :: Dynamic -> DimVal -> DimVal -> IO Dynamic
  newTranspose t a b = withDynamicState t $ \s' t' -> Sig.c_newTranspose s' t' (fromIntegral a) (fromIntegral b) >>= mkDynamic s'

  newUnfold :: Dynamic -> DimVal -> Int64 -> Int64 -> IO Dynamic
  newUnfold t a b c = withDynamicState t $ \s' t' -> Sig.c_newUnfold s' t' (fromIntegral a) (fromIntegral b) (fromIntegral c) >>= mkDynamic s'

  newView :: Dynamic -> TH.IndexStorage -> IO Dynamic
  newView t ix = withDynamicState t $ \s' t' ->
    withCPUIxStorage ix (Sig.c_newView s' t' >=> mkDynamic s')

  newWithSize :: TH.IndexStorage -> TH.IndexStorage -> IO Dynamic
  newWithSize l0 l1 =
    withCPUIxStorage l0 $ \l0' ->
      withCPUIxStorage l1 $ \l1' ->
        mkDynamicIO $ \s ->
          Sig.c_newWithSize s l0' l1'

  newWithSize1d :: Size -> IO Dynamic
  newWithSize1d a0 = mkDynamicIO $ \s -> Sig.c_newWithSize1d s (fromIntegral a0)

  newWithSize2d :: Size -> Size -> IO Dynamic
  newWithSize2d a0 a1 = mkDynamicIO $ \s -> Sig.c_newWithSize2d s (fromIntegral a0) (fromIntegral a1)

  newWithSize3d :: Size -> Size -> Size -> IO Dynamic
  newWithSize3d a0 a1 a2 = mkDynamicIO $ \s -> Sig.c_newWithSize3d s (fromIntegral a0) (fromIntegral a1) (fromIntegral a2)

  newWithSize4d :: Size -> Size -> Size -> Size -> IO Dynamic
  newWithSize4d a0 a1 a2 a3 = mkDynamicIO $ \s -> Sig.c_newWithSize4d s (fromIntegral a0) (fromIntegral a1) (fromIntegral a2) (fromIntegral a3)

  newWithStorage :: Storage -> StorageOffset -> TH.IndexStorage -> TH.IndexStorage -> IO Dynamic
  newWithStorage s pd l0 l1 =
    withStorageState s $ \state' s' ->
      withForeignPtr (longCStorageTH l0) $ \l0' ->
        withForeignPtr (longCStorageTH l1) $ \l1' ->
          Sig.c_newWithStorage state' s' (fromIntegral pd) l0' l1'
          >>= mkDynamic state'

  newWithStorage1d :: Storage -> StorageOffset -> (Size, Stride) -> IO Dynamic
  newWithStorage1d s pd (d00,d01) =
    withStorageState s $ \state' s' ->
      Sig.c_newWithStorage1d state' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      >>= mkDynamic state'


  newWithStorage2d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO Dynamic
  newWithStorage2d s pd (d00,d01) (d10,d11) =
    withStorageState s $ \state' s' ->
      Sig.c_newWithStorage2d state' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)
      >>= mkDynamic state'


  newWithStorage3d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO Dynamic
  newWithStorage3d s pd (d00,d01) (d10,d11) (d20,d21) =
    withStorageState s $ \state' s' ->
      Sig.c_newWithStorage3d state' s' (fromIntegral pd)
        (fromIntegral d00) (fromIntegral d01)
        (fromIntegral d10) (fromIntegral d11)
        (fromIntegral d20) (fromIntegral d21)
      >>= mkDynamic state'


  newWithStorage4d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO Dynamic
  newWithStorage4d s pd (d00,d01) (d10,d11) (d20,d21) (d30,d31) =
    withStorageState s $ \state' s' ->
      Sig.c_newWithStorage4d state' s' (fromIntegral pd)
        (fromIntegral d00) (fromIntegral d01)
        (fromIntegral d10) (fromIntegral d11)
        (fromIntegral d20) (fromIntegral d21)
        (fromIntegral d30) (fromIntegral d31)
      >>= mkDynamic state'

  newWithTensor :: Dynamic -> IO Dynamic
  newWithTensor t = withDynamicState t $ \s' t' -> Sig.c_newWithTensor s' t' >>= mkDynamic s'

  resize_ :: Dynamic -> TH.IndexStorage -> TH.IndexStorage -> IO ()
  resize_ t l0 l1 = withDynamicState t $ \s' t' -> runManaged $ do
    l0' <- managed $ withCPUIxStorage l0
    l1' <- managed $ withCPUIxStorage l1
    liftIO $ Sig.c_resize s' t' l0' l1'

  resize1d_ :: Dynamic -> Int64 -> IO ()
  resize1d_ t l0 = withDynamicState t (\s' t' -> Sig.c_resize1d s' t' (fromIntegral l0))

  resize2d_ :: Dynamic -> Int64 -> Int64 -> IO ()
  resize2d_ t l0 l1 = withDynamicState t $ \s' t' -> Sig.c_resize2d s' t'
      (fromIntegral l0) (fromIntegral l1)

  resize3d_ :: Dynamic -> Int64 -> Int64 -> Int64 -> IO ()
  resize3d_ t l0 l1 l2 = withDynamicState t $ \s' t' -> Sig.c_resize3d s' t'
      (fromIntegral l0) (fromIntegral l1) (fromIntegral l2)

  resize4d_ :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize4d_ t l0 l1 l2 l3 = withDynamicState t $ \s' t' -> Sig.c_resize4d s' t'
      (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (fromIntegral l3)

  resize5d_ :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize5d_ t l0 l1 l2 l3 l4 = withDynamicState t $ \s' t' -> Sig.c_resize5d s' t'
      (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (fromIntegral l3) (fromIntegral l4)

  resizeAs_ :: Dynamic -> Dynamic -> IO ()
  resizeAs_ t0 t1 = with2DynamicState t0 t1 Sig.c_resizeAs

  resizeNd_ :: Dynamic -> Int32 -> [Size] -> [Stride] -> IO ()
  resizeNd_ t i l0' l1' = do
    l0 <- FM.newArray (coerce l0' :: [CLLong])
    l1 <- FM.newArray (coerce l1' :: [CLLong])
    withDynamicState t $ \s' t' -> Sig.c_resizeNd s' t' (fromIntegral i) l0 l1

  retain :: Dynamic -> IO ()
  retain t = withDynamicState t Sig.c_retain

  select_ :: Dynamic -> Dynamic -> DimVal -> Int64 -> IO ()
  select_ t0 t1 a b = with2DynamicState t0 t1 $ \s' t0' t1' ->
    Sig.c_select s' t0' t1' (fromIntegral a) (fromIntegral b)

  set_ :: Dynamic -> Dynamic -> IO ()
  set_ t0 t1 = with2DynamicState t0 t1 Sig.c_set

  set1d_ :: Dynamic -> Int64 -> HsReal -> IO ()
  set1d_ t l0 v = withDynamicState t $ \s' t' -> Sig.c_set1d s' t' (fromIntegral l0) (hs2cReal v)

  set2d_ :: Dynamic -> Int64 -> Int64 -> HsReal -> IO ()
  set2d_ t l0 l1 v = withDynamicState t $ \s' t' -> Sig.c_set2d s' t' (fromIntegral l0) (fromIntegral l1) (hs2cReal v)

  set3d_ :: Dynamic -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
  set3d_ t l0 l1 l2 v = withDynamicState t $ \s' t' -> Sig.c_set3d s' t' (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (hs2cReal v)

  set4d_ :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
  set4d_ t l0 l1 l2 l3 v = withDynamicState t $ \s' t' -> Sig.c_set4d s' t' (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (fromIntegral l3) (hs2cReal v)

  setFlag_ :: Dynamic -> Int8 -> IO ()
  setFlag_ t l0 = withDynamicState t $ shuffle2 Sig.c_setFlag (CChar l0)

  setStorage_ :: Dynamic -> Storage -> StorageOffset -> TH.IndexStorage -> TH.IndexStorage -> IO ()
  setStorage_ t s a b c = withDynamicStateAndStorage t s $ \st' t' s' ->
    runManaged $ do
      b' <- managed $ withCPUIxStorage b
      c' <- managed $ withCPUIxStorage c
      liftIO $ Sig.c_setStorage st' t' s' (fromIntegral a) b' c'

  setStorage1d_ :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> IO ()
  setStorage1d_ t s pd (d00,d01) =
    withDynamicStateAndStorage t s $ \st' t' s' ->
      Sig.c_setStorage1d st' t' s' (fromIntegral pd)
        (fromIntegral d00) (fromIntegral d01)

  setStorage2d_ :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage2d_ t s pd (d00,d01) (d10,d11) =
    withDynamicStateAndStorage t s $ \st' t' s' ->
      Sig.c_setStorage2d st' t' s' (fromIntegral pd)
        (fromIntegral d00) (fromIntegral d01)
        (fromIntegral d10) (fromIntegral d11)


  setStorage3d_ :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage3d_ t s pd (d00,d01) (d10,d11) (d20,d21) =
    withDynamicStateAndStorage t s $ \st' t' s' ->
      Sig.c_setStorage3d st' t' s' (fromIntegral pd)
        (fromIntegral d00) (fromIntegral d01)
        (fromIntegral d10) (fromIntegral d11)
        (fromIntegral d20) (fromIntegral d21)

  setStorage4d_ :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage4d_ t s pd (d00,d01) (d10,d11) (d20,d21) (d30,d31) =
    withDynamicStateAndStorage t s $ \st' t' s' ->
      Sig.c_setStorage4d st' t' s' (fromIntegral pd)
        (fromIntegral d00) (fromIntegral d01)
        (fromIntegral d10) (fromIntegral d11)
        (fromIntegral d20) (fromIntegral d21)
        (fromIntegral d30) (fromIntegral d31)

  setStorageNd_ :: Dynamic -> Storage -> StorageOffset -> DimVal -> [Size] -> [Stride] -> IO ()
  setStorageNd_ t s a b hsc hsd = do
    c <- FM.newArray (coerce hsc :: [CLLong])
    d <- FM.newArray (coerce hsd :: [CLLong])
    withDynamicStateAndStorage t s $ \st' t' s' ->
      Sig.c_setStorageNd st' t' s' (fromIntegral a) (fromIntegral b) c d

  size :: Dynamic -> DimVal -> IO Size
  size t l0 = withDynamicState t $ \st' t' -> fromIntegral <$> Sig.c_size st' t' (fromIntegral l0)

  sizeDesc :: Dynamic -> IO DescBuff
  sizeDesc t = withDynamicState t $ \s' t' -> Sig.c_sizeDesc s' t' >>= Sig.descBuff

  squeeze_ :: Dynamic -> Dynamic -> IO ()
  squeeze_ t0 t1 = with2DynamicState t0 t1 Sig.c_squeeze

  squeeze1d_ :: Dynamic -> Dynamic -> DimVal -> IO ()
  squeeze1d_ t0 t1 d = with2DynamicState t0 t1 (shuffle3 Sig.c_squeeze1d (fromIntegral d))

  storage :: Dynamic -> IO Storage
  storage t = withDynamicState t $ \s' t' -> Sig.c_storage s' t' >>= mkStorage s'

  storageOffset :: Dynamic -> IO StorageOffset
  storageOffset t = withDynamicState t (fmap fromIntegral .: Sig.c_storageOffset)

  stride :: Dynamic -> DimVal -> IO Stride
  stride t a = withDynamicState t (fmap fromIntegral .: shuffle2 Sig.c_stride (fromIntegral a))

  transpose_ :: Dynamic -> Dynamic -> DimVal -> DimVal -> IO ()
  transpose_ t0 t1 a b = with2DynamicState t0 t1 $ \s' t0' t1' ->
    Sig.c_transpose s' t0' t1' (fromIntegral a) (fromIntegral b)

  unfold_ :: Dynamic -> Dynamic -> DimVal -> Size -> Step -> IO ()
  unfold_ t0 t1 a b c = with2DynamicState t0 t1 $ \s' t0' t1' ->
    Sig.c_unfold s' t0' t1' (fromIntegral a) (fromIntegral b) (fromIntegral c)

  unsqueeze1d_ :: Dynamic -> Dynamic -> DimVal -> IO ()
  unsqueeze1d_ t0 t1 d = with2DynamicState t0 t1 $
    shuffle3 Sig.c_unsqueeze1d (fromIntegral d)
