{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE InstanceSigs #-}
module Torch.Indef.Static.Tensor () where

import GHC.Int
import Torch.Dimensions
import qualified Torch.Class.Tensor as Dynamic
import qualified Torch.Class.Tensor.Static as Class
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Types as Sig
import Data.Coerce

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor ()

-- Definitely don't export this. Make sure these gory details never see the day of light.
sudo :: Tensor d -> Tensor d'
sudo t = Sig.asStatic ((Sig.asDynamic t) :: Dynamic)

instance Class.Tensor Tensor where
  free_ :: Dimensions d => Tensor d -> IO ()
  freeCopyTo_ :: (Dimensions d, Dimensions d') => Tensor d -> Tensor d' -> IO ()
  get1d :: Tensor d -> Int64 -> IO (HsReal)
  get2d :: Tensor d -> Int64 -> Int64 -> IO (HsReal)
  get3d :: Tensor d -> Int64 -> Int64 -> Int64 -> IO (HsReal)
  get4d :: Tensor d -> Int64 -> Int64 -> Int64 -> Int64 -> IO (HsReal)
  isContiguous :: Tensor d -> IO Bool

  isSameSizeAs :: forall t d d' . (Dimensions d', Dimensions d) => t d -> t d' -> Bool
  isSameSizeAs _ _ = dimVals (dim :: Dim d) == dimVals (dim :: Dim d')

  isSetTo :: Tensor d -> Tensor d' -> IO Bool
  isSize :: Tensor d -> TH.IndexStorage -> IO Bool
  nDimension :: Tensor d -> IO Int32
  nElement :: Tensor d -> IO Int64
  narrow_ :: Tensor d -> Tensor d' -> DimVal -> Int64 -> Size -> IO ()
  empty :: IO (Tensor d)
  newClone :: (Tensor d) -> IO (Tensor d)
  newContiguous :: Tensor d -> IO (Tensor d')
  newNarrow :: Tensor d -> DimVal -> Int64 -> Size -> IO (Tensor d')
  newSelect :: Tensor d -> DimVal -> Int64 -> IO (Tensor d')
  newSizeOf :: Tensor d -> IO TH.IndexStorage
  newStrideOf :: Tensor d -> IO TH.IndexStorage
  newTranspose :: Tensor d -> DimVal -> DimVal -> IO (Tensor d')
  newUnfold :: Tensor d -> DimVal -> Int64 -> Int64 -> IO (Tensor d')
  newView :: Tensor d -> TH.LongStorage -> IO (Tensor d')
  newWithSize :: TH.LongStorage -> TH.LongStorage -> IO (Tensor d)
  newWithSize1d :: Size -> IO (Tensor d)
  newWithSize2d :: Size -> Size -> IO (Tensor d)
  newWithSize3d :: Size -> Size -> Size -> IO (Tensor d)
  newWithSize4d :: Size -> Size -> Size -> Size -> IO (Tensor d)
  newWithStorage :: Storage -> StorageOffset -> TH.LongStorage -> TH.LongStorage -> IO (Tensor d)
  newWithStorage1d :: Storage -> StorageOffset -> (Size, Stride) -> IO (Tensor d)
  newWithStorage2d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO (Tensor d)
  newWithStorage3d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO (Tensor d)
  newWithStorage4d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO (Tensor d)
  newWithTensor :: Tensor d -> IO (Tensor d)

  resize_ :: Tensor d -> TH.LongStorage -> TH.LongStorage -> IO (Tensor d')
  resize_ t a b = Dynamic.resize_ (asDynamic t) a b >> pure (sudo t)

  resize1d_ :: Tensor d -> Int64 -> IO (Tensor d')
  resize1d_ t a = Dynamic.resize1d_ (asDynamic t) a >> pure (sudo t)

  resize2d_ :: Tensor d -> Int64 -> Int64 -> IO (Tensor d')
  resize2d_ t a b = Dynamic.resize2d_ (asDynamic t) a b >> pure (sudo t)

  resize3d_ :: Tensor d -> Int64 -> Int64 -> Int64 -> IO (Tensor d')
  resize3d_ t a b c = Dynamic.resize3d_ (asDynamic t) a b c >> pure (sudo t)

  resize4d_ :: Tensor d -> Int64 -> Int64 -> Int64 -> Int64 -> IO (Tensor d')
  resize4d_ t a b c d = Dynamic.resize4d_ (asDynamic t) a b c d >> pure (sudo t)

  resize5d_ :: Tensor d -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO (Tensor d')
  resize5d_ t a b c d e = Dynamic.resize5d_ (asDynamic t) a b c d e >> pure (sudo t)

  resizeAs_ :: Tensor d -> Tensor d' -> IO (Tensor d')
  resizeAs_ src tar = Dynamic.resizeAs_ (asDynamic src) (asDynamic tar) >> pure (sudo src)

  resizeNd_ :: Tensor d -> Int32 -> [Size] -> [Stride] -> IO (Tensor d')
  resizeNd_ src a b c = Dynamic.resizeNd_ (asDynamic src) a b c >> pure (sudo src)

  retain :: Tensor d -> IO ()
  retain t = Dynamic.retain (asDynamic t)

  select_ :: Tensor d -> Tensor d' -> DimVal -> Int64 -> IO ()
  set_ :: Tensor d -> Tensor d -> IO ()
  set1d_ :: Tensor d -> Int64 -> HsReal -> IO ()
  set2d_ :: Tensor d -> Int64 -> Int64 -> HsReal -> IO ()
  set3d_ :: Tensor d -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
  set4d_ :: Tensor d -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
  setFlag_ :: Tensor d -> Int8 -> IO ()
  setStorage_   :: Tensor d -> Storage -> StorageOffset -> TH.LongStorage -> TH.LongStorage -> IO ()
  setStorage1d_ :: Tensor d -> Storage -> StorageOffset -> (Size, Stride) -> IO ()
  setStorage2d_ :: Tensor d -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage3d_ :: Tensor d -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage4d_ :: Tensor d -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorageNd_ :: Tensor d -> Storage -> StorageOffset -> DimVal -> [Size] -> [Stride] -> IO ()
  size :: Tensor d -> DimVal -> IO Size
  sizeDesc :: Tensor d -> IO DescBuff
  squeeze_ :: Tensor d -> Tensor d' -> IO ()
  squeeze1d_ :: Tensor d -> Tensor d' -> DimVal -> IO ()
  storage :: Tensor d -> IO (Storage)
  storageOffset :: Tensor d -> IO StorageOffset
  stride :: Tensor d -> DimVal -> IO Stride
  transpose_ :: Tensor d -> Tensor d' -> DimVal -> DimVal -> IO ()
  unfold_ :: Tensor d -> Tensor d' -> DimVal -> Size -> Step -> IO ()
  unsqueeze1d_ :: Tensor d -> Tensor d' -> DimVal -> IO ()
  clearFlag_ :: Dimensions d => Tensor d -> Int8 -> IO ()
  clearFlag_ t = Dynamic.clearFlag_ (asDynamic t)
  tensordata :: Dimensions d => Tensor d -> IO [HsReal]
  tensordata t = Dynamic.tensordata (asDynamic t)
  free_ t = Dynamic.free_ (asDynamic t)
  freeCopyTo_ t0 t1 = Dynamic.freeCopyTo_ (asDynamic t0) (asDynamic t1)
  get1d t = Dynamic.get1d (asDynamic t)
  get2d t = Dynamic.get2d (asDynamic t)
  get3d t = Dynamic.get3d (asDynamic t)
  get4d t = Dynamic.get4d (asDynamic t)
  isContiguous t = Dynamic.isContiguous (asDynamic t)

  isSetTo t0 t1 = Dynamic.isSetTo (asDynamic t0) (asDynamic t1)
  isSize t = Dynamic.isSize (asDynamic t)
  nDimension t = Dynamic.nDimension (asDynamic t)
  nElement t = Dynamic.nElement (asDynamic t)
  narrow_ t0 t1 = Dynamic.narrow_ (asDynamic t0) (asDynamic t1)
  empty = asStatic <$> Dynamic.empty
  newClone t = asStatic <$> Dynamic.newClone (asDynamic t)
  newContiguous t = asStatic <$> Dynamic.newContiguous (asDynamic t)
  newNarrow t a b c = asStatic <$> Dynamic.newNarrow (asDynamic t) a b c
  newSelect t a b = asStatic <$> Dynamic.newSelect (asDynamic t) a b
  newSizeOf t = Dynamic.newSizeOf (asDynamic t)
  newStrideOf t = Dynamic.newStrideOf (asDynamic t)
  newTranspose t a b = asStatic <$> Dynamic.newTranspose (asDynamic t) a b
  newUnfold t a b c = asStatic <$> Dynamic.newUnfold (asDynamic t) a b c
  newView t a = asStatic <$> Dynamic.newView (asDynamic t) a
  newWithSize a0 a1 = asStatic <$> Dynamic.newWithSize a0 a1
  newWithSize1d a0 = asStatic <$> Dynamic.newWithSize1d a0
  newWithSize2d a0 a1 = asStatic <$> Dynamic.newWithSize2d a0 a1
  newWithSize3d a0 a1 a2 = asStatic <$> Dynamic.newWithSize3d a0 a1 a2
  newWithSize4d a0 a1 a2 a3 = asStatic <$> Dynamic.newWithSize4d a0 a1 a2 a3
  newWithStorage a0 a1 a2 a3 = asStatic <$> Dynamic.newWithStorage a0 a1 a2 a3
  newWithStorage1d a0 a1 a2 = asStatic <$> Dynamic.newWithStorage1d a0 a1 a2
  newWithStorage2d a0 a1 a2 a3 = asStatic <$> Dynamic.newWithStorage2d a0 a1 a2 a3
  newWithStorage3d a0 a1 a2 a3 a4 = asStatic <$> Dynamic.newWithStorage3d a0 a1 a2 a3 a4
  newWithStorage4d a0 a1 a2 a3 a4 a5 = asStatic <$> Dynamic.newWithStorage4d a0 a1 a2 a3 a4 a5
  newWithTensor t = asStatic <$> Dynamic.newWithTensor (asDynamic t)
  select_ t0 t1 = Dynamic.select_ (asDynamic t0) (asDynamic t1)
  set_ t0 t1 = Dynamic.set_ (asDynamic t0) (asDynamic t1)
  set1d_ t = Dynamic.set1d_ (asDynamic t)
  set2d_ t = Dynamic.set2d_ (asDynamic t)
  set3d_ t = Dynamic.set3d_ (asDynamic t)
  set4d_ t = Dynamic.set4d_ (asDynamic t)
  setFlag_ t = Dynamic.setFlag_ (asDynamic t)
  setStorage_ t = Dynamic.setStorage_ (asDynamic t)
  setStorage1d_ t = Dynamic.setStorage1d_ (asDynamic t)
  setStorage2d_ t = Dynamic.setStorage2d_ (asDynamic t)
  setStorage3d_ t = Dynamic.setStorage3d_ (asDynamic t)
  setStorage4d_ t = Dynamic.setStorage4d_ (asDynamic t)
  setStorageNd_ t = Dynamic.setStorageNd_ (asDynamic t)
  size t = Dynamic.size (asDynamic t)
  sizeDesc t = Dynamic.sizeDesc (asDynamic t)
  squeeze_ t0 t1 = Dynamic.squeeze_ (asDynamic t0) (asDynamic t1)
  squeeze1d_ t0 t1 = Dynamic.squeeze1d_ (asDynamic t0) (asDynamic t1)
  storage t = Dynamic.storage (asDynamic t)
  storageOffset t = Dynamic.storageOffset (asDynamic t)
  stride t = Dynamic.stride (asDynamic t)
  transpose_ t0 t1 = Dynamic.transpose_ (asDynamic t0) (asDynamic t1)
  unfold_ t0 t1 = Dynamic.unfold_ (asDynamic t0) (asDynamic t1)
  unsqueeze1d_ t0 t1 = Dynamic.unsqueeze1d_ (asDynamic t0) (asDynamic t1)

  fromList1d :: [HsReal] -> IO (Tensor '[n])
  fromList1d l = asStatic <$> (Dynamic.fromList1d l)


