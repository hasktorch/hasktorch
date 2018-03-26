{-# LANGUAGE InstanceSigs #-}
module Torch.Indef.Tensor.Static.IsTensor
  ( Class.IsTensor(..)
  ) where

import GHC.Int
import Torch.Types.TH (CTHDescBuff)

import Torch.Dimensions (DimVal)
import qualified Torch.Types.TH.Long as Long
import qualified Torch.FFI.TH.Long.Storage as Long
import qualified Torch.Class.IsTensor as Class
import Torch.Class.Internal (Stride, Size, StorageOffset, Step, SizesStorage, StridesStorage)

import Torch.Sig.Types -- for the types
import Torch.Indef.Tensor.Dynamic.IsTensor () -- for the downcasting
import Torch.Indef.Tensor.Static (tensor) -- for the type families

-- TODO: replace some of this with static-aware versions
instance Class.IsTensor (Tensor d) where
  clearFlag_ :: Tensor d -> Int8 -> IO ()
  clearFlag_ t = Class.clearFlag_ (dynamic t)
  tensordata :: Tensor d -> IO [HsReal]
  tensordata t = Class.tensordata (dynamic t)
  desc :: Tensor d -> IO CTHDescBuff
  desc t = Class.desc (dynamic t)
  expand_ :: Tensor d -> Tensor d -> Long.Storage -> IO ()
  expand_ t0 t1 = Class.expand_ (dynamic t0) (dynamic t1)
  expandNd_ :: [Tensor d] -> [Tensor d] -> Int32 -> IO ()
  expandNd_ a0 a1 = Class.expandNd_ (fmap dynamic a0) (fmap dynamic a1)
  free_ :: Tensor d -> IO ()
  free_ t = Class.free_ (dynamic t)
  freeCopyTo_ :: Tensor d -> Tensor d -> IO ()
  freeCopyTo_ t0 t1 = Class.freeCopyTo_ (dynamic t0) (dynamic t1)
  get1d :: Tensor d -> Int64 -> IO HsReal
  get1d t = Class.get1d (dynamic t)
  get2d :: Tensor d -> Int64 -> Int64 -> IO HsReal
  get2d t = Class.get2d (dynamic t)
  get3d :: Tensor d -> Int64 -> Int64 -> Int64 -> IO HsReal
  get3d t = Class.get3d (dynamic t)
  get4d :: Tensor d -> Int64 -> Int64 -> Int64 -> Int64 -> IO HsReal
  get4d t = Class.get4d (dynamic t)
  isContiguous :: Tensor d -> IO Bool
  isContiguous t = Class.isContiguous (dynamic t)
  isSameSizeAs :: Tensor d -> Tensor d -> IO Bool
  isSameSizeAs t0 t1 = Class.isSameSizeAs (dynamic t0) (dynamic t1)
  isSetTo :: Tensor d -> Tensor d -> IO Bool
  isSetTo t0 t1 = Class.isSetTo (dynamic t0) (dynamic t1)
  isSize :: Tensor d -> Long.Storage -> IO Bool
  isSize t = Class.isSize (dynamic t)
  nDimension :: Tensor d -> IO Int32
  nDimension t = Class.nDimension (dynamic t)
  nElement :: Tensor d -> IO Int64
  nElement t = Class.nElement (dynamic t)
  narrow_ :: Tensor d -> Tensor d -> DimVal -> Int64 -> Size -> IO ()
  narrow_ t0 t1 = Class.narrow_ (dynamic t0) (dynamic t1)
  empty :: IO (Tensor d)
  empty = asStatic <$> Class.empty
  newClone :: Tensor d -> IO (Tensor d)
  newClone t = asStatic <$> Class.newClone (dynamic t)
  newContiguous :: Tensor d -> IO (Tensor d)
  newContiguous t = asStatic <$> Class.newContiguous (dynamic t)
  newExpand :: Tensor d -> Long.Storage -> IO (Tensor d)
  newExpand t a = asStatic <$> Class.newExpand (dynamic t) a
  newNarrow :: Tensor d -> DimVal -> Int64 -> Size -> IO (Tensor d)
  newNarrow t a b c = asStatic <$> Class.newNarrow (dynamic t) a b c
  newSelect :: Tensor d -> DimVal -> Int64 -> IO (Tensor d)
  newSelect t a b = asStatic <$> Class.newSelect (dynamic t) a b
  newSizeOf :: Tensor d -> IO Long.Storage
  newSizeOf t = Class.newSizeOf (dynamic t)
  newStrideOf :: Tensor d -> IO Long.Storage
  newStrideOf t = Class.newStrideOf (dynamic t)
  newTranspose :: Tensor d -> DimVal -> DimVal -> IO (Tensor d)
  newTranspose t a b = asStatic <$> Class.newTranspose (dynamic t) a b
  newUnfold :: Tensor d -> DimVal -> Int64 -> Int64 -> IO (Tensor d)
  newUnfold t a b c = asStatic <$> Class.newUnfold (dynamic t) a b c
  newView :: Tensor d -> Long.Storage -> IO (Tensor d)
  newView t a = asStatic <$> Class.newView (dynamic t) a
  newWithSize :: Long.Storage -> Long.Storage -> IO (Tensor d)
  newWithSize a0 a1 = asStatic <$> Class.newWithSize a0 a1
  newWithSize1d :: Size -> IO (Tensor d)
  newWithSize1d a0 = asStatic <$> Class.newWithSize1d a0
  newWithSize2d :: Size -> Size -> IO (Tensor d)
  newWithSize2d a0 a1 = asStatic <$> Class.newWithSize2d a0 a1
  newWithSize3d :: Size -> Size -> Size -> IO (Tensor d)
  newWithSize3d a0 a1 a2 = asStatic <$> Class.newWithSize3d a0 a1 a2
  newWithSize4d :: Size -> Size -> Size -> Size -> IO (Tensor d)
  newWithSize4d a0 a1 a2 a3 = asStatic <$> Class.newWithSize4d a0 a1 a2 a3
  newWithStorage :: Storage -> StorageOffset -> Long.Storage -> Long.Storage -> IO (Tensor d)
  newWithStorage a0 a1 a2 a3 = asStatic <$> Class.newWithStorage a0 a1 a2 a3
  newWithStorage1d :: Storage -> StorageOffset -> (Size, Stride) -> IO (Tensor d)
  newWithStorage1d a0 a1 a2 = asStatic <$> Class.newWithStorage1d a0 a1 a2
  newWithStorage2d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO (Tensor d)
  newWithStorage2d a0 a1 a2 a3 = asStatic <$> Class.newWithStorage2d a0 a1 a2 a3
  newWithStorage3d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO (Tensor d)
  newWithStorage3d a0 a1 a2 a3 a4 = asStatic <$> Class.newWithStorage3d a0 a1 a2 a3 a4
  newWithStorage4d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO (Tensor d)
  newWithStorage4d a0 a1 a2 a3 a4 a5 = asStatic <$> Class.newWithStorage4d a0 a1 a2 a3 a4 a5
  newWithTensor :: Tensor d -> IO (Tensor d)
  newWithTensor t = asStatic <$> Class.newWithTensor (dynamic t)
  resize_ :: Tensor d -> Long.Storage -> Long.Storage -> IO ()
  resize_ t = Class.resize_ (dynamic t)
  resize1d_ :: Tensor d -> Int64 -> IO ()
  resize1d_ t = Class.resize1d_ (dynamic t)
  resize2d_ :: Tensor d -> Int64 -> Int64 -> IO ()
  resize2d_ t = Class.resize2d_ (dynamic t)
  resize3d_ :: Tensor d -> Int64 -> Int64 -> Int64 -> IO ()
  resize3d_ t = Class.resize3d_ (dynamic t)
  resize4d_ :: Tensor d -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize4d_ t = Class.resize4d_ (dynamic t)
  resize5d_ :: Tensor d -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize5d_ t = Class.resize5d_ (dynamic t)
  resizeAs_ :: Tensor d -> Tensor d -> IO ()
  resizeAs_ t0 t1 = Class.resizeAs_ (dynamic t0) (dynamic t1)
  resizeNd_ :: Tensor d -> Int32 -> [Size] -> [Stride] -> IO ()
  resizeNd_ t = Class.resizeNd_ (dynamic t)
  retain :: Tensor d -> IO ()
  retain t = Class.retain (dynamic t)

  select_ :: Tensor d -> Tensor d -> DimVal -> Int64 -> IO ()
  select_ t0 t1 = Class.select_ (dynamic t0) (dynamic t1)
  set_ :: Tensor d -> Tensor d -> IO ()
  set_ t0 t1 = Class.set_ (dynamic t0) (dynamic t1)

  set1d_ :: Tensor d -> Int64 -> HsReal -> IO ()
  set1d_ t = Class.set1d_ (dynamic t)
  set2d_ :: Tensor d -> Int64 -> Int64 -> HsReal -> IO ()
  set2d_ t = Class.set2d_ (dynamic t)
  set3d_ :: Tensor d -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
  set3d_ t = Class.set3d_ (dynamic t)
  set4d_ :: Tensor d -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
  set4d_ t = Class.set4d_ (dynamic t)
  setFlag_ :: Tensor d -> Int8 -> IO ()
  setFlag_ t = Class.setFlag_ (dynamic t)
  setStorage_ :: Tensor d -> Storage -> StorageOffset -> Long.Storage -> Long.Storage -> IO ()
  setStorage_ t = Class.setStorage_ (dynamic t)
  setStorage1d_ :: Tensor d -> Storage -> StorageOffset -> (Size, Stride) -> IO ()
  setStorage1d_ t = Class.setStorage1d_ (dynamic t)
  setStorage2d_ :: Tensor d -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage2d_ t = Class.setStorage2d_ (dynamic t)
  setStorage3d_ :: Tensor d -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage3d_ t = Class.setStorage3d_ (dynamic t)
  setStorage4d_ :: Tensor d -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage4d_ t = Class.setStorage4d_ (dynamic t)
  setStorageNd_ :: Tensor d -> Storage -> StorageOffset -> DimVal -> [Size] -> [Stride] -> IO ()
  setStorageNd_ t = Class.setStorageNd_ (dynamic t)
  size :: Tensor d -> DimVal -> IO Size
  size t = Class.size (dynamic t)
  sizeDesc :: Tensor d -> IO CTHDescBuff
  sizeDesc t = Class.sizeDesc (dynamic t)
  squeeze_ :: Tensor d -> Tensor d -> IO ()
  squeeze_ t0 t1 = Class.squeeze_ (dynamic t0) (dynamic t1)
  squeeze1d_ :: Tensor d -> Tensor d -> DimVal -> IO ()
  squeeze1d_ t0 t1 = Class.squeeze1d_ (dynamic t0) (dynamic t1)
  storage :: Tensor d -> IO Storage
  storage t = Class.storage (dynamic t)
  storageOffset :: Tensor d -> IO StorageOffset
  storageOffset t = Class.storageOffset (dynamic t)
  stride :: Tensor d -> DimVal -> IO Stride
  stride t = Class.stride (dynamic t)
  transpose_ :: Tensor d -> Tensor d -> DimVal -> DimVal -> IO ()
  transpose_ t0 t1 = Class.transpose_ (dynamic t0) (dynamic t1)
  unfold_ :: Tensor d -> Tensor d -> DimVal -> Size -> Step -> IO ()
  unfold_ t0 t1 = Class.unfold_ (dynamic t0) (dynamic t1)
  unsqueeze1d_ :: Tensor d -> Tensor d -> DimVal -> IO ()
  unsqueeze1d_ t0 t1 = Class.unsqueeze1d_ (dynamic t0) (dynamic t1)

