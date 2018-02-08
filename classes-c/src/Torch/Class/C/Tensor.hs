module Torch.Class.C.Tensor where

import THTypes
import Foreign
import Foreign.C.Types
import Torch.Class.C.Internal

class IsTensor t where
  clearFlag :: t -> Int8 -> IO ()
  tensordata :: t -> IO [HsReal t]
  desc :: t -> IO CTHDescBuff
  expand :: t -> t -> Ptr CTHLongStorage -> IO ()
  expandNd :: Ptr t -> Ptr t -> Int32 -> IO ()
  free :: t -> IO ()
  freeCopyTo :: t -> t -> IO ()
  get1d :: t -> Int64 -> IO (HsReal t)
  get2d :: t -> Int64 -> Int64 -> IO (HsReal t)
  get3d :: t -> Int64 -> Int64 -> Int64 -> IO (HsReal t)
  get4d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> IO (HsReal t)
  isContiguous :: t -> IO Bool
  isSameSizeAs :: t -> t -> IO Bool
  isSetTo :: t -> t -> IO Bool
  isSize :: t -> Ptr CTHLongStorage -> IO Bool
  nDimension :: t -> IO Int32
  nElement :: t -> IO Int64
  narrow :: t -> t -> Int32 -> Int64 -> Int64 -> IO ()
  new :: IO t
  newClone :: t -> IO t
  newContiguous :: t -> IO t
  newExpand :: t -> Ptr CTHLongStorage -> IO t
  newNarrow :: t -> Int32 -> Int64 -> Int64 -> IO t
  newSelect :: t -> Int32 -> Int64 -> IO t
  newSizeOf :: t -> IO (Ptr CTHLongStorage)
  newStrideOf :: t -> IO (Ptr CTHLongStorage)
  newTranspose :: t -> Int32 -> Int32 -> IO t
  newUnfold :: t -> Int32 -> Int64 -> Int64 -> IO t
  newView :: t -> Ptr CTHLongStorage -> IO t
  newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO t
  newWithSize1d :: Int64 -> IO t
  newWithSize2d :: Int64 -> Int64 -> IO t
  newWithSize3d :: Int64 -> Int64 -> Int64 -> IO t
  newWithSize4d :: Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage :: HsStorage t -> Int64 -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO t
  newWithStorage1d :: HsStorage t -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage2d :: HsStorage t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage3d :: HsStorage t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage4d :: HsStorage t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithTensor :: t -> IO t
  resize :: t -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  resize1d :: t -> Int64 -> IO ()
  resize2d :: t -> Int64 -> Int64 -> IO ()
  resize3d :: t -> Int64 -> Int64 -> Int64 -> IO ()
  resize4d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize5d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resizeAs :: t -> t -> IO ()
  resizeNd :: t -> Int32 -> Ptr CLLong -> Ptr CLLong -> IO ()
  retain :: t -> IO ()
  select :: t -> t -> Int32 -> Int64 -> IO ()
  set :: t -> t -> IO ()
  set1d :: t -> Int64 -> HsReal t -> IO ()
  set2d :: t -> Int64 -> Int64 -> HsReal t -> IO ()
  set3d :: t -> Int64 -> Int64 -> Int64 -> HsReal t -> IO ()
  set4d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal t -> IO ()
  setFlag :: t -> Int8 -> IO ()
  setStorage :: t -> HsStorage t -> Int64 -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  setStorage1d :: t -> HsStorage t -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage2d :: t -> HsStorage t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage3d :: t -> HsStorage t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage4d :: t -> HsStorage t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorageNd :: t -> HsStorage t -> Int64 -> Int32 -> Ptr CLLong -> Ptr CLLong -> IO ()
  size :: t -> Int32 -> IO Int64
  sizeDesc :: t -> IO CTHDescBuff
  squeeze :: t -> t -> IO ()
  squeeze1d :: t -> t -> Int32 -> IO ()
  storage :: t -> IO (HsStorage t)
  storageOffset :: t -> IO Int64
  stride :: t -> Int32 -> IO Int64
  transpose :: t -> t -> Int32 -> Int32 -> IO ()
  unfold :: t -> t -> Int32 -> Int64 -> Int64 -> IO ()
  unsqueeze1d :: t -> t -> Int32 -> IO ()
