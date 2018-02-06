module Torch.Class.Tensor where

import THTypes
import Foreign
import Foreign.C.Types
import Torch.Class.Internal

class Tensor t where
  clearFlag :: t -> CChar -> IO ()
  tensordata :: t -> IO (Ptr (HsReal t))
  desc :: t -> CTHDescBuff
  expand :: t -> t -> Ptr CTHLongStorage -> IO ()
  expandNd :: Ptr t -> Ptr t -> CInt -> IO ()
  free :: t -> IO ()
  freeCopyTo :: t -> t -> IO ()
  get1d :: t -> CLLong -> HsReal t
  get2d :: t -> CLLong -> CLLong -> HsReal t
  get3d :: t -> CLLong -> CLLong -> CLLong -> HsReal t
  get4d :: t -> CLLong -> CLLong -> CLLong -> CLLong -> HsReal t
  isContiguous :: t -> CInt
  isSameSizeAs :: t -> t -> CInt
  isSetTo :: t -> t -> CInt
  isSize :: t -> Ptr CTHLongStorage -> CInt
  nDimension :: t -> CInt
  nElement :: t -> CPtrdiff
  narrow :: t -> t -> CInt -> CLLong -> CLLong -> IO ()
  new :: IO t
  newClone :: t -> IO t
  newContiguous :: t -> IO t
  newExpand :: t -> Ptr CTHLongStorage -> IO t
  newNarrow :: t -> CInt -> CLLong -> CLLong -> IO t
  newSelect :: t -> CInt -> CLLong -> IO t
  newSizeOf :: t -> IO (Ptr CTHLongStorage)
  newStrideOf :: t -> IO (Ptr CTHLongStorage)
  newTranspose :: t -> CInt -> CInt -> IO t
  newUnfold :: t -> CInt -> CLLong -> CLLong -> IO t
  newView :: t -> Ptr CTHLongStorage -> IO t
  newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO t
  newWithSize1d :: CLLong -> IO t
  newWithSize2d :: CLLong -> CLLong -> IO t
  newWithSize3d :: CLLong -> CLLong -> CLLong -> IO t
  newWithSize4d :: CLLong -> CLLong -> CLLong -> CLLong -> IO t
  newWithStorage :: HsStorage t -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO t
  newWithStorage1d :: HsStorage t -> CPtrdiff -> CLLong -> CLLong -> IO t
  newWithStorage2d :: HsStorage t -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO t
  newWithStorage3d :: HsStorage t -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO t
  newWithStorage4d :: HsStorage t -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO t
  newWithTensor :: t -> IO t
  resize :: t -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  resize1d :: t -> CLLong -> IO ()
  resize2d :: t -> CLLong -> CLLong -> IO ()
  resize3d :: t -> CLLong -> CLLong -> CLLong -> IO ()
  resize4d :: t -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  resize5d :: t -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  resizeAs :: t -> t -> IO ()
  resizeNd :: t -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
  retain :: t -> IO ()
  select :: t -> t -> CInt -> CLLong -> IO ()
  set :: t -> t -> IO ()
  set1d :: t -> CLLong -> HsReal t -> IO ()
  set2d :: t -> CLLong -> CLLong -> HsReal t -> IO ()
  set3d :: t -> CLLong -> CLLong -> CLLong -> HsReal t -> IO ()
  set4d :: t -> CLLong -> CLLong -> CLLong -> CLLong -> HsReal t -> IO ()
  setFlag :: t -> CChar -> IO ()
  setStorage :: t -> HsStorage t -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  setStorage1d :: t -> HsStorage t -> CPtrdiff -> CLLong -> CLLong -> IO ()
  setStorage2d :: t -> HsStorage t -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorage3d :: t -> HsStorage t -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorage4d :: t -> HsStorage t -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorageNd :: t -> HsStorage t -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
  size :: t -> CInt -> CLLong
  sizeDesc :: t -> CTHDescBuff
  squeeze :: t -> t -> IO ()
  squeeze1d :: t -> t -> CInt -> IO ()
  storage :: t -> IO (HsStorage t)
  storageOffset :: t -> CPtrdiff
  stride :: t -> CInt -> CLLong
  transpose :: t -> t -> CInt -> CInt -> IO ()
  unfold :: t -> t -> CInt -> CLLong -> CLLong -> IO ()
  unsqueeze1d :: t -> t -> CInt -> IO ()
