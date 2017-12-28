{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Generic.Ops where

import Torch.Core.Tensor.Generic.Internal

import qualified THByteTensor as T
import qualified THDoubleTensor as T
import qualified THFloatTensor as T
import qualified THIntTensor as T
import qualified THLongTensor as T
import qualified THShortTensor as T

class GenericOps t where
  -- C-functions
  clearFlag :: Ptr t -> CChar -> IO ()
  tensordata :: Ptr t -> IO (Ptr (HaskType t))
  desc :: Ptr t -> CTHDescBuff
  expand :: Ptr t -> Ptr t -> Ptr CTHLongStorage -> IO ()
  expandNd :: Ptr (Ptr t) -> Ptr (Ptr t) -> CInt -> IO ()
  free :: Ptr t -> IO ()
  freeCopyTo :: Ptr t -> Ptr t -> IO ()
  get1d :: Ptr t -> CLLong -> HaskType t
  get2d :: Ptr t -> CLLong -> CLLong -> HaskType t
  get3d :: Ptr t -> CLLong -> CLLong -> CLLong -> HaskType t
  get4d :: Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> HaskType t
  isContiguous :: Ptr t -> CInt
  isSameSizeAs :: Ptr t -> Ptr t -> CInt
  isSetTo :: Ptr t -> Ptr t -> CInt
  isSize :: Ptr t -> Ptr CTHLongStorage -> CInt
  nDimension :: Ptr t -> CInt
  nElement :: Ptr t -> CPtrdiff
  narrow :: Ptr t -> Ptr t -> CInt -> CLLong -> CLLong -> IO ()
  new :: IO (Ptr t)
  newClone :: Ptr t -> IO (Ptr t)
  newContiguous :: Ptr t -> IO (Ptr t)
  newExpand :: Ptr t -> Ptr CTHLongStorage -> IO (Ptr t)
  newNarrow :: Ptr t -> CInt -> CLLong -> CLLong -> IO (Ptr t)
  newSelect :: Ptr t -> CInt -> CLLong -> IO (Ptr t)
  newSizeOf :: Ptr t -> IO (Ptr CTHLongStorage)
  newStrideOf :: Ptr t -> IO (Ptr CTHLongStorage)
  newTranspose :: Ptr t -> CInt -> CInt -> IO (Ptr t)
  newUnfold :: Ptr t -> CInt -> CLLong -> CLLong -> IO (Ptr t)
  newView :: Ptr t -> Ptr CTHLongStorage -> IO (Ptr t)
  newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr t)
  newWithSize1d :: CLLong -> IO (Ptr t)
  newWithSize2d :: CLLong -> CLLong -> IO (Ptr t)
  newWithSize3d :: CLLong -> CLLong -> CLLong -> IO (Ptr t)
  newWithSize4d :: CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t)
  newWithStorage :: Ptr (Storage t) -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr t)
  newWithStorage1d :: Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr t)
  newWithStorage2d :: Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t)
  newWithStorage3d :: Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t)
  newWithStorage4d :: Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t)
  newWithTensor :: Ptr t -> IO (Ptr t)
  resize :: Ptr t -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  resize1d :: Ptr t -> CLLong -> IO ()
  resize2d :: Ptr t -> CLLong -> CLLong -> IO ()
  resize3d :: Ptr t -> CLLong -> CLLong -> CLLong -> IO ()
  resize4d :: Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  resize5d :: Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  resizeAs :: Ptr t -> Ptr t -> IO ()
  resizeNd :: Ptr t -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
  retain :: Ptr t -> IO ()
  select :: Ptr t -> Ptr t -> CInt -> CLLong -> IO ()
  set :: Ptr t -> Ptr t -> IO ()
  set1d :: Ptr t -> CLLong -> HaskType t -> IO ()
  set2d :: Ptr t -> CLLong -> CLLong -> HaskType t -> IO ()
  set3d :: Ptr t -> CLLong -> CLLong -> CLLong -> HaskType t -> IO ()
  set4d :: Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> HaskType t -> IO ()
  setFlag :: Ptr t -> CChar -> IO ()
  setStorage :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  setStorage1d :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> IO ()
  setStorage2d :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorage3d :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorage4d :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorageNd :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
  size :: Ptr t -> CInt -> CLLong
  sizeDesc :: Ptr t -> CTHDescBuff
  squeeze :: Ptr t -> Ptr t -> IO ()
  squeeze1d :: Ptr t -> Ptr t -> CInt -> IO ()
  storage :: Ptr t -> IO (Ptr (Storage t))
  storageOffset :: Ptr t -> CPtrdiff
  stride :: Ptr t -> CInt -> CLLong
  transpose :: Ptr t -> Ptr t -> CInt -> CInt -> IO ()
  unfold :: Ptr t -> Ptr t -> CInt -> CLLong -> CLLong -> IO ()
  unsqueeze1d :: Ptr t -> Ptr t -> CInt -> IO ()

  -- C-Function Pointers

  _clearFlag :: FunPtr (Ptr t -> CChar -> IO ())
  _tensordata :: FunPtr (Ptr t -> IO (Ptr (HaskType t)))
  _desc :: FunPtr (Ptr t -> CTHDescBuff)
  _expand :: FunPtr (Ptr t -> Ptr t -> Ptr CTHLongStorage -> IO ())
  _expandNd :: FunPtr (Ptr (Ptr t) -> Ptr (Ptr t) -> CInt -> IO ())
  _free :: FunPtr (Ptr t -> IO ())
  _freeCopyTo :: FunPtr (Ptr t -> Ptr t -> IO ())
  _get1d :: FunPtr (Ptr t -> CLLong -> HaskType t)
  _get2d :: FunPtr (Ptr t -> CLLong -> CLLong -> HaskType t)
  _get3d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> HaskType t)
  _get4d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> HaskType t)
  _isContiguous :: FunPtr (Ptr t -> CInt)
  _isSameSizeAs :: FunPtr (Ptr t -> Ptr t -> CInt)
  _isSetTo :: FunPtr (Ptr t -> Ptr t -> CInt)
  _isSize :: FunPtr (Ptr t -> Ptr CTHLongStorage -> CInt)
  _nDimension :: FunPtr (Ptr t -> CInt)
  _nElement :: FunPtr (Ptr t -> CPtrdiff)
  _narrow :: FunPtr (Ptr t -> Ptr t -> CInt -> CLLong -> CLLong -> IO ())
  _new :: FunPtr (IO (Ptr t))
  _newClone :: FunPtr (Ptr t -> IO (Ptr t))
  _newContiguous :: FunPtr (Ptr t -> IO (Ptr t))
  _newExpand :: FunPtr (Ptr t -> Ptr CTHLongStorage -> IO (Ptr t))
  _newNarrow :: FunPtr (Ptr t -> CInt -> CLLong -> CLLong -> IO (Ptr t))
  _newSelect :: FunPtr (Ptr t -> CInt -> CLLong -> IO (Ptr t))
  _newSizeOf :: FunPtr (Ptr t -> IO (Ptr CTHLongStorage))
  _newStrideOf :: FunPtr (Ptr t -> IO (Ptr CTHLongStorage))
  _newTranspose :: FunPtr (Ptr t -> CInt -> CInt -> IO (Ptr t))
  _newUnfold :: FunPtr (Ptr t -> CInt -> CLLong -> CLLong -> IO (Ptr t))
  _newView :: FunPtr (Ptr t -> Ptr CTHLongStorage -> IO (Ptr t))
  _newWithSize :: FunPtr (Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr t))
  _newWithSize1d :: FunPtr (CLLong -> IO (Ptr t))
  _newWithSize2d :: FunPtr (CLLong -> CLLong -> IO (Ptr t))
  _newWithSize3d :: FunPtr (CLLong -> CLLong -> CLLong -> IO (Ptr t))
  _newWithSize4d :: FunPtr (CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t))
  _newWithStorage :: FunPtr (Ptr (Storage t) -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr t))
  _newWithStorage1d :: FunPtr (Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr t))
  _newWithStorage2d :: FunPtr (Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t))
  _newWithStorage3d :: FunPtr (Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t))
  _newWithStorage4d :: FunPtr (Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t))
  _newWithTensor :: FunPtr (Ptr t -> IO (Ptr t))
  _resize :: FunPtr (Ptr t -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())
  _resize1d :: FunPtr (Ptr t -> CLLong -> IO ())
  _resize2d :: FunPtr (Ptr t -> CLLong -> CLLong -> IO ())
  _resize3d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> IO ())
  _resize4d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  _resize5d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  _resizeAs :: FunPtr (Ptr t -> Ptr t -> IO ())
  _resizeNd :: FunPtr (Ptr t -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())
  _retain :: FunPtr (Ptr t -> IO ())
  _select :: FunPtr (Ptr t -> Ptr t -> CInt -> CLLong -> IO ())
  _set :: FunPtr (Ptr t -> Ptr t -> IO ())
  _set1d :: FunPtr (Ptr t -> CLLong -> HaskType t -> IO ())
  _set2d :: FunPtr (Ptr t -> CLLong -> CLLong -> HaskType t -> IO ())
  _set3d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> HaskType t -> IO ())
  _set4d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> HaskType t -> IO ())
  _setFlag :: FunPtr (Ptr t -> CChar -> IO ())
  _setStorage :: FunPtr (Ptr t -> Ptr (Storage t) -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())
  _setStorage1d :: FunPtr (Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> IO ())
  _setStorage2d :: FunPtr (Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  _setStorage3d :: FunPtr (Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  _setStorage4d :: FunPtr (Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  _setStorageNd :: FunPtr (Ptr t -> Ptr (Storage t) -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())
  _size :: FunPtr (Ptr t -> CInt -> CLLong)
  _sizeDesc :: FunPtr (Ptr t -> CTHDescBuff)
  _squeeze :: FunPtr (Ptr t -> Ptr t -> IO ())
  _squeeze1d :: FunPtr (Ptr t -> Ptr t -> CInt -> IO ())
  _storage :: FunPtr (Ptr t -> IO (Ptr (Storage t)))
  _storageOffset :: FunPtr (Ptr t -> CPtrdiff)
  _stride :: FunPtr (Ptr t -> CInt -> CLLong)
  _transpose :: FunPtr (Ptr t -> Ptr t -> CInt -> CInt -> IO ())
  _unfold :: FunPtr (Ptr t -> Ptr t -> CInt -> CLLong -> CLLong -> IO ())
  _unsqueeze1d :: FunPtr (Ptr t -> Ptr t -> CInt -> IO ())


instance GenericOps CTHByteTensor where
  storage = T.c_THByteTensor_storage
  storageOffset = T.c_THByteTensor_storageOffset
  nDimension = T.c_THByteTensor_nDimension
  size = T.c_THByteTensor_size
  stride = T.c_THByteTensor_stride
  newSizeOf = T.c_THByteTensor_newSizeOf
  newStrideOf = T.c_THByteTensor_newStrideOf
  tensordata = T.c_THByteTensor_data
  setFlag = T.c_THByteTensor_setFlag
  clearFlag = T.c_THByteTensor_clearFlag
  new = T.c_THByteTensor_new
  newWithTensor = T.c_THByteTensor_newWithTensor
  newWithStorage = T.c_THByteTensor_newWithStorage
  newWithStorage1d = T.c_THByteTensor_newWithStorage1d
  newWithStorage2d = T.c_THByteTensor_newWithStorage2d
  newWithStorage3d = T.c_THByteTensor_newWithStorage3d
  newWithStorage4d = T.c_THByteTensor_newWithStorage4d
  newWithSize = T.c_THByteTensor_newWithSize
  newWithSize1d = T.c_THByteTensor_newWithSize1d
  newWithSize2d = T.c_THByteTensor_newWithSize2d
  newWithSize3d = T.c_THByteTensor_newWithSize3d
  newWithSize4d = T.c_THByteTensor_newWithSize4d
  newClone = T.c_THByteTensor_newClone
  newContiguous = T.c_THByteTensor_newContiguous
  newSelect = T.c_THByteTensor_newSelect
  newNarrow = T.c_THByteTensor_newNarrow
  newTranspose = T.c_THByteTensor_newTranspose
  newUnfold = T.c_THByteTensor_newUnfold
  newView = T.c_THByteTensor_newView
  newExpand = T.c_THByteTensor_newExpand
  expand = T.c_THByteTensor_expand
  expandNd = T.c_THByteTensor_expandNd
  resize = T.c_THByteTensor_resize
  resizeAs = T.c_THByteTensor_resizeAs
  resizeNd = T.c_THByteTensor_resizeNd
  resize1d = T.c_THByteTensor_resize1d
  resize2d = T.c_THByteTensor_resize2d
  resize3d = T.c_THByteTensor_resize3d
  resize4d = T.c_THByteTensor_resize4d
  resize5d = T.c_THByteTensor_resize5d
  set = T.c_THByteTensor_set
  setStorage = T.c_THByteTensor_setStorage
  setStorageNd = T.c_THByteTensor_setStorageNd
  setStorage1d = T.c_THByteTensor_setStorage1d
  setStorage2d = T.c_THByteTensor_setStorage2d
  setStorage3d = T.c_THByteTensor_setStorage3d
  setStorage4d = T.c_THByteTensor_setStorage4d
  narrow = T.c_THByteTensor_narrow
  select = T.c_THByteTensor_select
  transpose = T.c_THByteTensor_transpose
  unfold = T.c_THByteTensor_unfold
  squeeze = T.c_THByteTensor_squeeze
  squeeze1d = T.c_THByteTensor_squeeze1d
  unsqueeze1d = T.c_THByteTensor_unsqueeze1d
  isContiguous = T.c_THByteTensor_isContiguous
  isSameSizeAs = T.c_THByteTensor_isSameSizeAs
  isSetTo = T.c_THByteTensor_isSetTo
  isSize = T.c_THByteTensor_isSize
  nElement = T.c_THByteTensor_nElement
  retain = T.c_THByteTensor_retain
  free = T.c_THByteTensor_free
  freeCopyTo = T.c_THByteTensor_freeCopyTo
  set1d = T.c_THByteTensor_set1d
  set2d = T.c_THByteTensor_set2d
  set3d = T.c_THByteTensor_set3d
  set4d = T.c_THByteTensor_set4d
  get1d = T.c_THByteTensor_get1d
  get2d = T.c_THByteTensor_get2d
  get3d = T.c_THByteTensor_get3d
  get4d = T.c_THByteTensor_get4d
  desc = T.c_THByteTensor_desc
  sizeDesc = T.c_THByteTensor_sizeDesc
  _storage = T.p_THByteTensor_storage
  _storageOffset = T.p_THByteTensor_storageOffset
  _nDimension = T.p_THByteTensor_nDimension
  _size = T.p_THByteTensor_size
  _stride = T.p_THByteTensor_stride
  _newSizeOf = T.p_THByteTensor_newSizeOf
  _newStrideOf = T.p_THByteTensor_newStrideOf
  _tensordata = T.p_THByteTensor_data
  _setFlag = T.p_THByteTensor_setFlag
  _clearFlag = T.p_THByteTensor_clearFlag
  _new = T.p_THByteTensor_new
  _newWithTensor = T.p_THByteTensor_newWithTensor
  _newWithStorage = T.p_THByteTensor_newWithStorage
  _newWithStorage1d = T.p_THByteTensor_newWithStorage1d
  _newWithStorage2d = T.p_THByteTensor_newWithStorage2d
  _newWithStorage3d = T.p_THByteTensor_newWithStorage3d
  _newWithStorage4d = T.p_THByteTensor_newWithStorage4d
  _newWithSize = T.p_THByteTensor_newWithSize
  _newWithSize1d = T.p_THByteTensor_newWithSize1d
  _newWithSize2d = T.p_THByteTensor_newWithSize2d
  _newWithSize3d = T.p_THByteTensor_newWithSize3d
  _newWithSize4d = T.p_THByteTensor_newWithSize4d
  _newClone = T.p_THByteTensor_newClone
  _newContiguous = T.p_THByteTensor_newContiguous
  _newSelect = T.p_THByteTensor_newSelect
  _newNarrow = T.p_THByteTensor_newNarrow
  _newTranspose = T.p_THByteTensor_newTranspose
  _newUnfold = T.p_THByteTensor_newUnfold
  _newView = T.p_THByteTensor_newView
  _newExpand = T.p_THByteTensor_newExpand
  _expand = T.p_THByteTensor_expand
  _expandNd = T.p_THByteTensor_expandNd
  _resize = T.p_THByteTensor_resize
  _resizeAs = T.p_THByteTensor_resizeAs
  _resizeNd = T.p_THByteTensor_resizeNd
  _resize1d = T.p_THByteTensor_resize1d
  _resize2d = T.p_THByteTensor_resize2d
  _resize3d = T.p_THByteTensor_resize3d
  _resize4d = T.p_THByteTensor_resize4d
  _resize5d = T.p_THByteTensor_resize5d
  _set = T.p_THByteTensor_set
  _setStorage = T.p_THByteTensor_setStorage
  _setStorageNd = T.p_THByteTensor_setStorageNd
  _setStorage1d = T.p_THByteTensor_setStorage1d
  _setStorage2d = T.p_THByteTensor_setStorage2d
  _setStorage3d = T.p_THByteTensor_setStorage3d
  _setStorage4d = T.p_THByteTensor_setStorage4d
  _narrow = T.p_THByteTensor_narrow
  _select = T.p_THByteTensor_select
  _transpose = T.p_THByteTensor_transpose
  _unfold = T.p_THByteTensor_unfold
  _squeeze = T.p_THByteTensor_squeeze
  _squeeze1d = T.p_THByteTensor_squeeze1d
  _unsqueeze1d = T.p_THByteTensor_unsqueeze1d
  _isContiguous = T.p_THByteTensor_isContiguous
  _isSameSizeAs = T.p_THByteTensor_isSameSizeAs
  _isSetTo = T.p_THByteTensor_isSetTo
  _isSize = T.p_THByteTensor_isSize
  _nElement = T.p_THByteTensor_nElement
  _retain = T.p_THByteTensor_retain
  _free = T.p_THByteTensor_free
  _freeCopyTo = T.p_THByteTensor_freeCopyTo
  _set1d = T.p_THByteTensor_set1d
  _set2d = T.p_THByteTensor_set2d
  _set3d = T.p_THByteTensor_set3d
  _set4d = T.p_THByteTensor_set4d
  _get1d = T.p_THByteTensor_get1d
  _get2d = T.p_THByteTensor_get2d
  _get3d = T.p_THByteTensor_get3d
  _get4d = T.p_THByteTensor_get4d
  _desc = T.p_THByteTensor_desc
  _sizeDesc = T.p_THByteTensor_sizeDesc

instance GenericOps CTHDoubleTensor where
  clearFlag = T.c_THDoubleTensor_clearFlag
  tensordata = T.c_THDoubleTensor_data
  desc = T.c_THDoubleTensor_desc
  expand = T.c_THDoubleTensor_expand
  expandNd = T.c_THDoubleTensor_expandNd
  free = T.c_THDoubleTensor_free
  freeCopyTo = T.c_THDoubleTensor_freeCopyTo
  get1d = T.c_THDoubleTensor_get1d
  get2d = T.c_THDoubleTensor_get2d
  get3d = T.c_THDoubleTensor_get3d
  get4d = T.c_THDoubleTensor_get4d
  isContiguous = T.c_THDoubleTensor_isContiguous
  isSameSizeAs = T.c_THDoubleTensor_isSameSizeAs
  isSetTo = T.c_THDoubleTensor_isSetTo
  isSize = T.c_THDoubleTensor_isSize
  nDimension = T.c_THDoubleTensor_nDimension
  nElement = T.c_THDoubleTensor_nElement
  narrow = T.c_THDoubleTensor_narrow
  new = T.c_THDoubleTensor_new
  newClone = T.c_THDoubleTensor_newClone
  newContiguous = T.c_THDoubleTensor_newContiguous
  newExpand = T.c_THDoubleTensor_newExpand
  newNarrow = T.c_THDoubleTensor_newNarrow
  newSelect = T.c_THDoubleTensor_newSelect
  newSizeOf = T.c_THDoubleTensor_newSizeOf
  newStrideOf = T.c_THDoubleTensor_newStrideOf
  newTranspose = T.c_THDoubleTensor_newTranspose
  newUnfold = T.c_THDoubleTensor_newUnfold
  newView = T.c_THDoubleTensor_newView
  newWithSize = T.c_THDoubleTensor_newWithSize
  newWithSize1d = T.c_THDoubleTensor_newWithSize1d
  newWithSize2d = T.c_THDoubleTensor_newWithSize2d
  newWithSize3d = T.c_THDoubleTensor_newWithSize3d
  newWithSize4d = T.c_THDoubleTensor_newWithSize4d
  newWithStorage = T.c_THDoubleTensor_newWithStorage
  newWithStorage1d = T.c_THDoubleTensor_newWithStorage1d
  newWithStorage2d = T.c_THDoubleTensor_newWithStorage2d
  newWithStorage3d = T.c_THDoubleTensor_newWithStorage3d
  newWithStorage4d = T.c_THDoubleTensor_newWithStorage4d
  newWithTensor = T.c_THDoubleTensor_newWithTensor
  resize = T.c_THDoubleTensor_resize
  resize1d = T.c_THDoubleTensor_resize1d
  resize2d = T.c_THDoubleTensor_resize2d
  resize3d = T.c_THDoubleTensor_resize3d
  resize4d = T.c_THDoubleTensor_resize4d
  resize5d = T.c_THDoubleTensor_resize5d
  resizeAs = T.c_THDoubleTensor_resizeAs
  resizeNd = T.c_THDoubleTensor_resizeNd
  retain = T.c_THDoubleTensor_retain
  select = T.c_THDoubleTensor_select
  set = T.c_THDoubleTensor_set
  set1d = T.c_THDoubleTensor_set1d
  set2d = T.c_THDoubleTensor_set2d
  set3d = T.c_THDoubleTensor_set3d
  set4d = T.c_THDoubleTensor_set4d
  setFlag = T.c_THDoubleTensor_setFlag
  setStorage = T.c_THDoubleTensor_setStorage
  setStorage1d = T.c_THDoubleTensor_setStorage1d
  setStorage2d = T.c_THDoubleTensor_setStorage2d
  setStorage3d = T.c_THDoubleTensor_setStorage3d
  setStorage4d = T.c_THDoubleTensor_setStorage4d
  setStorageNd = T.c_THDoubleTensor_setStorageNd
  size = T.c_THDoubleTensor_size
  sizeDesc = T.c_THDoubleTensor_sizeDesc
  squeeze = T.c_THDoubleTensor_squeeze
  squeeze1d = T.c_THDoubleTensor_squeeze1d
  storage = T.c_THDoubleTensor_storage
  storageOffset = T.c_THDoubleTensor_storageOffset
  stride = T.c_THDoubleTensor_stride
  transpose = T.c_THDoubleTensor_transpose
  unfold = T.c_THDoubleTensor_unfold
  unsqueeze1d = T.c_THDoubleTensor_unsqueeze1d

  _clearFlag = T.p_THDoubleTensor_clearFlag
  _tensordata = T.p_THDoubleTensor_data
  _desc = T.p_THDoubleTensor_desc
  _expand = T.p_THDoubleTensor_expand
  _expandNd = T.p_THDoubleTensor_expandNd
  _free = T.p_THDoubleTensor_free
  _freeCopyTo = T.p_THDoubleTensor_freeCopyTo
  _get1d = T.p_THDoubleTensor_get1d
  _get2d = T.p_THDoubleTensor_get2d
  _get3d = T.p_THDoubleTensor_get3d
  _get4d = T.p_THDoubleTensor_get4d
  _isContiguous = T.p_THDoubleTensor_isContiguous
  _isSameSizeAs = T.p_THDoubleTensor_isSameSizeAs
  _isSetTo = T.p_THDoubleTensor_isSetTo
  _isSize = T.p_THDoubleTensor_isSize
  _nDimension = T.p_THDoubleTensor_nDimension
  _nElement = T.p_THDoubleTensor_nElement
  _narrow = T.p_THDoubleTensor_narrow
  _new = T.p_THDoubleTensor_new
  _newClone = T.p_THDoubleTensor_newClone
  _newContiguous = T.p_THDoubleTensor_newContiguous
  _newExpand = T.p_THDoubleTensor_newExpand
  _newNarrow = T.p_THDoubleTensor_newNarrow
  _newSelect = T.p_THDoubleTensor_newSelect
  _newSizeOf = T.p_THDoubleTensor_newSizeOf
  _newStrideOf = T.p_THDoubleTensor_newStrideOf
  _newTranspose = T.p_THDoubleTensor_newTranspose
  _newUnfold = T.p_THDoubleTensor_newUnfold
  _newView = T.p_THDoubleTensor_newView
  _newWithSize = T.p_THDoubleTensor_newWithSize
  _newWithSize1d = T.p_THDoubleTensor_newWithSize1d
  _newWithSize2d = T.p_THDoubleTensor_newWithSize2d
  _newWithSize3d = T.p_THDoubleTensor_newWithSize3d
  _newWithSize4d = T.p_THDoubleTensor_newWithSize4d
  _newWithStorage = T.p_THDoubleTensor_newWithStorage
  _newWithStorage1d = T.p_THDoubleTensor_newWithStorage1d
  _newWithStorage2d = T.p_THDoubleTensor_newWithStorage2d
  _newWithStorage3d = T.p_THDoubleTensor_newWithStorage3d
  _newWithStorage4d = T.p_THDoubleTensor_newWithStorage4d
  _newWithTensor = T.p_THDoubleTensor_newWithTensor
  _resize = T.p_THDoubleTensor_resize
  _resize1d = T.p_THDoubleTensor_resize1d
  _resize2d = T.p_THDoubleTensor_resize2d
  _resize3d = T.p_THDoubleTensor_resize3d
  _resize4d = T.p_THDoubleTensor_resize4d
  _resize5d = T.p_THDoubleTensor_resize5d
  _resizeAs = T.p_THDoubleTensor_resizeAs
  _resizeNd = T.p_THDoubleTensor_resizeNd
  _retain = T.p_THDoubleTensor_retain
  _select = T.p_THDoubleTensor_select
  _set = T.p_THDoubleTensor_set
  _set1d = T.p_THDoubleTensor_set1d
  _set2d = T.p_THDoubleTensor_set2d
  _set3d = T.p_THDoubleTensor_set3d
  _set4d = T.p_THDoubleTensor_set4d
  _setFlag = T.p_THDoubleTensor_setFlag
  _setStorage = T.p_THDoubleTensor_setStorage
  _setStorage1d = T.p_THDoubleTensor_setStorage1d
  _setStorage2d = T.p_THDoubleTensor_setStorage2d
  _setStorage3d = T.p_THDoubleTensor_setStorage3d
  _setStorage4d = T.p_THDoubleTensor_setStorage4d
  _setStorageNd = T.p_THDoubleTensor_setStorageNd
  _size = T.p_THDoubleTensor_size
  _sizeDesc = T.p_THDoubleTensor_sizeDesc
  _squeeze = T.p_THDoubleTensor_squeeze
  _squeeze1d = T.p_THDoubleTensor_squeeze1d
  _storage = T.p_THDoubleTensor_storage
  _storageOffset = T.p_THDoubleTensor_storageOffset
  _stride = T.p_THDoubleTensor_stride
  _transpose = T.p_THDoubleTensor_transpose
  _unfold = T.p_THDoubleTensor_unfold
  _unsqueeze1d = T.p_THDoubleTensor_unsqueeze1d


instance GenericOps CTHFloatTensor where
  clearFlag = T.c_THFloatTensor_clearFlag
  tensordata = T.c_THFloatTensor_data
  desc = T.c_THFloatTensor_desc
  expand = T.c_THFloatTensor_expand
  expandNd = T.c_THFloatTensor_expandNd
  free = T.c_THFloatTensor_free
  freeCopyTo = T.c_THFloatTensor_freeCopyTo
  get1d = T.c_THFloatTensor_get1d
  get2d = T.c_THFloatTensor_get2d
  get3d = T.c_THFloatTensor_get3d
  get4d = T.c_THFloatTensor_get4d
  isContiguous = T.c_THFloatTensor_isContiguous
  isSameSizeAs = T.c_THFloatTensor_isSameSizeAs
  isSetTo = T.c_THFloatTensor_isSetTo
  isSize = T.c_THFloatTensor_isSize
  nDimension = T.c_THFloatTensor_nDimension
  nElement = T.c_THFloatTensor_nElement
  narrow = T.c_THFloatTensor_narrow
  new = T.c_THFloatTensor_new
  newClone = T.c_THFloatTensor_newClone
  newContiguous = T.c_THFloatTensor_newContiguous
  newExpand = T.c_THFloatTensor_newExpand
  newNarrow = T.c_THFloatTensor_newNarrow
  newSelect = T.c_THFloatTensor_newSelect
  newSizeOf = T.c_THFloatTensor_newSizeOf
  newStrideOf = T.c_THFloatTensor_newStrideOf
  newTranspose = T.c_THFloatTensor_newTranspose
  newUnfold = T.c_THFloatTensor_newUnfold
  newView = T.c_THFloatTensor_newView
  newWithSize = T.c_THFloatTensor_newWithSize
  newWithSize1d = T.c_THFloatTensor_newWithSize1d
  newWithSize2d = T.c_THFloatTensor_newWithSize2d
  newWithSize3d = T.c_THFloatTensor_newWithSize3d
  newWithSize4d = T.c_THFloatTensor_newWithSize4d
  newWithStorage = T.c_THFloatTensor_newWithStorage
  newWithStorage1d = T.c_THFloatTensor_newWithStorage1d
  newWithStorage2d = T.c_THFloatTensor_newWithStorage2d
  newWithStorage3d = T.c_THFloatTensor_newWithStorage3d
  newWithStorage4d = T.c_THFloatTensor_newWithStorage4d
  newWithTensor = T.c_THFloatTensor_newWithTensor
  resize = T.c_THFloatTensor_resize
  resize1d = T.c_THFloatTensor_resize1d
  resize2d = T.c_THFloatTensor_resize2d
  resize3d = T.c_THFloatTensor_resize3d
  resize4d = T.c_THFloatTensor_resize4d
  resize5d = T.c_THFloatTensor_resize5d
  resizeAs = T.c_THFloatTensor_resizeAs
  resizeNd = T.c_THFloatTensor_resizeNd
  retain = T.c_THFloatTensor_retain
  select = T.c_THFloatTensor_select
  set = T.c_THFloatTensor_set
  set1d = T.c_THFloatTensor_set1d
  set2d = T.c_THFloatTensor_set2d
  set3d = T.c_THFloatTensor_set3d
  set4d = T.c_THFloatTensor_set4d
  setFlag = T.c_THFloatTensor_setFlag
  setStorage = T.c_THFloatTensor_setStorage
  setStorage1d = T.c_THFloatTensor_setStorage1d
  setStorage2d = T.c_THFloatTensor_setStorage2d
  setStorage3d = T.c_THFloatTensor_setStorage3d
  setStorage4d = T.c_THFloatTensor_setStorage4d
  setStorageNd = T.c_THFloatTensor_setStorageNd
  size = T.c_THFloatTensor_size
  sizeDesc = T.c_THFloatTensor_sizeDesc
  squeeze = T.c_THFloatTensor_squeeze
  squeeze1d = T.c_THFloatTensor_squeeze1d
  storage = T.c_THFloatTensor_storage
  storageOffset = T.c_THFloatTensor_storageOffset
  stride = T.c_THFloatTensor_stride
  transpose = T.c_THFloatTensor_transpose
  unfold = T.c_THFloatTensor_unfold
  unsqueeze1d = T.c_THFloatTensor_unsqueeze1d

  _clearFlag = T.p_THFloatTensor_clearFlag
  _tensordata = T.p_THFloatTensor_data
  _desc = T.p_THFloatTensor_desc
  _expand = T.p_THFloatTensor_expand
  _expandNd = T.p_THFloatTensor_expandNd
  _free = T.p_THFloatTensor_free
  _freeCopyTo = T.p_THFloatTensor_freeCopyTo
  _get1d = T.p_THFloatTensor_get1d
  _get2d = T.p_THFloatTensor_get2d
  _get3d = T.p_THFloatTensor_get3d
  _get4d = T.p_THFloatTensor_get4d
  _isContiguous = T.p_THFloatTensor_isContiguous
  _isSameSizeAs = T.p_THFloatTensor_isSameSizeAs
  _isSetTo = T.p_THFloatTensor_isSetTo
  _isSize = T.p_THFloatTensor_isSize
  _nDimension = T.p_THFloatTensor_nDimension
  _nElement = T.p_THFloatTensor_nElement
  _narrow = T.p_THFloatTensor_narrow
  _new = T.p_THFloatTensor_new
  _newClone = T.p_THFloatTensor_newClone
  _newContiguous = T.p_THFloatTensor_newContiguous
  _newExpand = T.p_THFloatTensor_newExpand
  _newNarrow = T.p_THFloatTensor_newNarrow
  _newSelect = T.p_THFloatTensor_newSelect
  _newSizeOf = T.p_THFloatTensor_newSizeOf
  _newStrideOf = T.p_THFloatTensor_newStrideOf
  _newTranspose = T.p_THFloatTensor_newTranspose
  _newUnfold = T.p_THFloatTensor_newUnfold
  _newView = T.p_THFloatTensor_newView
  _newWithSize = T.p_THFloatTensor_newWithSize
  _newWithSize1d = T.p_THFloatTensor_newWithSize1d
  _newWithSize2d = T.p_THFloatTensor_newWithSize2d
  _newWithSize3d = T.p_THFloatTensor_newWithSize3d
  _newWithSize4d = T.p_THFloatTensor_newWithSize4d
  _newWithStorage = T.p_THFloatTensor_newWithStorage
  _newWithStorage1d = T.p_THFloatTensor_newWithStorage1d
  _newWithStorage2d = T.p_THFloatTensor_newWithStorage2d
  _newWithStorage3d = T.p_THFloatTensor_newWithStorage3d
  _newWithStorage4d = T.p_THFloatTensor_newWithStorage4d
  _newWithTensor = T.p_THFloatTensor_newWithTensor
  _resize = T.p_THFloatTensor_resize
  _resize1d = T.p_THFloatTensor_resize1d
  _resize2d = T.p_THFloatTensor_resize2d
  _resize3d = T.p_THFloatTensor_resize3d
  _resize4d = T.p_THFloatTensor_resize4d
  _resize5d = T.p_THFloatTensor_resize5d
  _resizeAs = T.p_THFloatTensor_resizeAs
  _resizeNd = T.p_THFloatTensor_resizeNd
  _retain = T.p_THFloatTensor_retain
  _select = T.p_THFloatTensor_select
  _set = T.p_THFloatTensor_set
  _set1d = T.p_THFloatTensor_set1d
  _set2d = T.p_THFloatTensor_set2d
  _set3d = T.p_THFloatTensor_set3d
  _set4d = T.p_THFloatTensor_set4d
  _setFlag = T.p_THFloatTensor_setFlag
  _setStorage = T.p_THFloatTensor_setStorage
  _setStorage1d = T.p_THFloatTensor_setStorage1d
  _setStorage2d = T.p_THFloatTensor_setStorage2d
  _setStorage3d = T.p_THFloatTensor_setStorage3d
  _setStorage4d = T.p_THFloatTensor_setStorage4d
  _setStorageNd = T.p_THFloatTensor_setStorageNd
  _size = T.p_THFloatTensor_size
  _sizeDesc = T.p_THFloatTensor_sizeDesc
  _squeeze = T.p_THFloatTensor_squeeze
  _squeeze1d = T.p_THFloatTensor_squeeze1d
  _storage = T.p_THFloatTensor_storage
  _storageOffset = T.p_THFloatTensor_storageOffset
  _stride = T.p_THFloatTensor_stride
  _transpose = T.p_THFloatTensor_transpose
  _unfold = T.p_THFloatTensor_unfold
  _unsqueeze1d = T.p_THFloatTensor_unsqueeze1d


instance GenericOps CTHIntTensor where
  clearFlag = T.c_THIntTensor_clearFlag
  tensordata = T.c_THIntTensor_data
  desc = T.c_THIntTensor_desc
  expand = T.c_THIntTensor_expand
  expandNd = T.c_THIntTensor_expandNd
  free = T.c_THIntTensor_free
  freeCopyTo = T.c_THIntTensor_freeCopyTo
  get1d = T.c_THIntTensor_get1d
  get2d = T.c_THIntTensor_get2d
  get3d = T.c_THIntTensor_get3d
  get4d = T.c_THIntTensor_get4d
  isContiguous = T.c_THIntTensor_isContiguous
  isSameSizeAs = T.c_THIntTensor_isSameSizeAs
  isSetTo = T.c_THIntTensor_isSetTo
  isSize = T.c_THIntTensor_isSize
  nDimension = T.c_THIntTensor_nDimension
  nElement = T.c_THIntTensor_nElement
  narrow = T.c_THIntTensor_narrow
  new = T.c_THIntTensor_new
  newClone = T.c_THIntTensor_newClone
  newContiguous = T.c_THIntTensor_newContiguous
  newExpand = T.c_THIntTensor_newExpand
  newNarrow = T.c_THIntTensor_newNarrow
  newSelect = T.c_THIntTensor_newSelect
  newSizeOf = T.c_THIntTensor_newSizeOf
  newStrideOf = T.c_THIntTensor_newStrideOf
  newTranspose = T.c_THIntTensor_newTranspose
  newUnfold = T.c_THIntTensor_newUnfold
  newView = T.c_THIntTensor_newView
  newWithSize = T.c_THIntTensor_newWithSize
  newWithSize1d = T.c_THIntTensor_newWithSize1d
  newWithSize2d = T.c_THIntTensor_newWithSize2d
  newWithSize3d = T.c_THIntTensor_newWithSize3d
  newWithSize4d = T.c_THIntTensor_newWithSize4d
  newWithStorage = T.c_THIntTensor_newWithStorage
  newWithStorage1d = T.c_THIntTensor_newWithStorage1d
  newWithStorage2d = T.c_THIntTensor_newWithStorage2d
  newWithStorage3d = T.c_THIntTensor_newWithStorage3d
  newWithStorage4d = T.c_THIntTensor_newWithStorage4d
  newWithTensor = T.c_THIntTensor_newWithTensor
  resize = T.c_THIntTensor_resize
  resize1d = T.c_THIntTensor_resize1d
  resize2d = T.c_THIntTensor_resize2d
  resize3d = T.c_THIntTensor_resize3d
  resize4d = T.c_THIntTensor_resize4d
  resize5d = T.c_THIntTensor_resize5d
  resizeAs = T.c_THIntTensor_resizeAs
  resizeNd = T.c_THIntTensor_resizeNd
  retain = T.c_THIntTensor_retain
  select = T.c_THIntTensor_select
  set = T.c_THIntTensor_set
  set1d = T.c_THIntTensor_set1d
  set2d = T.c_THIntTensor_set2d
  set3d = T.c_THIntTensor_set3d
  set4d = T.c_THIntTensor_set4d
  setFlag = T.c_THIntTensor_setFlag
  setStorage = T.c_THIntTensor_setStorage
  setStorage1d = T.c_THIntTensor_setStorage1d
  setStorage2d = T.c_THIntTensor_setStorage2d
  setStorage3d = T.c_THIntTensor_setStorage3d
  setStorage4d = T.c_THIntTensor_setStorage4d
  setStorageNd = T.c_THIntTensor_setStorageNd
  size = T.c_THIntTensor_size
  sizeDesc = T.c_THIntTensor_sizeDesc
  squeeze = T.c_THIntTensor_squeeze
  squeeze1d = T.c_THIntTensor_squeeze1d
  storage = T.c_THIntTensor_storage
  storageOffset = T.c_THIntTensor_storageOffset
  stride = T.c_THIntTensor_stride
  transpose = T.c_THIntTensor_transpose
  unfold = T.c_THIntTensor_unfold
  unsqueeze1d = T.c_THIntTensor_unsqueeze1d

  _clearFlag = T.p_THIntTensor_clearFlag
  _tensordata = T.p_THIntTensor_data
  _desc = T.p_THIntTensor_desc
  _expand = T.p_THIntTensor_expand
  _expandNd = T.p_THIntTensor_expandNd
  _free = T.p_THIntTensor_free
  _freeCopyTo = T.p_THIntTensor_freeCopyTo
  _get1d = T.p_THIntTensor_get1d
  _get2d = T.p_THIntTensor_get2d
  _get3d = T.p_THIntTensor_get3d
  _get4d = T.p_THIntTensor_get4d
  _isContiguous = T.p_THIntTensor_isContiguous
  _isSameSizeAs = T.p_THIntTensor_isSameSizeAs
  _isSetTo = T.p_THIntTensor_isSetTo
  _isSize = T.p_THIntTensor_isSize
  _nDimension = T.p_THIntTensor_nDimension
  _nElement = T.p_THIntTensor_nElement
  _narrow = T.p_THIntTensor_narrow
  _new = T.p_THIntTensor_new
  _newClone = T.p_THIntTensor_newClone
  _newContiguous = T.p_THIntTensor_newContiguous
  _newExpand = T.p_THIntTensor_newExpand
  _newNarrow = T.p_THIntTensor_newNarrow
  _newSelect = T.p_THIntTensor_newSelect
  _newSizeOf = T.p_THIntTensor_newSizeOf
  _newStrideOf = T.p_THIntTensor_newStrideOf
  _newTranspose = T.p_THIntTensor_newTranspose
  _newUnfold = T.p_THIntTensor_newUnfold
  _newView = T.p_THIntTensor_newView
  _newWithSize = T.p_THIntTensor_newWithSize
  _newWithSize1d = T.p_THIntTensor_newWithSize1d
  _newWithSize2d = T.p_THIntTensor_newWithSize2d
  _newWithSize3d = T.p_THIntTensor_newWithSize3d
  _newWithSize4d = T.p_THIntTensor_newWithSize4d
  _newWithStorage = T.p_THIntTensor_newWithStorage
  _newWithStorage1d = T.p_THIntTensor_newWithStorage1d
  _newWithStorage2d = T.p_THIntTensor_newWithStorage2d
  _newWithStorage3d = T.p_THIntTensor_newWithStorage3d
  _newWithStorage4d = T.p_THIntTensor_newWithStorage4d
  _newWithTensor = T.p_THIntTensor_newWithTensor
  _resize = T.p_THIntTensor_resize
  _resize1d = T.p_THIntTensor_resize1d
  _resize2d = T.p_THIntTensor_resize2d
  _resize3d = T.p_THIntTensor_resize3d
  _resize4d = T.p_THIntTensor_resize4d
  _resize5d = T.p_THIntTensor_resize5d
  _resizeAs = T.p_THIntTensor_resizeAs
  _resizeNd = T.p_THIntTensor_resizeNd
  _retain = T.p_THIntTensor_retain
  _select = T.p_THIntTensor_select
  _set = T.p_THIntTensor_set
  _set1d = T.p_THIntTensor_set1d
  _set2d = T.p_THIntTensor_set2d
  _set3d = T.p_THIntTensor_set3d
  _set4d = T.p_THIntTensor_set4d
  _setFlag = T.p_THIntTensor_setFlag
  _setStorage = T.p_THIntTensor_setStorage
  _setStorage1d = T.p_THIntTensor_setStorage1d
  _setStorage2d = T.p_THIntTensor_setStorage2d
  _setStorage3d = T.p_THIntTensor_setStorage3d
  _setStorage4d = T.p_THIntTensor_setStorage4d
  _setStorageNd = T.p_THIntTensor_setStorageNd
  _size = T.p_THIntTensor_size
  _sizeDesc = T.p_THIntTensor_sizeDesc
  _squeeze = T.p_THIntTensor_squeeze
  _squeeze1d = T.p_THIntTensor_squeeze1d
  _storage = T.p_THIntTensor_storage
  _storageOffset = T.p_THIntTensor_storageOffset
  _stride = T.p_THIntTensor_stride
  _transpose = T.p_THIntTensor_transpose
  _unfold = T.p_THIntTensor_unfold
  _unsqueeze1d = T.p_THIntTensor_unsqueeze1d


instance GenericOps CTHLongTensor where
  clearFlag = T.c_THLongTensor_clearFlag
  tensordata = T.c_THLongTensor_data
  desc = T.c_THLongTensor_desc
  expand = T.c_THLongTensor_expand
  expandNd = T.c_THLongTensor_expandNd
  free = T.c_THLongTensor_free
  freeCopyTo = T.c_THLongTensor_freeCopyTo
  get1d = T.c_THLongTensor_get1d
  get2d = T.c_THLongTensor_get2d
  get3d = T.c_THLongTensor_get3d
  get4d = T.c_THLongTensor_get4d
  isContiguous = T.c_THLongTensor_isContiguous
  isSameSizeAs = T.c_THLongTensor_isSameSizeAs
  isSetTo = T.c_THLongTensor_isSetTo
  isSize = T.c_THLongTensor_isSize
  nDimension = T.c_THLongTensor_nDimension
  nElement = T.c_THLongTensor_nElement
  narrow = T.c_THLongTensor_narrow
  new = T.c_THLongTensor_new
  newClone = T.c_THLongTensor_newClone
  newContiguous = T.c_THLongTensor_newContiguous
  newExpand = T.c_THLongTensor_newExpand
  newNarrow = T.c_THLongTensor_newNarrow
  newSelect = T.c_THLongTensor_newSelect
  newSizeOf = T.c_THLongTensor_newSizeOf
  newStrideOf = T.c_THLongTensor_newStrideOf
  newTranspose = T.c_THLongTensor_newTranspose
  newUnfold = T.c_THLongTensor_newUnfold
  newView = T.c_THLongTensor_newView
  newWithSize = T.c_THLongTensor_newWithSize
  newWithSize1d = T.c_THLongTensor_newWithSize1d
  newWithSize2d = T.c_THLongTensor_newWithSize2d
  newWithSize3d = T.c_THLongTensor_newWithSize3d
  newWithSize4d = T.c_THLongTensor_newWithSize4d
  newWithStorage = T.c_THLongTensor_newWithStorage
  newWithStorage1d = T.c_THLongTensor_newWithStorage1d
  newWithStorage2d = T.c_THLongTensor_newWithStorage2d
  newWithStorage3d = T.c_THLongTensor_newWithStorage3d
  newWithStorage4d = T.c_THLongTensor_newWithStorage4d
  newWithTensor = T.c_THLongTensor_newWithTensor
  resize = T.c_THLongTensor_resize
  resize1d = T.c_THLongTensor_resize1d
  resize2d = T.c_THLongTensor_resize2d
  resize3d = T.c_THLongTensor_resize3d
  resize4d = T.c_THLongTensor_resize4d
  resize5d = T.c_THLongTensor_resize5d
  resizeAs = T.c_THLongTensor_resizeAs
  resizeNd = T.c_THLongTensor_resizeNd
  retain = T.c_THLongTensor_retain
  select = T.c_THLongTensor_select
  set = T.c_THLongTensor_set
  set1d = T.c_THLongTensor_set1d
  set2d = T.c_THLongTensor_set2d
  set3d = T.c_THLongTensor_set3d
  set4d = T.c_THLongTensor_set4d
  setFlag = T.c_THLongTensor_setFlag
  setStorage = T.c_THLongTensor_setStorage
  setStorage1d = T.c_THLongTensor_setStorage1d
  setStorage2d = T.c_THLongTensor_setStorage2d
  setStorage3d = T.c_THLongTensor_setStorage3d
  setStorage4d = T.c_THLongTensor_setStorage4d
  setStorageNd = T.c_THLongTensor_setStorageNd
  size = T.c_THLongTensor_size
  sizeDesc = T.c_THLongTensor_sizeDesc
  squeeze = T.c_THLongTensor_squeeze
  squeeze1d = T.c_THLongTensor_squeeze1d
  storage = T.c_THLongTensor_storage
  storageOffset = T.c_THLongTensor_storageOffset
  stride = T.c_THLongTensor_stride
  transpose = T.c_THLongTensor_transpose
  unfold = T.c_THLongTensor_unfold
  unsqueeze1d = T.c_THLongTensor_unsqueeze1d

  _clearFlag = T.p_THLongTensor_clearFlag
  _tensordata = T.p_THLongTensor_data
  _desc = T.p_THLongTensor_desc
  _expand = T.p_THLongTensor_expand
  _expandNd = T.p_THLongTensor_expandNd
  _free = T.p_THLongTensor_free
  _freeCopyTo = T.p_THLongTensor_freeCopyTo
  _get1d = T.p_THLongTensor_get1d
  _get2d = T.p_THLongTensor_get2d
  _get3d = T.p_THLongTensor_get3d
  _get4d = T.p_THLongTensor_get4d
  _isContiguous = T.p_THLongTensor_isContiguous
  _isSameSizeAs = T.p_THLongTensor_isSameSizeAs
  _isSetTo = T.p_THLongTensor_isSetTo
  _isSize = T.p_THLongTensor_isSize
  _nDimension = T.p_THLongTensor_nDimension
  _nElement = T.p_THLongTensor_nElement
  _narrow = T.p_THLongTensor_narrow
  _new = T.p_THLongTensor_new
  _newClone = T.p_THLongTensor_newClone
  _newContiguous = T.p_THLongTensor_newContiguous
  _newExpand = T.p_THLongTensor_newExpand
  _newNarrow = T.p_THLongTensor_newNarrow
  _newSelect = T.p_THLongTensor_newSelect
  _newSizeOf = T.p_THLongTensor_newSizeOf
  _newStrideOf = T.p_THLongTensor_newStrideOf
  _newTranspose = T.p_THLongTensor_newTranspose
  _newUnfold = T.p_THLongTensor_newUnfold
  _newView = T.p_THLongTensor_newView
  _newWithSize = T.p_THLongTensor_newWithSize
  _newWithSize1d = T.p_THLongTensor_newWithSize1d
  _newWithSize2d = T.p_THLongTensor_newWithSize2d
  _newWithSize3d = T.p_THLongTensor_newWithSize3d
  _newWithSize4d = T.p_THLongTensor_newWithSize4d
  _newWithStorage = T.p_THLongTensor_newWithStorage
  _newWithStorage1d = T.p_THLongTensor_newWithStorage1d
  _newWithStorage2d = T.p_THLongTensor_newWithStorage2d
  _newWithStorage3d = T.p_THLongTensor_newWithStorage3d
  _newWithStorage4d = T.p_THLongTensor_newWithStorage4d
  _newWithTensor = T.p_THLongTensor_newWithTensor
  _resize = T.p_THLongTensor_resize
  _resize1d = T.p_THLongTensor_resize1d
  _resize2d = T.p_THLongTensor_resize2d
  _resize3d = T.p_THLongTensor_resize3d
  _resize4d = T.p_THLongTensor_resize4d
  _resize5d = T.p_THLongTensor_resize5d
  _resizeAs = T.p_THLongTensor_resizeAs
  _resizeNd = T.p_THLongTensor_resizeNd
  _retain = T.p_THLongTensor_retain
  _select = T.p_THLongTensor_select
  _set = T.p_THLongTensor_set
  _set1d = T.p_THLongTensor_set1d
  _set2d = T.p_THLongTensor_set2d
  _set3d = T.p_THLongTensor_set3d
  _set4d = T.p_THLongTensor_set4d
  _setFlag = T.p_THLongTensor_setFlag
  _setStorage = T.p_THLongTensor_setStorage
  _setStorage1d = T.p_THLongTensor_setStorage1d
  _setStorage2d = T.p_THLongTensor_setStorage2d
  _setStorage3d = T.p_THLongTensor_setStorage3d
  _setStorage4d = T.p_THLongTensor_setStorage4d
  _setStorageNd = T.p_THLongTensor_setStorageNd
  _size = T.p_THLongTensor_size
  _sizeDesc = T.p_THLongTensor_sizeDesc
  _squeeze = T.p_THLongTensor_squeeze
  _squeeze1d = T.p_THLongTensor_squeeze1d
  _storage = T.p_THLongTensor_storage
  _storageOffset = T.p_THLongTensor_storageOffset
  _stride = T.p_THLongTensor_stride
  _transpose = T.p_THLongTensor_transpose
  _unfold = T.p_THLongTensor_unfold
  _unsqueeze1d = T.p_THLongTensor_unsqueeze1d


instance GenericOps CTHShortTensor where
  clearFlag = T.c_THShortTensor_clearFlag
  tensordata = T.c_THShortTensor_data
  desc = T.c_THShortTensor_desc
  expand = T.c_THShortTensor_expand
  expandNd = T.c_THShortTensor_expandNd
  free = T.c_THShortTensor_free
  freeCopyTo = T.c_THShortTensor_freeCopyTo
  get1d = T.c_THShortTensor_get1d
  get2d = T.c_THShortTensor_get2d
  get3d = T.c_THShortTensor_get3d
  get4d = T.c_THShortTensor_get4d
  isContiguous = T.c_THShortTensor_isContiguous
  isSameSizeAs = T.c_THShortTensor_isSameSizeAs
  isSetTo = T.c_THShortTensor_isSetTo
  isSize = T.c_THShortTensor_isSize
  nDimension = T.c_THShortTensor_nDimension
  nElement = T.c_THShortTensor_nElement
  narrow = T.c_THShortTensor_narrow
  new = T.c_THShortTensor_new
  newClone = T.c_THShortTensor_newClone
  newContiguous = T.c_THShortTensor_newContiguous
  newExpand = T.c_THShortTensor_newExpand
  newNarrow = T.c_THShortTensor_newNarrow
  newSelect = T.c_THShortTensor_newSelect
  newSizeOf = T.c_THShortTensor_newSizeOf
  newStrideOf = T.c_THShortTensor_newStrideOf
  newTranspose = T.c_THShortTensor_newTranspose
  newUnfold = T.c_THShortTensor_newUnfold
  newView = T.c_THShortTensor_newView
  newWithSize = T.c_THShortTensor_newWithSize
  newWithSize1d = T.c_THShortTensor_newWithSize1d
  newWithSize2d = T.c_THShortTensor_newWithSize2d
  newWithSize3d = T.c_THShortTensor_newWithSize3d
  newWithSize4d = T.c_THShortTensor_newWithSize4d
  newWithStorage = T.c_THShortTensor_newWithStorage
  newWithStorage1d = T.c_THShortTensor_newWithStorage1d
  newWithStorage2d = T.c_THShortTensor_newWithStorage2d
  newWithStorage3d = T.c_THShortTensor_newWithStorage3d
  newWithStorage4d = T.c_THShortTensor_newWithStorage4d
  newWithTensor = T.c_THShortTensor_newWithTensor
  resize = T.c_THShortTensor_resize
  resize1d = T.c_THShortTensor_resize1d
  resize2d = T.c_THShortTensor_resize2d
  resize3d = T.c_THShortTensor_resize3d
  resize4d = T.c_THShortTensor_resize4d
  resize5d = T.c_THShortTensor_resize5d
  resizeAs = T.c_THShortTensor_resizeAs
  resizeNd = T.c_THShortTensor_resizeNd
  retain = T.c_THShortTensor_retain
  select = T.c_THShortTensor_select
  set = T.c_THShortTensor_set
  set1d = T.c_THShortTensor_set1d
  set2d = T.c_THShortTensor_set2d
  set3d = T.c_THShortTensor_set3d
  set4d = T.c_THShortTensor_set4d
  setFlag = T.c_THShortTensor_setFlag
  setStorage = T.c_THShortTensor_setStorage
  setStorage1d = T.c_THShortTensor_setStorage1d
  setStorage2d = T.c_THShortTensor_setStorage2d
  setStorage3d = T.c_THShortTensor_setStorage3d
  setStorage4d = T.c_THShortTensor_setStorage4d
  setStorageNd = T.c_THShortTensor_setStorageNd
  size = T.c_THShortTensor_size
  sizeDesc = T.c_THShortTensor_sizeDesc
  squeeze = T.c_THShortTensor_squeeze
  squeeze1d = T.c_THShortTensor_squeeze1d
  storage = T.c_THShortTensor_storage
  storageOffset = T.c_THShortTensor_storageOffset
  stride = T.c_THShortTensor_stride
  transpose = T.c_THShortTensor_transpose
  unfold = T.c_THShortTensor_unfold
  unsqueeze1d = T.c_THShortTensor_unsqueeze1d

  _clearFlag = T.p_THShortTensor_clearFlag
  _tensordata = T.p_THShortTensor_data
  _desc = T.p_THShortTensor_desc
  _expand = T.p_THShortTensor_expand
  _expandNd = T.p_THShortTensor_expandNd
  _free = T.p_THShortTensor_free
  _freeCopyTo = T.p_THShortTensor_freeCopyTo
  _get1d = T.p_THShortTensor_get1d
  _get2d = T.p_THShortTensor_get2d
  _get3d = T.p_THShortTensor_get3d
  _get4d = T.p_THShortTensor_get4d
  _isContiguous = T.p_THShortTensor_isContiguous
  _isSameSizeAs = T.p_THShortTensor_isSameSizeAs
  _isSetTo = T.p_THShortTensor_isSetTo
  _isSize = T.p_THShortTensor_isSize
  _nDimension = T.p_THShortTensor_nDimension
  _nElement = T.p_THShortTensor_nElement
  _narrow = T.p_THShortTensor_narrow
  _new = T.p_THShortTensor_new
  _newClone = T.p_THShortTensor_newClone
  _newContiguous = T.p_THShortTensor_newContiguous
  _newExpand = T.p_THShortTensor_newExpand
  _newNarrow = T.p_THShortTensor_newNarrow
  _newSelect = T.p_THShortTensor_newSelect
  _newSizeOf = T.p_THShortTensor_newSizeOf
  _newStrideOf = T.p_THShortTensor_newStrideOf
  _newTranspose = T.p_THShortTensor_newTranspose
  _newUnfold = T.p_THShortTensor_newUnfold
  _newView = T.p_THShortTensor_newView
  _newWithSize = T.p_THShortTensor_newWithSize
  _newWithSize1d = T.p_THShortTensor_newWithSize1d
  _newWithSize2d = T.p_THShortTensor_newWithSize2d
  _newWithSize3d = T.p_THShortTensor_newWithSize3d
  _newWithSize4d = T.p_THShortTensor_newWithSize4d
  _newWithStorage = T.p_THShortTensor_newWithStorage
  _newWithStorage1d = T.p_THShortTensor_newWithStorage1d
  _newWithStorage2d = T.p_THShortTensor_newWithStorage2d
  _newWithStorage3d = T.p_THShortTensor_newWithStorage3d
  _newWithStorage4d = T.p_THShortTensor_newWithStorage4d
  _newWithTensor = T.p_THShortTensor_newWithTensor
  _resize = T.p_THShortTensor_resize
  _resize1d = T.p_THShortTensor_resize1d
  _resize2d = T.p_THShortTensor_resize2d
  _resize3d = T.p_THShortTensor_resize3d
  _resize4d = T.p_THShortTensor_resize4d
  _resize5d = T.p_THShortTensor_resize5d
  _resizeAs = T.p_THShortTensor_resizeAs
  _resizeNd = T.p_THShortTensor_resizeNd
  _retain = T.p_THShortTensor_retain
  _select = T.p_THShortTensor_select
  _set = T.p_THShortTensor_set
  _set1d = T.p_THShortTensor_set1d
  _set2d = T.p_THShortTensor_set2d
  _set3d = T.p_THShortTensor_set3d
  _set4d = T.p_THShortTensor_set4d
  _setFlag = T.p_THShortTensor_setFlag
  _setStorage = T.p_THShortTensor_setStorage
  _setStorage1d = T.p_THShortTensor_setStorage1d
  _setStorage2d = T.p_THShortTensor_setStorage2d
  _setStorage3d = T.p_THShortTensor_setStorage3d
  _setStorage4d = T.p_THShortTensor_setStorage4d
  _setStorageNd = T.p_THShortTensor_setStorageNd
  _size = T.p_THShortTensor_size
  _sizeDesc = T.p_THShortTensor_sizeDesc
  _squeeze = T.p_THShortTensor_squeeze
  _squeeze1d = T.p_THShortTensor_squeeze1d
  _storage = T.p_THShortTensor_storage
  _storageOffset = T.p_THShortTensor_storageOffset
  _stride = T.p_THShortTensor_stride
  _transpose = T.p_THShortTensor_transpose
  _unfold = T.p_THShortTensor_unfold
  _unsqueeze1d = T.p_THShortTensor_unsqueeze1d

