{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Raw.Tensor
  ( THTensor(..)
  , module X
  ) where

import Torch.Raw.Internal as X

import qualified THByteTensor as T
import qualified THDoubleTensor as T
import qualified THFloatTensor as T
import qualified THIntTensor as T
import qualified THLongTensor as T
import qualified THShortTensor as T

class THTensor t where
  -- C-functions
  c_clearFlag :: Ptr t -> CChar -> IO ()
  c_tensordata :: Ptr t -> IO (Ptr (HaskReal t))
  c_desc :: Ptr t -> CTHDescBuff
  c_expand :: Ptr t -> Ptr t -> Ptr CTHLongStorage -> IO ()
  c_expandNd :: Ptr (Ptr t) -> Ptr (Ptr t) -> CInt -> IO ()
  c_free :: Ptr t -> IO ()
  c_freeCopyTo :: Ptr t -> Ptr t -> IO ()
  c_get1d :: Ptr t -> CLLong -> HaskReal t
  c_get2d :: Ptr t -> CLLong -> CLLong -> HaskReal t
  c_get3d :: Ptr t -> CLLong -> CLLong -> CLLong -> HaskReal t
  c_get4d :: Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> HaskReal t
  c_isContiguous :: Ptr t -> CInt
  c_isSameSizeAs :: Ptr t -> Ptr t -> CInt
  c_isSetTo :: Ptr t -> Ptr t -> CInt
  c_isSize :: Ptr t -> Ptr CTHLongStorage -> CInt
  c_nDimension :: Ptr t -> CInt
  c_nElement :: Ptr t -> CPtrdiff
  c_narrow :: Ptr t -> Ptr t -> CInt -> CLLong -> CLLong -> IO ()
  c_new :: IO (Ptr t)
  c_newClone :: Ptr t -> IO (Ptr t)
  c_newContiguous :: Ptr t -> IO (Ptr t)
  c_newExpand :: Ptr t -> Ptr CTHLongStorage -> IO (Ptr t)
  c_newNarrow :: Ptr t -> CInt -> CLLong -> CLLong -> IO (Ptr t)
  c_newSelect :: Ptr t -> CInt -> CLLong -> IO (Ptr t)
  c_newSizeOf :: Ptr t -> IO (Ptr CTHLongStorage)
  c_newStrideOf :: Ptr t -> IO (Ptr CTHLongStorage)
  c_newTranspose :: Ptr t -> CInt -> CInt -> IO (Ptr t)
  c_newUnfold :: Ptr t -> CInt -> CLLong -> CLLong -> IO (Ptr t)
  c_newView :: Ptr t -> Ptr CTHLongStorage -> IO (Ptr t)
  c_newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr t)
  c_newWithSize1d :: CLLong -> IO (Ptr t)
  c_newWithSize2d :: CLLong -> CLLong -> IO (Ptr t)
  c_newWithSize3d :: CLLong -> CLLong -> CLLong -> IO (Ptr t)
  c_newWithSize4d :: CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t)
  c_newWithStorage :: Ptr (Storage t) -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr t)
  c_newWithStorage1d :: Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr t)
  c_newWithStorage2d :: Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t)
  c_newWithStorage3d :: Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t)
  c_newWithStorage4d :: Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t)
  c_newWithTensor :: Ptr t -> IO (Ptr t)
  c_resize :: Ptr t -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  c_resize1d :: Ptr t -> CLLong -> IO ()
  c_resize2d :: Ptr t -> CLLong -> CLLong -> IO ()
  c_resize3d :: Ptr t -> CLLong -> CLLong -> CLLong -> IO ()
  c_resize4d :: Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_resize5d :: Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_resizeAs :: Ptr t -> Ptr t -> IO ()
  c_resizeNd :: Ptr t -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
  c_retain :: Ptr t -> IO ()
  c_select :: Ptr t -> Ptr t -> CInt -> CLLong -> IO ()
  c_set :: Ptr t -> Ptr t -> IO ()
  c_set1d :: Ptr t -> CLLong -> HaskReal t -> IO ()
  c_set2d :: Ptr t -> CLLong -> CLLong -> HaskReal t -> IO ()
  c_set3d :: Ptr t -> CLLong -> CLLong -> CLLong -> HaskReal t -> IO ()
  c_set4d :: Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> HaskReal t -> IO ()
  c_setFlag :: Ptr t -> CChar -> IO ()
  c_setStorage :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  c_setStorage1d :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> IO ()
  c_setStorage2d :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_setStorage3d :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_setStorage4d :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_setStorageNd :: Ptr t -> Ptr (Storage t) -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
  c_size :: Ptr t -> CInt -> CLLong
  c_sizeDesc :: Ptr t -> CTHDescBuff
  c_squeeze :: Ptr t -> Ptr t -> IO ()
  c_squeeze1d :: Ptr t -> Ptr t -> CInt -> IO ()
  c_storage :: Ptr t -> IO (Ptr (Storage t))
  c_storageOffset :: Ptr t -> CPtrdiff
  c_stride :: Ptr t -> CInt -> CLLong
  c_transpose :: Ptr t -> Ptr t -> CInt -> CInt -> IO ()
  c_unfold :: Ptr t -> Ptr t -> CInt -> CLLong -> CLLong -> IO ()
  c_unsqueeze1d :: Ptr t -> Ptr t -> CInt -> IO ()

  -- C-Function Pointers

  p_clearFlag :: FunPtr (Ptr t -> CChar -> IO ())
  p_tensordata :: FunPtr (Ptr t -> IO (Ptr (HaskReal t)))
  p_desc :: FunPtr (Ptr t -> CTHDescBuff)
  p_expand :: FunPtr (Ptr t -> Ptr t -> Ptr CTHLongStorage -> IO ())
  p_expandNd :: FunPtr (Ptr (Ptr t) -> Ptr (Ptr t) -> CInt -> IO ())
  p_free :: FunPtr (Ptr t -> IO ())
  p_freeCopyTo :: FunPtr (Ptr t -> Ptr t -> IO ())
  p_get1d :: FunPtr (Ptr t -> CLLong -> HaskReal t)
  p_get2d :: FunPtr (Ptr t -> CLLong -> CLLong -> HaskReal t)
  p_get3d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> HaskReal t)
  p_get4d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> HaskReal t)
  p_isContiguous :: FunPtr (Ptr t -> CInt)
  p_isSameSizeAs :: FunPtr (Ptr t -> Ptr t -> CInt)
  p_isSetTo :: FunPtr (Ptr t -> Ptr t -> CInt)
  p_isSize :: FunPtr (Ptr t -> Ptr CTHLongStorage -> CInt)
  p_nDimension :: FunPtr (Ptr t -> CInt)
  p_nElement :: FunPtr (Ptr t -> CPtrdiff)
  p_narrow :: FunPtr (Ptr t -> Ptr t -> CInt -> CLLong -> CLLong -> IO ())
  p_new :: FunPtr (IO (Ptr t))
  p_newClone :: FunPtr (Ptr t -> IO (Ptr t))
  p_newContiguous :: FunPtr (Ptr t -> IO (Ptr t))
  p_newExpand :: FunPtr (Ptr t -> Ptr CTHLongStorage -> IO (Ptr t))
  p_newNarrow :: FunPtr (Ptr t -> CInt -> CLLong -> CLLong -> IO (Ptr t))
  p_newSelect :: FunPtr (Ptr t -> CInt -> CLLong -> IO (Ptr t))
  p_newSizeOf :: FunPtr (Ptr t -> IO (Ptr CTHLongStorage))
  p_newStrideOf :: FunPtr (Ptr t -> IO (Ptr CTHLongStorage))
  p_newTranspose :: FunPtr (Ptr t -> CInt -> CInt -> IO (Ptr t))
  p_newUnfold :: FunPtr (Ptr t -> CInt -> CLLong -> CLLong -> IO (Ptr t))
  p_newView :: FunPtr (Ptr t -> Ptr CTHLongStorage -> IO (Ptr t))
  p_newWithSize :: FunPtr (Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr t))
  p_newWithSize1d :: FunPtr (CLLong -> IO (Ptr t))
  p_newWithSize2d :: FunPtr (CLLong -> CLLong -> IO (Ptr t))
  p_newWithSize3d :: FunPtr (CLLong -> CLLong -> CLLong -> IO (Ptr t))
  p_newWithSize4d :: FunPtr (CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t))
  p_newWithStorage :: FunPtr (Ptr (Storage t) -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr t))
  p_newWithStorage1d :: FunPtr (Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr t))
  p_newWithStorage2d :: FunPtr (Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t))
  p_newWithStorage3d :: FunPtr (Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t))
  p_newWithStorage4d :: FunPtr (Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr t))
  p_newWithTensor :: FunPtr (Ptr t -> IO (Ptr t))
  p_resize :: FunPtr (Ptr t -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())
  p_resize1d :: FunPtr (Ptr t -> CLLong -> IO ())
  p_resize2d :: FunPtr (Ptr t -> CLLong -> CLLong -> IO ())
  p_resize3d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> IO ())
  p_resize4d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_resize5d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_resizeAs :: FunPtr (Ptr t -> Ptr t -> IO ())
  p_resizeNd :: FunPtr (Ptr t -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())
  p_retain :: FunPtr (Ptr t -> IO ())
  p_select :: FunPtr (Ptr t -> Ptr t -> CInt -> CLLong -> IO ())
  p_set :: FunPtr (Ptr t -> Ptr t -> IO ())
  p_set1d :: FunPtr (Ptr t -> CLLong -> HaskReal t -> IO ())
  p_set2d :: FunPtr (Ptr t -> CLLong -> CLLong -> HaskReal t -> IO ())
  p_set3d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> HaskReal t -> IO ())
  p_set4d :: FunPtr (Ptr t -> CLLong -> CLLong -> CLLong -> CLLong -> HaskReal t -> IO ())
  p_setFlag :: FunPtr (Ptr t -> CChar -> IO ())
  p_setStorage :: FunPtr (Ptr t -> Ptr (Storage t) -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())
  p_setStorage1d :: FunPtr (Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> IO ())
  p_setStorage2d :: FunPtr (Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_setStorage3d :: FunPtr (Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_setStorage4d :: FunPtr (Ptr t -> Ptr (Storage t) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_setStorageNd :: FunPtr (Ptr t -> Ptr (Storage t) -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())
  p_size :: FunPtr (Ptr t -> CInt -> CLLong)
  p_sizeDesc :: FunPtr (Ptr t -> CTHDescBuff)
  p_squeeze :: FunPtr (Ptr t -> Ptr t -> IO ())
  p_squeeze1d :: FunPtr (Ptr t -> Ptr t -> CInt -> IO ())
  p_storage :: FunPtr (Ptr t -> IO (Ptr (Storage t)))
  p_storageOffset :: FunPtr (Ptr t -> CPtrdiff)
  p_stride :: FunPtr (Ptr t -> CInt -> CLLong)
  p_transpose :: FunPtr (Ptr t -> Ptr t -> CInt -> CInt -> IO ())
  p_unfold :: FunPtr (Ptr t -> Ptr t -> CInt -> CLLong -> CLLong -> IO ())
  p_unsqueeze1d :: FunPtr (Ptr t -> Ptr t -> CInt -> IO ())


instance THTensor CTHByteTensor where
  c_storage = T.c_THByteTensor_storage
  c_storageOffset = T.c_THByteTensor_storageOffset
  c_nDimension = T.c_THByteTensor_nDimension
  c_size = T.c_THByteTensor_size
  c_stride = T.c_THByteTensor_stride
  c_newSizeOf = T.c_THByteTensor_newSizeOf
  c_newStrideOf = T.c_THByteTensor_newStrideOf
  c_tensordata = T.c_THByteTensor_data
  c_setFlag = T.c_THByteTensor_setFlag
  c_clearFlag = T.c_THByteTensor_clearFlag
  c_new = T.c_THByteTensor_new
  c_newWithTensor = T.c_THByteTensor_newWithTensor
  c_newWithStorage = T.c_THByteTensor_newWithStorage
  c_newWithStorage1d = T.c_THByteTensor_newWithStorage1d
  c_newWithStorage2d = T.c_THByteTensor_newWithStorage2d
  c_newWithStorage3d = T.c_THByteTensor_newWithStorage3d
  c_newWithStorage4d = T.c_THByteTensor_newWithStorage4d
  c_newWithSize = T.c_THByteTensor_newWithSize
  c_newWithSize1d = T.c_THByteTensor_newWithSize1d
  c_newWithSize2d = T.c_THByteTensor_newWithSize2d
  c_newWithSize3d = T.c_THByteTensor_newWithSize3d
  c_newWithSize4d = T.c_THByteTensor_newWithSize4d
  c_newClone = T.c_THByteTensor_newClone
  c_newContiguous = T.c_THByteTensor_newContiguous
  c_newSelect = T.c_THByteTensor_newSelect
  c_newNarrow = T.c_THByteTensor_newNarrow
  c_newTranspose = T.c_THByteTensor_newTranspose
  c_newUnfold = T.c_THByteTensor_newUnfold
  c_newView = T.c_THByteTensor_newView
  c_newExpand = T.c_THByteTensor_newExpand
  c_expand = T.c_THByteTensor_expand
  c_expandNd = T.c_THByteTensor_expandNd
  c_resize = T.c_THByteTensor_resize
  c_resizeAs = T.c_THByteTensor_resizeAs
  c_resizeNd = T.c_THByteTensor_resizeNd
  c_resize1d = T.c_THByteTensor_resize1d
  c_resize2d = T.c_THByteTensor_resize2d
  c_resize3d = T.c_THByteTensor_resize3d
  c_resize4d = T.c_THByteTensor_resize4d
  c_resize5d = T.c_THByteTensor_resize5d
  c_set = T.c_THByteTensor_set
  c_setStorage = T.c_THByteTensor_setStorage
  c_setStorageNd = T.c_THByteTensor_setStorageNd
  c_setStorage1d = T.c_THByteTensor_setStorage1d
  c_setStorage2d = T.c_THByteTensor_setStorage2d
  c_setStorage3d = T.c_THByteTensor_setStorage3d
  c_setStorage4d = T.c_THByteTensor_setStorage4d
  c_narrow = T.c_THByteTensor_narrow
  c_select = T.c_THByteTensor_select
  c_transpose = T.c_THByteTensor_transpose
  c_unfold = T.c_THByteTensor_unfold
  c_squeeze = T.c_THByteTensor_squeeze
  c_squeeze1d = T.c_THByteTensor_squeeze1d
  c_unsqueeze1d = T.c_THByteTensor_unsqueeze1d
  c_isContiguous = T.c_THByteTensor_isContiguous
  c_isSameSizeAs = T.c_THByteTensor_isSameSizeAs
  c_isSetTo = T.c_THByteTensor_isSetTo
  c_isSize = T.c_THByteTensor_isSize
  c_nElement = T.c_THByteTensor_nElement
  c_retain = T.c_THByteTensor_retain
  c_free = T.c_THByteTensor_free
  c_freeCopyTo = T.c_THByteTensor_freeCopyTo
  c_set1d = T.c_THByteTensor_set1d
  c_set2d = T.c_THByteTensor_set2d
  c_set3d = T.c_THByteTensor_set3d
  c_set4d = T.c_THByteTensor_set4d
  c_get1d = T.c_THByteTensor_get1d
  c_get2d = T.c_THByteTensor_get2d
  c_get3d = T.c_THByteTensor_get3d
  c_get4d = T.c_THByteTensor_get4d
  c_desc = T.c_THByteTensor_desc
  c_sizeDesc = T.c_THByteTensor_sizeDesc
  p_storage = T.p_THByteTensor_storage
  p_storageOffset = T.p_THByteTensor_storageOffset
  p_nDimension = T.p_THByteTensor_nDimension
  p_size = T.p_THByteTensor_size
  p_stride = T.p_THByteTensor_stride
  p_newSizeOf = T.p_THByteTensor_newSizeOf
  p_newStrideOf = T.p_THByteTensor_newStrideOf
  p_tensordata = T.p_THByteTensor_data
  p_setFlag = T.p_THByteTensor_setFlag
  p_clearFlag = T.p_THByteTensor_clearFlag
  p_new = T.p_THByteTensor_new
  p_newWithTensor = T.p_THByteTensor_newWithTensor
  p_newWithStorage = T.p_THByteTensor_newWithStorage
  p_newWithStorage1d = T.p_THByteTensor_newWithStorage1d
  p_newWithStorage2d = T.p_THByteTensor_newWithStorage2d
  p_newWithStorage3d = T.p_THByteTensor_newWithStorage3d
  p_newWithStorage4d = T.p_THByteTensor_newWithStorage4d
  p_newWithSize = T.p_THByteTensor_newWithSize
  p_newWithSize1d = T.p_THByteTensor_newWithSize1d
  p_newWithSize2d = T.p_THByteTensor_newWithSize2d
  p_newWithSize3d = T.p_THByteTensor_newWithSize3d
  p_newWithSize4d = T.p_THByteTensor_newWithSize4d
  p_newClone = T.p_THByteTensor_newClone
  p_newContiguous = T.p_THByteTensor_newContiguous
  p_newSelect = T.p_THByteTensor_newSelect
  p_newNarrow = T.p_THByteTensor_newNarrow
  p_newTranspose = T.p_THByteTensor_newTranspose
  p_newUnfold = T.p_THByteTensor_newUnfold
  p_newView = T.p_THByteTensor_newView
  p_newExpand = T.p_THByteTensor_newExpand
  p_expand = T.p_THByteTensor_expand
  p_expandNd = T.p_THByteTensor_expandNd
  p_resize = T.p_THByteTensor_resize
  p_resizeAs = T.p_THByteTensor_resizeAs
  p_resizeNd = T.p_THByteTensor_resizeNd
  p_resize1d = T.p_THByteTensor_resize1d
  p_resize2d = T.p_THByteTensor_resize2d
  p_resize3d = T.p_THByteTensor_resize3d
  p_resize4d = T.p_THByteTensor_resize4d
  p_resize5d = T.p_THByteTensor_resize5d
  p_set = T.p_THByteTensor_set
  p_setStorage = T.p_THByteTensor_setStorage
  p_setStorageNd = T.p_THByteTensor_setStorageNd
  p_setStorage1d = T.p_THByteTensor_setStorage1d
  p_setStorage2d = T.p_THByteTensor_setStorage2d
  p_setStorage3d = T.p_THByteTensor_setStorage3d
  p_setStorage4d = T.p_THByteTensor_setStorage4d
  p_narrow = T.p_THByteTensor_narrow
  p_select = T.p_THByteTensor_select
  p_transpose = T.p_THByteTensor_transpose
  p_unfold = T.p_THByteTensor_unfold
  p_squeeze = T.p_THByteTensor_squeeze
  p_squeeze1d = T.p_THByteTensor_squeeze1d
  p_unsqueeze1d = T.p_THByteTensor_unsqueeze1d
  p_isContiguous = T.p_THByteTensor_isContiguous
  p_isSameSizeAs = T.p_THByteTensor_isSameSizeAs
  p_isSetTo = T.p_THByteTensor_isSetTo
  p_isSize = T.p_THByteTensor_isSize
  p_nElement = T.p_THByteTensor_nElement
  p_retain = T.p_THByteTensor_retain
  p_free = T.p_THByteTensor_free
  p_freeCopyTo = T.p_THByteTensor_freeCopyTo
  p_set1d = T.p_THByteTensor_set1d
  p_set2d = T.p_THByteTensor_set2d
  p_set3d = T.p_THByteTensor_set3d
  p_set4d = T.p_THByteTensor_set4d
  p_get1d = T.p_THByteTensor_get1d
  p_get2d = T.p_THByteTensor_get2d
  p_get3d = T.p_THByteTensor_get3d
  p_get4d = T.p_THByteTensor_get4d
  p_desc = T.p_THByteTensor_desc
  p_sizeDesc = T.p_THByteTensor_sizeDesc

instance THTensor CTHDoubleTensor where
  c_clearFlag = T.c_THDoubleTensor_clearFlag
  c_tensordata = T.c_THDoubleTensor_data
  c_desc = T.c_THDoubleTensor_desc
  c_expand = T.c_THDoubleTensor_expand
  c_expandNd = T.c_THDoubleTensor_expandNd
  c_free = T.c_THDoubleTensor_free
  c_freeCopyTo = T.c_THDoubleTensor_freeCopyTo
  c_get1d = T.c_THDoubleTensor_get1d
  c_get2d = T.c_THDoubleTensor_get2d
  c_get3d = T.c_THDoubleTensor_get3d
  c_get4d = T.c_THDoubleTensor_get4d
  c_isContiguous = T.c_THDoubleTensor_isContiguous
  c_isSameSizeAs = T.c_THDoubleTensor_isSameSizeAs
  c_isSetTo = T.c_THDoubleTensor_isSetTo
  c_isSize = T.c_THDoubleTensor_isSize
  c_nDimension = T.c_THDoubleTensor_nDimension
  c_nElement = T.c_THDoubleTensor_nElement
  c_narrow = T.c_THDoubleTensor_narrow
  c_new = T.c_THDoubleTensor_new
  c_newClone = T.c_THDoubleTensor_newClone
  c_newContiguous = T.c_THDoubleTensor_newContiguous
  c_newExpand = T.c_THDoubleTensor_newExpand
  c_newNarrow = T.c_THDoubleTensor_newNarrow
  c_newSelect = T.c_THDoubleTensor_newSelect
  c_newSizeOf = T.c_THDoubleTensor_newSizeOf
  c_newStrideOf = T.c_THDoubleTensor_newStrideOf
  c_newTranspose = T.c_THDoubleTensor_newTranspose
  c_newUnfold = T.c_THDoubleTensor_newUnfold
  c_newView = T.c_THDoubleTensor_newView
  c_newWithSize = T.c_THDoubleTensor_newWithSize
  c_newWithSize1d = T.c_THDoubleTensor_newWithSize1d
  c_newWithSize2d = T.c_THDoubleTensor_newWithSize2d
  c_newWithSize3d = T.c_THDoubleTensor_newWithSize3d
  c_newWithSize4d = T.c_THDoubleTensor_newWithSize4d
  c_newWithStorage = T.c_THDoubleTensor_newWithStorage
  c_newWithStorage1d = T.c_THDoubleTensor_newWithStorage1d
  c_newWithStorage2d = T.c_THDoubleTensor_newWithStorage2d
  c_newWithStorage3d = T.c_THDoubleTensor_newWithStorage3d
  c_newWithStorage4d = T.c_THDoubleTensor_newWithStorage4d
  c_newWithTensor = T.c_THDoubleTensor_newWithTensor
  c_resize = T.c_THDoubleTensor_resize
  c_resize1d = T.c_THDoubleTensor_resize1d
  c_resize2d = T.c_THDoubleTensor_resize2d
  c_resize3d = T.c_THDoubleTensor_resize3d
  c_resize4d = T.c_THDoubleTensor_resize4d
  c_resize5d = T.c_THDoubleTensor_resize5d
  c_resizeAs = T.c_THDoubleTensor_resizeAs
  c_resizeNd = T.c_THDoubleTensor_resizeNd
  c_retain = T.c_THDoubleTensor_retain
  c_select = T.c_THDoubleTensor_select
  c_set = T.c_THDoubleTensor_set
  c_set1d = T.c_THDoubleTensor_set1d
  c_set2d = T.c_THDoubleTensor_set2d
  c_set3d = T.c_THDoubleTensor_set3d
  c_set4d = T.c_THDoubleTensor_set4d
  c_setFlag = T.c_THDoubleTensor_setFlag
  c_setStorage = T.c_THDoubleTensor_setStorage
  c_setStorage1d = T.c_THDoubleTensor_setStorage1d
  c_setStorage2d = T.c_THDoubleTensor_setStorage2d
  c_setStorage3d = T.c_THDoubleTensor_setStorage3d
  c_setStorage4d = T.c_THDoubleTensor_setStorage4d
  c_setStorageNd = T.c_THDoubleTensor_setStorageNd
  c_size = T.c_THDoubleTensor_size
  c_sizeDesc = T.c_THDoubleTensor_sizeDesc
  c_squeeze = T.c_THDoubleTensor_squeeze
  c_squeeze1d = T.c_THDoubleTensor_squeeze1d
  c_storage = T.c_THDoubleTensor_storage
  c_storageOffset = T.c_THDoubleTensor_storageOffset
  c_stride = T.c_THDoubleTensor_stride
  c_transpose = T.c_THDoubleTensor_transpose
  c_unfold = T.c_THDoubleTensor_unfold
  c_unsqueeze1d = T.c_THDoubleTensor_unsqueeze1d

  p_clearFlag = T.p_THDoubleTensor_clearFlag
  p_tensordata = T.p_THDoubleTensor_data
  p_desc = T.p_THDoubleTensor_desc
  p_expand = T.p_THDoubleTensor_expand
  p_expandNd = T.p_THDoubleTensor_expandNd
  p_free = T.p_THDoubleTensor_free
  p_freeCopyTo = T.p_THDoubleTensor_freeCopyTo
  p_get1d = T.p_THDoubleTensor_get1d
  p_get2d = T.p_THDoubleTensor_get2d
  p_get3d = T.p_THDoubleTensor_get3d
  p_get4d = T.p_THDoubleTensor_get4d
  p_isContiguous = T.p_THDoubleTensor_isContiguous
  p_isSameSizeAs = T.p_THDoubleTensor_isSameSizeAs
  p_isSetTo = T.p_THDoubleTensor_isSetTo
  p_isSize = T.p_THDoubleTensor_isSize
  p_nDimension = T.p_THDoubleTensor_nDimension
  p_nElement = T.p_THDoubleTensor_nElement
  p_narrow = T.p_THDoubleTensor_narrow
  p_new = T.p_THDoubleTensor_new
  p_newClone = T.p_THDoubleTensor_newClone
  p_newContiguous = T.p_THDoubleTensor_newContiguous
  p_newExpand = T.p_THDoubleTensor_newExpand
  p_newNarrow = T.p_THDoubleTensor_newNarrow
  p_newSelect = T.p_THDoubleTensor_newSelect
  p_newSizeOf = T.p_THDoubleTensor_newSizeOf
  p_newStrideOf = T.p_THDoubleTensor_newStrideOf
  p_newTranspose = T.p_THDoubleTensor_newTranspose
  p_newUnfold = T.p_THDoubleTensor_newUnfold
  p_newView = T.p_THDoubleTensor_newView
  p_newWithSize = T.p_THDoubleTensor_newWithSize
  p_newWithSize1d = T.p_THDoubleTensor_newWithSize1d
  p_newWithSize2d = T.p_THDoubleTensor_newWithSize2d
  p_newWithSize3d = T.p_THDoubleTensor_newWithSize3d
  p_newWithSize4d = T.p_THDoubleTensor_newWithSize4d
  p_newWithStorage = T.p_THDoubleTensor_newWithStorage
  p_newWithStorage1d = T.p_THDoubleTensor_newWithStorage1d
  p_newWithStorage2d = T.p_THDoubleTensor_newWithStorage2d
  p_newWithStorage3d = T.p_THDoubleTensor_newWithStorage3d
  p_newWithStorage4d = T.p_THDoubleTensor_newWithStorage4d
  p_newWithTensor = T.p_THDoubleTensor_newWithTensor
  p_resize = T.p_THDoubleTensor_resize
  p_resize1d = T.p_THDoubleTensor_resize1d
  p_resize2d = T.p_THDoubleTensor_resize2d
  p_resize3d = T.p_THDoubleTensor_resize3d
  p_resize4d = T.p_THDoubleTensor_resize4d
  p_resize5d = T.p_THDoubleTensor_resize5d
  p_resizeAs = T.p_THDoubleTensor_resizeAs
  p_resizeNd = T.p_THDoubleTensor_resizeNd
  p_retain = T.p_THDoubleTensor_retain
  p_select = T.p_THDoubleTensor_select
  p_set = T.p_THDoubleTensor_set
  p_set1d = T.p_THDoubleTensor_set1d
  p_set2d = T.p_THDoubleTensor_set2d
  p_set3d = T.p_THDoubleTensor_set3d
  p_set4d = T.p_THDoubleTensor_set4d
  p_setFlag = T.p_THDoubleTensor_setFlag
  p_setStorage = T.p_THDoubleTensor_setStorage
  p_setStorage1d = T.p_THDoubleTensor_setStorage1d
  p_setStorage2d = T.p_THDoubleTensor_setStorage2d
  p_setStorage3d = T.p_THDoubleTensor_setStorage3d
  p_setStorage4d = T.p_THDoubleTensor_setStorage4d
  p_setStorageNd = T.p_THDoubleTensor_setStorageNd
  p_size = T.p_THDoubleTensor_size
  p_sizeDesc = T.p_THDoubleTensor_sizeDesc
  p_squeeze = T.p_THDoubleTensor_squeeze
  p_squeeze1d = T.p_THDoubleTensor_squeeze1d
  p_storage = T.p_THDoubleTensor_storage
  p_storageOffset = T.p_THDoubleTensor_storageOffset
  p_stride = T.p_THDoubleTensor_stride
  p_transpose = T.p_THDoubleTensor_transpose
  p_unfold = T.p_THDoubleTensor_unfold
  p_unsqueeze1d = T.p_THDoubleTensor_unsqueeze1d


instance THTensor CTHFloatTensor where
  c_clearFlag = T.c_THFloatTensor_clearFlag
  c_tensordata = T.c_THFloatTensor_data
  c_desc = T.c_THFloatTensor_desc
  c_expand = T.c_THFloatTensor_expand
  c_expandNd = T.c_THFloatTensor_expandNd
  c_free = T.c_THFloatTensor_free
  c_freeCopyTo = T.c_THFloatTensor_freeCopyTo
  c_get1d = T.c_THFloatTensor_get1d
  c_get2d = T.c_THFloatTensor_get2d
  c_get3d = T.c_THFloatTensor_get3d
  c_get4d = T.c_THFloatTensor_get4d
  c_isContiguous = T.c_THFloatTensor_isContiguous
  c_isSameSizeAs = T.c_THFloatTensor_isSameSizeAs
  c_isSetTo = T.c_THFloatTensor_isSetTo
  c_isSize = T.c_THFloatTensor_isSize
  c_nDimension = T.c_THFloatTensor_nDimension
  c_nElement = T.c_THFloatTensor_nElement
  c_narrow = T.c_THFloatTensor_narrow
  c_new = T.c_THFloatTensor_new
  c_newClone = T.c_THFloatTensor_newClone
  c_newContiguous = T.c_THFloatTensor_newContiguous
  c_newExpand = T.c_THFloatTensor_newExpand
  c_newNarrow = T.c_THFloatTensor_newNarrow
  c_newSelect = T.c_THFloatTensor_newSelect
  c_newSizeOf = T.c_THFloatTensor_newSizeOf
  c_newStrideOf = T.c_THFloatTensor_newStrideOf
  c_newTranspose = T.c_THFloatTensor_newTranspose
  c_newUnfold = T.c_THFloatTensor_newUnfold
  c_newView = T.c_THFloatTensor_newView
  c_newWithSize = T.c_THFloatTensor_newWithSize
  c_newWithSize1d = T.c_THFloatTensor_newWithSize1d
  c_newWithSize2d = T.c_THFloatTensor_newWithSize2d
  c_newWithSize3d = T.c_THFloatTensor_newWithSize3d
  c_newWithSize4d = T.c_THFloatTensor_newWithSize4d
  c_newWithStorage = T.c_THFloatTensor_newWithStorage
  c_newWithStorage1d = T.c_THFloatTensor_newWithStorage1d
  c_newWithStorage2d = T.c_THFloatTensor_newWithStorage2d
  c_newWithStorage3d = T.c_THFloatTensor_newWithStorage3d
  c_newWithStorage4d = T.c_THFloatTensor_newWithStorage4d
  c_newWithTensor = T.c_THFloatTensor_newWithTensor
  c_resize = T.c_THFloatTensor_resize
  c_resize1d = T.c_THFloatTensor_resize1d
  c_resize2d = T.c_THFloatTensor_resize2d
  c_resize3d = T.c_THFloatTensor_resize3d
  c_resize4d = T.c_THFloatTensor_resize4d
  c_resize5d = T.c_THFloatTensor_resize5d
  c_resizeAs = T.c_THFloatTensor_resizeAs
  c_resizeNd = T.c_THFloatTensor_resizeNd
  c_retain = T.c_THFloatTensor_retain
  c_select = T.c_THFloatTensor_select
  c_set = T.c_THFloatTensor_set
  c_set1d = T.c_THFloatTensor_set1d
  c_set2d = T.c_THFloatTensor_set2d
  c_set3d = T.c_THFloatTensor_set3d
  c_set4d = T.c_THFloatTensor_set4d
  c_setFlag = T.c_THFloatTensor_setFlag
  c_setStorage = T.c_THFloatTensor_setStorage
  c_setStorage1d = T.c_THFloatTensor_setStorage1d
  c_setStorage2d = T.c_THFloatTensor_setStorage2d
  c_setStorage3d = T.c_THFloatTensor_setStorage3d
  c_setStorage4d = T.c_THFloatTensor_setStorage4d
  c_setStorageNd = T.c_THFloatTensor_setStorageNd
  c_size = T.c_THFloatTensor_size
  c_sizeDesc = T.c_THFloatTensor_sizeDesc
  c_squeeze = T.c_THFloatTensor_squeeze
  c_squeeze1d = T.c_THFloatTensor_squeeze1d
  c_storage = T.c_THFloatTensor_storage
  c_storageOffset = T.c_THFloatTensor_storageOffset
  c_stride = T.c_THFloatTensor_stride
  c_transpose = T.c_THFloatTensor_transpose
  c_unfold = T.c_THFloatTensor_unfold
  c_unsqueeze1d = T.c_THFloatTensor_unsqueeze1d

  p_clearFlag = T.p_THFloatTensor_clearFlag
  p_tensordata = T.p_THFloatTensor_data
  p_desc = T.p_THFloatTensor_desc
  p_expand = T.p_THFloatTensor_expand
  p_expandNd = T.p_THFloatTensor_expandNd
  p_free = T.p_THFloatTensor_free
  p_freeCopyTo = T.p_THFloatTensor_freeCopyTo
  p_get1d = T.p_THFloatTensor_get1d
  p_get2d = T.p_THFloatTensor_get2d
  p_get3d = T.p_THFloatTensor_get3d
  p_get4d = T.p_THFloatTensor_get4d
  p_isContiguous = T.p_THFloatTensor_isContiguous
  p_isSameSizeAs = T.p_THFloatTensor_isSameSizeAs
  p_isSetTo = T.p_THFloatTensor_isSetTo
  p_isSize = T.p_THFloatTensor_isSize
  p_nDimension = T.p_THFloatTensor_nDimension
  p_nElement = T.p_THFloatTensor_nElement
  p_narrow = T.p_THFloatTensor_narrow
  p_new = T.p_THFloatTensor_new
  p_newClone = T.p_THFloatTensor_newClone
  p_newContiguous = T.p_THFloatTensor_newContiguous
  p_newExpand = T.p_THFloatTensor_newExpand
  p_newNarrow = T.p_THFloatTensor_newNarrow
  p_newSelect = T.p_THFloatTensor_newSelect
  p_newSizeOf = T.p_THFloatTensor_newSizeOf
  p_newStrideOf = T.p_THFloatTensor_newStrideOf
  p_newTranspose = T.p_THFloatTensor_newTranspose
  p_newUnfold = T.p_THFloatTensor_newUnfold
  p_newView = T.p_THFloatTensor_newView
  p_newWithSize = T.p_THFloatTensor_newWithSize
  p_newWithSize1d = T.p_THFloatTensor_newWithSize1d
  p_newWithSize2d = T.p_THFloatTensor_newWithSize2d
  p_newWithSize3d = T.p_THFloatTensor_newWithSize3d
  p_newWithSize4d = T.p_THFloatTensor_newWithSize4d
  p_newWithStorage = T.p_THFloatTensor_newWithStorage
  p_newWithStorage1d = T.p_THFloatTensor_newWithStorage1d
  p_newWithStorage2d = T.p_THFloatTensor_newWithStorage2d
  p_newWithStorage3d = T.p_THFloatTensor_newWithStorage3d
  p_newWithStorage4d = T.p_THFloatTensor_newWithStorage4d
  p_newWithTensor = T.p_THFloatTensor_newWithTensor
  p_resize = T.p_THFloatTensor_resize
  p_resize1d = T.p_THFloatTensor_resize1d
  p_resize2d = T.p_THFloatTensor_resize2d
  p_resize3d = T.p_THFloatTensor_resize3d
  p_resize4d = T.p_THFloatTensor_resize4d
  p_resize5d = T.p_THFloatTensor_resize5d
  p_resizeAs = T.p_THFloatTensor_resizeAs
  p_resizeNd = T.p_THFloatTensor_resizeNd
  p_retain = T.p_THFloatTensor_retain
  p_select = T.p_THFloatTensor_select
  p_set = T.p_THFloatTensor_set
  p_set1d = T.p_THFloatTensor_set1d
  p_set2d = T.p_THFloatTensor_set2d
  p_set3d = T.p_THFloatTensor_set3d
  p_set4d = T.p_THFloatTensor_set4d
  p_setFlag = T.p_THFloatTensor_setFlag
  p_setStorage = T.p_THFloatTensor_setStorage
  p_setStorage1d = T.p_THFloatTensor_setStorage1d
  p_setStorage2d = T.p_THFloatTensor_setStorage2d
  p_setStorage3d = T.p_THFloatTensor_setStorage3d
  p_setStorage4d = T.p_THFloatTensor_setStorage4d
  p_setStorageNd = T.p_THFloatTensor_setStorageNd
  p_size = T.p_THFloatTensor_size
  p_sizeDesc = T.p_THFloatTensor_sizeDesc
  p_squeeze = T.p_THFloatTensor_squeeze
  p_squeeze1d = T.p_THFloatTensor_squeeze1d
  p_storage = T.p_THFloatTensor_storage
  p_storageOffset = T.p_THFloatTensor_storageOffset
  p_stride = T.p_THFloatTensor_stride
  p_transpose = T.p_THFloatTensor_transpose
  p_unfold = T.p_THFloatTensor_unfold
  p_unsqueeze1d = T.p_THFloatTensor_unsqueeze1d


instance THTensor CTHIntTensor where
  c_clearFlag = T.c_THIntTensor_clearFlag
  c_tensordata = T.c_THIntTensor_data
  c_desc = T.c_THIntTensor_desc
  c_expand = T.c_THIntTensor_expand
  c_expandNd = T.c_THIntTensor_expandNd
  c_free = T.c_THIntTensor_free
  c_freeCopyTo = T.c_THIntTensor_freeCopyTo
  c_get1d = T.c_THIntTensor_get1d
  c_get2d = T.c_THIntTensor_get2d
  c_get3d = T.c_THIntTensor_get3d
  c_get4d = T.c_THIntTensor_get4d
  c_isContiguous = T.c_THIntTensor_isContiguous
  c_isSameSizeAs = T.c_THIntTensor_isSameSizeAs
  c_isSetTo = T.c_THIntTensor_isSetTo
  c_isSize = T.c_THIntTensor_isSize
  c_nDimension = T.c_THIntTensor_nDimension
  c_nElement = T.c_THIntTensor_nElement
  c_narrow = T.c_THIntTensor_narrow
  c_new = T.c_THIntTensor_new
  c_newClone = T.c_THIntTensor_newClone
  c_newContiguous = T.c_THIntTensor_newContiguous
  c_newExpand = T.c_THIntTensor_newExpand
  c_newNarrow = T.c_THIntTensor_newNarrow
  c_newSelect = T.c_THIntTensor_newSelect
  c_newSizeOf = T.c_THIntTensor_newSizeOf
  c_newStrideOf = T.c_THIntTensor_newStrideOf
  c_newTranspose = T.c_THIntTensor_newTranspose
  c_newUnfold = T.c_THIntTensor_newUnfold
  c_newView = T.c_THIntTensor_newView
  c_newWithSize = T.c_THIntTensor_newWithSize
  c_newWithSize1d = T.c_THIntTensor_newWithSize1d
  c_newWithSize2d = T.c_THIntTensor_newWithSize2d
  c_newWithSize3d = T.c_THIntTensor_newWithSize3d
  c_newWithSize4d = T.c_THIntTensor_newWithSize4d
  c_newWithStorage = T.c_THIntTensor_newWithStorage
  c_newWithStorage1d = T.c_THIntTensor_newWithStorage1d
  c_newWithStorage2d = T.c_THIntTensor_newWithStorage2d
  c_newWithStorage3d = T.c_THIntTensor_newWithStorage3d
  c_newWithStorage4d = T.c_THIntTensor_newWithStorage4d
  c_newWithTensor = T.c_THIntTensor_newWithTensor
  c_resize = T.c_THIntTensor_resize
  c_resize1d = T.c_THIntTensor_resize1d
  c_resize2d = T.c_THIntTensor_resize2d
  c_resize3d = T.c_THIntTensor_resize3d
  c_resize4d = T.c_THIntTensor_resize4d
  c_resize5d = T.c_THIntTensor_resize5d
  c_resizeAs = T.c_THIntTensor_resizeAs
  c_resizeNd = T.c_THIntTensor_resizeNd
  c_retain = T.c_THIntTensor_retain
  c_select = T.c_THIntTensor_select
  c_set = T.c_THIntTensor_set
  c_set1d = T.c_THIntTensor_set1d
  c_set2d = T.c_THIntTensor_set2d
  c_set3d = T.c_THIntTensor_set3d
  c_set4d = T.c_THIntTensor_set4d
  c_setFlag = T.c_THIntTensor_setFlag
  c_setStorage = T.c_THIntTensor_setStorage
  c_setStorage1d = T.c_THIntTensor_setStorage1d
  c_setStorage2d = T.c_THIntTensor_setStorage2d
  c_setStorage3d = T.c_THIntTensor_setStorage3d
  c_setStorage4d = T.c_THIntTensor_setStorage4d
  c_setStorageNd = T.c_THIntTensor_setStorageNd
  c_size = T.c_THIntTensor_size
  c_sizeDesc = T.c_THIntTensor_sizeDesc
  c_squeeze = T.c_THIntTensor_squeeze
  c_squeeze1d = T.c_THIntTensor_squeeze1d
  c_storage = T.c_THIntTensor_storage
  c_storageOffset = T.c_THIntTensor_storageOffset
  c_stride = T.c_THIntTensor_stride
  c_transpose = T.c_THIntTensor_transpose
  c_unfold = T.c_THIntTensor_unfold
  c_unsqueeze1d = T.c_THIntTensor_unsqueeze1d

  p_clearFlag = T.p_THIntTensor_clearFlag
  p_tensordata = T.p_THIntTensor_data
  p_desc = T.p_THIntTensor_desc
  p_expand = T.p_THIntTensor_expand
  p_expandNd = T.p_THIntTensor_expandNd
  p_free = T.p_THIntTensor_free
  p_freeCopyTo = T.p_THIntTensor_freeCopyTo
  p_get1d = T.p_THIntTensor_get1d
  p_get2d = T.p_THIntTensor_get2d
  p_get3d = T.p_THIntTensor_get3d
  p_get4d = T.p_THIntTensor_get4d
  p_isContiguous = T.p_THIntTensor_isContiguous
  p_isSameSizeAs = T.p_THIntTensor_isSameSizeAs
  p_isSetTo = T.p_THIntTensor_isSetTo
  p_isSize = T.p_THIntTensor_isSize
  p_nDimension = T.p_THIntTensor_nDimension
  p_nElement = T.p_THIntTensor_nElement
  p_narrow = T.p_THIntTensor_narrow
  p_new = T.p_THIntTensor_new
  p_newClone = T.p_THIntTensor_newClone
  p_newContiguous = T.p_THIntTensor_newContiguous
  p_newExpand = T.p_THIntTensor_newExpand
  p_newNarrow = T.p_THIntTensor_newNarrow
  p_newSelect = T.p_THIntTensor_newSelect
  p_newSizeOf = T.p_THIntTensor_newSizeOf
  p_newStrideOf = T.p_THIntTensor_newStrideOf
  p_newTranspose = T.p_THIntTensor_newTranspose
  p_newUnfold = T.p_THIntTensor_newUnfold
  p_newView = T.p_THIntTensor_newView
  p_newWithSize = T.p_THIntTensor_newWithSize
  p_newWithSize1d = T.p_THIntTensor_newWithSize1d
  p_newWithSize2d = T.p_THIntTensor_newWithSize2d
  p_newWithSize3d = T.p_THIntTensor_newWithSize3d
  p_newWithSize4d = T.p_THIntTensor_newWithSize4d
  p_newWithStorage = T.p_THIntTensor_newWithStorage
  p_newWithStorage1d = T.p_THIntTensor_newWithStorage1d
  p_newWithStorage2d = T.p_THIntTensor_newWithStorage2d
  p_newWithStorage3d = T.p_THIntTensor_newWithStorage3d
  p_newWithStorage4d = T.p_THIntTensor_newWithStorage4d
  p_newWithTensor = T.p_THIntTensor_newWithTensor
  p_resize = T.p_THIntTensor_resize
  p_resize1d = T.p_THIntTensor_resize1d
  p_resize2d = T.p_THIntTensor_resize2d
  p_resize3d = T.p_THIntTensor_resize3d
  p_resize4d = T.p_THIntTensor_resize4d
  p_resize5d = T.p_THIntTensor_resize5d
  p_resizeAs = T.p_THIntTensor_resizeAs
  p_resizeNd = T.p_THIntTensor_resizeNd
  p_retain = T.p_THIntTensor_retain
  p_select = T.p_THIntTensor_select
  p_set = T.p_THIntTensor_set
  p_set1d = T.p_THIntTensor_set1d
  p_set2d = T.p_THIntTensor_set2d
  p_set3d = T.p_THIntTensor_set3d
  p_set4d = T.p_THIntTensor_set4d
  p_setFlag = T.p_THIntTensor_setFlag
  p_setStorage = T.p_THIntTensor_setStorage
  p_setStorage1d = T.p_THIntTensor_setStorage1d
  p_setStorage2d = T.p_THIntTensor_setStorage2d
  p_setStorage3d = T.p_THIntTensor_setStorage3d
  p_setStorage4d = T.p_THIntTensor_setStorage4d
  p_setStorageNd = T.p_THIntTensor_setStorageNd
  p_size = T.p_THIntTensor_size
  p_sizeDesc = T.p_THIntTensor_sizeDesc
  p_squeeze = T.p_THIntTensor_squeeze
  p_squeeze1d = T.p_THIntTensor_squeeze1d
  p_storage = T.p_THIntTensor_storage
  p_storageOffset = T.p_THIntTensor_storageOffset
  p_stride = T.p_THIntTensor_stride
  p_transpose = T.p_THIntTensor_transpose
  p_unfold = T.p_THIntTensor_unfold
  p_unsqueeze1d = T.p_THIntTensor_unsqueeze1d


instance THTensor CTHLongTensor where
  c_clearFlag = T.c_THLongTensor_clearFlag
  c_tensordata = T.c_THLongTensor_data
  c_desc = T.c_THLongTensor_desc
  c_expand = T.c_THLongTensor_expand
  c_expandNd = T.c_THLongTensor_expandNd
  c_free = T.c_THLongTensor_free
  c_freeCopyTo = T.c_THLongTensor_freeCopyTo
  c_get1d = T.c_THLongTensor_get1d
  c_get2d = T.c_THLongTensor_get2d
  c_get3d = T.c_THLongTensor_get3d
  c_get4d = T.c_THLongTensor_get4d
  c_isContiguous = T.c_THLongTensor_isContiguous
  c_isSameSizeAs = T.c_THLongTensor_isSameSizeAs
  c_isSetTo = T.c_THLongTensor_isSetTo
  c_isSize = T.c_THLongTensor_isSize
  c_nDimension = T.c_THLongTensor_nDimension
  c_nElement = T.c_THLongTensor_nElement
  c_narrow = T.c_THLongTensor_narrow
  c_new = T.c_THLongTensor_new
  c_newClone = T.c_THLongTensor_newClone
  c_newContiguous = T.c_THLongTensor_newContiguous
  c_newExpand = T.c_THLongTensor_newExpand
  c_newNarrow = T.c_THLongTensor_newNarrow
  c_newSelect = T.c_THLongTensor_newSelect
  c_newSizeOf = T.c_THLongTensor_newSizeOf
  c_newStrideOf = T.c_THLongTensor_newStrideOf
  c_newTranspose = T.c_THLongTensor_newTranspose
  c_newUnfold = T.c_THLongTensor_newUnfold
  c_newView = T.c_THLongTensor_newView
  c_newWithSize = T.c_THLongTensor_newWithSize
  c_newWithSize1d = T.c_THLongTensor_newWithSize1d
  c_newWithSize2d = T.c_THLongTensor_newWithSize2d
  c_newWithSize3d = T.c_THLongTensor_newWithSize3d
  c_newWithSize4d = T.c_THLongTensor_newWithSize4d
  c_newWithStorage = T.c_THLongTensor_newWithStorage
  c_newWithStorage1d = T.c_THLongTensor_newWithStorage1d
  c_newWithStorage2d = T.c_THLongTensor_newWithStorage2d
  c_newWithStorage3d = T.c_THLongTensor_newWithStorage3d
  c_newWithStorage4d = T.c_THLongTensor_newWithStorage4d
  c_newWithTensor = T.c_THLongTensor_newWithTensor
  c_resize = T.c_THLongTensor_resize
  c_resize1d = T.c_THLongTensor_resize1d
  c_resize2d = T.c_THLongTensor_resize2d
  c_resize3d = T.c_THLongTensor_resize3d
  c_resize4d = T.c_THLongTensor_resize4d
  c_resize5d = T.c_THLongTensor_resize5d
  c_resizeAs = T.c_THLongTensor_resizeAs
  c_resizeNd = T.c_THLongTensor_resizeNd
  c_retain = T.c_THLongTensor_retain
  c_select = T.c_THLongTensor_select
  c_set = T.c_THLongTensor_set
  c_set1d = T.c_THLongTensor_set1d
  c_set2d = T.c_THLongTensor_set2d
  c_set3d = T.c_THLongTensor_set3d
  c_set4d = T.c_THLongTensor_set4d
  c_setFlag = T.c_THLongTensor_setFlag
  c_setStorage = T.c_THLongTensor_setStorage
  c_setStorage1d = T.c_THLongTensor_setStorage1d
  c_setStorage2d = T.c_THLongTensor_setStorage2d
  c_setStorage3d = T.c_THLongTensor_setStorage3d
  c_setStorage4d = T.c_THLongTensor_setStorage4d
  c_setStorageNd = T.c_THLongTensor_setStorageNd
  c_size = T.c_THLongTensor_size
  c_sizeDesc = T.c_THLongTensor_sizeDesc
  c_squeeze = T.c_THLongTensor_squeeze
  c_squeeze1d = T.c_THLongTensor_squeeze1d
  c_storage = T.c_THLongTensor_storage
  c_storageOffset = T.c_THLongTensor_storageOffset
  c_stride = T.c_THLongTensor_stride
  c_transpose = T.c_THLongTensor_transpose
  c_unfold = T.c_THLongTensor_unfold
  c_unsqueeze1d = T.c_THLongTensor_unsqueeze1d

  p_clearFlag = T.p_THLongTensor_clearFlag
  p_tensordata = T.p_THLongTensor_data
  p_desc = T.p_THLongTensor_desc
  p_expand = T.p_THLongTensor_expand
  p_expandNd = T.p_THLongTensor_expandNd
  p_free = T.p_THLongTensor_free
  p_freeCopyTo = T.p_THLongTensor_freeCopyTo
  p_get1d = T.p_THLongTensor_get1d
  p_get2d = T.p_THLongTensor_get2d
  p_get3d = T.p_THLongTensor_get3d
  p_get4d = T.p_THLongTensor_get4d
  p_isContiguous = T.p_THLongTensor_isContiguous
  p_isSameSizeAs = T.p_THLongTensor_isSameSizeAs
  p_isSetTo = T.p_THLongTensor_isSetTo
  p_isSize = T.p_THLongTensor_isSize
  p_nDimension = T.p_THLongTensor_nDimension
  p_nElement = T.p_THLongTensor_nElement
  p_narrow = T.p_THLongTensor_narrow
  p_new = T.p_THLongTensor_new
  p_newClone = T.p_THLongTensor_newClone
  p_newContiguous = T.p_THLongTensor_newContiguous
  p_newExpand = T.p_THLongTensor_newExpand
  p_newNarrow = T.p_THLongTensor_newNarrow
  p_newSelect = T.p_THLongTensor_newSelect
  p_newSizeOf = T.p_THLongTensor_newSizeOf
  p_newStrideOf = T.p_THLongTensor_newStrideOf
  p_newTranspose = T.p_THLongTensor_newTranspose
  p_newUnfold = T.p_THLongTensor_newUnfold
  p_newView = T.p_THLongTensor_newView
  p_newWithSize = T.p_THLongTensor_newWithSize
  p_newWithSize1d = T.p_THLongTensor_newWithSize1d
  p_newWithSize2d = T.p_THLongTensor_newWithSize2d
  p_newWithSize3d = T.p_THLongTensor_newWithSize3d
  p_newWithSize4d = T.p_THLongTensor_newWithSize4d
  p_newWithStorage = T.p_THLongTensor_newWithStorage
  p_newWithStorage1d = T.p_THLongTensor_newWithStorage1d
  p_newWithStorage2d = T.p_THLongTensor_newWithStorage2d
  p_newWithStorage3d = T.p_THLongTensor_newWithStorage3d
  p_newWithStorage4d = T.p_THLongTensor_newWithStorage4d
  p_newWithTensor = T.p_THLongTensor_newWithTensor
  p_resize = T.p_THLongTensor_resize
  p_resize1d = T.p_THLongTensor_resize1d
  p_resize2d = T.p_THLongTensor_resize2d
  p_resize3d = T.p_THLongTensor_resize3d
  p_resize4d = T.p_THLongTensor_resize4d
  p_resize5d = T.p_THLongTensor_resize5d
  p_resizeAs = T.p_THLongTensor_resizeAs
  p_resizeNd = T.p_THLongTensor_resizeNd
  p_retain = T.p_THLongTensor_retain
  p_select = T.p_THLongTensor_select
  p_set = T.p_THLongTensor_set
  p_set1d = T.p_THLongTensor_set1d
  p_set2d = T.p_THLongTensor_set2d
  p_set3d = T.p_THLongTensor_set3d
  p_set4d = T.p_THLongTensor_set4d
  p_setFlag = T.p_THLongTensor_setFlag
  p_setStorage = T.p_THLongTensor_setStorage
  p_setStorage1d = T.p_THLongTensor_setStorage1d
  p_setStorage2d = T.p_THLongTensor_setStorage2d
  p_setStorage3d = T.p_THLongTensor_setStorage3d
  p_setStorage4d = T.p_THLongTensor_setStorage4d
  p_setStorageNd = T.p_THLongTensor_setStorageNd
  p_size = T.p_THLongTensor_size
  p_sizeDesc = T.p_THLongTensor_sizeDesc
  p_squeeze = T.p_THLongTensor_squeeze
  p_squeeze1d = T.p_THLongTensor_squeeze1d
  p_storage = T.p_THLongTensor_storage
  p_storageOffset = T.p_THLongTensor_storageOffset
  p_stride = T.p_THLongTensor_stride
  p_transpose = T.p_THLongTensor_transpose
  p_unfold = T.p_THLongTensor_unfold
  p_unsqueeze1d = T.p_THLongTensor_unsqueeze1d


instance THTensor CTHShortTensor where
  c_clearFlag = T.c_THShortTensor_clearFlag
  c_tensordata = T.c_THShortTensor_data
  c_desc = T.c_THShortTensor_desc
  c_expand = T.c_THShortTensor_expand
  c_expandNd = T.c_THShortTensor_expandNd
  c_free = T.c_THShortTensor_free
  c_freeCopyTo = T.c_THShortTensor_freeCopyTo
  c_get1d = T.c_THShortTensor_get1d
  c_get2d = T.c_THShortTensor_get2d
  c_get3d = T.c_THShortTensor_get3d
  c_get4d = T.c_THShortTensor_get4d
  c_isContiguous = T.c_THShortTensor_isContiguous
  c_isSameSizeAs = T.c_THShortTensor_isSameSizeAs
  c_isSetTo = T.c_THShortTensor_isSetTo
  c_isSize = T.c_THShortTensor_isSize
  c_nDimension = T.c_THShortTensor_nDimension
  c_nElement = T.c_THShortTensor_nElement
  c_narrow = T.c_THShortTensor_narrow
  c_new = T.c_THShortTensor_new
  c_newClone = T.c_THShortTensor_newClone
  c_newContiguous = T.c_THShortTensor_newContiguous
  c_newExpand = T.c_THShortTensor_newExpand
  c_newNarrow = T.c_THShortTensor_newNarrow
  c_newSelect = T.c_THShortTensor_newSelect
  c_newSizeOf = T.c_THShortTensor_newSizeOf
  c_newStrideOf = T.c_THShortTensor_newStrideOf
  c_newTranspose = T.c_THShortTensor_newTranspose
  c_newUnfold = T.c_THShortTensor_newUnfold
  c_newView = T.c_THShortTensor_newView
  c_newWithSize = T.c_THShortTensor_newWithSize
  c_newWithSize1d = T.c_THShortTensor_newWithSize1d
  c_newWithSize2d = T.c_THShortTensor_newWithSize2d
  c_newWithSize3d = T.c_THShortTensor_newWithSize3d
  c_newWithSize4d = T.c_THShortTensor_newWithSize4d
  c_newWithStorage = T.c_THShortTensor_newWithStorage
  c_newWithStorage1d = T.c_THShortTensor_newWithStorage1d
  c_newWithStorage2d = T.c_THShortTensor_newWithStorage2d
  c_newWithStorage3d = T.c_THShortTensor_newWithStorage3d
  c_newWithStorage4d = T.c_THShortTensor_newWithStorage4d
  c_newWithTensor = T.c_THShortTensor_newWithTensor
  c_resize = T.c_THShortTensor_resize
  c_resize1d = T.c_THShortTensor_resize1d
  c_resize2d = T.c_THShortTensor_resize2d
  c_resize3d = T.c_THShortTensor_resize3d
  c_resize4d = T.c_THShortTensor_resize4d
  c_resize5d = T.c_THShortTensor_resize5d
  c_resizeAs = T.c_THShortTensor_resizeAs
  c_resizeNd = T.c_THShortTensor_resizeNd
  c_retain = T.c_THShortTensor_retain
  c_select = T.c_THShortTensor_select
  c_set = T.c_THShortTensor_set
  c_set1d = T.c_THShortTensor_set1d
  c_set2d = T.c_THShortTensor_set2d
  c_set3d = T.c_THShortTensor_set3d
  c_set4d = T.c_THShortTensor_set4d
  c_setFlag = T.c_THShortTensor_setFlag
  c_setStorage = T.c_THShortTensor_setStorage
  c_setStorage1d = T.c_THShortTensor_setStorage1d
  c_setStorage2d = T.c_THShortTensor_setStorage2d
  c_setStorage3d = T.c_THShortTensor_setStorage3d
  c_setStorage4d = T.c_THShortTensor_setStorage4d
  c_setStorageNd = T.c_THShortTensor_setStorageNd
  c_size = T.c_THShortTensor_size
  c_sizeDesc = T.c_THShortTensor_sizeDesc
  c_squeeze = T.c_THShortTensor_squeeze
  c_squeeze1d = T.c_THShortTensor_squeeze1d
  c_storage = T.c_THShortTensor_storage
  c_storageOffset = T.c_THShortTensor_storageOffset
  c_stride = T.c_THShortTensor_stride
  c_transpose = T.c_THShortTensor_transpose
  c_unfold = T.c_THShortTensor_unfold
  c_unsqueeze1d = T.c_THShortTensor_unsqueeze1d

  p_clearFlag = T.p_THShortTensor_clearFlag
  p_tensordata = T.p_THShortTensor_data
  p_desc = T.p_THShortTensor_desc
  p_expand = T.p_THShortTensor_expand
  p_expandNd = T.p_THShortTensor_expandNd
  p_free = T.p_THShortTensor_free
  p_freeCopyTo = T.p_THShortTensor_freeCopyTo
  p_get1d = T.p_THShortTensor_get1d
  p_get2d = T.p_THShortTensor_get2d
  p_get3d = T.p_THShortTensor_get3d
  p_get4d = T.p_THShortTensor_get4d
  p_isContiguous = T.p_THShortTensor_isContiguous
  p_isSameSizeAs = T.p_THShortTensor_isSameSizeAs
  p_isSetTo = T.p_THShortTensor_isSetTo
  p_isSize = T.p_THShortTensor_isSize
  p_nDimension = T.p_THShortTensor_nDimension
  p_nElement = T.p_THShortTensor_nElement
  p_narrow = T.p_THShortTensor_narrow
  p_new = T.p_THShortTensor_new
  p_newClone = T.p_THShortTensor_newClone
  p_newContiguous = T.p_THShortTensor_newContiguous
  p_newExpand = T.p_THShortTensor_newExpand
  p_newNarrow = T.p_THShortTensor_newNarrow
  p_newSelect = T.p_THShortTensor_newSelect
  p_newSizeOf = T.p_THShortTensor_newSizeOf
  p_newStrideOf = T.p_THShortTensor_newStrideOf
  p_newTranspose = T.p_THShortTensor_newTranspose
  p_newUnfold = T.p_THShortTensor_newUnfold
  p_newView = T.p_THShortTensor_newView
  p_newWithSize = T.p_THShortTensor_newWithSize
  p_newWithSize1d = T.p_THShortTensor_newWithSize1d
  p_newWithSize2d = T.p_THShortTensor_newWithSize2d
  p_newWithSize3d = T.p_THShortTensor_newWithSize3d
  p_newWithSize4d = T.p_THShortTensor_newWithSize4d
  p_newWithStorage = T.p_THShortTensor_newWithStorage
  p_newWithStorage1d = T.p_THShortTensor_newWithStorage1d
  p_newWithStorage2d = T.p_THShortTensor_newWithStorage2d
  p_newWithStorage3d = T.p_THShortTensor_newWithStorage3d
  p_newWithStorage4d = T.p_THShortTensor_newWithStorage4d
  p_newWithTensor = T.p_THShortTensor_newWithTensor
  p_resize = T.p_THShortTensor_resize
  p_resize1d = T.p_THShortTensor_resize1d
  p_resize2d = T.p_THShortTensor_resize2d
  p_resize3d = T.p_THShortTensor_resize3d
  p_resize4d = T.p_THShortTensor_resize4d
  p_resize5d = T.p_THShortTensor_resize5d
  p_resizeAs = T.p_THShortTensor_resizeAs
  p_resizeNd = T.p_THShortTensor_resizeNd
  p_retain = T.p_THShortTensor_retain
  p_select = T.p_THShortTensor_select
  p_set = T.p_THShortTensor_set
  p_set1d = T.p_THShortTensor_set1d
  p_set2d = T.p_THShortTensor_set2d
  p_set3d = T.p_THShortTensor_set3d
  p_set4d = T.p_THShortTensor_set4d
  p_setFlag = T.p_THShortTensor_setFlag
  p_setStorage = T.p_THShortTensor_setStorage
  p_setStorage1d = T.p_THShortTensor_setStorage1d
  p_setStorage2d = T.p_THShortTensor_setStorage2d
  p_setStorage3d = T.p_THShortTensor_setStorage3d
  p_setStorage4d = T.p_THShortTensor_setStorage4d
  p_setStorageNd = T.p_THShortTensor_setStorageNd
  p_size = T.p_THShortTensor_size
  p_sizeDesc = T.p_THShortTensor_sizeDesc
  p_squeeze = T.p_THShortTensor_squeeze
  p_squeeze1d = T.p_THShortTensor_squeeze1d
  p_storage = T.p_THShortTensor_storage
  p_storageOffset = T.p_THShortTensor_storageOffset
  p_stride = T.p_THShortTensor_stride
  p_transpose = T.p_THShortTensor_transpose
  p_unfold = T.p_THShortTensor_unfold
  p_unsqueeze1d = T.p_THShortTensor_unsqueeze1d

