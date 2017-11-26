{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfTensor (
    c_THHalfTensor_storage,
    c_THHalfTensor_storageOffset,
    c_THHalfTensor_nDimension,
    c_THHalfTensor_size,
    c_THHalfTensor_stride,
    c_THHalfTensor_newSizeOf,
    c_THHalfTensor_newStrideOf,
    c_THHalfTensor_data,
    c_THHalfTensor_setFlag,
    c_THHalfTensor_clearFlag,
    c_THHalfTensor_new,
    c_THHalfTensor_newWithTensor,
    c_THHalfTensor_newWithStorage,
    c_THHalfTensor_newWithStorage1d,
    c_THHalfTensor_newWithStorage2d,
    c_THHalfTensor_newWithStorage3d,
    c_THHalfTensor_newWithStorage4d,
    c_THHalfTensor_newWithSize,
    c_THHalfTensor_newWithSize1d,
    c_THHalfTensor_newWithSize2d,
    c_THHalfTensor_newWithSize3d,
    c_THHalfTensor_newWithSize4d,
    c_THHalfTensor_newClone,
    c_THHalfTensor_newContiguous,
    c_THHalfTensor_newSelect,
    c_THHalfTensor_newNarrow,
    c_THHalfTensor_newTranspose,
    c_THHalfTensor_newUnfold,
    c_THHalfTensor_newView,
    c_THHalfTensor_newExpand,
    c_THHalfTensor_expand,
    c_THHalfTensor_expandNd,
    c_THHalfTensor_resize,
    c_THHalfTensor_resizeAs,
    c_THHalfTensor_resizeNd,
    c_THHalfTensor_resize1d,
    c_THHalfTensor_resize2d,
    c_THHalfTensor_resize3d,
    c_THHalfTensor_resize4d,
    c_THHalfTensor_resize5d,
    c_THHalfTensor_set,
    c_THHalfTensor_setStorage,
    c_THHalfTensor_setStorageNd,
    c_THHalfTensor_setStorage1d,
    c_THHalfTensor_setStorage2d,
    c_THHalfTensor_setStorage3d,
    c_THHalfTensor_setStorage4d,
    c_THHalfTensor_narrow,
    c_THHalfTensor_select,
    c_THHalfTensor_transpose,
    c_THHalfTensor_unfold,
    c_THHalfTensor_squeeze,
    c_THHalfTensor_squeeze1d,
    c_THHalfTensor_unsqueeze1d,
    c_THHalfTensor_isContiguous,
    c_THHalfTensor_isSameSizeAs,
    c_THHalfTensor_isSetTo,
    c_THHalfTensor_isSize,
    c_THHalfTensor_nElement,
    c_THHalfTensor_retain,
    c_THHalfTensor_free,
    c_THHalfTensor_freeCopyTo,
    c_THHalfTensor_set1d,
    c_THHalfTensor_set2d,
    c_THHalfTensor_set3d,
    c_THHalfTensor_set4d,
    c_THHalfTensor_get1d,
    c_THHalfTensor_get2d,
    c_THHalfTensor_get3d,
    c_THHalfTensor_get4d,
    c_THHalfTensor_desc,
    c_THHalfTensor_sizeDesc,
    p_THHalfTensor_storage,
    p_THHalfTensor_storageOffset,
    p_THHalfTensor_nDimension,
    p_THHalfTensor_size,
    p_THHalfTensor_stride,
    p_THHalfTensor_newSizeOf,
    p_THHalfTensor_newStrideOf,
    p_THHalfTensor_data,
    p_THHalfTensor_setFlag,
    p_THHalfTensor_clearFlag,
    p_THHalfTensor_new,
    p_THHalfTensor_newWithTensor,
    p_THHalfTensor_newWithStorage,
    p_THHalfTensor_newWithStorage1d,
    p_THHalfTensor_newWithStorage2d,
    p_THHalfTensor_newWithStorage3d,
    p_THHalfTensor_newWithStorage4d,
    p_THHalfTensor_newWithSize,
    p_THHalfTensor_newWithSize1d,
    p_THHalfTensor_newWithSize2d,
    p_THHalfTensor_newWithSize3d,
    p_THHalfTensor_newWithSize4d,
    p_THHalfTensor_newClone,
    p_THHalfTensor_newContiguous,
    p_THHalfTensor_newSelect,
    p_THHalfTensor_newNarrow,
    p_THHalfTensor_newTranspose,
    p_THHalfTensor_newUnfold,
    p_THHalfTensor_newView,
    p_THHalfTensor_newExpand,
    p_THHalfTensor_expand,
    p_THHalfTensor_expandNd,
    p_THHalfTensor_resize,
    p_THHalfTensor_resizeAs,
    p_THHalfTensor_resizeNd,
    p_THHalfTensor_resize1d,
    p_THHalfTensor_resize2d,
    p_THHalfTensor_resize3d,
    p_THHalfTensor_resize4d,
    p_THHalfTensor_resize5d,
    p_THHalfTensor_set,
    p_THHalfTensor_setStorage,
    p_THHalfTensor_setStorageNd,
    p_THHalfTensor_setStorage1d,
    p_THHalfTensor_setStorage2d,
    p_THHalfTensor_setStorage3d,
    p_THHalfTensor_setStorage4d,
    p_THHalfTensor_narrow,
    p_THHalfTensor_select,
    p_THHalfTensor_transpose,
    p_THHalfTensor_unfold,
    p_THHalfTensor_squeeze,
    p_THHalfTensor_squeeze1d,
    p_THHalfTensor_unsqueeze1d,
    p_THHalfTensor_isContiguous,
    p_THHalfTensor_isSameSizeAs,
    p_THHalfTensor_isSetTo,
    p_THHalfTensor_isSize,
    p_THHalfTensor_nElement,
    p_THHalfTensor_retain,
    p_THHalfTensor_free,
    p_THHalfTensor_freeCopyTo,
    p_THHalfTensor_set1d,
    p_THHalfTensor_set2d,
    p_THHalfTensor_set3d,
    p_THHalfTensor_set4d,
    p_THHalfTensor_get1d,
    p_THHalfTensor_get2d,
    p_THHalfTensor_get3d,
    p_THHalfTensor_get4d,
    p_THHalfTensor_desc,
    p_THHalfTensor_sizeDesc) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THHalfTensor_storage : self -> THStorage *
foreign import ccall "THTensor.h THHalfTensor_storage"
  c_THHalfTensor_storage :: (Ptr CTHHalfTensor) -> IO (Ptr CTHHalfStorage)

-- |c_THHalfTensor_storageOffset : self -> ptrdiff_t
foreign import ccall "THTensor.h THHalfTensor_storageOffset"
  c_THHalfTensor_storageOffset :: (Ptr CTHHalfTensor) -> CPtrdiff

-- |c_THHalfTensor_nDimension : self -> int
foreign import ccall "THTensor.h THHalfTensor_nDimension"
  c_THHalfTensor_nDimension :: (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensor_size : self dim -> int64_t
foreign import ccall "THTensor.h THHalfTensor_size"
  c_THHalfTensor_size :: (Ptr CTHHalfTensor) -> CInt -> CLLong

-- |c_THHalfTensor_stride : self dim -> int64_t
foreign import ccall "THTensor.h THHalfTensor_stride"
  c_THHalfTensor_stride :: (Ptr CTHHalfTensor) -> CInt -> CLLong

-- |c_THHalfTensor_newSizeOf : self -> THLongStorage *
foreign import ccall "THTensor.h THHalfTensor_newSizeOf"
  c_THHalfTensor_newSizeOf :: (Ptr CTHHalfTensor) -> IO (Ptr CTHLongStorage)

-- |c_THHalfTensor_newStrideOf : self -> THLongStorage *
foreign import ccall "THTensor.h THHalfTensor_newStrideOf"
  c_THHalfTensor_newStrideOf :: (Ptr CTHHalfTensor) -> IO (Ptr CTHLongStorage)

-- |c_THHalfTensor_data : self -> real *
foreign import ccall "THTensor.h THHalfTensor_data"
  c_THHalfTensor_data :: (Ptr CTHHalfTensor) -> IO (Ptr THHalf)

-- |c_THHalfTensor_setFlag : self flag -> void
foreign import ccall "THTensor.h THHalfTensor_setFlag"
  c_THHalfTensor_setFlag :: (Ptr CTHHalfTensor) -> CChar -> IO ()

-- |c_THHalfTensor_clearFlag : self flag -> void
foreign import ccall "THTensor.h THHalfTensor_clearFlag"
  c_THHalfTensor_clearFlag :: (Ptr CTHHalfTensor) -> CChar -> IO ()

-- |c_THHalfTensor_new :  -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_new"
  c_THHalfTensor_new :: IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithTensor : tensor -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newWithTensor"
  c_THHalfTensor_newWithTensor :: (Ptr CTHHalfTensor) -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithStorage : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newWithStorage"
  c_THHalfTensor_newWithStorage :: Ptr CTHHalfStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newWithStorage1d"
  c_THHalfTensor_newWithStorage1d :: Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newWithStorage2d"
  c_THHalfTensor_newWithStorage2d :: Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newWithStorage3d"
  c_THHalfTensor_newWithStorage3d :: Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newWithStorage4d"
  c_THHalfTensor_newWithStorage4d :: Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithSize : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newWithSize"
  c_THHalfTensor_newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithSize1d : size0_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newWithSize1d"
  c_THHalfTensor_newWithSize1d :: CLLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newWithSize2d"
  c_THHalfTensor_newWithSize2d :: CLLong -> CLLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newWithSize3d"
  c_THHalfTensor_newWithSize3d :: CLLong -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newWithSize4d"
  c_THHalfTensor_newWithSize4d :: CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newClone : self -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newClone"
  c_THHalfTensor_newClone :: (Ptr CTHHalfTensor) -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newContiguous : tensor -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newContiguous"
  c_THHalfTensor_newContiguous :: (Ptr CTHHalfTensor) -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newSelect"
  c_THHalfTensor_newSelect :: (Ptr CTHHalfTensor) -> CInt -> CLLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newNarrow"
  c_THHalfTensor_newNarrow :: (Ptr CTHHalfTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newTranspose"
  c_THHalfTensor_newTranspose :: (Ptr CTHHalfTensor) -> CInt -> CInt -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newUnfold"
  c_THHalfTensor_newUnfold :: (Ptr CTHHalfTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newView : tensor size -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newView"
  c_THHalfTensor_newView :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newExpand : tensor size -> THTensor *
foreign import ccall "THTensor.h THHalfTensor_newExpand"
  c_THHalfTensor_newExpand :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_expand : r tensor size -> void
foreign import ccall "THTensor.h THHalfTensor_expand"
  c_THHalfTensor_expand :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_expandNd : rets ops count -> void
foreign import ccall "THTensor.h THHalfTensor_expandNd"
  c_THHalfTensor_expandNd :: Ptr (Ptr CTHHalfTensor) -> Ptr (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_resize : tensor size stride -> void
foreign import ccall "THTensor.h THHalfTensor_resize"
  c_THHalfTensor_resize :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_resizeAs : tensor src -> void
foreign import ccall "THTensor.h THHalfTensor_resizeAs"
  c_THHalfTensor_resizeAs :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_resizeNd : tensor nDimension size stride -> void
foreign import ccall "THTensor.h THHalfTensor_resizeNd"
  c_THHalfTensor_resizeNd :: (Ptr CTHHalfTensor) -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- |c_THHalfTensor_resize1d : tensor size0_ -> void
foreign import ccall "THTensor.h THHalfTensor_resize1d"
  c_THHalfTensor_resize1d :: (Ptr CTHHalfTensor) -> CLLong -> IO ()

-- |c_THHalfTensor_resize2d : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h THHalfTensor_resize2d"
  c_THHalfTensor_resize2d :: (Ptr CTHHalfTensor) -> CLLong -> CLLong -> IO ()

-- |c_THHalfTensor_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h THHalfTensor_resize3d"
  c_THHalfTensor_resize3d :: (Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THHalfTensor_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h THHalfTensor_resize4d"
  c_THHalfTensor_resize4d :: (Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THHalfTensor_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h THHalfTensor_resize5d"
  c_THHalfTensor_resize5d :: (Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THHalfTensor_set : self src -> void
foreign import ccall "THTensor.h THHalfTensor_set"
  c_THHalfTensor_set :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h THHalfTensor_setStorage"
  c_THHalfTensor_setStorage :: (Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h THHalfTensor_setStorageNd"
  c_THHalfTensor_setStorageNd :: (Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- |c_THHalfTensor_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h THHalfTensor_setStorage1d"
  c_THHalfTensor_setStorage1d :: (Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> IO ()

-- |c_THHalfTensor_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h THHalfTensor_setStorage2d"
  c_THHalfTensor_setStorage2d :: (Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THHalfTensor_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h THHalfTensor_setStorage3d"
  c_THHalfTensor_setStorage3d :: (Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THHalfTensor_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h THHalfTensor_setStorage4d"
  c_THHalfTensor_setStorage4d :: (Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THHalfTensor_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h THHalfTensor_narrow"
  c_THHalfTensor_narrow :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CLLong -> CLLong -> IO ()

-- |c_THHalfTensor_select : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h THHalfTensor_select"
  c_THHalfTensor_select :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CLLong -> IO ()

-- |c_THHalfTensor_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h THHalfTensor_transpose"
  c_THHalfTensor_transpose :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_unfold : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h THHalfTensor_unfold"
  c_THHalfTensor_unfold :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CLLong -> CLLong -> IO ()

-- |c_THHalfTensor_squeeze : self src -> void
foreign import ccall "THTensor.h THHalfTensor_squeeze"
  c_THHalfTensor_squeeze :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_squeeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h THHalfTensor_squeeze1d"
  c_THHalfTensor_squeeze1d :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_unsqueeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h THHalfTensor_unsqueeze1d"
  c_THHalfTensor_unsqueeze1d :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_isContiguous : self -> int
foreign import ccall "THTensor.h THHalfTensor_isContiguous"
  c_THHalfTensor_isContiguous :: (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensor_isSameSizeAs : self src -> int
foreign import ccall "THTensor.h THHalfTensor_isSameSizeAs"
  c_THHalfTensor_isSameSizeAs :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensor_isSetTo : self src -> int
foreign import ccall "THTensor.h THHalfTensor_isSetTo"
  c_THHalfTensor_isSetTo :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensor_isSize : self dims -> int
foreign import ccall "THTensor.h THHalfTensor_isSize"
  c_THHalfTensor_isSize :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> CInt

-- |c_THHalfTensor_nElement : self -> ptrdiff_t
foreign import ccall "THTensor.h THHalfTensor_nElement"
  c_THHalfTensor_nElement :: (Ptr CTHHalfTensor) -> CPtrdiff

-- |c_THHalfTensor_retain : self -> void
foreign import ccall "THTensor.h THHalfTensor_retain"
  c_THHalfTensor_retain :: (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_free : self -> void
foreign import ccall "THTensor.h THHalfTensor_free"
  c_THHalfTensor_free :: (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_freeCopyTo : self dst -> void
foreign import ccall "THTensor.h THHalfTensor_freeCopyTo"
  c_THHalfTensor_freeCopyTo :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_set1d : tensor x0 value -> void
foreign import ccall "THTensor.h THHalfTensor_set1d"
  c_THHalfTensor_set1d :: (Ptr CTHHalfTensor) -> CLLong -> THHalf -> IO ()

-- |c_THHalfTensor_set2d : tensor x0 x1 value -> void
foreign import ccall "THTensor.h THHalfTensor_set2d"
  c_THHalfTensor_set2d :: (Ptr CTHHalfTensor) -> CLLong -> CLLong -> THHalf -> IO ()

-- |c_THHalfTensor_set3d : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h THHalfTensor_set3d"
  c_THHalfTensor_set3d :: (Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> THHalf -> IO ()

-- |c_THHalfTensor_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h THHalfTensor_set4d"
  c_THHalfTensor_set4d :: (Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> THHalf -> IO ()

-- |c_THHalfTensor_get1d : tensor x0 -> real
foreign import ccall "THTensor.h THHalfTensor_get1d"
  c_THHalfTensor_get1d :: (Ptr CTHHalfTensor) -> CLLong -> THHalf

-- |c_THHalfTensor_get2d : tensor x0 x1 -> real
foreign import ccall "THTensor.h THHalfTensor_get2d"
  c_THHalfTensor_get2d :: (Ptr CTHHalfTensor) -> CLLong -> CLLong -> THHalf

-- |c_THHalfTensor_get3d : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h THHalfTensor_get3d"
  c_THHalfTensor_get3d :: (Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> THHalf

-- |c_THHalfTensor_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h THHalfTensor_get4d"
  c_THHalfTensor_get4d :: (Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> THHalf

-- |c_THHalfTensor_desc : tensor -> THDescBuff
foreign import ccall "THTensor.h THHalfTensor_desc"
  c_THHalfTensor_desc :: (Ptr CTHHalfTensor) -> CTHDescBuff

-- |c_THHalfTensor_sizeDesc : tensor -> THDescBuff
foreign import ccall "THTensor.h THHalfTensor_sizeDesc"
  c_THHalfTensor_sizeDesc :: (Ptr CTHHalfTensor) -> CTHDescBuff

-- |p_THHalfTensor_storage : Pointer to function : self -> THStorage *
foreign import ccall "THTensor.h &THHalfTensor_storage"
  p_THHalfTensor_storage :: FunPtr ((Ptr CTHHalfTensor) -> IO (Ptr CTHHalfStorage))

-- |p_THHalfTensor_storageOffset : Pointer to function : self -> ptrdiff_t
foreign import ccall "THTensor.h &THHalfTensor_storageOffset"
  p_THHalfTensor_storageOffset :: FunPtr ((Ptr CTHHalfTensor) -> CPtrdiff)

-- |p_THHalfTensor_nDimension : Pointer to function : self -> int
foreign import ccall "THTensor.h &THHalfTensor_nDimension"
  p_THHalfTensor_nDimension :: FunPtr ((Ptr CTHHalfTensor) -> CInt)

-- |p_THHalfTensor_size : Pointer to function : self dim -> int64_t
foreign import ccall "THTensor.h &THHalfTensor_size"
  p_THHalfTensor_size :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> CLLong)

-- |p_THHalfTensor_stride : Pointer to function : self dim -> int64_t
foreign import ccall "THTensor.h &THHalfTensor_stride"
  p_THHalfTensor_stride :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> CLLong)

-- |p_THHalfTensor_newSizeOf : Pointer to function : self -> THLongStorage *
foreign import ccall "THTensor.h &THHalfTensor_newSizeOf"
  p_THHalfTensor_newSizeOf :: FunPtr ((Ptr CTHHalfTensor) -> IO (Ptr CTHLongStorage))

-- |p_THHalfTensor_newStrideOf : Pointer to function : self -> THLongStorage *
foreign import ccall "THTensor.h &THHalfTensor_newStrideOf"
  p_THHalfTensor_newStrideOf :: FunPtr ((Ptr CTHHalfTensor) -> IO (Ptr CTHLongStorage))

-- |p_THHalfTensor_data : Pointer to function : self -> real *
foreign import ccall "THTensor.h &THHalfTensor_data"
  p_THHalfTensor_data :: FunPtr ((Ptr CTHHalfTensor) -> IO (Ptr THHalf))

-- |p_THHalfTensor_setFlag : Pointer to function : self flag -> void
foreign import ccall "THTensor.h &THHalfTensor_setFlag"
  p_THHalfTensor_setFlag :: FunPtr ((Ptr CTHHalfTensor) -> CChar -> IO ())

-- |p_THHalfTensor_clearFlag : Pointer to function : self flag -> void
foreign import ccall "THTensor.h &THHalfTensor_clearFlag"
  p_THHalfTensor_clearFlag :: FunPtr ((Ptr CTHHalfTensor) -> CChar -> IO ())

-- |p_THHalfTensor_new : Pointer to function :  -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_new"
  p_THHalfTensor_new :: FunPtr (IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newWithTensor : Pointer to function : tensor -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newWithTensor"
  p_THHalfTensor_newWithTensor :: FunPtr ((Ptr CTHHalfTensor) -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newWithStorage : Pointer to function : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newWithStorage"
  p_THHalfTensor_newWithStorage :: FunPtr (Ptr CTHHalfStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newWithStorage1d : Pointer to function : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newWithStorage1d"
  p_THHalfTensor_newWithStorage1d :: FunPtr (Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newWithStorage2d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newWithStorage2d"
  p_THHalfTensor_newWithStorage2d :: FunPtr (Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newWithStorage3d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newWithStorage3d"
  p_THHalfTensor_newWithStorage3d :: FunPtr (Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newWithStorage4d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newWithStorage4d"
  p_THHalfTensor_newWithStorage4d :: FunPtr (Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newWithSize : Pointer to function : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newWithSize"
  p_THHalfTensor_newWithSize :: FunPtr (Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newWithSize1d : Pointer to function : size0_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newWithSize1d"
  p_THHalfTensor_newWithSize1d :: FunPtr (CLLong -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newWithSize2d : Pointer to function : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newWithSize2d"
  p_THHalfTensor_newWithSize2d :: FunPtr (CLLong -> CLLong -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newWithSize3d : Pointer to function : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newWithSize3d"
  p_THHalfTensor_newWithSize3d :: FunPtr (CLLong -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newWithSize4d : Pointer to function : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newWithSize4d"
  p_THHalfTensor_newWithSize4d :: FunPtr (CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newClone : Pointer to function : self -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newClone"
  p_THHalfTensor_newClone :: FunPtr ((Ptr CTHHalfTensor) -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newContiguous : Pointer to function : tensor -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newContiguous"
  p_THHalfTensor_newContiguous :: FunPtr ((Ptr CTHHalfTensor) -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newSelect : Pointer to function : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newSelect"
  p_THHalfTensor_newSelect :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> CLLong -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newNarrow : Pointer to function : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newNarrow"
  p_THHalfTensor_newNarrow :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newTranspose : Pointer to function : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newTranspose"
  p_THHalfTensor_newTranspose :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> CInt -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newUnfold : Pointer to function : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newUnfold"
  p_THHalfTensor_newUnfold :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newView : Pointer to function : tensor size -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newView"
  p_THHalfTensor_newView :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_newExpand : Pointer to function : tensor size -> THTensor *
foreign import ccall "THTensor.h &THHalfTensor_newExpand"
  p_THHalfTensor_newExpand :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHHalfTensor))

-- |p_THHalfTensor_expand : Pointer to function : r tensor size -> void
foreign import ccall "THTensor.h &THHalfTensor_expand"
  p_THHalfTensor_expand :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THHalfTensor_expandNd : Pointer to function : rets ops count -> void
foreign import ccall "THTensor.h &THHalfTensor_expandNd"
  p_THHalfTensor_expandNd :: FunPtr (Ptr (Ptr CTHHalfTensor) -> Ptr (Ptr CTHHalfTensor) -> CInt -> IO ())

-- |p_THHalfTensor_resize : Pointer to function : tensor size stride -> void
foreign import ccall "THTensor.h &THHalfTensor_resize"
  p_THHalfTensor_resize :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THHalfTensor_resizeAs : Pointer to function : tensor src -> void
foreign import ccall "THTensor.h &THHalfTensor_resizeAs"
  p_THHalfTensor_resizeAs :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_resizeNd : Pointer to function : tensor nDimension size stride -> void
foreign import ccall "THTensor.h &THHalfTensor_resizeNd"
  p_THHalfTensor_resizeNd :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- |p_THHalfTensor_resize1d : Pointer to function : tensor size0_ -> void
foreign import ccall "THTensor.h &THHalfTensor_resize1d"
  p_THHalfTensor_resize1d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> IO ())

-- |p_THHalfTensor_resize2d : Pointer to function : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h &THHalfTensor_resize2d"
  p_THHalfTensor_resize2d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> CLLong -> IO ())

-- |p_THHalfTensor_resize3d : Pointer to function : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h &THHalfTensor_resize3d"
  p_THHalfTensor_resize3d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THHalfTensor_resize4d : Pointer to function : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h &THHalfTensor_resize4d"
  p_THHalfTensor_resize4d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THHalfTensor_resize5d : Pointer to function : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h &THHalfTensor_resize5d"
  p_THHalfTensor_resize5d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THHalfTensor_set : Pointer to function : self src -> void
foreign import ccall "THTensor.h &THHalfTensor_set"
  p_THHalfTensor_set :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_setStorage : Pointer to function : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h &THHalfTensor_setStorage"
  p_THHalfTensor_setStorage :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THHalfTensor_setStorageNd : Pointer to function : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h &THHalfTensor_setStorageNd"
  p_THHalfTensor_setStorageNd :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- |p_THHalfTensor_setStorage1d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h &THHalfTensor_setStorage1d"
  p_THHalfTensor_setStorage1d :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> IO ())

-- |p_THHalfTensor_setStorage2d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h &THHalfTensor_setStorage2d"
  p_THHalfTensor_setStorage2d :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THHalfTensor_setStorage3d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h &THHalfTensor_setStorage3d"
  p_THHalfTensor_setStorage3d :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THHalfTensor_setStorage4d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h &THHalfTensor_setStorage4d"
  p_THHalfTensor_setStorage4d :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THHalfTensor_narrow : Pointer to function : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h &THHalfTensor_narrow"
  p_THHalfTensor_narrow :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CLLong -> CLLong -> IO ())

-- |p_THHalfTensor_select : Pointer to function : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h &THHalfTensor_select"
  p_THHalfTensor_select :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CLLong -> IO ())

-- |p_THHalfTensor_transpose : Pointer to function : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h &THHalfTensor_transpose"
  p_THHalfTensor_transpose :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ())

-- |p_THHalfTensor_unfold : Pointer to function : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h &THHalfTensor_unfold"
  p_THHalfTensor_unfold :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CLLong -> CLLong -> IO ())

-- |p_THHalfTensor_squeeze : Pointer to function : self src -> void
foreign import ccall "THTensor.h &THHalfTensor_squeeze"
  p_THHalfTensor_squeeze :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_squeeze1d : Pointer to function : self src dimension_ -> void
foreign import ccall "THTensor.h &THHalfTensor_squeeze1d"
  p_THHalfTensor_squeeze1d :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ())

-- |p_THHalfTensor_unsqueeze1d : Pointer to function : self src dimension_ -> void
foreign import ccall "THTensor.h &THHalfTensor_unsqueeze1d"
  p_THHalfTensor_unsqueeze1d :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ())

-- |p_THHalfTensor_isContiguous : Pointer to function : self -> int
foreign import ccall "THTensor.h &THHalfTensor_isContiguous"
  p_THHalfTensor_isContiguous :: FunPtr ((Ptr CTHHalfTensor) -> CInt)

-- |p_THHalfTensor_isSameSizeAs : Pointer to function : self src -> int
foreign import ccall "THTensor.h &THHalfTensor_isSameSizeAs"
  p_THHalfTensor_isSameSizeAs :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt)

-- |p_THHalfTensor_isSetTo : Pointer to function : self src -> int
foreign import ccall "THTensor.h &THHalfTensor_isSetTo"
  p_THHalfTensor_isSetTo :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt)

-- |p_THHalfTensor_isSize : Pointer to function : self dims -> int
foreign import ccall "THTensor.h &THHalfTensor_isSize"
  p_THHalfTensor_isSize :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> CInt)

-- |p_THHalfTensor_nElement : Pointer to function : self -> ptrdiff_t
foreign import ccall "THTensor.h &THHalfTensor_nElement"
  p_THHalfTensor_nElement :: FunPtr ((Ptr CTHHalfTensor) -> CPtrdiff)

-- |p_THHalfTensor_retain : Pointer to function : self -> void
foreign import ccall "THTensor.h &THHalfTensor_retain"
  p_THHalfTensor_retain :: FunPtr ((Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_free : Pointer to function : self -> void
foreign import ccall "THTensor.h &THHalfTensor_free"
  p_THHalfTensor_free :: FunPtr ((Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_freeCopyTo : Pointer to function : self dst -> void
foreign import ccall "THTensor.h &THHalfTensor_freeCopyTo"
  p_THHalfTensor_freeCopyTo :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_set1d : Pointer to function : tensor x0 value -> void
foreign import ccall "THTensor.h &THHalfTensor_set1d"
  p_THHalfTensor_set1d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> THHalf -> IO ())

-- |p_THHalfTensor_set2d : Pointer to function : tensor x0 x1 value -> void
foreign import ccall "THTensor.h &THHalfTensor_set2d"
  p_THHalfTensor_set2d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> CLLong -> THHalf -> IO ())

-- |p_THHalfTensor_set3d : Pointer to function : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h &THHalfTensor_set3d"
  p_THHalfTensor_set3d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> THHalf -> IO ())

-- |p_THHalfTensor_set4d : Pointer to function : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h &THHalfTensor_set4d"
  p_THHalfTensor_set4d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> THHalf -> IO ())

-- |p_THHalfTensor_get1d : Pointer to function : tensor x0 -> real
foreign import ccall "THTensor.h &THHalfTensor_get1d"
  p_THHalfTensor_get1d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> THHalf)

-- |p_THHalfTensor_get2d : Pointer to function : tensor x0 x1 -> real
foreign import ccall "THTensor.h &THHalfTensor_get2d"
  p_THHalfTensor_get2d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> CLLong -> THHalf)

-- |p_THHalfTensor_get3d : Pointer to function : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h &THHalfTensor_get3d"
  p_THHalfTensor_get3d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> THHalf)

-- |p_THHalfTensor_get4d : Pointer to function : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h &THHalfTensor_get4d"
  p_THHalfTensor_get4d :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> THHalf)

-- |p_THHalfTensor_desc : Pointer to function : tensor -> THDescBuff
foreign import ccall "THTensor.h &THHalfTensor_desc"
  p_THHalfTensor_desc :: FunPtr ((Ptr CTHHalfTensor) -> CTHDescBuff)

-- |p_THHalfTensor_sizeDesc : Pointer to function : tensor -> THDescBuff
foreign import ccall "THTensor.h &THHalfTensor_sizeDesc"
  p_THHalfTensor_sizeDesc :: FunPtr ((Ptr CTHHalfTensor) -> CTHDescBuff)