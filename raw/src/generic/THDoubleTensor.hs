{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleTensor (
    c_THDoubleTensor_storage,
    c_THDoubleTensor_storageOffset,
    c_THDoubleTensor_nDimension,
    c_THDoubleTensor_size,
    c_THDoubleTensor_stride,
    c_THDoubleTensor_newSizeOf,
    c_THDoubleTensor_newStrideOf,
    c_THDoubleTensor_data,
    c_THDoubleTensor_setFlag,
    c_THDoubleTensor_clearFlag,
    c_THDoubleTensor_new,
    c_THDoubleTensor_newWithTensor,
    c_THDoubleTensor_newWithStorage,
    c_THDoubleTensor_newWithStorage1d,
    c_THDoubleTensor_newWithStorage2d,
    c_THDoubleTensor_newWithStorage3d,
    c_THDoubleTensor_newWithStorage4d,
    c_THDoubleTensor_newWithSize,
    c_THDoubleTensor_newWithSize1d,
    c_THDoubleTensor_newWithSize2d,
    c_THDoubleTensor_newWithSize3d,
    c_THDoubleTensor_newWithSize4d,
    c_THDoubleTensor_newClone,
    c_THDoubleTensor_newContiguous,
    c_THDoubleTensor_newSelect,
    c_THDoubleTensor_newNarrow,
    c_THDoubleTensor_newTranspose,
    c_THDoubleTensor_newUnfold,
    c_THDoubleTensor_newView,
    c_THDoubleTensor_newExpand,
    c_THDoubleTensor_expand,
    c_THDoubleTensor_expandNd,
    c_THDoubleTensor_resize,
    c_THDoubleTensor_resizeAs,
    c_THDoubleTensor_resizeNd,
    c_THDoubleTensor_resize1d,
    c_THDoubleTensor_resize2d,
    c_THDoubleTensor_resize3d,
    c_THDoubleTensor_resize4d,
    c_THDoubleTensor_resize5d,
    c_THDoubleTensor_set,
    c_THDoubleTensor_setStorage,
    c_THDoubleTensor_setStorageNd,
    c_THDoubleTensor_setStorage1d,
    c_THDoubleTensor_setStorage2d,
    c_THDoubleTensor_setStorage3d,
    c_THDoubleTensor_setStorage4d,
    c_THDoubleTensor_narrow,
    c_THDoubleTensor_select,
    c_THDoubleTensor_transpose,
    c_THDoubleTensor_unfold,
    c_THDoubleTensor_squeeze,
    c_THDoubleTensor_squeeze1d,
    c_THDoubleTensor_unsqueeze1d,
    c_THDoubleTensor_isContiguous,
    c_THDoubleTensor_isSameSizeAs,
    c_THDoubleTensor_isSetTo,
    c_THDoubleTensor_isSize,
    c_THDoubleTensor_nElement,
    c_THDoubleTensor_retain,
    c_THDoubleTensor_free,
    c_THDoubleTensor_freeCopyTo,
    c_THDoubleTensor_set1d,
    c_THDoubleTensor_set2d,
    c_THDoubleTensor_set3d,
    c_THDoubleTensor_set4d,
    c_THDoubleTensor_get1d,
    c_THDoubleTensor_get2d,
    c_THDoubleTensor_get3d,
    c_THDoubleTensor_get4d,
    c_THDoubleTensor_desc,
    c_THDoubleTensor_sizeDesc,
    p_THDoubleTensor_storage,
    p_THDoubleTensor_storageOffset,
    p_THDoubleTensor_nDimension,
    p_THDoubleTensor_size,
    p_THDoubleTensor_stride,
    p_THDoubleTensor_newSizeOf,
    p_THDoubleTensor_newStrideOf,
    p_THDoubleTensor_data,
    p_THDoubleTensor_setFlag,
    p_THDoubleTensor_clearFlag,
    p_THDoubleTensor_new,
    p_THDoubleTensor_newWithTensor,
    p_THDoubleTensor_newWithStorage,
    p_THDoubleTensor_newWithStorage1d,
    p_THDoubleTensor_newWithStorage2d,
    p_THDoubleTensor_newWithStorage3d,
    p_THDoubleTensor_newWithStorage4d,
    p_THDoubleTensor_newWithSize,
    p_THDoubleTensor_newWithSize1d,
    p_THDoubleTensor_newWithSize2d,
    p_THDoubleTensor_newWithSize3d,
    p_THDoubleTensor_newWithSize4d,
    p_THDoubleTensor_newClone,
    p_THDoubleTensor_newContiguous,
    p_THDoubleTensor_newSelect,
    p_THDoubleTensor_newNarrow,
    p_THDoubleTensor_newTranspose,
    p_THDoubleTensor_newUnfold,
    p_THDoubleTensor_newView,
    p_THDoubleTensor_newExpand,
    p_THDoubleTensor_expand,
    p_THDoubleTensor_expandNd,
    p_THDoubleTensor_resize,
    p_THDoubleTensor_resizeAs,
    p_THDoubleTensor_resizeNd,
    p_THDoubleTensor_resize1d,
    p_THDoubleTensor_resize2d,
    p_THDoubleTensor_resize3d,
    p_THDoubleTensor_resize4d,
    p_THDoubleTensor_resize5d,
    p_THDoubleTensor_set,
    p_THDoubleTensor_setStorage,
    p_THDoubleTensor_setStorageNd,
    p_THDoubleTensor_setStorage1d,
    p_THDoubleTensor_setStorage2d,
    p_THDoubleTensor_setStorage3d,
    p_THDoubleTensor_setStorage4d,
    p_THDoubleTensor_narrow,
    p_THDoubleTensor_select,
    p_THDoubleTensor_transpose,
    p_THDoubleTensor_unfold,
    p_THDoubleTensor_squeeze,
    p_THDoubleTensor_squeeze1d,
    p_THDoubleTensor_unsqueeze1d,
    p_THDoubleTensor_isContiguous,
    p_THDoubleTensor_isSameSizeAs,
    p_THDoubleTensor_isSetTo,
    p_THDoubleTensor_isSize,
    p_THDoubleTensor_nElement,
    p_THDoubleTensor_retain,
    p_THDoubleTensor_free,
    p_THDoubleTensor_freeCopyTo,
    p_THDoubleTensor_set1d,
    p_THDoubleTensor_set2d,
    p_THDoubleTensor_set3d,
    p_THDoubleTensor_set4d,
    p_THDoubleTensor_get1d,
    p_THDoubleTensor_get2d,
    p_THDoubleTensor_get3d,
    p_THDoubleTensor_get4d,
    p_THDoubleTensor_desc,
    p_THDoubleTensor_sizeDesc) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleTensor_storage : self -> THStorage *
foreign import ccall unsafe "THTensor.h THDoubleTensor_storage"
  c_THDoubleTensor_storage :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleTensor_storageOffset : self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h THDoubleTensor_storageOffset"
  c_THDoubleTensor_storageOffset :: (Ptr CTHDoubleTensor) -> CPtrdiff

-- |c_THDoubleTensor_nDimension : self -> int
foreign import ccall unsafe "THTensor.h THDoubleTensor_nDimension"
  c_THDoubleTensor_nDimension :: (Ptr CTHDoubleTensor) -> CInt

-- |c_THDoubleTensor_size : self dim -> long
foreign import ccall unsafe "THTensor.h THDoubleTensor_size"
  c_THDoubleTensor_size :: (Ptr CTHDoubleTensor) -> CInt -> CLong

-- |c_THDoubleTensor_stride : self dim -> long
foreign import ccall unsafe "THTensor.h THDoubleTensor_stride"
  c_THDoubleTensor_stride :: (Ptr CTHDoubleTensor) -> CInt -> CLong

-- |c_THDoubleTensor_newSizeOf : self -> THLongStorage *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newSizeOf"
  c_THDoubleTensor_newSizeOf :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHLongStorage)

-- |c_THDoubleTensor_newStrideOf : self -> THLongStorage *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newStrideOf"
  c_THDoubleTensor_newStrideOf :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHLongStorage)

-- |c_THDoubleTensor_data : self -> real *
foreign import ccall unsafe "THTensor.h THDoubleTensor_data"
  c_THDoubleTensor_data :: (Ptr CTHDoubleTensor) -> IO (Ptr CDouble)

-- |c_THDoubleTensor_setFlag : self flag -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_setFlag"
  c_THDoubleTensor_setFlag :: (Ptr CTHDoubleTensor) -> CChar -> IO ()

-- |c_THDoubleTensor_clearFlag : self flag -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_clearFlag"
  c_THDoubleTensor_clearFlag :: (Ptr CTHDoubleTensor) -> CChar -> IO ()

-- |c_THDoubleTensor_new :  -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_new"
  c_THDoubleTensor_new :: IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithTensor : tensor -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newWithTensor"
  c_THDoubleTensor_newWithTensor :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithStorage : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newWithStorage"
  c_THDoubleTensor_newWithStorage :: Ptr CTHDoubleStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newWithStorage1d"
  c_THDoubleTensor_newWithStorage1d :: Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newWithStorage2d"
  c_THDoubleTensor_newWithStorage2d :: Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newWithStorage3d"
  c_THDoubleTensor_newWithStorage3d :: Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newWithStorage4d"
  c_THDoubleTensor_newWithStorage4d :: Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithSize : size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newWithSize"
  c_THDoubleTensor_newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithSize1d : size0_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newWithSize1d"
  c_THDoubleTensor_newWithSize1d :: CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newWithSize2d"
  c_THDoubleTensor_newWithSize2d :: CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newWithSize3d"
  c_THDoubleTensor_newWithSize3d :: CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newWithSize4d"
  c_THDoubleTensor_newWithSize4d :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newClone : self -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newClone"
  c_THDoubleTensor_newClone :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newContiguous : tensor -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newContiguous"
  c_THDoubleTensor_newContiguous :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newSelect"
  c_THDoubleTensor_newSelect :: (Ptr CTHDoubleTensor) -> CInt -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newNarrow"
  c_THDoubleTensor_newNarrow :: (Ptr CTHDoubleTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newTranspose"
  c_THDoubleTensor_newTranspose :: (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newUnfold"
  c_THDoubleTensor_newUnfold :: (Ptr CTHDoubleTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newView : tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newView"
  c_THDoubleTensor_newView :: (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newExpand : tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h THDoubleTensor_newExpand"
  c_THDoubleTensor_newExpand :: (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_expand : r tensor size -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_expand"
  c_THDoubleTensor_expand :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleTensor_expandNd : rets ops count -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_expandNd"
  c_THDoubleTensor_expandNd :: Ptr (Ptr CTHDoubleTensor) -> Ptr (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensor_resize : tensor size stride -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_resize"
  c_THDoubleTensor_resize :: (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleTensor_resizeAs : tensor src -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_resizeAs"
  c_THDoubleTensor_resizeAs :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_resizeNd : tensor nDimension size stride -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_resizeNd"
  c_THDoubleTensor_resizeNd :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THDoubleTensor_resize1d : tensor size0_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_resize1d"
  c_THDoubleTensor_resize1d :: (Ptr CTHDoubleTensor) -> CLong -> IO ()

-- |c_THDoubleTensor_resize2d : tensor size0_ size1_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_resize2d"
  c_THDoubleTensor_resize2d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_resize3d"
  c_THDoubleTensor_resize3d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_resize4d"
  c_THDoubleTensor_resize4d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_resize5d"
  c_THDoubleTensor_resize5d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_set : self src -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_set"
  c_THDoubleTensor_set :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_setStorage"
  c_THDoubleTensor_setStorage :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleTensor_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_setStorageNd"
  c_THDoubleTensor_setStorageNd :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> CPtrdiff -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THDoubleTensor_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_setStorage1d"
  c_THDoubleTensor_setStorage1d :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_setStorage2d"
  c_THDoubleTensor_setStorage2d :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_setStorage3d"
  c_THDoubleTensor_setStorage3d :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_setStorage4d"
  c_THDoubleTensor_setStorage4d :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_narrow"
  c_THDoubleTensor_narrow :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_select : self src dimension_ sliceIndex_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_select"
  c_THDoubleTensor_select :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CLong -> IO ()

-- |c_THDoubleTensor_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_transpose"
  c_THDoubleTensor_transpose :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_unfold : self src dimension_ size_ step_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_unfold"
  c_THDoubleTensor_unfold :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_squeeze : self src -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_squeeze"
  c_THDoubleTensor_squeeze :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_squeeze1d : self src dimension_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_squeeze1d"
  c_THDoubleTensor_squeeze1d :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensor_unsqueeze1d : self src dimension_ -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_unsqueeze1d"
  c_THDoubleTensor_unsqueeze1d :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensor_isContiguous : self -> int
foreign import ccall unsafe "THTensor.h THDoubleTensor_isContiguous"
  c_THDoubleTensor_isContiguous :: (Ptr CTHDoubleTensor) -> CInt

-- |c_THDoubleTensor_isSameSizeAs : self src -> int
foreign import ccall unsafe "THTensor.h THDoubleTensor_isSameSizeAs"
  c_THDoubleTensor_isSameSizeAs :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt

-- |c_THDoubleTensor_isSetTo : self src -> int
foreign import ccall unsafe "THTensor.h THDoubleTensor_isSetTo"
  c_THDoubleTensor_isSetTo :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt

-- |c_THDoubleTensor_isSize : self dims -> int
foreign import ccall unsafe "THTensor.h THDoubleTensor_isSize"
  c_THDoubleTensor_isSize :: (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> CInt

-- |c_THDoubleTensor_nElement : self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h THDoubleTensor_nElement"
  c_THDoubleTensor_nElement :: (Ptr CTHDoubleTensor) -> CPtrdiff

-- |c_THDoubleTensor_retain : self -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_retain"
  c_THDoubleTensor_retain :: (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_free : self -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_free"
  c_THDoubleTensor_free :: (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_freeCopyTo : self dst -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_freeCopyTo"
  c_THDoubleTensor_freeCopyTo :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_set1d : tensor x0 value -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_set1d"
  c_THDoubleTensor_set1d :: (Ptr CTHDoubleTensor) -> CLong -> CDouble -> IO ()

-- |c_THDoubleTensor_set2d : tensor x0 x1 value -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_set2d"
  c_THDoubleTensor_set2d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CDouble -> IO ()

-- |c_THDoubleTensor_set3d : tensor x0 x1 x2 value -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_set3d"
  c_THDoubleTensor_set3d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CDouble -> IO ()

-- |c_THDoubleTensor_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall unsafe "THTensor.h THDoubleTensor_set4d"
  c_THDoubleTensor_set4d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CLong -> CDouble -> IO ()

-- |c_THDoubleTensor_get1d : tensor x0 -> real
foreign import ccall unsafe "THTensor.h THDoubleTensor_get1d"
  c_THDoubleTensor_get1d :: (Ptr CTHDoubleTensor) -> CLong -> CDouble

-- |c_THDoubleTensor_get2d : tensor x0 x1 -> real
foreign import ccall unsafe "THTensor.h THDoubleTensor_get2d"
  c_THDoubleTensor_get2d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CDouble

-- |c_THDoubleTensor_get3d : tensor x0 x1 x2 -> real
foreign import ccall unsafe "THTensor.h THDoubleTensor_get3d"
  c_THDoubleTensor_get3d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CDouble

-- |c_THDoubleTensor_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall unsafe "THTensor.h THDoubleTensor_get4d"
  c_THDoubleTensor_get4d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CLong -> CDouble

-- |c_THDoubleTensor_desc : tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h THDoubleTensor_desc"
  c_THDoubleTensor_desc :: (Ptr CTHDoubleTensor) -> CTHDescBuff

-- |c_THDoubleTensor_sizeDesc : tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h THDoubleTensor_sizeDesc"
  c_THDoubleTensor_sizeDesc :: (Ptr CTHDoubleTensor) -> CTHDescBuff

-- |p_THDoubleTensor_storage : Pointer to function self -> THStorage *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_storage"
  p_THDoubleTensor_storage :: FunPtr ((Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleStorage))

-- |p_THDoubleTensor_storageOffset : Pointer to function self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h &THDoubleTensor_storageOffset"
  p_THDoubleTensor_storageOffset :: FunPtr ((Ptr CTHDoubleTensor) -> CPtrdiff)

-- |p_THDoubleTensor_nDimension : Pointer to function self -> int
foreign import ccall unsafe "THTensor.h &THDoubleTensor_nDimension"
  p_THDoubleTensor_nDimension :: FunPtr ((Ptr CTHDoubleTensor) -> CInt)

-- |p_THDoubleTensor_size : Pointer to function self dim -> long
foreign import ccall unsafe "THTensor.h &THDoubleTensor_size"
  p_THDoubleTensor_size :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> CLong)

-- |p_THDoubleTensor_stride : Pointer to function self dim -> long
foreign import ccall unsafe "THTensor.h &THDoubleTensor_stride"
  p_THDoubleTensor_stride :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> CLong)

-- |p_THDoubleTensor_newSizeOf : Pointer to function self -> THLongStorage *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newSizeOf"
  p_THDoubleTensor_newSizeOf :: FunPtr ((Ptr CTHDoubleTensor) -> IO (Ptr CTHLongStorage))

-- |p_THDoubleTensor_newStrideOf : Pointer to function self -> THLongStorage *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newStrideOf"
  p_THDoubleTensor_newStrideOf :: FunPtr ((Ptr CTHDoubleTensor) -> IO (Ptr CTHLongStorage))

-- |p_THDoubleTensor_data : Pointer to function self -> real *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_data"
  p_THDoubleTensor_data :: FunPtr ((Ptr CTHDoubleTensor) -> IO (Ptr CDouble))

-- |p_THDoubleTensor_setFlag : Pointer to function self flag -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_setFlag"
  p_THDoubleTensor_setFlag :: FunPtr ((Ptr CTHDoubleTensor) -> CChar -> IO ())

-- |p_THDoubleTensor_clearFlag : Pointer to function self flag -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_clearFlag"
  p_THDoubleTensor_clearFlag :: FunPtr ((Ptr CTHDoubleTensor) -> CChar -> IO ())

-- |p_THDoubleTensor_new : Pointer to function  -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_new"
  p_THDoubleTensor_new :: FunPtr (IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newWithTensor : Pointer to function tensor -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newWithTensor"
  p_THDoubleTensor_newWithTensor :: FunPtr ((Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newWithStorage : Pointer to function storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newWithStorage"
  p_THDoubleTensor_newWithStorage :: FunPtr (Ptr CTHDoubleStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newWithStorage1d : Pointer to function storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newWithStorage1d"
  p_THDoubleTensor_newWithStorage1d :: FunPtr (Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newWithStorage2d : Pointer to function storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newWithStorage2d"
  p_THDoubleTensor_newWithStorage2d :: FunPtr (Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newWithStorage3d : Pointer to function storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newWithStorage3d"
  p_THDoubleTensor_newWithStorage3d :: FunPtr (Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newWithStorage4d : Pointer to function storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newWithStorage4d"
  p_THDoubleTensor_newWithStorage4d :: FunPtr (Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newWithSize : Pointer to function size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newWithSize"
  p_THDoubleTensor_newWithSize :: FunPtr (Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newWithSize1d : Pointer to function size0_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newWithSize1d"
  p_THDoubleTensor_newWithSize1d :: FunPtr (CLong -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newWithSize2d : Pointer to function size0_ size1_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newWithSize2d"
  p_THDoubleTensor_newWithSize2d :: FunPtr (CLong -> CLong -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newWithSize3d : Pointer to function size0_ size1_ size2_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newWithSize3d"
  p_THDoubleTensor_newWithSize3d :: FunPtr (CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newWithSize4d : Pointer to function size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newWithSize4d"
  p_THDoubleTensor_newWithSize4d :: FunPtr (CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newClone : Pointer to function self -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newClone"
  p_THDoubleTensor_newClone :: FunPtr ((Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newContiguous : Pointer to function tensor -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newContiguous"
  p_THDoubleTensor_newContiguous :: FunPtr ((Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newSelect : Pointer to function tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newSelect"
  p_THDoubleTensor_newSelect :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> CLong -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newNarrow : Pointer to function tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newNarrow"
  p_THDoubleTensor_newNarrow :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newTranspose : Pointer to function tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newTranspose"
  p_THDoubleTensor_newTranspose :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> CInt -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newUnfold : Pointer to function tensor dimension_ size_ step_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newUnfold"
  p_THDoubleTensor_newUnfold :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newView : Pointer to function tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newView"
  p_THDoubleTensor_newView :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_newExpand : Pointer to function tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h &THDoubleTensor_newExpand"
  p_THDoubleTensor_newExpand :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHDoubleTensor))

-- |p_THDoubleTensor_expand : Pointer to function r tensor size -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_expand"
  p_THDoubleTensor_expand :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THDoubleTensor_expandNd : Pointer to function rets ops count -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_expandNd"
  p_THDoubleTensor_expandNd :: FunPtr (Ptr (Ptr CTHDoubleTensor) -> Ptr (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleTensor_resize : Pointer to function tensor size stride -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_resize"
  p_THDoubleTensor_resize :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THDoubleTensor_resizeAs : Pointer to function tensor src -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_resizeAs"
  p_THDoubleTensor_resizeAs :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_resizeNd : Pointer to function tensor nDimension size stride -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_resizeNd"
  p_THDoubleTensor_resizeNd :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ())

-- |p_THDoubleTensor_resize1d : Pointer to function tensor size0_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_resize1d"
  p_THDoubleTensor_resize1d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> IO ())

-- |p_THDoubleTensor_resize2d : Pointer to function tensor size0_ size1_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_resize2d"
  p_THDoubleTensor_resize2d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_resize3d : Pointer to function tensor size0_ size1_ size2_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_resize3d"
  p_THDoubleTensor_resize3d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_resize4d : Pointer to function tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_resize4d"
  p_THDoubleTensor_resize4d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_resize5d : Pointer to function tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_resize5d"
  p_THDoubleTensor_resize5d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_set : Pointer to function self src -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_set"
  p_THDoubleTensor_set :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_setStorage : Pointer to function self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_setStorage"
  p_THDoubleTensor_setStorage :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THDoubleTensor_setStorageNd : Pointer to function self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_setStorageNd"
  p_THDoubleTensor_setStorageNd :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> CPtrdiff -> CInt -> Ptr CLong -> Ptr CLong -> IO ())

-- |p_THDoubleTensor_setStorage1d : Pointer to function self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_setStorage1d"
  p_THDoubleTensor_setStorage1d :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_setStorage2d : Pointer to function self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_setStorage2d"
  p_THDoubleTensor_setStorage2d :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_setStorage3d : Pointer to function self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_setStorage3d"
  p_THDoubleTensor_setStorage3d :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_setStorage4d : Pointer to function self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_setStorage4d"
  p_THDoubleTensor_setStorage4d :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_narrow : Pointer to function self src dimension_ firstIndex_ size_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_narrow"
  p_THDoubleTensor_narrow :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_select : Pointer to function self src dimension_ sliceIndex_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_select"
  p_THDoubleTensor_select :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CLong -> IO ())

-- |p_THDoubleTensor_transpose : Pointer to function self src dimension1_ dimension2_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_transpose"
  p_THDoubleTensor_transpose :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_unfold : Pointer to function self src dimension_ size_ step_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_unfold"
  p_THDoubleTensor_unfold :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_squeeze : Pointer to function self src -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_squeeze"
  p_THDoubleTensor_squeeze :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_squeeze1d : Pointer to function self src dimension_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_squeeze1d"
  p_THDoubleTensor_squeeze1d :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleTensor_unsqueeze1d : Pointer to function self src dimension_ -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_unsqueeze1d"
  p_THDoubleTensor_unsqueeze1d :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleTensor_isContiguous : Pointer to function self -> int
foreign import ccall unsafe "THTensor.h &THDoubleTensor_isContiguous"
  p_THDoubleTensor_isContiguous :: FunPtr ((Ptr CTHDoubleTensor) -> CInt)

-- |p_THDoubleTensor_isSameSizeAs : Pointer to function self src -> int
foreign import ccall unsafe "THTensor.h &THDoubleTensor_isSameSizeAs"
  p_THDoubleTensor_isSameSizeAs :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt)

-- |p_THDoubleTensor_isSetTo : Pointer to function self src -> int
foreign import ccall unsafe "THTensor.h &THDoubleTensor_isSetTo"
  p_THDoubleTensor_isSetTo :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt)

-- |p_THDoubleTensor_isSize : Pointer to function self dims -> int
foreign import ccall unsafe "THTensor.h &THDoubleTensor_isSize"
  p_THDoubleTensor_isSize :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> CInt)

-- |p_THDoubleTensor_nElement : Pointer to function self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h &THDoubleTensor_nElement"
  p_THDoubleTensor_nElement :: FunPtr ((Ptr CTHDoubleTensor) -> CPtrdiff)

-- |p_THDoubleTensor_retain : Pointer to function self -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_retain"
  p_THDoubleTensor_retain :: FunPtr ((Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_free : Pointer to function self -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_free"
  p_THDoubleTensor_free :: FunPtr ((Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_freeCopyTo : Pointer to function self dst -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_freeCopyTo"
  p_THDoubleTensor_freeCopyTo :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_set1d : Pointer to function tensor x0 value -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_set1d"
  p_THDoubleTensor_set1d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CDouble -> IO ())

-- |p_THDoubleTensor_set2d : Pointer to function tensor x0 x1 value -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_set2d"
  p_THDoubleTensor_set2d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CLong -> CDouble -> IO ())

-- |p_THDoubleTensor_set3d : Pointer to function tensor x0 x1 x2 value -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_set3d"
  p_THDoubleTensor_set3d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CDouble -> IO ())

-- |p_THDoubleTensor_set4d : Pointer to function tensor x0 x1 x2 x3 value -> void
foreign import ccall unsafe "THTensor.h &THDoubleTensor_set4d"
  p_THDoubleTensor_set4d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CLong -> CDouble -> IO ())

-- |p_THDoubleTensor_get1d : Pointer to function tensor x0 -> real
foreign import ccall unsafe "THTensor.h &THDoubleTensor_get1d"
  p_THDoubleTensor_get1d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CDouble)

-- |p_THDoubleTensor_get2d : Pointer to function tensor x0 x1 -> real
foreign import ccall unsafe "THTensor.h &THDoubleTensor_get2d"
  p_THDoubleTensor_get2d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CLong -> CDouble)

-- |p_THDoubleTensor_get3d : Pointer to function tensor x0 x1 x2 -> real
foreign import ccall unsafe "THTensor.h &THDoubleTensor_get3d"
  p_THDoubleTensor_get3d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CDouble)

-- |p_THDoubleTensor_get4d : Pointer to function tensor x0 x1 x2 x3 -> real
foreign import ccall unsafe "THTensor.h &THDoubleTensor_get4d"
  p_THDoubleTensor_get4d :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CLong -> CDouble)

-- |p_THDoubleTensor_desc : Pointer to function tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h &THDoubleTensor_desc"
  p_THDoubleTensor_desc :: FunPtr ((Ptr CTHDoubleTensor) -> CTHDescBuff)

-- |p_THDoubleTensor_sizeDesc : Pointer to function tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h &THDoubleTensor_sizeDesc"
  p_THDoubleTensor_sizeDesc :: FunPtr ((Ptr CTHDoubleTensor) -> CTHDescBuff)