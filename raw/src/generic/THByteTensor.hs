{-# LANGUAGE ForeignFunctionInterface #-}

module THByteTensor (
    c_THByteTensor_storage,
    c_THByteTensor_storageOffset,
    c_THByteTensor_nDimension,
    c_THByteTensor_size,
    c_THByteTensor_stride,
    c_THByteTensor_newSizeOf,
    c_THByteTensor_newStrideOf,
    c_THByteTensor_data,
    c_THByteTensor_setFlag,
    c_THByteTensor_clearFlag,
    c_THByteTensor_new,
    c_THByteTensor_newWithTensor,
    c_THByteTensor_newWithStorage,
    c_THByteTensor_newWithStorage1d,
    c_THByteTensor_newWithStorage2d,
    c_THByteTensor_newWithStorage3d,
    c_THByteTensor_newWithStorage4d,
    c_THByteTensor_newWithSize,
    c_THByteTensor_newWithSize1d,
    c_THByteTensor_newWithSize2d,
    c_THByteTensor_newWithSize3d,
    c_THByteTensor_newWithSize4d,
    c_THByteTensor_newClone,
    c_THByteTensor_newContiguous,
    c_THByteTensor_newSelect,
    c_THByteTensor_newNarrow,
    c_THByteTensor_newTranspose,
    c_THByteTensor_newUnfold,
    c_THByteTensor_newView,
    c_THByteTensor_newExpand,
    c_THByteTensor_expand,
    c_THByteTensor_expandNd,
    c_THByteTensor_resize,
    c_THByteTensor_resizeAs,
    c_THByteTensor_resizeNd,
    c_THByteTensor_resize1d,
    c_THByteTensor_resize2d,
    c_THByteTensor_resize3d,
    c_THByteTensor_resize4d,
    c_THByteTensor_resize5d,
    c_THByteTensor_set,
    c_THByteTensor_setStorage,
    c_THByteTensor_setStorageNd,
    c_THByteTensor_setStorage1d,
    c_THByteTensor_setStorage2d,
    c_THByteTensor_setStorage3d,
    c_THByteTensor_setStorage4d,
    c_THByteTensor_narrow,
    c_THByteTensor_select,
    c_THByteTensor_transpose,
    c_THByteTensor_unfold,
    c_THByteTensor_squeeze,
    c_THByteTensor_squeeze1d,
    c_THByteTensor_unsqueeze1d,
    c_THByteTensor_isContiguous,
    c_THByteTensor_isSameSizeAs,
    c_THByteTensor_isSetTo,
    c_THByteTensor_isSize,
    c_THByteTensor_nElement,
    c_THByteTensor_retain,
    c_THByteTensor_free,
    c_THByteTensor_freeCopyTo,
    c_THByteTensor_set1d,
    c_THByteTensor_set2d,
    c_THByteTensor_set3d,
    c_THByteTensor_set4d,
    c_THByteTensor_get1d,
    c_THByteTensor_get2d,
    c_THByteTensor_get3d,
    c_THByteTensor_get4d,
    c_THByteTensor_desc,
    c_THByteTensor_sizeDesc,
    p_THByteTensor_storage,
    p_THByteTensor_storageOffset,
    p_THByteTensor_nDimension,
    p_THByteTensor_size,
    p_THByteTensor_stride,
    p_THByteTensor_newSizeOf,
    p_THByteTensor_newStrideOf,
    p_THByteTensor_data,
    p_THByteTensor_setFlag,
    p_THByteTensor_clearFlag,
    p_THByteTensor_new,
    p_THByteTensor_newWithTensor,
    p_THByteTensor_newWithStorage,
    p_THByteTensor_newWithStorage1d,
    p_THByteTensor_newWithStorage2d,
    p_THByteTensor_newWithStorage3d,
    p_THByteTensor_newWithStorage4d,
    p_THByteTensor_newWithSize,
    p_THByteTensor_newWithSize1d,
    p_THByteTensor_newWithSize2d,
    p_THByteTensor_newWithSize3d,
    p_THByteTensor_newWithSize4d,
    p_THByteTensor_newClone,
    p_THByteTensor_newContiguous,
    p_THByteTensor_newSelect,
    p_THByteTensor_newNarrow,
    p_THByteTensor_newTranspose,
    p_THByteTensor_newUnfold,
    p_THByteTensor_newView,
    p_THByteTensor_newExpand,
    p_THByteTensor_expand,
    p_THByteTensor_expandNd,
    p_THByteTensor_resize,
    p_THByteTensor_resizeAs,
    p_THByteTensor_resizeNd,
    p_THByteTensor_resize1d,
    p_THByteTensor_resize2d,
    p_THByteTensor_resize3d,
    p_THByteTensor_resize4d,
    p_THByteTensor_resize5d,
    p_THByteTensor_set,
    p_THByteTensor_setStorage,
    p_THByteTensor_setStorageNd,
    p_THByteTensor_setStorage1d,
    p_THByteTensor_setStorage2d,
    p_THByteTensor_setStorage3d,
    p_THByteTensor_setStorage4d,
    p_THByteTensor_narrow,
    p_THByteTensor_select,
    p_THByteTensor_transpose,
    p_THByteTensor_unfold,
    p_THByteTensor_squeeze,
    p_THByteTensor_squeeze1d,
    p_THByteTensor_unsqueeze1d,
    p_THByteTensor_isContiguous,
    p_THByteTensor_isSameSizeAs,
    p_THByteTensor_isSetTo,
    p_THByteTensor_isSize,
    p_THByteTensor_nElement,
    p_THByteTensor_retain,
    p_THByteTensor_free,
    p_THByteTensor_freeCopyTo,
    p_THByteTensor_set1d,
    p_THByteTensor_set2d,
    p_THByteTensor_set3d,
    p_THByteTensor_set4d,
    p_THByteTensor_get1d,
    p_THByteTensor_get2d,
    p_THByteTensor_get3d,
    p_THByteTensor_get4d,
    p_THByteTensor_desc,
    p_THByteTensor_sizeDesc) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteTensor_storage : self -> THStorage *
foreign import ccall unsafe "THTensor.h THByteTensor_storage"
  c_THByteTensor_storage :: (Ptr CTHByteTensor) -> IO (Ptr CTHByteStorage)

-- |c_THByteTensor_storageOffset : self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h THByteTensor_storageOffset"
  c_THByteTensor_storageOffset :: (Ptr CTHByteTensor) -> CPtrdiff

-- |c_THByteTensor_nDimension : self -> int
foreign import ccall unsafe "THTensor.h THByteTensor_nDimension"
  c_THByteTensor_nDimension :: (Ptr CTHByteTensor) -> CInt

-- |c_THByteTensor_size : self dim -> long
foreign import ccall unsafe "THTensor.h THByteTensor_size"
  c_THByteTensor_size :: (Ptr CTHByteTensor) -> CInt -> CLong

-- |c_THByteTensor_stride : self dim -> long
foreign import ccall unsafe "THTensor.h THByteTensor_stride"
  c_THByteTensor_stride :: (Ptr CTHByteTensor) -> CInt -> CLong

-- |c_THByteTensor_newSizeOf : self -> THLongStorage *
foreign import ccall unsafe "THTensor.h THByteTensor_newSizeOf"
  c_THByteTensor_newSizeOf :: (Ptr CTHByteTensor) -> IO (Ptr CTHLongStorage)

-- |c_THByteTensor_newStrideOf : self -> THLongStorage *
foreign import ccall unsafe "THTensor.h THByteTensor_newStrideOf"
  c_THByteTensor_newStrideOf :: (Ptr CTHByteTensor) -> IO (Ptr CTHLongStorage)

-- |c_THByteTensor_data : self -> real *
foreign import ccall unsafe "THTensor.h THByteTensor_data"
  c_THByteTensor_data :: (Ptr CTHByteTensor) -> IO (Ptr CChar)

-- |c_THByteTensor_setFlag : self flag -> void
foreign import ccall unsafe "THTensor.h THByteTensor_setFlag"
  c_THByteTensor_setFlag :: (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_clearFlag : self flag -> void
foreign import ccall unsafe "THTensor.h THByteTensor_clearFlag"
  c_THByteTensor_clearFlag :: (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_new :  -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_new"
  c_THByteTensor_new :: IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithTensor : tensor -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newWithTensor"
  c_THByteTensor_newWithTensor :: (Ptr CTHByteTensor) -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithStorage : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newWithStorage"
  c_THByteTensor_newWithStorage :: Ptr CTHByteStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newWithStorage1d"
  c_THByteTensor_newWithStorage1d :: Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newWithStorage2d"
  c_THByteTensor_newWithStorage2d :: Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newWithStorage3d"
  c_THByteTensor_newWithStorage3d :: Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newWithStorage4d"
  c_THByteTensor_newWithStorage4d :: Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithSize : size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newWithSize"
  c_THByteTensor_newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithSize1d : size0_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newWithSize1d"
  c_THByteTensor_newWithSize1d :: CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newWithSize2d"
  c_THByteTensor_newWithSize2d :: CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newWithSize3d"
  c_THByteTensor_newWithSize3d :: CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newWithSize4d"
  c_THByteTensor_newWithSize4d :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newClone : self -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newClone"
  c_THByteTensor_newClone :: (Ptr CTHByteTensor) -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newContiguous : tensor -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newContiguous"
  c_THByteTensor_newContiguous :: (Ptr CTHByteTensor) -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newSelect"
  c_THByteTensor_newSelect :: (Ptr CTHByteTensor) -> CInt -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newNarrow"
  c_THByteTensor_newNarrow :: (Ptr CTHByteTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newTranspose"
  c_THByteTensor_newTranspose :: (Ptr CTHByteTensor) -> CInt -> CInt -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newUnfold"
  c_THByteTensor_newUnfold :: (Ptr CTHByteTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newView : tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newView"
  c_THByteTensor_newView :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newExpand : tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h THByteTensor_newExpand"
  c_THByteTensor_newExpand :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_expand : r tensor size -> void
foreign import ccall unsafe "THTensor.h THByteTensor_expand"
  c_THByteTensor_expand :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensor_expandNd : rets ops count -> void
foreign import ccall unsafe "THTensor.h THByteTensor_expandNd"
  c_THByteTensor_expandNd :: Ptr (Ptr CTHByteTensor) -> Ptr (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_resize : tensor size stride -> void
foreign import ccall unsafe "THTensor.h THByteTensor_resize"
  c_THByteTensor_resize :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensor_resizeAs : tensor src -> void
foreign import ccall unsafe "THTensor.h THByteTensor_resizeAs"
  c_THByteTensor_resizeAs :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_resizeNd : tensor nDimension size stride -> void
foreign import ccall unsafe "THTensor.h THByteTensor_resizeNd"
  c_THByteTensor_resizeNd :: (Ptr CTHByteTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THByteTensor_resize1d : tensor size0_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_resize1d"
  c_THByteTensor_resize1d :: (Ptr CTHByteTensor) -> CLong -> IO ()

-- |c_THByteTensor_resize2d : tensor size0_ size1_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_resize2d"
  c_THByteTensor_resize2d :: (Ptr CTHByteTensor) -> CLong -> CLong -> IO ()

-- |c_THByteTensor_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_resize3d"
  c_THByteTensor_resize3d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_resize4d"
  c_THByteTensor_resize4d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_resize5d"
  c_THByteTensor_resize5d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_set : self src -> void
foreign import ccall unsafe "THTensor.h THByteTensor_set"
  c_THByteTensor_set :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_setStorage"
  c_THByteTensor_setStorage :: (Ptr CTHByteTensor) -> Ptr CTHByteStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensor_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall unsafe "THTensor.h THByteTensor_setStorageNd"
  c_THByteTensor_setStorageNd :: (Ptr CTHByteTensor) -> Ptr CTHByteStorage -> CPtrdiff -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THByteTensor_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_setStorage1d"
  c_THByteTensor_setStorage1d :: (Ptr CTHByteTensor) -> Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> IO ()

-- |c_THByteTensor_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_setStorage2d"
  c_THByteTensor_setStorage2d :: (Ptr CTHByteTensor) -> Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_setStorage3d"
  c_THByteTensor_setStorage3d :: (Ptr CTHByteTensor) -> Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_setStorage4d"
  c_THByteTensor_setStorage4d :: (Ptr CTHByteTensor) -> Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_narrow"
  c_THByteTensor_narrow :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THByteTensor_select : self src dimension_ sliceIndex_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_select"
  c_THByteTensor_select :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CLong -> IO ()

-- |c_THByteTensor_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_transpose"
  c_THByteTensor_transpose :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_unfold : self src dimension_ size_ step_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_unfold"
  c_THByteTensor_unfold :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THByteTensor_squeeze : self src -> void
foreign import ccall unsafe "THTensor.h THByteTensor_squeeze"
  c_THByteTensor_squeeze :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_squeeze1d : self src dimension_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_squeeze1d"
  c_THByteTensor_squeeze1d :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_unsqueeze1d : self src dimension_ -> void
foreign import ccall unsafe "THTensor.h THByteTensor_unsqueeze1d"
  c_THByteTensor_unsqueeze1d :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_isContiguous : self -> int
foreign import ccall unsafe "THTensor.h THByteTensor_isContiguous"
  c_THByteTensor_isContiguous :: (Ptr CTHByteTensor) -> CInt

-- |c_THByteTensor_isSameSizeAs : self src -> int
foreign import ccall unsafe "THTensor.h THByteTensor_isSameSizeAs"
  c_THByteTensor_isSameSizeAs :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt

-- |c_THByteTensor_isSetTo : self src -> int
foreign import ccall unsafe "THTensor.h THByteTensor_isSetTo"
  c_THByteTensor_isSetTo :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt

-- |c_THByteTensor_isSize : self dims -> int
foreign import ccall unsafe "THTensor.h THByteTensor_isSize"
  c_THByteTensor_isSize :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> CInt

-- |c_THByteTensor_nElement : self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h THByteTensor_nElement"
  c_THByteTensor_nElement :: (Ptr CTHByteTensor) -> CPtrdiff

-- |c_THByteTensor_retain : self -> void
foreign import ccall unsafe "THTensor.h THByteTensor_retain"
  c_THByteTensor_retain :: (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_free : self -> void
foreign import ccall unsafe "THTensor.h THByteTensor_free"
  c_THByteTensor_free :: (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_freeCopyTo : self dst -> void
foreign import ccall unsafe "THTensor.h THByteTensor_freeCopyTo"
  c_THByteTensor_freeCopyTo :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_set1d : tensor x0 value -> void
foreign import ccall unsafe "THTensor.h THByteTensor_set1d"
  c_THByteTensor_set1d :: (Ptr CTHByteTensor) -> CLong -> CChar -> IO ()

-- |c_THByteTensor_set2d : tensor x0 x1 value -> void
foreign import ccall unsafe "THTensor.h THByteTensor_set2d"
  c_THByteTensor_set2d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CChar -> IO ()

-- |c_THByteTensor_set3d : tensor x0 x1 x2 value -> void
foreign import ccall unsafe "THTensor.h THByteTensor_set3d"
  c_THByteTensor_set3d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CChar -> IO ()

-- |c_THByteTensor_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall unsafe "THTensor.h THByteTensor_set4d"
  c_THByteTensor_set4d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CLong -> CChar -> IO ()

-- |c_THByteTensor_get1d : tensor x0 -> real
foreign import ccall unsafe "THTensor.h THByteTensor_get1d"
  c_THByteTensor_get1d :: (Ptr CTHByteTensor) -> CLong -> CChar

-- |c_THByteTensor_get2d : tensor x0 x1 -> real
foreign import ccall unsafe "THTensor.h THByteTensor_get2d"
  c_THByteTensor_get2d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CChar

-- |c_THByteTensor_get3d : tensor x0 x1 x2 -> real
foreign import ccall unsafe "THTensor.h THByteTensor_get3d"
  c_THByteTensor_get3d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CChar

-- |c_THByteTensor_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall unsafe "THTensor.h THByteTensor_get4d"
  c_THByteTensor_get4d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CLong -> CChar

-- |c_THByteTensor_desc : tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h THByteTensor_desc"
  c_THByteTensor_desc :: (Ptr CTHByteTensor) -> CTHDescBuff

-- |c_THByteTensor_sizeDesc : tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h THByteTensor_sizeDesc"
  c_THByteTensor_sizeDesc :: (Ptr CTHByteTensor) -> CTHDescBuff

-- |p_THByteTensor_storage : Pointer to function self -> THStorage *
foreign import ccall unsafe "THTensor.h &THByteTensor_storage"
  p_THByteTensor_storage :: FunPtr ((Ptr CTHByteTensor) -> IO (Ptr CTHByteStorage))

-- |p_THByteTensor_storageOffset : Pointer to function self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h &THByteTensor_storageOffset"
  p_THByteTensor_storageOffset :: FunPtr ((Ptr CTHByteTensor) -> CPtrdiff)

-- |p_THByteTensor_nDimension : Pointer to function self -> int
foreign import ccall unsafe "THTensor.h &THByteTensor_nDimension"
  p_THByteTensor_nDimension :: FunPtr ((Ptr CTHByteTensor) -> CInt)

-- |p_THByteTensor_size : Pointer to function self dim -> long
foreign import ccall unsafe "THTensor.h &THByteTensor_size"
  p_THByteTensor_size :: FunPtr ((Ptr CTHByteTensor) -> CInt -> CLong)

-- |p_THByteTensor_stride : Pointer to function self dim -> long
foreign import ccall unsafe "THTensor.h &THByteTensor_stride"
  p_THByteTensor_stride :: FunPtr ((Ptr CTHByteTensor) -> CInt -> CLong)

-- |p_THByteTensor_newSizeOf : Pointer to function self -> THLongStorage *
foreign import ccall unsafe "THTensor.h &THByteTensor_newSizeOf"
  p_THByteTensor_newSizeOf :: FunPtr ((Ptr CTHByteTensor) -> IO (Ptr CTHLongStorage))

-- |p_THByteTensor_newStrideOf : Pointer to function self -> THLongStorage *
foreign import ccall unsafe "THTensor.h &THByteTensor_newStrideOf"
  p_THByteTensor_newStrideOf :: FunPtr ((Ptr CTHByteTensor) -> IO (Ptr CTHLongStorage))

-- |p_THByteTensor_data : Pointer to function self -> real *
foreign import ccall unsafe "THTensor.h &THByteTensor_data"
  p_THByteTensor_data :: FunPtr ((Ptr CTHByteTensor) -> IO (Ptr CChar))

-- |p_THByteTensor_setFlag : Pointer to function self flag -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_setFlag"
  p_THByteTensor_setFlag :: FunPtr ((Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_clearFlag : Pointer to function self flag -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_clearFlag"
  p_THByteTensor_clearFlag :: FunPtr ((Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_new : Pointer to function  -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_new"
  p_THByteTensor_new :: FunPtr (IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newWithTensor : Pointer to function tensor -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newWithTensor"
  p_THByteTensor_newWithTensor :: FunPtr ((Ptr CTHByteTensor) -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newWithStorage : Pointer to function storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newWithStorage"
  p_THByteTensor_newWithStorage :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newWithStorage1d : Pointer to function storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newWithStorage1d"
  p_THByteTensor_newWithStorage1d :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newWithStorage2d : Pointer to function storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newWithStorage2d"
  p_THByteTensor_newWithStorage2d :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newWithStorage3d : Pointer to function storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newWithStorage3d"
  p_THByteTensor_newWithStorage3d :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newWithStorage4d : Pointer to function storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newWithStorage4d"
  p_THByteTensor_newWithStorage4d :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newWithSize : Pointer to function size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newWithSize"
  p_THByteTensor_newWithSize :: FunPtr (Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newWithSize1d : Pointer to function size0_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newWithSize1d"
  p_THByteTensor_newWithSize1d :: FunPtr (CLong -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newWithSize2d : Pointer to function size0_ size1_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newWithSize2d"
  p_THByteTensor_newWithSize2d :: FunPtr (CLong -> CLong -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newWithSize3d : Pointer to function size0_ size1_ size2_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newWithSize3d"
  p_THByteTensor_newWithSize3d :: FunPtr (CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newWithSize4d : Pointer to function size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newWithSize4d"
  p_THByteTensor_newWithSize4d :: FunPtr (CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newClone : Pointer to function self -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newClone"
  p_THByteTensor_newClone :: FunPtr ((Ptr CTHByteTensor) -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newContiguous : Pointer to function tensor -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newContiguous"
  p_THByteTensor_newContiguous :: FunPtr ((Ptr CTHByteTensor) -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newSelect : Pointer to function tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newSelect"
  p_THByteTensor_newSelect :: FunPtr ((Ptr CTHByteTensor) -> CInt -> CLong -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newNarrow : Pointer to function tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newNarrow"
  p_THByteTensor_newNarrow :: FunPtr ((Ptr CTHByteTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newTranspose : Pointer to function tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newTranspose"
  p_THByteTensor_newTranspose :: FunPtr ((Ptr CTHByteTensor) -> CInt -> CInt -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newUnfold : Pointer to function tensor dimension_ size_ step_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newUnfold"
  p_THByteTensor_newUnfold :: FunPtr ((Ptr CTHByteTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newView : Pointer to function tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newView"
  p_THByteTensor_newView :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_newExpand : Pointer to function tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h &THByteTensor_newExpand"
  p_THByteTensor_newExpand :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor))

-- |p_THByteTensor_expand : Pointer to function r tensor size -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_expand"
  p_THByteTensor_expand :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THByteTensor_expandNd : Pointer to function rets ops count -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_expandNd"
  p_THByteTensor_expandNd :: FunPtr (Ptr (Ptr CTHByteTensor) -> Ptr (Ptr CTHByteTensor) -> CInt -> IO ())

-- |p_THByteTensor_resize : Pointer to function tensor size stride -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_resize"
  p_THByteTensor_resize :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THByteTensor_resizeAs : Pointer to function tensor src -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_resizeAs"
  p_THByteTensor_resizeAs :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_resizeNd : Pointer to function tensor nDimension size stride -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_resizeNd"
  p_THByteTensor_resizeNd :: FunPtr ((Ptr CTHByteTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ())

-- |p_THByteTensor_resize1d : Pointer to function tensor size0_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_resize1d"
  p_THByteTensor_resize1d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> IO ())

-- |p_THByteTensor_resize2d : Pointer to function tensor size0_ size1_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_resize2d"
  p_THByteTensor_resize2d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> IO ())

-- |p_THByteTensor_resize3d : Pointer to function tensor size0_ size1_ size2_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_resize3d"
  p_THByteTensor_resize3d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THByteTensor_resize4d : Pointer to function tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_resize4d"
  p_THByteTensor_resize4d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THByteTensor_resize5d : Pointer to function tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_resize5d"
  p_THByteTensor_resize5d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THByteTensor_set : Pointer to function self src -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_set"
  p_THByteTensor_set :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_setStorage : Pointer to function self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_setStorage"
  p_THByteTensor_setStorage :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHByteStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THByteTensor_setStorageNd : Pointer to function self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_setStorageNd"
  p_THByteTensor_setStorageNd :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHByteStorage -> CPtrdiff -> CInt -> Ptr CLong -> Ptr CLong -> IO ())

-- |p_THByteTensor_setStorage1d : Pointer to function self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_setStorage1d"
  p_THByteTensor_setStorage1d :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> IO ())

-- |p_THByteTensor_setStorage2d : Pointer to function self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_setStorage2d"
  p_THByteTensor_setStorage2d :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THByteTensor_setStorage3d : Pointer to function self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_setStorage3d"
  p_THByteTensor_setStorage3d :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THByteTensor_setStorage4d : Pointer to function self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_setStorage4d"
  p_THByteTensor_setStorage4d :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHByteStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THByteTensor_narrow : Pointer to function self src dimension_ firstIndex_ size_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_narrow"
  p_THByteTensor_narrow :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CLong -> CLong -> IO ())

-- |p_THByteTensor_select : Pointer to function self src dimension_ sliceIndex_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_select"
  p_THByteTensor_select :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CLong -> IO ())

-- |p_THByteTensor_transpose : Pointer to function self src dimension1_ dimension2_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_transpose"
  p_THByteTensor_transpose :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ())

-- |p_THByteTensor_unfold : Pointer to function self src dimension_ size_ step_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_unfold"
  p_THByteTensor_unfold :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CLong -> CLong -> IO ())

-- |p_THByteTensor_squeeze : Pointer to function self src -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_squeeze"
  p_THByteTensor_squeeze :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_squeeze1d : Pointer to function self src dimension_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_squeeze1d"
  p_THByteTensor_squeeze1d :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ())

-- |p_THByteTensor_unsqueeze1d : Pointer to function self src dimension_ -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_unsqueeze1d"
  p_THByteTensor_unsqueeze1d :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ())

-- |p_THByteTensor_isContiguous : Pointer to function self -> int
foreign import ccall unsafe "THTensor.h &THByteTensor_isContiguous"
  p_THByteTensor_isContiguous :: FunPtr ((Ptr CTHByteTensor) -> CInt)

-- |p_THByteTensor_isSameSizeAs : Pointer to function self src -> int
foreign import ccall unsafe "THTensor.h &THByteTensor_isSameSizeAs"
  p_THByteTensor_isSameSizeAs :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt)

-- |p_THByteTensor_isSetTo : Pointer to function self src -> int
foreign import ccall unsafe "THTensor.h &THByteTensor_isSetTo"
  p_THByteTensor_isSetTo :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt)

-- |p_THByteTensor_isSize : Pointer to function self dims -> int
foreign import ccall unsafe "THTensor.h &THByteTensor_isSize"
  p_THByteTensor_isSize :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongStorage -> CInt)

-- |p_THByteTensor_nElement : Pointer to function self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h &THByteTensor_nElement"
  p_THByteTensor_nElement :: FunPtr ((Ptr CTHByteTensor) -> CPtrdiff)

-- |p_THByteTensor_retain : Pointer to function self -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_retain"
  p_THByteTensor_retain :: FunPtr ((Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_free : Pointer to function self -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_free"
  p_THByteTensor_free :: FunPtr ((Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_freeCopyTo : Pointer to function self dst -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_freeCopyTo"
  p_THByteTensor_freeCopyTo :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_set1d : Pointer to function tensor x0 value -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_set1d"
  p_THByteTensor_set1d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CChar -> IO ())

-- |p_THByteTensor_set2d : Pointer to function tensor x0 x1 value -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_set2d"
  p_THByteTensor_set2d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> CChar -> IO ())

-- |p_THByteTensor_set3d : Pointer to function tensor x0 x1 x2 value -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_set3d"
  p_THByteTensor_set3d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CChar -> IO ())

-- |p_THByteTensor_set4d : Pointer to function tensor x0 x1 x2 x3 value -> void
foreign import ccall unsafe "THTensor.h &THByteTensor_set4d"
  p_THByteTensor_set4d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CLong -> CChar -> IO ())

-- |p_THByteTensor_get1d : Pointer to function tensor x0 -> real
foreign import ccall unsafe "THTensor.h &THByteTensor_get1d"
  p_THByteTensor_get1d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CChar)

-- |p_THByteTensor_get2d : Pointer to function tensor x0 x1 -> real
foreign import ccall unsafe "THTensor.h &THByteTensor_get2d"
  p_THByteTensor_get2d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> CChar)

-- |p_THByteTensor_get3d : Pointer to function tensor x0 x1 x2 -> real
foreign import ccall unsafe "THTensor.h &THByteTensor_get3d"
  p_THByteTensor_get3d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CChar)

-- |p_THByteTensor_get4d : Pointer to function tensor x0 x1 x2 x3 -> real
foreign import ccall unsafe "THTensor.h &THByteTensor_get4d"
  p_THByteTensor_get4d :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CLong -> CChar)

-- |p_THByteTensor_desc : Pointer to function tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h &THByteTensor_desc"
  p_THByteTensor_desc :: FunPtr ((Ptr CTHByteTensor) -> CTHDescBuff)

-- |p_THByteTensor_sizeDesc : Pointer to function tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h &THByteTensor_sizeDesc"
  p_THByteTensor_sizeDesc :: FunPtr ((Ptr CTHByteTensor) -> CTHDescBuff)