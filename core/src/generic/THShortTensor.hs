{-# LANGUAGE ForeignFunctionInterface #-}

module THShortTensor (
    c_THShortTensor_storage,
    c_THShortTensor_storageOffset,
    c_THShortTensor_nDimension,
    c_THShortTensor_size,
    c_THShortTensor_stride,
    c_THShortTensor_newSizeOf,
    c_THShortTensor_newStrideOf,
    c_THShortTensor_data,
    c_THShortTensor_setFlag,
    c_THShortTensor_clearFlag,
    c_THShortTensor_new,
    c_THShortTensor_newWithTensor,
    c_THShortTensor_newWithStorage,
    c_THShortTensor_newWithStorage1d,
    c_THShortTensor_newWithStorage2d,
    c_THShortTensor_newWithStorage3d,
    c_THShortTensor_newWithStorage4d,
    c_THShortTensor_newWithSize,
    c_THShortTensor_newWithSize1d,
    c_THShortTensor_newWithSize2d,
    c_THShortTensor_newWithSize3d,
    c_THShortTensor_newWithSize4d,
    c_THShortTensor_newClone,
    c_THShortTensor_newContiguous,
    c_THShortTensor_newSelect,
    c_THShortTensor_newNarrow,
    c_THShortTensor_newTranspose,
    c_THShortTensor_newUnfold,
    c_THShortTensor_newView,
    c_THShortTensor_newExpand,
    c_THShortTensor_expand,
    c_THShortTensor_expandNd,
    c_THShortTensor_resize,
    c_THShortTensor_resizeAs,
    c_THShortTensor_resizeNd,
    c_THShortTensor_resize1d,
    c_THShortTensor_resize2d,
    c_THShortTensor_resize3d,
    c_THShortTensor_resize4d,
    c_THShortTensor_resize5d,
    c_THShortTensor_set,
    c_THShortTensor_setStorage,
    c_THShortTensor_setStorageNd,
    c_THShortTensor_setStorage1d,
    c_THShortTensor_setStorage2d,
    c_THShortTensor_setStorage3d,
    c_THShortTensor_setStorage4d,
    c_THShortTensor_narrow,
    c_THShortTensor_select,
    c_THShortTensor_transpose,
    c_THShortTensor_unfold,
    c_THShortTensor_squeeze,
    c_THShortTensor_squeeze1d,
    c_THShortTensor_unsqueeze1d,
    c_THShortTensor_isContiguous,
    c_THShortTensor_isSameSizeAs,
    c_THShortTensor_isSetTo,
    c_THShortTensor_isSize,
    c_THShortTensor_nElement,
    c_THShortTensor_retain,
    c_THShortTensor_free,
    c_THShortTensor_freeCopyTo,
    c_THShortTensor_set1d,
    c_THShortTensor_set2d,
    c_THShortTensor_set3d,
    c_THShortTensor_set4d,
    c_THShortTensor_get1d,
    c_THShortTensor_get2d,
    c_THShortTensor_get3d,
    c_THShortTensor_get4d,
    c_THShortTensor_desc,
    c_THShortTensor_sizeDesc,
    p_THShortTensor_storage,
    p_THShortTensor_storageOffset,
    p_THShortTensor_nDimension,
    p_THShortTensor_size,
    p_THShortTensor_stride,
    p_THShortTensor_newSizeOf,
    p_THShortTensor_newStrideOf,
    p_THShortTensor_data,
    p_THShortTensor_setFlag,
    p_THShortTensor_clearFlag,
    p_THShortTensor_new,
    p_THShortTensor_newWithTensor,
    p_THShortTensor_newWithStorage,
    p_THShortTensor_newWithStorage1d,
    p_THShortTensor_newWithStorage2d,
    p_THShortTensor_newWithStorage3d,
    p_THShortTensor_newWithStorage4d,
    p_THShortTensor_newWithSize,
    p_THShortTensor_newWithSize1d,
    p_THShortTensor_newWithSize2d,
    p_THShortTensor_newWithSize3d,
    p_THShortTensor_newWithSize4d,
    p_THShortTensor_newClone,
    p_THShortTensor_newContiguous,
    p_THShortTensor_newSelect,
    p_THShortTensor_newNarrow,
    p_THShortTensor_newTranspose,
    p_THShortTensor_newUnfold,
    p_THShortTensor_newView,
    p_THShortTensor_newExpand,
    p_THShortTensor_expand,
    p_THShortTensor_expandNd,
    p_THShortTensor_resize,
    p_THShortTensor_resizeAs,
    p_THShortTensor_resizeNd,
    p_THShortTensor_resize1d,
    p_THShortTensor_resize2d,
    p_THShortTensor_resize3d,
    p_THShortTensor_resize4d,
    p_THShortTensor_resize5d,
    p_THShortTensor_set,
    p_THShortTensor_setStorage,
    p_THShortTensor_setStorageNd,
    p_THShortTensor_setStorage1d,
    p_THShortTensor_setStorage2d,
    p_THShortTensor_setStorage3d,
    p_THShortTensor_setStorage4d,
    p_THShortTensor_narrow,
    p_THShortTensor_select,
    p_THShortTensor_transpose,
    p_THShortTensor_unfold,
    p_THShortTensor_squeeze,
    p_THShortTensor_squeeze1d,
    p_THShortTensor_unsqueeze1d,
    p_THShortTensor_isContiguous,
    p_THShortTensor_isSameSizeAs,
    p_THShortTensor_isSetTo,
    p_THShortTensor_isSize,
    p_THShortTensor_nElement,
    p_THShortTensor_retain,
    p_THShortTensor_free,
    p_THShortTensor_freeCopyTo,
    p_THShortTensor_set1d,
    p_THShortTensor_set2d,
    p_THShortTensor_set3d,
    p_THShortTensor_set4d,
    p_THShortTensor_get1d,
    p_THShortTensor_get2d,
    p_THShortTensor_get3d,
    p_THShortTensor_get4d,
    p_THShortTensor_desc,
    p_THShortTensor_sizeDesc) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortTensor_storage : self -> THStorage *
foreign import ccall unsafe "THTensor.h THShortTensor_storage"
  c_THShortTensor_storage :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortStorage)

-- |c_THShortTensor_storageOffset : self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h THShortTensor_storageOffset"
  c_THShortTensor_storageOffset :: (Ptr CTHShortTensor) -> CPtrdiff

-- |c_THShortTensor_nDimension : self -> int
foreign import ccall unsafe "THTensor.h THShortTensor_nDimension"
  c_THShortTensor_nDimension :: (Ptr CTHShortTensor) -> CInt

-- |c_THShortTensor_size : self dim -> long
foreign import ccall unsafe "THTensor.h THShortTensor_size"
  c_THShortTensor_size :: (Ptr CTHShortTensor) -> CInt -> CLong

-- |c_THShortTensor_stride : self dim -> long
foreign import ccall unsafe "THTensor.h THShortTensor_stride"
  c_THShortTensor_stride :: (Ptr CTHShortTensor) -> CInt -> CLong

-- |c_THShortTensor_newSizeOf : self -> THLongStorage *
foreign import ccall unsafe "THTensor.h THShortTensor_newSizeOf"
  c_THShortTensor_newSizeOf :: (Ptr CTHShortTensor) -> IO (Ptr CTHLongStorage)

-- |c_THShortTensor_newStrideOf : self -> THLongStorage *
foreign import ccall unsafe "THTensor.h THShortTensor_newStrideOf"
  c_THShortTensor_newStrideOf :: (Ptr CTHShortTensor) -> IO (Ptr CTHLongStorage)

-- |c_THShortTensor_data : self -> real *
foreign import ccall unsafe "THTensor.h THShortTensor_data"
  c_THShortTensor_data :: (Ptr CTHShortTensor) -> IO (Ptr CShort)

-- |c_THShortTensor_setFlag : self flag -> void
foreign import ccall unsafe "THTensor.h THShortTensor_setFlag"
  c_THShortTensor_setFlag :: (Ptr CTHShortTensor) -> CChar -> IO ()

-- |c_THShortTensor_clearFlag : self flag -> void
foreign import ccall unsafe "THTensor.h THShortTensor_clearFlag"
  c_THShortTensor_clearFlag :: (Ptr CTHShortTensor) -> CChar -> IO ()

-- |c_THShortTensor_new :  -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_new"
  c_THShortTensor_new :: IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithTensor : tensor -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newWithTensor"
  c_THShortTensor_newWithTensor :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithStorage : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newWithStorage"
  c_THShortTensor_newWithStorage :: Ptr CTHShortStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newWithStorage1d"
  c_THShortTensor_newWithStorage1d :: Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newWithStorage2d"
  c_THShortTensor_newWithStorage2d :: Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newWithStorage3d"
  c_THShortTensor_newWithStorage3d :: Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newWithStorage4d"
  c_THShortTensor_newWithStorage4d :: Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithSize : size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newWithSize"
  c_THShortTensor_newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithSize1d : size0_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newWithSize1d"
  c_THShortTensor_newWithSize1d :: CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newWithSize2d"
  c_THShortTensor_newWithSize2d :: CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newWithSize3d"
  c_THShortTensor_newWithSize3d :: CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newWithSize4d"
  c_THShortTensor_newWithSize4d :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newClone : self -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newClone"
  c_THShortTensor_newClone :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newContiguous : tensor -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newContiguous"
  c_THShortTensor_newContiguous :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newSelect"
  c_THShortTensor_newSelect :: (Ptr CTHShortTensor) -> CInt -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newNarrow"
  c_THShortTensor_newNarrow :: (Ptr CTHShortTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newTranspose"
  c_THShortTensor_newTranspose :: (Ptr CTHShortTensor) -> CInt -> CInt -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newUnfold"
  c_THShortTensor_newUnfold :: (Ptr CTHShortTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newView : tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newView"
  c_THShortTensor_newView :: (Ptr CTHShortTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newExpand : tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h THShortTensor_newExpand"
  c_THShortTensor_newExpand :: (Ptr CTHShortTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_expand : r tensor size -> void
foreign import ccall unsafe "THTensor.h THShortTensor_expand"
  c_THShortTensor_expand :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THShortTensor_expandNd : rets ops count -> void
foreign import ccall unsafe "THTensor.h THShortTensor_expandNd"
  c_THShortTensor_expandNd :: Ptr (Ptr CTHShortTensor) -> Ptr (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensor_resize : tensor size stride -> void
foreign import ccall unsafe "THTensor.h THShortTensor_resize"
  c_THShortTensor_resize :: (Ptr CTHShortTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THShortTensor_resizeAs : tensor src -> void
foreign import ccall unsafe "THTensor.h THShortTensor_resizeAs"
  c_THShortTensor_resizeAs :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_resizeNd : tensor nDimension size stride -> void
foreign import ccall unsafe "THTensor.h THShortTensor_resizeNd"
  c_THShortTensor_resizeNd :: (Ptr CTHShortTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THShortTensor_resize1d : tensor size0_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_resize1d"
  c_THShortTensor_resize1d :: (Ptr CTHShortTensor) -> CLong -> IO ()

-- |c_THShortTensor_resize2d : tensor size0_ size1_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_resize2d"
  c_THShortTensor_resize2d :: (Ptr CTHShortTensor) -> CLong -> CLong -> IO ()

-- |c_THShortTensor_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_resize3d"
  c_THShortTensor_resize3d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_resize4d"
  c_THShortTensor_resize4d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_resize5d"
  c_THShortTensor_resize5d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_set : self src -> void
foreign import ccall unsafe "THTensor.h THShortTensor_set"
  c_THShortTensor_set :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_setStorage"
  c_THShortTensor_setStorage :: (Ptr CTHShortTensor) -> Ptr CTHShortStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THShortTensor_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall unsafe "THTensor.h THShortTensor_setStorageNd"
  c_THShortTensor_setStorageNd :: (Ptr CTHShortTensor) -> Ptr CTHShortStorage -> CPtrdiff -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THShortTensor_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_setStorage1d"
  c_THShortTensor_setStorage1d :: (Ptr CTHShortTensor) -> Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> IO ()

-- |c_THShortTensor_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_setStorage2d"
  c_THShortTensor_setStorage2d :: (Ptr CTHShortTensor) -> Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_setStorage3d"
  c_THShortTensor_setStorage3d :: (Ptr CTHShortTensor) -> Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_setStorage4d"
  c_THShortTensor_setStorage4d :: (Ptr CTHShortTensor) -> Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_narrow"
  c_THShortTensor_narrow :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THShortTensor_select : self src dimension_ sliceIndex_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_select"
  c_THShortTensor_select :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CLong -> IO ()

-- |c_THShortTensor_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_transpose"
  c_THShortTensor_transpose :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensor_unfold : self src dimension_ size_ step_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_unfold"
  c_THShortTensor_unfold :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THShortTensor_squeeze : self src -> void
foreign import ccall unsafe "THTensor.h THShortTensor_squeeze"
  c_THShortTensor_squeeze :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_squeeze1d : self src dimension_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_squeeze1d"
  c_THShortTensor_squeeze1d :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensor_unsqueeze1d : self src dimension_ -> void
foreign import ccall unsafe "THTensor.h THShortTensor_unsqueeze1d"
  c_THShortTensor_unsqueeze1d :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensor_isContiguous : self -> int
foreign import ccall unsafe "THTensor.h THShortTensor_isContiguous"
  c_THShortTensor_isContiguous :: (Ptr CTHShortTensor) -> CInt

-- |c_THShortTensor_isSameSizeAs : self src -> int
foreign import ccall unsafe "THTensor.h THShortTensor_isSameSizeAs"
  c_THShortTensor_isSameSizeAs :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt

-- |c_THShortTensor_isSetTo : self src -> int
foreign import ccall unsafe "THTensor.h THShortTensor_isSetTo"
  c_THShortTensor_isSetTo :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt

-- |c_THShortTensor_isSize : self dims -> int
foreign import ccall unsafe "THTensor.h THShortTensor_isSize"
  c_THShortTensor_isSize :: (Ptr CTHShortTensor) -> Ptr CTHLongStorage -> CInt

-- |c_THShortTensor_nElement : self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h THShortTensor_nElement"
  c_THShortTensor_nElement :: (Ptr CTHShortTensor) -> CPtrdiff

-- |c_THShortTensor_retain : self -> void
foreign import ccall unsafe "THTensor.h THShortTensor_retain"
  c_THShortTensor_retain :: (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_free : self -> void
foreign import ccall unsafe "THTensor.h THShortTensor_free"
  c_THShortTensor_free :: (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_freeCopyTo : self dst -> void
foreign import ccall unsafe "THTensor.h THShortTensor_freeCopyTo"
  c_THShortTensor_freeCopyTo :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_set1d : tensor x0 value -> void
foreign import ccall unsafe "THTensor.h THShortTensor_set1d"
  c_THShortTensor_set1d :: (Ptr CTHShortTensor) -> CLong -> CShort -> IO ()

-- |c_THShortTensor_set2d : tensor x0 x1 value -> void
foreign import ccall unsafe "THTensor.h THShortTensor_set2d"
  c_THShortTensor_set2d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CShort -> IO ()

-- |c_THShortTensor_set3d : tensor x0 x1 x2 value -> void
foreign import ccall unsafe "THTensor.h THShortTensor_set3d"
  c_THShortTensor_set3d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CShort -> IO ()

-- |c_THShortTensor_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall unsafe "THTensor.h THShortTensor_set4d"
  c_THShortTensor_set4d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CLong -> CShort -> IO ()

-- |c_THShortTensor_get1d : tensor x0 -> real
foreign import ccall unsafe "THTensor.h THShortTensor_get1d"
  c_THShortTensor_get1d :: (Ptr CTHShortTensor) -> CLong -> CShort

-- |c_THShortTensor_get2d : tensor x0 x1 -> real
foreign import ccall unsafe "THTensor.h THShortTensor_get2d"
  c_THShortTensor_get2d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CShort

-- |c_THShortTensor_get3d : tensor x0 x1 x2 -> real
foreign import ccall unsafe "THTensor.h THShortTensor_get3d"
  c_THShortTensor_get3d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CShort

-- |c_THShortTensor_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall unsafe "THTensor.h THShortTensor_get4d"
  c_THShortTensor_get4d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CLong -> CShort

-- |c_THShortTensor_desc : tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h THShortTensor_desc"
  c_THShortTensor_desc :: (Ptr CTHShortTensor) -> CTHDescBuff

-- |c_THShortTensor_sizeDesc : tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h THShortTensor_sizeDesc"
  c_THShortTensor_sizeDesc :: (Ptr CTHShortTensor) -> CTHDescBuff

-- |p_THShortTensor_storage : Pointer to function self -> THStorage *
foreign import ccall unsafe "THTensor.h &THShortTensor_storage"
  p_THShortTensor_storage :: FunPtr ((Ptr CTHShortTensor) -> IO (Ptr CTHShortStorage))

-- |p_THShortTensor_storageOffset : Pointer to function self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h &THShortTensor_storageOffset"
  p_THShortTensor_storageOffset :: FunPtr ((Ptr CTHShortTensor) -> CPtrdiff)

-- |p_THShortTensor_nDimension : Pointer to function self -> int
foreign import ccall unsafe "THTensor.h &THShortTensor_nDimension"
  p_THShortTensor_nDimension :: FunPtr ((Ptr CTHShortTensor) -> CInt)

-- |p_THShortTensor_size : Pointer to function self dim -> long
foreign import ccall unsafe "THTensor.h &THShortTensor_size"
  p_THShortTensor_size :: FunPtr ((Ptr CTHShortTensor) -> CInt -> CLong)

-- |p_THShortTensor_stride : Pointer to function self dim -> long
foreign import ccall unsafe "THTensor.h &THShortTensor_stride"
  p_THShortTensor_stride :: FunPtr ((Ptr CTHShortTensor) -> CInt -> CLong)

-- |p_THShortTensor_newSizeOf : Pointer to function self -> THLongStorage *
foreign import ccall unsafe "THTensor.h &THShortTensor_newSizeOf"
  p_THShortTensor_newSizeOf :: FunPtr ((Ptr CTHShortTensor) -> IO (Ptr CTHLongStorage))

-- |p_THShortTensor_newStrideOf : Pointer to function self -> THLongStorage *
foreign import ccall unsafe "THTensor.h &THShortTensor_newStrideOf"
  p_THShortTensor_newStrideOf :: FunPtr ((Ptr CTHShortTensor) -> IO (Ptr CTHLongStorage))

-- |p_THShortTensor_data : Pointer to function self -> real *
foreign import ccall unsafe "THTensor.h &THShortTensor_data"
  p_THShortTensor_data :: FunPtr ((Ptr CTHShortTensor) -> IO (Ptr CShort))

-- |p_THShortTensor_setFlag : Pointer to function self flag -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_setFlag"
  p_THShortTensor_setFlag :: FunPtr ((Ptr CTHShortTensor) -> CChar -> IO ())

-- |p_THShortTensor_clearFlag : Pointer to function self flag -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_clearFlag"
  p_THShortTensor_clearFlag :: FunPtr ((Ptr CTHShortTensor) -> CChar -> IO ())

-- |p_THShortTensor_new : Pointer to function  -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_new"
  p_THShortTensor_new :: FunPtr (IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newWithTensor : Pointer to function tensor -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newWithTensor"
  p_THShortTensor_newWithTensor :: FunPtr ((Ptr CTHShortTensor) -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newWithStorage : Pointer to function storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newWithStorage"
  p_THShortTensor_newWithStorage :: FunPtr (Ptr CTHShortStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newWithStorage1d : Pointer to function storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newWithStorage1d"
  p_THShortTensor_newWithStorage1d :: FunPtr (Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newWithStorage2d : Pointer to function storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newWithStorage2d"
  p_THShortTensor_newWithStorage2d :: FunPtr (Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newWithStorage3d : Pointer to function storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newWithStorage3d"
  p_THShortTensor_newWithStorage3d :: FunPtr (Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newWithStorage4d : Pointer to function storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newWithStorage4d"
  p_THShortTensor_newWithStorage4d :: FunPtr (Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newWithSize : Pointer to function size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newWithSize"
  p_THShortTensor_newWithSize :: FunPtr (Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newWithSize1d : Pointer to function size0_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newWithSize1d"
  p_THShortTensor_newWithSize1d :: FunPtr (CLong -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newWithSize2d : Pointer to function size0_ size1_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newWithSize2d"
  p_THShortTensor_newWithSize2d :: FunPtr (CLong -> CLong -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newWithSize3d : Pointer to function size0_ size1_ size2_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newWithSize3d"
  p_THShortTensor_newWithSize3d :: FunPtr (CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newWithSize4d : Pointer to function size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newWithSize4d"
  p_THShortTensor_newWithSize4d :: FunPtr (CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newClone : Pointer to function self -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newClone"
  p_THShortTensor_newClone :: FunPtr ((Ptr CTHShortTensor) -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newContiguous : Pointer to function tensor -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newContiguous"
  p_THShortTensor_newContiguous :: FunPtr ((Ptr CTHShortTensor) -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newSelect : Pointer to function tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newSelect"
  p_THShortTensor_newSelect :: FunPtr ((Ptr CTHShortTensor) -> CInt -> CLong -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newNarrow : Pointer to function tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newNarrow"
  p_THShortTensor_newNarrow :: FunPtr ((Ptr CTHShortTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newTranspose : Pointer to function tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newTranspose"
  p_THShortTensor_newTranspose :: FunPtr ((Ptr CTHShortTensor) -> CInt -> CInt -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newUnfold : Pointer to function tensor dimension_ size_ step_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newUnfold"
  p_THShortTensor_newUnfold :: FunPtr ((Ptr CTHShortTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newView : Pointer to function tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newView"
  p_THShortTensor_newView :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_newExpand : Pointer to function tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h &THShortTensor_newExpand"
  p_THShortTensor_newExpand :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHShortTensor))

-- |p_THShortTensor_expand : Pointer to function r tensor size -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_expand"
  p_THShortTensor_expand :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THShortTensor_expandNd : Pointer to function rets ops count -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_expandNd"
  p_THShortTensor_expandNd :: FunPtr (Ptr (Ptr CTHShortTensor) -> Ptr (Ptr CTHShortTensor) -> CInt -> IO ())

-- |p_THShortTensor_resize : Pointer to function tensor size stride -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_resize"
  p_THShortTensor_resize :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THShortTensor_resizeAs : Pointer to function tensor src -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_resizeAs"
  p_THShortTensor_resizeAs :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ())

-- |p_THShortTensor_resizeNd : Pointer to function tensor nDimension size stride -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_resizeNd"
  p_THShortTensor_resizeNd :: FunPtr ((Ptr CTHShortTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ())

-- |p_THShortTensor_resize1d : Pointer to function tensor size0_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_resize1d"
  p_THShortTensor_resize1d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> IO ())

-- |p_THShortTensor_resize2d : Pointer to function tensor size0_ size1_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_resize2d"
  p_THShortTensor_resize2d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> CLong -> IO ())

-- |p_THShortTensor_resize3d : Pointer to function tensor size0_ size1_ size2_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_resize3d"
  p_THShortTensor_resize3d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THShortTensor_resize4d : Pointer to function tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_resize4d"
  p_THShortTensor_resize4d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THShortTensor_resize5d : Pointer to function tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_resize5d"
  p_THShortTensor_resize5d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THShortTensor_set : Pointer to function self src -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_set"
  p_THShortTensor_set :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ())

-- |p_THShortTensor_setStorage : Pointer to function self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_setStorage"
  p_THShortTensor_setStorage :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHShortStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THShortTensor_setStorageNd : Pointer to function self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_setStorageNd"
  p_THShortTensor_setStorageNd :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHShortStorage -> CPtrdiff -> CInt -> Ptr CLong -> Ptr CLong -> IO ())

-- |p_THShortTensor_setStorage1d : Pointer to function self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_setStorage1d"
  p_THShortTensor_setStorage1d :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> IO ())

-- |p_THShortTensor_setStorage2d : Pointer to function self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_setStorage2d"
  p_THShortTensor_setStorage2d :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THShortTensor_setStorage3d : Pointer to function self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_setStorage3d"
  p_THShortTensor_setStorage3d :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THShortTensor_setStorage4d : Pointer to function self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_setStorage4d"
  p_THShortTensor_setStorage4d :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHShortStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THShortTensor_narrow : Pointer to function self src dimension_ firstIndex_ size_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_narrow"
  p_THShortTensor_narrow :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CLong -> CLong -> IO ())

-- |p_THShortTensor_select : Pointer to function self src dimension_ sliceIndex_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_select"
  p_THShortTensor_select :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CLong -> IO ())

-- |p_THShortTensor_transpose : Pointer to function self src dimension1_ dimension2_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_transpose"
  p_THShortTensor_transpose :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ())

-- |p_THShortTensor_unfold : Pointer to function self src dimension_ size_ step_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_unfold"
  p_THShortTensor_unfold :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CLong -> CLong -> IO ())

-- |p_THShortTensor_squeeze : Pointer to function self src -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_squeeze"
  p_THShortTensor_squeeze :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ())

-- |p_THShortTensor_squeeze1d : Pointer to function self src dimension_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_squeeze1d"
  p_THShortTensor_squeeze1d :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ())

-- |p_THShortTensor_unsqueeze1d : Pointer to function self src dimension_ -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_unsqueeze1d"
  p_THShortTensor_unsqueeze1d :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ())

-- |p_THShortTensor_isContiguous : Pointer to function self -> int
foreign import ccall unsafe "THTensor.h &THShortTensor_isContiguous"
  p_THShortTensor_isContiguous :: FunPtr ((Ptr CTHShortTensor) -> CInt)

-- |p_THShortTensor_isSameSizeAs : Pointer to function self src -> int
foreign import ccall unsafe "THTensor.h &THShortTensor_isSameSizeAs"
  p_THShortTensor_isSameSizeAs :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt)

-- |p_THShortTensor_isSetTo : Pointer to function self src -> int
foreign import ccall unsafe "THTensor.h &THShortTensor_isSetTo"
  p_THShortTensor_isSetTo :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt)

-- |p_THShortTensor_isSize : Pointer to function self dims -> int
foreign import ccall unsafe "THTensor.h &THShortTensor_isSize"
  p_THShortTensor_isSize :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHLongStorage -> CInt)

-- |p_THShortTensor_nElement : Pointer to function self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h &THShortTensor_nElement"
  p_THShortTensor_nElement :: FunPtr ((Ptr CTHShortTensor) -> CPtrdiff)

-- |p_THShortTensor_retain : Pointer to function self -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_retain"
  p_THShortTensor_retain :: FunPtr ((Ptr CTHShortTensor) -> IO ())

-- |p_THShortTensor_free : Pointer to function self -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_free"
  p_THShortTensor_free :: FunPtr ((Ptr CTHShortTensor) -> IO ())

-- |p_THShortTensor_freeCopyTo : Pointer to function self dst -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_freeCopyTo"
  p_THShortTensor_freeCopyTo :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ())

-- |p_THShortTensor_set1d : Pointer to function tensor x0 value -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_set1d"
  p_THShortTensor_set1d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> CShort -> IO ())

-- |p_THShortTensor_set2d : Pointer to function tensor x0 x1 value -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_set2d"
  p_THShortTensor_set2d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> CLong -> CShort -> IO ())

-- |p_THShortTensor_set3d : Pointer to function tensor x0 x1 x2 value -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_set3d"
  p_THShortTensor_set3d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CShort -> IO ())

-- |p_THShortTensor_set4d : Pointer to function tensor x0 x1 x2 x3 value -> void
foreign import ccall unsafe "THTensor.h &THShortTensor_set4d"
  p_THShortTensor_set4d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CLong -> CShort -> IO ())

-- |p_THShortTensor_get1d : Pointer to function tensor x0 -> real
foreign import ccall unsafe "THTensor.h &THShortTensor_get1d"
  p_THShortTensor_get1d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> CShort)

-- |p_THShortTensor_get2d : Pointer to function tensor x0 x1 -> real
foreign import ccall unsafe "THTensor.h &THShortTensor_get2d"
  p_THShortTensor_get2d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> CLong -> CShort)

-- |p_THShortTensor_get3d : Pointer to function tensor x0 x1 x2 -> real
foreign import ccall unsafe "THTensor.h &THShortTensor_get3d"
  p_THShortTensor_get3d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CShort)

-- |p_THShortTensor_get4d : Pointer to function tensor x0 x1 x2 x3 -> real
foreign import ccall unsafe "THTensor.h &THShortTensor_get4d"
  p_THShortTensor_get4d :: FunPtr ((Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CLong -> CShort)

-- |p_THShortTensor_desc : Pointer to function tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h &THShortTensor_desc"
  p_THShortTensor_desc :: FunPtr ((Ptr CTHShortTensor) -> CTHDescBuff)

-- |p_THShortTensor_sizeDesc : Pointer to function tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h &THShortTensor_sizeDesc"
  p_THShortTensor_sizeDesc :: FunPtr ((Ptr CTHShortTensor) -> CTHDescBuff)