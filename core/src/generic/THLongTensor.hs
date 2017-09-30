{-# LANGUAGE ForeignFunctionInterface #-}

module THLongTensor (
    c_THLongTensor_storage,
    c_THLongTensor_storageOffset,
    c_THLongTensor_nDimension,
    c_THLongTensor_size,
    c_THLongTensor_stride,
    c_THLongTensor_newSizeOf,
    c_THLongTensor_newStrideOf,
    c_THLongTensor_data,
    c_THLongTensor_setFlag,
    c_THLongTensor_clearFlag,
    c_THLongTensor_new,
    c_THLongTensor_newWithTensor,
    c_THLongTensor_newWithStorage,
    c_THLongTensor_newWithStorage1d,
    c_THLongTensor_newWithStorage2d,
    c_THLongTensor_newWithStorage3d,
    c_THLongTensor_newWithStorage4d,
    c_THLongTensor_newWithSize,
    c_THLongTensor_newWithSize1d,
    c_THLongTensor_newWithSize2d,
    c_THLongTensor_newWithSize3d,
    c_THLongTensor_newWithSize4d,
    c_THLongTensor_newClone,
    c_THLongTensor_newContiguous,
    c_THLongTensor_newSelect,
    c_THLongTensor_newNarrow,
    c_THLongTensor_newTranspose,
    c_THLongTensor_newUnfold,
    c_THLongTensor_newView,
    c_THLongTensor_newExpand,
    c_THLongTensor_expand,
    c_THLongTensor_expandNd,
    c_THLongTensor_resize,
    c_THLongTensor_resizeAs,
    c_THLongTensor_resizeNd,
    c_THLongTensor_resize1d,
    c_THLongTensor_resize2d,
    c_THLongTensor_resize3d,
    c_THLongTensor_resize4d,
    c_THLongTensor_resize5d,
    c_THLongTensor_set,
    c_THLongTensor_setStorage,
    c_THLongTensor_setStorageNd,
    c_THLongTensor_setStorage1d,
    c_THLongTensor_setStorage2d,
    c_THLongTensor_setStorage3d,
    c_THLongTensor_setStorage4d,
    c_THLongTensor_narrow,
    c_THLongTensor_select,
    c_THLongTensor_transpose,
    c_THLongTensor_unfold,
    c_THLongTensor_squeeze,
    c_THLongTensor_squeeze1d,
    c_THLongTensor_unsqueeze1d,
    c_THLongTensor_isContiguous,
    c_THLongTensor_isSameSizeAs,
    c_THLongTensor_isSetTo,
    c_THLongTensor_isSize,
    c_THLongTensor_nElement,
    c_THLongTensor_retain,
    c_THLongTensor_free,
    c_THLongTensor_freeCopyTo,
    c_THLongTensor_set1d,
    c_THLongTensor_set2d,
    c_THLongTensor_set3d,
    c_THLongTensor_set4d,
    c_THLongTensor_get1d,
    c_THLongTensor_get2d,
    c_THLongTensor_get3d,
    c_THLongTensor_get4d,
    c_THLongTensor_desc,
    c_THLongTensor_sizeDesc,
    p_THLongTensor_storage,
    p_THLongTensor_storageOffset,
    p_THLongTensor_nDimension,
    p_THLongTensor_size,
    p_THLongTensor_stride,
    p_THLongTensor_newSizeOf,
    p_THLongTensor_newStrideOf,
    p_THLongTensor_data,
    p_THLongTensor_setFlag,
    p_THLongTensor_clearFlag,
    p_THLongTensor_new,
    p_THLongTensor_newWithTensor,
    p_THLongTensor_newWithStorage,
    p_THLongTensor_newWithStorage1d,
    p_THLongTensor_newWithStorage2d,
    p_THLongTensor_newWithStorage3d,
    p_THLongTensor_newWithStorage4d,
    p_THLongTensor_newWithSize,
    p_THLongTensor_newWithSize1d,
    p_THLongTensor_newWithSize2d,
    p_THLongTensor_newWithSize3d,
    p_THLongTensor_newWithSize4d,
    p_THLongTensor_newClone,
    p_THLongTensor_newContiguous,
    p_THLongTensor_newSelect,
    p_THLongTensor_newNarrow,
    p_THLongTensor_newTranspose,
    p_THLongTensor_newUnfold,
    p_THLongTensor_newView,
    p_THLongTensor_newExpand,
    p_THLongTensor_expand,
    p_THLongTensor_expandNd,
    p_THLongTensor_resize,
    p_THLongTensor_resizeAs,
    p_THLongTensor_resizeNd,
    p_THLongTensor_resize1d,
    p_THLongTensor_resize2d,
    p_THLongTensor_resize3d,
    p_THLongTensor_resize4d,
    p_THLongTensor_resize5d,
    p_THLongTensor_set,
    p_THLongTensor_setStorage,
    p_THLongTensor_setStorageNd,
    p_THLongTensor_setStorage1d,
    p_THLongTensor_setStorage2d,
    p_THLongTensor_setStorage3d,
    p_THLongTensor_setStorage4d,
    p_THLongTensor_narrow,
    p_THLongTensor_select,
    p_THLongTensor_transpose,
    p_THLongTensor_unfold,
    p_THLongTensor_squeeze,
    p_THLongTensor_squeeze1d,
    p_THLongTensor_unsqueeze1d,
    p_THLongTensor_isContiguous,
    p_THLongTensor_isSameSizeAs,
    p_THLongTensor_isSetTo,
    p_THLongTensor_isSize,
    p_THLongTensor_nElement,
    p_THLongTensor_retain,
    p_THLongTensor_free,
    p_THLongTensor_freeCopyTo,
    p_THLongTensor_set1d,
    p_THLongTensor_set2d,
    p_THLongTensor_set3d,
    p_THLongTensor_set4d,
    p_THLongTensor_get1d,
    p_THLongTensor_get2d,
    p_THLongTensor_get3d,
    p_THLongTensor_get4d,
    p_THLongTensor_desc,
    p_THLongTensor_sizeDesc) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongTensor_storage : self -> THStorage *
foreign import ccall unsafe "THTensor.h THLongTensor_storage"
  c_THLongTensor_storage :: (Ptr CTHLongTensor) -> IO (Ptr CTHLongStorage)

-- |c_THLongTensor_storageOffset : self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h THLongTensor_storageOffset"
  c_THLongTensor_storageOffset :: (Ptr CTHLongTensor) -> CPtrdiff

-- |c_THLongTensor_nDimension : self -> int
foreign import ccall unsafe "THTensor.h THLongTensor_nDimension"
  c_THLongTensor_nDimension :: (Ptr CTHLongTensor) -> CInt

-- |c_THLongTensor_size : self dim -> long
foreign import ccall unsafe "THTensor.h THLongTensor_size"
  c_THLongTensor_size :: (Ptr CTHLongTensor) -> CInt -> CLong

-- |c_THLongTensor_stride : self dim -> long
foreign import ccall unsafe "THTensor.h THLongTensor_stride"
  c_THLongTensor_stride :: (Ptr CTHLongTensor) -> CInt -> CLong

-- |c_THLongTensor_newSizeOf : self -> THLongStorage *
foreign import ccall unsafe "THTensor.h THLongTensor_newSizeOf"
  c_THLongTensor_newSizeOf :: (Ptr CTHLongTensor) -> IO (Ptr CTHLongStorage)

-- |c_THLongTensor_newStrideOf : self -> THLongStorage *
foreign import ccall unsafe "THTensor.h THLongTensor_newStrideOf"
  c_THLongTensor_newStrideOf :: (Ptr CTHLongTensor) -> IO (Ptr CTHLongStorage)

-- |c_THLongTensor_data : self -> real *
foreign import ccall unsafe "THTensor.h THLongTensor_data"
  c_THLongTensor_data :: (Ptr CTHLongTensor) -> IO (Ptr CLong)

-- |c_THLongTensor_setFlag : self flag -> void
foreign import ccall unsafe "THTensor.h THLongTensor_setFlag"
  c_THLongTensor_setFlag :: (Ptr CTHLongTensor) -> CChar -> IO ()

-- |c_THLongTensor_clearFlag : self flag -> void
foreign import ccall unsafe "THTensor.h THLongTensor_clearFlag"
  c_THLongTensor_clearFlag :: (Ptr CTHLongTensor) -> CChar -> IO ()

-- |c_THLongTensor_new :  -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_new"
  c_THLongTensor_new :: IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newWithTensor : tensor -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newWithTensor"
  c_THLongTensor_newWithTensor :: (Ptr CTHLongTensor) -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newWithStorage : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newWithStorage"
  c_THLongTensor_newWithStorage :: Ptr CTHLongStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newWithStorage1d"
  c_THLongTensor_newWithStorage1d :: Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newWithStorage2d"
  c_THLongTensor_newWithStorage2d :: Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newWithStorage3d"
  c_THLongTensor_newWithStorage3d :: Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newWithStorage4d"
  c_THLongTensor_newWithStorage4d :: Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newWithSize : size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newWithSize"
  c_THLongTensor_newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newWithSize1d : size0_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newWithSize1d"
  c_THLongTensor_newWithSize1d :: CLong -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newWithSize2d"
  c_THLongTensor_newWithSize2d :: CLong -> CLong -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newWithSize3d"
  c_THLongTensor_newWithSize3d :: CLong -> CLong -> CLong -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newWithSize4d"
  c_THLongTensor_newWithSize4d :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newClone : self -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newClone"
  c_THLongTensor_newClone :: (Ptr CTHLongTensor) -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newContiguous : tensor -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newContiguous"
  c_THLongTensor_newContiguous :: (Ptr CTHLongTensor) -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newSelect"
  c_THLongTensor_newSelect :: (Ptr CTHLongTensor) -> CInt -> CLong -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newNarrow"
  c_THLongTensor_newNarrow :: (Ptr CTHLongTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newTranspose"
  c_THLongTensor_newTranspose :: (Ptr CTHLongTensor) -> CInt -> CInt -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newUnfold"
  c_THLongTensor_newUnfold :: (Ptr CTHLongTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newView : tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newView"
  c_THLongTensor_newView :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_newExpand : tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h THLongTensor_newExpand"
  c_THLongTensor_newExpand :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHLongTensor)

-- |c_THLongTensor_expand : r tensor size -> void
foreign import ccall unsafe "THTensor.h THLongTensor_expand"
  c_THLongTensor_expand :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensor_expandNd : rets ops count -> void
foreign import ccall unsafe "THTensor.h THLongTensor_expandNd"
  c_THLongTensor_expandNd :: Ptr (Ptr CTHLongTensor) -> Ptr (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_resize : tensor size stride -> void
foreign import ccall unsafe "THTensor.h THLongTensor_resize"
  c_THLongTensor_resize :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensor_resizeAs : tensor src -> void
foreign import ccall unsafe "THTensor.h THLongTensor_resizeAs"
  c_THLongTensor_resizeAs :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_resizeNd : tensor nDimension size stride -> void
foreign import ccall unsafe "THTensor.h THLongTensor_resizeNd"
  c_THLongTensor_resizeNd :: (Ptr CTHLongTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THLongTensor_resize1d : tensor size0_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_resize1d"
  c_THLongTensor_resize1d :: (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_resize2d : tensor size0_ size1_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_resize2d"
  c_THLongTensor_resize2d :: (Ptr CTHLongTensor) -> CLong -> CLong -> IO ()

-- |c_THLongTensor_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_resize3d"
  c_THLongTensor_resize3d :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_resize4d"
  c_THLongTensor_resize4d :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_resize5d"
  c_THLongTensor_resize5d :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_set : self src -> void
foreign import ccall unsafe "THTensor.h THLongTensor_set"
  c_THLongTensor_set :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_setStorage"
  c_THLongTensor_setStorage :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensor_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall unsafe "THTensor.h THLongTensor_setStorageNd"
  c_THLongTensor_setStorageNd :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CPtrdiff -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THLongTensor_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_setStorage1d"
  c_THLongTensor_setStorage1d :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> IO ()

-- |c_THLongTensor_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_setStorage2d"
  c_THLongTensor_setStorage2d :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_setStorage3d"
  c_THLongTensor_setStorage3d :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_setStorage4d"
  c_THLongTensor_setStorage4d :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_narrow"
  c_THLongTensor_narrow :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THLongTensor_select : self src dimension_ sliceIndex_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_select"
  c_THLongTensor_select :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CLong -> IO ()

-- |c_THLongTensor_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_transpose"
  c_THLongTensor_transpose :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_unfold : self src dimension_ size_ step_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_unfold"
  c_THLongTensor_unfold :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THLongTensor_squeeze : self src -> void
foreign import ccall unsafe "THTensor.h THLongTensor_squeeze"
  c_THLongTensor_squeeze :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_squeeze1d : self src dimension_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_squeeze1d"
  c_THLongTensor_squeeze1d :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_unsqueeze1d : self src dimension_ -> void
foreign import ccall unsafe "THTensor.h THLongTensor_unsqueeze1d"
  c_THLongTensor_unsqueeze1d :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_isContiguous : self -> int
foreign import ccall unsafe "THTensor.h THLongTensor_isContiguous"
  c_THLongTensor_isContiguous :: (Ptr CTHLongTensor) -> CInt

-- |c_THLongTensor_isSameSizeAs : self src -> int
foreign import ccall unsafe "THTensor.h THLongTensor_isSameSizeAs"
  c_THLongTensor_isSameSizeAs :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt

-- |c_THLongTensor_isSetTo : self src -> int
foreign import ccall unsafe "THTensor.h THLongTensor_isSetTo"
  c_THLongTensor_isSetTo :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt

-- |c_THLongTensor_isSize : self dims -> int
foreign import ccall unsafe "THTensor.h THLongTensor_isSize"
  c_THLongTensor_isSize :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CInt

-- |c_THLongTensor_nElement : self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h THLongTensor_nElement"
  c_THLongTensor_nElement :: (Ptr CTHLongTensor) -> CPtrdiff

-- |c_THLongTensor_retain : self -> void
foreign import ccall unsafe "THTensor.h THLongTensor_retain"
  c_THLongTensor_retain :: (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_free : self -> void
foreign import ccall unsafe "THTensor.h THLongTensor_free"
  c_THLongTensor_free :: (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_freeCopyTo : self dst -> void
foreign import ccall unsafe "THTensor.h THLongTensor_freeCopyTo"
  c_THLongTensor_freeCopyTo :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_set1d : tensor x0 value -> void
foreign import ccall unsafe "THTensor.h THLongTensor_set1d"
  c_THLongTensor_set1d :: (Ptr CTHLongTensor) -> CLong -> CLong -> IO ()

-- |c_THLongTensor_set2d : tensor x0 x1 value -> void
foreign import ccall unsafe "THTensor.h THLongTensor_set2d"
  c_THLongTensor_set2d :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_set3d : tensor x0 x1 x2 value -> void
foreign import ccall unsafe "THTensor.h THLongTensor_set3d"
  c_THLongTensor_set3d :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall unsafe "THTensor.h THLongTensor_set4d"
  c_THLongTensor_set4d :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_get1d : tensor x0 -> real
foreign import ccall unsafe "THTensor.h THLongTensor_get1d"
  c_THLongTensor_get1d :: (Ptr CTHLongTensor) -> CLong -> CLong

-- |c_THLongTensor_get2d : tensor x0 x1 -> real
foreign import ccall unsafe "THTensor.h THLongTensor_get2d"
  c_THLongTensor_get2d :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong

-- |c_THLongTensor_get3d : tensor x0 x1 x2 -> real
foreign import ccall unsafe "THTensor.h THLongTensor_get3d"
  c_THLongTensor_get3d :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CLong

-- |c_THLongTensor_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall unsafe "THTensor.h THLongTensor_get4d"
  c_THLongTensor_get4d :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CLong -> CLong

-- |c_THLongTensor_desc : tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h THLongTensor_desc"
  c_THLongTensor_desc :: (Ptr CTHLongTensor) -> CTHDescBuff

-- |c_THLongTensor_sizeDesc : tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h THLongTensor_sizeDesc"
  c_THLongTensor_sizeDesc :: (Ptr CTHLongTensor) -> CTHDescBuff

-- |p_THLongTensor_storage : Pointer to self -> THStorage *
foreign import ccall unsafe "THTensor.h &THLongTensor_storage"
  p_THLongTensor_storage :: FunPtr ((Ptr CTHLongTensor) -> IO (Ptr CTHLongStorage))

-- |p_THLongTensor_storageOffset : Pointer to self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h &THLongTensor_storageOffset"
  p_THLongTensor_storageOffset :: FunPtr ((Ptr CTHLongTensor) -> CPtrdiff)

-- |p_THLongTensor_nDimension : Pointer to self -> int
foreign import ccall unsafe "THTensor.h &THLongTensor_nDimension"
  p_THLongTensor_nDimension :: FunPtr ((Ptr CTHLongTensor) -> CInt)

-- |p_THLongTensor_size : Pointer to self dim -> long
foreign import ccall unsafe "THTensor.h &THLongTensor_size"
  p_THLongTensor_size :: FunPtr ((Ptr CTHLongTensor) -> CInt -> CLong)

-- |p_THLongTensor_stride : Pointer to self dim -> long
foreign import ccall unsafe "THTensor.h &THLongTensor_stride"
  p_THLongTensor_stride :: FunPtr ((Ptr CTHLongTensor) -> CInt -> CLong)

-- |p_THLongTensor_newSizeOf : Pointer to self -> THLongStorage *
foreign import ccall unsafe "THTensor.h &THLongTensor_newSizeOf"
  p_THLongTensor_newSizeOf :: FunPtr ((Ptr CTHLongTensor) -> IO (Ptr CTHLongStorage))

-- |p_THLongTensor_newStrideOf : Pointer to self -> THLongStorage *
foreign import ccall unsafe "THTensor.h &THLongTensor_newStrideOf"
  p_THLongTensor_newStrideOf :: FunPtr ((Ptr CTHLongTensor) -> IO (Ptr CTHLongStorage))

-- |p_THLongTensor_data : Pointer to self -> real *
foreign import ccall unsafe "THTensor.h &THLongTensor_data"
  p_THLongTensor_data :: FunPtr ((Ptr CTHLongTensor) -> IO (Ptr CLong))

-- |p_THLongTensor_setFlag : Pointer to self flag -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_setFlag"
  p_THLongTensor_setFlag :: FunPtr ((Ptr CTHLongTensor) -> CChar -> IO ())

-- |p_THLongTensor_clearFlag : Pointer to self flag -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_clearFlag"
  p_THLongTensor_clearFlag :: FunPtr ((Ptr CTHLongTensor) -> CChar -> IO ())

-- |p_THLongTensor_new : Pointer to  -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_new"
  p_THLongTensor_new :: FunPtr (IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newWithTensor : Pointer to tensor -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newWithTensor"
  p_THLongTensor_newWithTensor :: FunPtr ((Ptr CTHLongTensor) -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newWithStorage : Pointer to storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newWithStorage"
  p_THLongTensor_newWithStorage :: FunPtr (Ptr CTHLongStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newWithStorage1d : Pointer to storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newWithStorage1d"
  p_THLongTensor_newWithStorage1d :: FunPtr (Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newWithStorage2d : Pointer to storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newWithStorage2d"
  p_THLongTensor_newWithStorage2d :: FunPtr (Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newWithStorage3d : Pointer to storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newWithStorage3d"
  p_THLongTensor_newWithStorage3d :: FunPtr (Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newWithStorage4d : Pointer to storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newWithStorage4d"
  p_THLongTensor_newWithStorage4d :: FunPtr (Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newWithSize : Pointer to size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newWithSize"
  p_THLongTensor_newWithSize :: FunPtr (Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newWithSize1d : Pointer to size0_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newWithSize1d"
  p_THLongTensor_newWithSize1d :: FunPtr (CLong -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newWithSize2d : Pointer to size0_ size1_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newWithSize2d"
  p_THLongTensor_newWithSize2d :: FunPtr (CLong -> CLong -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newWithSize3d : Pointer to size0_ size1_ size2_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newWithSize3d"
  p_THLongTensor_newWithSize3d :: FunPtr (CLong -> CLong -> CLong -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newWithSize4d : Pointer to size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newWithSize4d"
  p_THLongTensor_newWithSize4d :: FunPtr (CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newClone : Pointer to self -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newClone"
  p_THLongTensor_newClone :: FunPtr ((Ptr CTHLongTensor) -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newContiguous : Pointer to tensor -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newContiguous"
  p_THLongTensor_newContiguous :: FunPtr ((Ptr CTHLongTensor) -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newSelect : Pointer to tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newSelect"
  p_THLongTensor_newSelect :: FunPtr ((Ptr CTHLongTensor) -> CInt -> CLong -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newNarrow : Pointer to tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newNarrow"
  p_THLongTensor_newNarrow :: FunPtr ((Ptr CTHLongTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newTranspose : Pointer to tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newTranspose"
  p_THLongTensor_newTranspose :: FunPtr ((Ptr CTHLongTensor) -> CInt -> CInt -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newUnfold : Pointer to tensor dimension_ size_ step_ -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newUnfold"
  p_THLongTensor_newUnfold :: FunPtr ((Ptr CTHLongTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newView : Pointer to tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newView"
  p_THLongTensor_newView :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_newExpand : Pointer to tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h &THLongTensor_newExpand"
  p_THLongTensor_newExpand :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHLongTensor))

-- |p_THLongTensor_expand : Pointer to r tensor size -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_expand"
  p_THLongTensor_expand :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THLongTensor_expandNd : Pointer to rets ops count -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_expandNd"
  p_THLongTensor_expandNd :: FunPtr (Ptr (Ptr CTHLongTensor) -> Ptr (Ptr CTHLongTensor) -> CInt -> IO ())

-- |p_THLongTensor_resize : Pointer to tensor size stride -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_resize"
  p_THLongTensor_resize :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THLongTensor_resizeAs : Pointer to tensor src -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_resizeAs"
  p_THLongTensor_resizeAs :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_resizeNd : Pointer to tensor nDimension size stride -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_resizeNd"
  p_THLongTensor_resizeNd :: FunPtr ((Ptr CTHLongTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ())

-- |p_THLongTensor_resize1d : Pointer to tensor size0_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_resize1d"
  p_THLongTensor_resize1d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_resize2d : Pointer to tensor size0_ size1_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_resize2d"
  p_THLongTensor_resize2d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> IO ())

-- |p_THLongTensor_resize3d : Pointer to tensor size0_ size1_ size2_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_resize3d"
  p_THLongTensor_resize3d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_resize4d : Pointer to tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_resize4d"
  p_THLongTensor_resize4d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_resize5d : Pointer to tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_resize5d"
  p_THLongTensor_resize5d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_set : Pointer to self src -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_set"
  p_THLongTensor_set :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_setStorage : Pointer to self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_setStorage"
  p_THLongTensor_setStorage :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THLongTensor_setStorageNd : Pointer to self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_setStorageNd"
  p_THLongTensor_setStorageNd :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CPtrdiff -> CInt -> Ptr CLong -> Ptr CLong -> IO ())

-- |p_THLongTensor_setStorage1d : Pointer to self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_setStorage1d"
  p_THLongTensor_setStorage1d :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> IO ())

-- |p_THLongTensor_setStorage2d : Pointer to self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_setStorage2d"
  p_THLongTensor_setStorage2d :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_setStorage3d : Pointer to self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_setStorage3d"
  p_THLongTensor_setStorage3d :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_setStorage4d : Pointer to self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_setStorage4d"
  p_THLongTensor_setStorage4d :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_narrow : Pointer to self src dimension_ firstIndex_ size_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_narrow"
  p_THLongTensor_narrow :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CLong -> CLong -> IO ())

-- |p_THLongTensor_select : Pointer to self src dimension_ sliceIndex_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_select"
  p_THLongTensor_select :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CLong -> IO ())

-- |p_THLongTensor_transpose : Pointer to self src dimension1_ dimension2_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_transpose"
  p_THLongTensor_transpose :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ())

-- |p_THLongTensor_unfold : Pointer to self src dimension_ size_ step_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_unfold"
  p_THLongTensor_unfold :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CLong -> CLong -> IO ())

-- |p_THLongTensor_squeeze : Pointer to self src -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_squeeze"
  p_THLongTensor_squeeze :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_squeeze1d : Pointer to self src dimension_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_squeeze1d"
  p_THLongTensor_squeeze1d :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ())

-- |p_THLongTensor_unsqueeze1d : Pointer to self src dimension_ -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_unsqueeze1d"
  p_THLongTensor_unsqueeze1d :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ())

-- |p_THLongTensor_isContiguous : Pointer to self -> int
foreign import ccall unsafe "THTensor.h &THLongTensor_isContiguous"
  p_THLongTensor_isContiguous :: FunPtr ((Ptr CTHLongTensor) -> CInt)

-- |p_THLongTensor_isSameSizeAs : Pointer to self src -> int
foreign import ccall unsafe "THTensor.h &THLongTensor_isSameSizeAs"
  p_THLongTensor_isSameSizeAs :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt)

-- |p_THLongTensor_isSetTo : Pointer to self src -> int
foreign import ccall unsafe "THTensor.h &THLongTensor_isSetTo"
  p_THLongTensor_isSetTo :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt)

-- |p_THLongTensor_isSize : Pointer to self dims -> int
foreign import ccall unsafe "THTensor.h &THLongTensor_isSize"
  p_THLongTensor_isSize :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongStorage -> CInt)

-- |p_THLongTensor_nElement : Pointer to self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h &THLongTensor_nElement"
  p_THLongTensor_nElement :: FunPtr ((Ptr CTHLongTensor) -> CPtrdiff)

-- |p_THLongTensor_retain : Pointer to self -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_retain"
  p_THLongTensor_retain :: FunPtr ((Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_free : Pointer to self -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_free"
  p_THLongTensor_free :: FunPtr ((Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_freeCopyTo : Pointer to self dst -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_freeCopyTo"
  p_THLongTensor_freeCopyTo :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_set1d : Pointer to tensor x0 value -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_set1d"
  p_THLongTensor_set1d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> IO ())

-- |p_THLongTensor_set2d : Pointer to tensor x0 x1 value -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_set2d"
  p_THLongTensor_set2d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_set3d : Pointer to tensor x0 x1 x2 value -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_set3d"
  p_THLongTensor_set3d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_set4d : Pointer to tensor x0 x1 x2 x3 value -> void
foreign import ccall unsafe "THTensor.h &THLongTensor_set4d"
  p_THLongTensor_set4d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_get1d : Pointer to tensor x0 -> real
foreign import ccall unsafe "THTensor.h &THLongTensor_get1d"
  p_THLongTensor_get1d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong)

-- |p_THLongTensor_get2d : Pointer to tensor x0 x1 -> real
foreign import ccall unsafe "THTensor.h &THLongTensor_get2d"
  p_THLongTensor_get2d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> CLong)

-- |p_THLongTensor_get3d : Pointer to tensor x0 x1 x2 -> real
foreign import ccall unsafe "THTensor.h &THLongTensor_get3d"
  p_THLongTensor_get3d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CLong)

-- |p_THLongTensor_get4d : Pointer to tensor x0 x1 x2 x3 -> real
foreign import ccall unsafe "THTensor.h &THLongTensor_get4d"
  p_THLongTensor_get4d :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CLong -> CLong)

-- |p_THLongTensor_desc : Pointer to tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h &THLongTensor_desc"
  p_THLongTensor_desc :: FunPtr ((Ptr CTHLongTensor) -> CTHDescBuff)

-- |p_THLongTensor_sizeDesc : Pointer to tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h &THLongTensor_sizeDesc"
  p_THLongTensor_sizeDesc :: FunPtr ((Ptr CTHLongTensor) -> CTHDescBuff)