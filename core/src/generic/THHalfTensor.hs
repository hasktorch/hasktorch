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
    c_THHalfTensor_sizeDesc) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfTensor_storage : self -> THStorage *
foreign import ccall unsafe "THTensor.h THHalfTensor_storage"
  c_THHalfTensor_storage :: (Ptr CTHHalfTensor) -> IO (Ptr CTHHalfStorage)

-- |c_THHalfTensor_storageOffset : self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h THHalfTensor_storageOffset"
  c_THHalfTensor_storageOffset :: (Ptr CTHHalfTensor) -> CPtrdiff

-- |c_THHalfTensor_nDimension : self -> int
foreign import ccall unsafe "THTensor.h THHalfTensor_nDimension"
  c_THHalfTensor_nDimension :: (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensor_size : self dim -> long
foreign import ccall unsafe "THTensor.h THHalfTensor_size"
  c_THHalfTensor_size :: (Ptr CTHHalfTensor) -> CInt -> CLong

-- |c_THHalfTensor_stride : self dim -> long
foreign import ccall unsafe "THTensor.h THHalfTensor_stride"
  c_THHalfTensor_stride :: (Ptr CTHHalfTensor) -> CInt -> CLong

-- |c_THHalfTensor_newSizeOf : self -> THLongStorage *
foreign import ccall unsafe "THTensor.h THHalfTensor_newSizeOf"
  c_THHalfTensor_newSizeOf :: (Ptr CTHHalfTensor) -> IO (Ptr CTHLongStorage)

-- |c_THHalfTensor_newStrideOf : self -> THLongStorage *
foreign import ccall unsafe "THTensor.h THHalfTensor_newStrideOf"
  c_THHalfTensor_newStrideOf :: (Ptr CTHHalfTensor) -> IO (Ptr CTHLongStorage)

-- |c_THHalfTensor_data : self -> real *
foreign import ccall unsafe "THTensor.h THHalfTensor_data"
  c_THHalfTensor_data :: (Ptr CTHHalfTensor) -> IO (Ptr THHalf)

-- |c_THHalfTensor_setFlag : self flag -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_setFlag"
  c_THHalfTensor_setFlag :: (Ptr CTHHalfTensor) -> CChar -> IO ()

-- |c_THHalfTensor_clearFlag : self flag -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_clearFlag"
  c_THHalfTensor_clearFlag :: (Ptr CTHHalfTensor) -> CChar -> IO ()

-- |c_THHalfTensor_new :  -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_new"
  c_THHalfTensor_new :: IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithTensor : tensor -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newWithTensor"
  c_THHalfTensor_newWithTensor :: (Ptr CTHHalfTensor) -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithStorage : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newWithStorage"
  c_THHalfTensor_newWithStorage :: Ptr CTHHalfStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newWithStorage1d"
  c_THHalfTensor_newWithStorage1d :: Ptr CTHHalfStorage -> CPtrdiff -> CLong -> CLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newWithStorage2d"
  c_THHalfTensor_newWithStorage2d :: Ptr CTHHalfStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newWithStorage3d"
  c_THHalfTensor_newWithStorage3d :: Ptr CTHHalfStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newWithStorage4d"
  c_THHalfTensor_newWithStorage4d :: Ptr CTHHalfStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithSize : size_ stride_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newWithSize"
  c_THHalfTensor_newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithSize1d : size0_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newWithSize1d"
  c_THHalfTensor_newWithSize1d :: CLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newWithSize2d"
  c_THHalfTensor_newWithSize2d :: CLong -> CLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newWithSize3d"
  c_THHalfTensor_newWithSize3d :: CLong -> CLong -> CLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newWithSize4d"
  c_THHalfTensor_newWithSize4d :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newClone : self -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newClone"
  c_THHalfTensor_newClone :: (Ptr CTHHalfTensor) -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newContiguous : tensor -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newContiguous"
  c_THHalfTensor_newContiguous :: (Ptr CTHHalfTensor) -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newSelect"
  c_THHalfTensor_newSelect :: (Ptr CTHHalfTensor) -> CInt -> CLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newNarrow"
  c_THHalfTensor_newNarrow :: (Ptr CTHHalfTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newTranspose"
  c_THHalfTensor_newTranspose :: (Ptr CTHHalfTensor) -> CInt -> CInt -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newUnfold"
  c_THHalfTensor_newUnfold :: (Ptr CTHHalfTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newView : tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newView"
  c_THHalfTensor_newView :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_newExpand : tensor size -> THTensor *
foreign import ccall unsafe "THTensor.h THHalfTensor_newExpand"
  c_THHalfTensor_newExpand :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHHalfTensor)

-- |c_THHalfTensor_expand : r tensor size -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_expand"
  c_THHalfTensor_expand :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_expandNd : rets ops count -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_expandNd"
  c_THHalfTensor_expandNd :: Ptr (Ptr CTHHalfTensor) -> Ptr (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_resize : tensor size stride -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_resize"
  c_THHalfTensor_resize :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_resizeAs : tensor src -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_resizeAs"
  c_THHalfTensor_resizeAs :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_resizeNd : tensor nDimension size stride -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_resizeNd"
  c_THHalfTensor_resizeNd :: (Ptr CTHHalfTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THHalfTensor_resize1d : tensor size0_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_resize1d"
  c_THHalfTensor_resize1d :: (Ptr CTHHalfTensor) -> CLong -> IO ()

-- |c_THHalfTensor_resize2d : tensor size0_ size1_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_resize2d"
  c_THHalfTensor_resize2d :: (Ptr CTHHalfTensor) -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_resize3d"
  c_THHalfTensor_resize3d :: (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_resize4d"
  c_THHalfTensor_resize4d :: (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_resize5d"
  c_THHalfTensor_resize5d :: (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_set : self src -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_set"
  c_THHalfTensor_set :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_setStorage"
  c_THHalfTensor_setStorage :: (Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_setStorageNd"
  c_THHalfTensor_setStorageNd :: (Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THHalfTensor_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_setStorage1d"
  c_THHalfTensor_setStorage1d :: (Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_setStorage2d"
  c_THHalfTensor_setStorage2d :: (Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_setStorage3d"
  c_THHalfTensor_setStorage3d :: (Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_setStorage4d"
  c_THHalfTensor_setStorage4d :: (Ptr CTHHalfTensor) -> Ptr CTHHalfStorage -> CPtrdiff -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_narrow"
  c_THHalfTensor_narrow :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_select : self src dimension_ sliceIndex_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_select"
  c_THHalfTensor_select :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CLong -> IO ()

-- |c_THHalfTensor_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_transpose"
  c_THHalfTensor_transpose :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_unfold : self src dimension_ size_ step_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_unfold"
  c_THHalfTensor_unfold :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_squeeze : self src -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_squeeze"
  c_THHalfTensor_squeeze :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_squeeze1d : self src dimension_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_squeeze1d"
  c_THHalfTensor_squeeze1d :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_unsqueeze1d : self src dimension_ -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_unsqueeze1d"
  c_THHalfTensor_unsqueeze1d :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_isContiguous : self -> int
foreign import ccall unsafe "THTensor.h THHalfTensor_isContiguous"
  c_THHalfTensor_isContiguous :: (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensor_isSameSizeAs : self src -> int
foreign import ccall unsafe "THTensor.h THHalfTensor_isSameSizeAs"
  c_THHalfTensor_isSameSizeAs :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensor_isSetTo : self src -> int
foreign import ccall unsafe "THTensor.h THHalfTensor_isSetTo"
  c_THHalfTensor_isSetTo :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensor_isSize : self dims -> int
foreign import ccall unsafe "THTensor.h THHalfTensor_isSize"
  c_THHalfTensor_isSize :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> CInt

-- |c_THHalfTensor_nElement : self -> ptrdiff_t
foreign import ccall unsafe "THTensor.h THHalfTensor_nElement"
  c_THHalfTensor_nElement :: (Ptr CTHHalfTensor) -> CPtrdiff

-- |c_THHalfTensor_retain : self -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_retain"
  c_THHalfTensor_retain :: (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_free : self -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_free"
  c_THHalfTensor_free :: (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_freeCopyTo : self dst -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_freeCopyTo"
  c_THHalfTensor_freeCopyTo :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_set1d : tensor x0 value -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_set1d"
  c_THHalfTensor_set1d :: (Ptr CTHHalfTensor) -> CLong -> THHalf -> IO ()

-- |c_THHalfTensor_set2d : tensor x0 x1 value -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_set2d"
  c_THHalfTensor_set2d :: (Ptr CTHHalfTensor) -> CLong -> CLong -> THHalf -> IO ()

-- |c_THHalfTensor_set3d : tensor x0 x1 x2 value -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_set3d"
  c_THHalfTensor_set3d :: (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> THHalf -> IO ()

-- |c_THHalfTensor_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall unsafe "THTensor.h THHalfTensor_set4d"
  c_THHalfTensor_set4d :: (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> CLong -> THHalf -> IO ()

-- |c_THHalfTensor_get1d : tensor x0 -> real
foreign import ccall unsafe "THTensor.h THHalfTensor_get1d"
  c_THHalfTensor_get1d :: (Ptr CTHHalfTensor) -> CLong -> THHalf

-- |c_THHalfTensor_get2d : tensor x0 x1 -> real
foreign import ccall unsafe "THTensor.h THHalfTensor_get2d"
  c_THHalfTensor_get2d :: (Ptr CTHHalfTensor) -> CLong -> CLong -> THHalf

-- |c_THHalfTensor_get3d : tensor x0 x1 x2 -> real
foreign import ccall unsafe "THTensor.h THHalfTensor_get3d"
  c_THHalfTensor_get3d :: (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> THHalf

-- |c_THHalfTensor_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall unsafe "THTensor.h THHalfTensor_get4d"
  c_THHalfTensor_get4d :: (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> CLong -> THHalf

-- |c_THHalfTensor_desc : tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h THHalfTensor_desc"
  c_THHalfTensor_desc :: (Ptr CTHHalfTensor) -> CTHDescBuff

-- |c_THHalfTensor_sizeDesc : tensor -> THDescBuff
foreign import ccall unsafe "THTensor.h THHalfTensor_sizeDesc"
  c_THHalfTensor_sizeDesc :: (Ptr CTHHalfTensor) -> CTHDescBuff