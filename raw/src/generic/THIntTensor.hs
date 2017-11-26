{-# LANGUAGE ForeignFunctionInterface #-}

module THIntTensor (
    c_THIntTensor_storage,
    c_THIntTensor_storageOffset,
    c_THIntTensor_nDimension,
    c_THIntTensor_size,
    c_THIntTensor_stride,
    c_THIntTensor_newSizeOf,
    c_THIntTensor_newStrideOf,
    c_THIntTensor_data,
    c_THIntTensor_setFlag,
    c_THIntTensor_clearFlag,
    c_THIntTensor_new,
    c_THIntTensor_newWithTensor,
    c_THIntTensor_newWithStorage,
    c_THIntTensor_newWithStorage1d,
    c_THIntTensor_newWithStorage2d,
    c_THIntTensor_newWithStorage3d,
    c_THIntTensor_newWithStorage4d,
    c_THIntTensor_newWithSize,
    c_THIntTensor_newWithSize1d,
    c_THIntTensor_newWithSize2d,
    c_THIntTensor_newWithSize3d,
    c_THIntTensor_newWithSize4d,
    c_THIntTensor_newClone,
    c_THIntTensor_newContiguous,
    c_THIntTensor_newSelect,
    c_THIntTensor_newNarrow,
    c_THIntTensor_newTranspose,
    c_THIntTensor_newUnfold,
    c_THIntTensor_newView,
    c_THIntTensor_newExpand,
    c_THIntTensor_expand,
    c_THIntTensor_expandNd,
    c_THIntTensor_resize,
    c_THIntTensor_resizeAs,
    c_THIntTensor_resizeNd,
    c_THIntTensor_resize1d,
    c_THIntTensor_resize2d,
    c_THIntTensor_resize3d,
    c_THIntTensor_resize4d,
    c_THIntTensor_resize5d,
    c_THIntTensor_set,
    c_THIntTensor_setStorage,
    c_THIntTensor_setStorageNd,
    c_THIntTensor_setStorage1d,
    c_THIntTensor_setStorage2d,
    c_THIntTensor_setStorage3d,
    c_THIntTensor_setStorage4d,
    c_THIntTensor_narrow,
    c_THIntTensor_select,
    c_THIntTensor_transpose,
    c_THIntTensor_unfold,
    c_THIntTensor_squeeze,
    c_THIntTensor_squeeze1d,
    c_THIntTensor_unsqueeze1d,
    c_THIntTensor_isContiguous,
    c_THIntTensor_isSameSizeAs,
    c_THIntTensor_isSetTo,
    c_THIntTensor_isSize,
    c_THIntTensor_nElement,
    c_THIntTensor_retain,
    c_THIntTensor_free,
    c_THIntTensor_freeCopyTo,
    c_THIntTensor_set1d,
    c_THIntTensor_set2d,
    c_THIntTensor_set3d,
    c_THIntTensor_set4d,
    c_THIntTensor_get1d,
    c_THIntTensor_get2d,
    c_THIntTensor_get3d,
    c_THIntTensor_get4d,
    c_THIntTensor_desc,
    c_THIntTensor_sizeDesc,
    p_THIntTensor_storage,
    p_THIntTensor_storageOffset,
    p_THIntTensor_nDimension,
    p_THIntTensor_size,
    p_THIntTensor_stride,
    p_THIntTensor_newSizeOf,
    p_THIntTensor_newStrideOf,
    p_THIntTensor_data,
    p_THIntTensor_setFlag,
    p_THIntTensor_clearFlag,
    p_THIntTensor_new,
    p_THIntTensor_newWithTensor,
    p_THIntTensor_newWithStorage,
    p_THIntTensor_newWithStorage1d,
    p_THIntTensor_newWithStorage2d,
    p_THIntTensor_newWithStorage3d,
    p_THIntTensor_newWithStorage4d,
    p_THIntTensor_newWithSize,
    p_THIntTensor_newWithSize1d,
    p_THIntTensor_newWithSize2d,
    p_THIntTensor_newWithSize3d,
    p_THIntTensor_newWithSize4d,
    p_THIntTensor_newClone,
    p_THIntTensor_newContiguous,
    p_THIntTensor_newSelect,
    p_THIntTensor_newNarrow,
    p_THIntTensor_newTranspose,
    p_THIntTensor_newUnfold,
    p_THIntTensor_newView,
    p_THIntTensor_newExpand,
    p_THIntTensor_expand,
    p_THIntTensor_expandNd,
    p_THIntTensor_resize,
    p_THIntTensor_resizeAs,
    p_THIntTensor_resizeNd,
    p_THIntTensor_resize1d,
    p_THIntTensor_resize2d,
    p_THIntTensor_resize3d,
    p_THIntTensor_resize4d,
    p_THIntTensor_resize5d,
    p_THIntTensor_set,
    p_THIntTensor_setStorage,
    p_THIntTensor_setStorageNd,
    p_THIntTensor_setStorage1d,
    p_THIntTensor_setStorage2d,
    p_THIntTensor_setStorage3d,
    p_THIntTensor_setStorage4d,
    p_THIntTensor_narrow,
    p_THIntTensor_select,
    p_THIntTensor_transpose,
    p_THIntTensor_unfold,
    p_THIntTensor_squeeze,
    p_THIntTensor_squeeze1d,
    p_THIntTensor_unsqueeze1d,
    p_THIntTensor_isContiguous,
    p_THIntTensor_isSameSizeAs,
    p_THIntTensor_isSetTo,
    p_THIntTensor_isSize,
    p_THIntTensor_nElement,
    p_THIntTensor_retain,
    p_THIntTensor_free,
    p_THIntTensor_freeCopyTo,
    p_THIntTensor_set1d,
    p_THIntTensor_set2d,
    p_THIntTensor_set3d,
    p_THIntTensor_set4d,
    p_THIntTensor_get1d,
    p_THIntTensor_get2d,
    p_THIntTensor_get3d,
    p_THIntTensor_get4d,
    p_THIntTensor_desc,
    p_THIntTensor_sizeDesc) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THIntTensor_storage : self -> THStorage *
foreign import ccall "THTensor.h THIntTensor_storage"
  c_THIntTensor_storage :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntStorage)

-- |c_THIntTensor_storageOffset : self -> ptrdiff_t
foreign import ccall "THTensor.h THIntTensor_storageOffset"
  c_THIntTensor_storageOffset :: (Ptr CTHIntTensor) -> CPtrdiff

-- |c_THIntTensor_nDimension : self -> int
foreign import ccall "THTensor.h THIntTensor_nDimension"
  c_THIntTensor_nDimension :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_size : self dim -> int64_t
foreign import ccall "THTensor.h THIntTensor_size"
  c_THIntTensor_size :: (Ptr CTHIntTensor) -> CInt -> CLLong

-- |c_THIntTensor_stride : self dim -> int64_t
foreign import ccall "THTensor.h THIntTensor_stride"
  c_THIntTensor_stride :: (Ptr CTHIntTensor) -> CInt -> CLLong

-- |c_THIntTensor_newSizeOf : self -> THLongStorage *
foreign import ccall "THTensor.h THIntTensor_newSizeOf"
  c_THIntTensor_newSizeOf :: (Ptr CTHIntTensor) -> IO (Ptr CTHLongStorage)

-- |c_THIntTensor_newStrideOf : self -> THLongStorage *
foreign import ccall "THTensor.h THIntTensor_newStrideOf"
  c_THIntTensor_newStrideOf :: (Ptr CTHIntTensor) -> IO (Ptr CTHLongStorage)

-- |c_THIntTensor_data : self -> real *
foreign import ccall "THTensor.h THIntTensor_data"
  c_THIntTensor_data :: (Ptr CTHIntTensor) -> IO (Ptr CInt)

-- |c_THIntTensor_setFlag : self flag -> void
foreign import ccall "THTensor.h THIntTensor_setFlag"
  c_THIntTensor_setFlag :: (Ptr CTHIntTensor) -> CChar -> IO ()

-- |c_THIntTensor_clearFlag : self flag -> void
foreign import ccall "THTensor.h THIntTensor_clearFlag"
  c_THIntTensor_clearFlag :: (Ptr CTHIntTensor) -> CChar -> IO ()

-- |c_THIntTensor_new :  -> THTensor *
foreign import ccall "THTensor.h THIntTensor_new"
  c_THIntTensor_new :: IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithTensor : tensor -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithTensor"
  c_THIntTensor_newWithTensor :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithStorage : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage"
  c_THIntTensor_newWithStorage :: Ptr CTHIntStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage1d"
  c_THIntTensor_newWithStorage1d :: Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage2d"
  c_THIntTensor_newWithStorage2d :: Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage3d"
  c_THIntTensor_newWithStorage3d :: Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage4d"
  c_THIntTensor_newWithStorage4d :: Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithSize : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize"
  c_THIntTensor_newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithSize1d : size0_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize1d"
  c_THIntTensor_newWithSize1d :: CLLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize2d"
  c_THIntTensor_newWithSize2d :: CLLong -> CLLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize3d"
  c_THIntTensor_newWithSize3d :: CLLong -> CLLong -> CLLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize4d"
  c_THIntTensor_newWithSize4d :: CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newClone : self -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newClone"
  c_THIntTensor_newClone :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newContiguous : tensor -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newContiguous"
  c_THIntTensor_newContiguous :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newSelect"
  c_THIntTensor_newSelect :: (Ptr CTHIntTensor) -> CInt -> CLLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newNarrow"
  c_THIntTensor_newNarrow :: (Ptr CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newTranspose"
  c_THIntTensor_newTranspose :: (Ptr CTHIntTensor) -> CInt -> CInt -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newUnfold"
  c_THIntTensor_newUnfold :: (Ptr CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newView : tensor size -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newView"
  c_THIntTensor_newView :: (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newExpand : tensor size -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newExpand"
  c_THIntTensor_newExpand :: (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_expand : r tensor size -> void
foreign import ccall "THTensor.h THIntTensor_expand"
  c_THIntTensor_expand :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensor_expandNd : rets ops count -> void
foreign import ccall "THTensor.h THIntTensor_expandNd"
  c_THIntTensor_expandNd :: Ptr (Ptr CTHIntTensor) -> Ptr (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_resize : tensor size stride -> void
foreign import ccall "THTensor.h THIntTensor_resize"
  c_THIntTensor_resize :: (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensor_resizeAs : tensor src -> void
foreign import ccall "THTensor.h THIntTensor_resizeAs"
  c_THIntTensor_resizeAs :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_resizeNd : tensor nDimension size stride -> void
foreign import ccall "THTensor.h THIntTensor_resizeNd"
  c_THIntTensor_resizeNd :: (Ptr CTHIntTensor) -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- |c_THIntTensor_resize1d : tensor size0_ -> void
foreign import ccall "THTensor.h THIntTensor_resize1d"
  c_THIntTensor_resize1d :: (Ptr CTHIntTensor) -> CLLong -> IO ()

-- |c_THIntTensor_resize2d : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h THIntTensor_resize2d"
  c_THIntTensor_resize2d :: (Ptr CTHIntTensor) -> CLLong -> CLLong -> IO ()

-- |c_THIntTensor_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h THIntTensor_resize3d"
  c_THIntTensor_resize3d :: (Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THIntTensor_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h THIntTensor_resize4d"
  c_THIntTensor_resize4d :: (Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THIntTensor_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h THIntTensor_resize5d"
  c_THIntTensor_resize5d :: (Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THIntTensor_set : self src -> void
foreign import ccall "THTensor.h THIntTensor_set"
  c_THIntTensor_set :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage"
  c_THIntTensor_setStorage :: (Ptr CTHIntTensor) -> Ptr CTHIntStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensor_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h THIntTensor_setStorageNd"
  c_THIntTensor_setStorageNd :: (Ptr CTHIntTensor) -> Ptr CTHIntStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- |c_THIntTensor_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage1d"
  c_THIntTensor_setStorage1d :: (Ptr CTHIntTensor) -> Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> IO ()

-- |c_THIntTensor_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage2d"
  c_THIntTensor_setStorage2d :: (Ptr CTHIntTensor) -> Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THIntTensor_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage3d"
  c_THIntTensor_setStorage3d :: (Ptr CTHIntTensor) -> Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THIntTensor_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage4d"
  c_THIntTensor_setStorage4d :: (Ptr CTHIntTensor) -> Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- |c_THIntTensor_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h THIntTensor_narrow"
  c_THIntTensor_narrow :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO ()

-- |c_THIntTensor_select : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h THIntTensor_select"
  c_THIntTensor_select :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CLLong -> IO ()

-- |c_THIntTensor_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h THIntTensor_transpose"
  c_THIntTensor_transpose :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_unfold : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h THIntTensor_unfold"
  c_THIntTensor_unfold :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO ()

-- |c_THIntTensor_squeeze : self src -> void
foreign import ccall "THTensor.h THIntTensor_squeeze"
  c_THIntTensor_squeeze :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_squeeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h THIntTensor_squeeze1d"
  c_THIntTensor_squeeze1d :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_unsqueeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h THIntTensor_unsqueeze1d"
  c_THIntTensor_unsqueeze1d :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_isContiguous : self -> int
foreign import ccall "THTensor.h THIntTensor_isContiguous"
  c_THIntTensor_isContiguous :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_isSameSizeAs : self src -> int
foreign import ccall "THTensor.h THIntTensor_isSameSizeAs"
  c_THIntTensor_isSameSizeAs :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_isSetTo : self src -> int
foreign import ccall "THTensor.h THIntTensor_isSetTo"
  c_THIntTensor_isSetTo :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_isSize : self dims -> int
foreign import ccall "THTensor.h THIntTensor_isSize"
  c_THIntTensor_isSize :: (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> CInt

-- |c_THIntTensor_nElement : self -> ptrdiff_t
foreign import ccall "THTensor.h THIntTensor_nElement"
  c_THIntTensor_nElement :: (Ptr CTHIntTensor) -> CPtrdiff

-- |c_THIntTensor_retain : self -> void
foreign import ccall "THTensor.h THIntTensor_retain"
  c_THIntTensor_retain :: (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_free : self -> void
foreign import ccall "THTensor.h THIntTensor_free"
  c_THIntTensor_free :: (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_freeCopyTo : self dst -> void
foreign import ccall "THTensor.h THIntTensor_freeCopyTo"
  c_THIntTensor_freeCopyTo :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_set1d : tensor x0 value -> void
foreign import ccall "THTensor.h THIntTensor_set1d"
  c_THIntTensor_set1d :: (Ptr CTHIntTensor) -> CLLong -> CInt -> IO ()

-- |c_THIntTensor_set2d : tensor x0 x1 value -> void
foreign import ccall "THTensor.h THIntTensor_set2d"
  c_THIntTensor_set2d :: (Ptr CTHIntTensor) -> CLLong -> CLLong -> CInt -> IO ()

-- |c_THIntTensor_set3d : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h THIntTensor_set3d"
  c_THIntTensor_set3d :: (Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CInt -> IO ()

-- |c_THIntTensor_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h THIntTensor_set4d"
  c_THIntTensor_set4d :: (Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CInt -> IO ()

-- |c_THIntTensor_get1d : tensor x0 -> real
foreign import ccall "THTensor.h THIntTensor_get1d"
  c_THIntTensor_get1d :: (Ptr CTHIntTensor) -> CLLong -> CInt

-- |c_THIntTensor_get2d : tensor x0 x1 -> real
foreign import ccall "THTensor.h THIntTensor_get2d"
  c_THIntTensor_get2d :: (Ptr CTHIntTensor) -> CLLong -> CLLong -> CInt

-- |c_THIntTensor_get3d : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h THIntTensor_get3d"
  c_THIntTensor_get3d :: (Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CInt

-- |c_THIntTensor_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h THIntTensor_get4d"
  c_THIntTensor_get4d :: (Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CInt

-- |c_THIntTensor_desc : tensor -> THDescBuff
foreign import ccall "THTensor.h THIntTensor_desc"
  c_THIntTensor_desc :: (Ptr CTHIntTensor) -> CTHDescBuff

-- |c_THIntTensor_sizeDesc : tensor -> THDescBuff
foreign import ccall "THTensor.h THIntTensor_sizeDesc"
  c_THIntTensor_sizeDesc :: (Ptr CTHIntTensor) -> CTHDescBuff

-- |p_THIntTensor_storage : Pointer to function : self -> THStorage *
foreign import ccall "THTensor.h &THIntTensor_storage"
  p_THIntTensor_storage :: FunPtr ((Ptr CTHIntTensor) -> IO (Ptr CTHIntStorage))

-- |p_THIntTensor_storageOffset : Pointer to function : self -> ptrdiff_t
foreign import ccall "THTensor.h &THIntTensor_storageOffset"
  p_THIntTensor_storageOffset :: FunPtr ((Ptr CTHIntTensor) -> CPtrdiff)

-- |p_THIntTensor_nDimension : Pointer to function : self -> int
foreign import ccall "THTensor.h &THIntTensor_nDimension"
  p_THIntTensor_nDimension :: FunPtr ((Ptr CTHIntTensor) -> CInt)

-- |p_THIntTensor_size : Pointer to function : self dim -> int64_t
foreign import ccall "THTensor.h &THIntTensor_size"
  p_THIntTensor_size :: FunPtr ((Ptr CTHIntTensor) -> CInt -> CLLong)

-- |p_THIntTensor_stride : Pointer to function : self dim -> int64_t
foreign import ccall "THTensor.h &THIntTensor_stride"
  p_THIntTensor_stride :: FunPtr ((Ptr CTHIntTensor) -> CInt -> CLLong)

-- |p_THIntTensor_newSizeOf : Pointer to function : self -> THLongStorage *
foreign import ccall "THTensor.h &THIntTensor_newSizeOf"
  p_THIntTensor_newSizeOf :: FunPtr ((Ptr CTHIntTensor) -> IO (Ptr CTHLongStorage))

-- |p_THIntTensor_newStrideOf : Pointer to function : self -> THLongStorage *
foreign import ccall "THTensor.h &THIntTensor_newStrideOf"
  p_THIntTensor_newStrideOf :: FunPtr ((Ptr CTHIntTensor) -> IO (Ptr CTHLongStorage))

-- |p_THIntTensor_data : Pointer to function : self -> real *
foreign import ccall "THTensor.h &THIntTensor_data"
  p_THIntTensor_data :: FunPtr ((Ptr CTHIntTensor) -> IO (Ptr CInt))

-- |p_THIntTensor_setFlag : Pointer to function : self flag -> void
foreign import ccall "THTensor.h &THIntTensor_setFlag"
  p_THIntTensor_setFlag :: FunPtr ((Ptr CTHIntTensor) -> CChar -> IO ())

-- |p_THIntTensor_clearFlag : Pointer to function : self flag -> void
foreign import ccall "THTensor.h &THIntTensor_clearFlag"
  p_THIntTensor_clearFlag :: FunPtr ((Ptr CTHIntTensor) -> CChar -> IO ())

-- |p_THIntTensor_new : Pointer to function :  -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_new"
  p_THIntTensor_new :: FunPtr (IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newWithTensor : Pointer to function : tensor -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithTensor"
  p_THIntTensor_newWithTensor :: FunPtr ((Ptr CTHIntTensor) -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newWithStorage : Pointer to function : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithStorage"
  p_THIntTensor_newWithStorage :: FunPtr (Ptr CTHIntStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newWithStorage1d : Pointer to function : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithStorage1d"
  p_THIntTensor_newWithStorage1d :: FunPtr (Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newWithStorage2d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithStorage2d"
  p_THIntTensor_newWithStorage2d :: FunPtr (Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newWithStorage3d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithStorage3d"
  p_THIntTensor_newWithStorage3d :: FunPtr (Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newWithStorage4d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithStorage4d"
  p_THIntTensor_newWithStorage4d :: FunPtr (Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newWithSize : Pointer to function : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithSize"
  p_THIntTensor_newWithSize :: FunPtr (Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newWithSize1d : Pointer to function : size0_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithSize1d"
  p_THIntTensor_newWithSize1d :: FunPtr (CLLong -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newWithSize2d : Pointer to function : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithSize2d"
  p_THIntTensor_newWithSize2d :: FunPtr (CLLong -> CLLong -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newWithSize3d : Pointer to function : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithSize3d"
  p_THIntTensor_newWithSize3d :: FunPtr (CLLong -> CLLong -> CLLong -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newWithSize4d : Pointer to function : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithSize4d"
  p_THIntTensor_newWithSize4d :: FunPtr (CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newClone : Pointer to function : self -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newClone"
  p_THIntTensor_newClone :: FunPtr ((Ptr CTHIntTensor) -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newContiguous : Pointer to function : tensor -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newContiguous"
  p_THIntTensor_newContiguous :: FunPtr ((Ptr CTHIntTensor) -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newSelect : Pointer to function : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newSelect"
  p_THIntTensor_newSelect :: FunPtr ((Ptr CTHIntTensor) -> CInt -> CLLong -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newNarrow : Pointer to function : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newNarrow"
  p_THIntTensor_newNarrow :: FunPtr ((Ptr CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newTranspose : Pointer to function : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newTranspose"
  p_THIntTensor_newTranspose :: FunPtr ((Ptr CTHIntTensor) -> CInt -> CInt -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newUnfold : Pointer to function : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newUnfold"
  p_THIntTensor_newUnfold :: FunPtr ((Ptr CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newView : Pointer to function : tensor size -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newView"
  p_THIntTensor_newView :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_newExpand : Pointer to function : tensor size -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newExpand"
  p_THIntTensor_newExpand :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHIntTensor))

-- |p_THIntTensor_expand : Pointer to function : r tensor size -> void
foreign import ccall "THTensor.h &THIntTensor_expand"
  p_THIntTensor_expand :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THIntTensor_expandNd : Pointer to function : rets ops count -> void
foreign import ccall "THTensor.h &THIntTensor_expandNd"
  p_THIntTensor_expandNd :: FunPtr (Ptr (Ptr CTHIntTensor) -> Ptr (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_resize : Pointer to function : tensor size stride -> void
foreign import ccall "THTensor.h &THIntTensor_resize"
  p_THIntTensor_resize :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THIntTensor_resizeAs : Pointer to function : tensor src -> void
foreign import ccall "THTensor.h &THIntTensor_resizeAs"
  p_THIntTensor_resizeAs :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_resizeNd : Pointer to function : tensor nDimension size stride -> void
foreign import ccall "THTensor.h &THIntTensor_resizeNd"
  p_THIntTensor_resizeNd :: FunPtr ((Ptr CTHIntTensor) -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- |p_THIntTensor_resize1d : Pointer to function : tensor size0_ -> void
foreign import ccall "THTensor.h &THIntTensor_resize1d"
  p_THIntTensor_resize1d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> IO ())

-- |p_THIntTensor_resize2d : Pointer to function : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h &THIntTensor_resize2d"
  p_THIntTensor_resize2d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> CLLong -> IO ())

-- |p_THIntTensor_resize3d : Pointer to function : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h &THIntTensor_resize3d"
  p_THIntTensor_resize3d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THIntTensor_resize4d : Pointer to function : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h &THIntTensor_resize4d"
  p_THIntTensor_resize4d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THIntTensor_resize5d : Pointer to function : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h &THIntTensor_resize5d"
  p_THIntTensor_resize5d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THIntTensor_set : Pointer to function : self src -> void
foreign import ccall "THTensor.h &THIntTensor_set"
  p_THIntTensor_set :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_setStorage : Pointer to function : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h &THIntTensor_setStorage"
  p_THIntTensor_setStorage :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHIntStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THIntTensor_setStorageNd : Pointer to function : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h &THIntTensor_setStorageNd"
  p_THIntTensor_setStorageNd :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHIntStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- |p_THIntTensor_setStorage1d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h &THIntTensor_setStorage1d"
  p_THIntTensor_setStorage1d :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> IO ())

-- |p_THIntTensor_setStorage2d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h &THIntTensor_setStorage2d"
  p_THIntTensor_setStorage2d :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THIntTensor_setStorage3d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h &THIntTensor_setStorage3d"
  p_THIntTensor_setStorage3d :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THIntTensor_setStorage4d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h &THIntTensor_setStorage4d"
  p_THIntTensor_setStorage4d :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_THIntTensor_narrow : Pointer to function : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h &THIntTensor_narrow"
  p_THIntTensor_narrow :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO ())

-- |p_THIntTensor_select : Pointer to function : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h &THIntTensor_select"
  p_THIntTensor_select :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CLLong -> IO ())

-- |p_THIntTensor_transpose : Pointer to function : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h &THIntTensor_transpose"
  p_THIntTensor_transpose :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ())

-- |p_THIntTensor_unfold : Pointer to function : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h &THIntTensor_unfold"
  p_THIntTensor_unfold :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO ())

-- |p_THIntTensor_squeeze : Pointer to function : self src -> void
foreign import ccall "THTensor.h &THIntTensor_squeeze"
  p_THIntTensor_squeeze :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_squeeze1d : Pointer to function : self src dimension_ -> void
foreign import ccall "THTensor.h &THIntTensor_squeeze1d"
  p_THIntTensor_squeeze1d :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_unsqueeze1d : Pointer to function : self src dimension_ -> void
foreign import ccall "THTensor.h &THIntTensor_unsqueeze1d"
  p_THIntTensor_unsqueeze1d :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_isContiguous : Pointer to function : self -> int
foreign import ccall "THTensor.h &THIntTensor_isContiguous"
  p_THIntTensor_isContiguous :: FunPtr ((Ptr CTHIntTensor) -> CInt)

-- |p_THIntTensor_isSameSizeAs : Pointer to function : self src -> int
foreign import ccall "THTensor.h &THIntTensor_isSameSizeAs"
  p_THIntTensor_isSameSizeAs :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt)

-- |p_THIntTensor_isSetTo : Pointer to function : self src -> int
foreign import ccall "THTensor.h &THIntTensor_isSetTo"
  p_THIntTensor_isSetTo :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt)

-- |p_THIntTensor_isSize : Pointer to function : self dims -> int
foreign import ccall "THTensor.h &THIntTensor_isSize"
  p_THIntTensor_isSize :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongStorage -> CInt)

-- |p_THIntTensor_nElement : Pointer to function : self -> ptrdiff_t
foreign import ccall "THTensor.h &THIntTensor_nElement"
  p_THIntTensor_nElement :: FunPtr ((Ptr CTHIntTensor) -> CPtrdiff)

-- |p_THIntTensor_retain : Pointer to function : self -> void
foreign import ccall "THTensor.h &THIntTensor_retain"
  p_THIntTensor_retain :: FunPtr ((Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_free : Pointer to function : self -> void
foreign import ccall "THTensor.h &THIntTensor_free"
  p_THIntTensor_free :: FunPtr ((Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_freeCopyTo : Pointer to function : self dst -> void
foreign import ccall "THTensor.h &THIntTensor_freeCopyTo"
  p_THIntTensor_freeCopyTo :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_set1d : Pointer to function : tensor x0 value -> void
foreign import ccall "THTensor.h &THIntTensor_set1d"
  p_THIntTensor_set1d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> CInt -> IO ())

-- |p_THIntTensor_set2d : Pointer to function : tensor x0 x1 value -> void
foreign import ccall "THTensor.h &THIntTensor_set2d"
  p_THIntTensor_set2d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> CLLong -> CInt -> IO ())

-- |p_THIntTensor_set3d : Pointer to function : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h &THIntTensor_set3d"
  p_THIntTensor_set3d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CInt -> IO ())

-- |p_THIntTensor_set4d : Pointer to function : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h &THIntTensor_set4d"
  p_THIntTensor_set4d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CInt -> IO ())

-- |p_THIntTensor_get1d : Pointer to function : tensor x0 -> real
foreign import ccall "THTensor.h &THIntTensor_get1d"
  p_THIntTensor_get1d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> CInt)

-- |p_THIntTensor_get2d : Pointer to function : tensor x0 x1 -> real
foreign import ccall "THTensor.h &THIntTensor_get2d"
  p_THIntTensor_get2d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> CLLong -> CInt)

-- |p_THIntTensor_get3d : Pointer to function : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h &THIntTensor_get3d"
  p_THIntTensor_get3d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CInt)

-- |p_THIntTensor_get4d : Pointer to function : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h &THIntTensor_get4d"
  p_THIntTensor_get4d :: FunPtr ((Ptr CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CInt)

-- |p_THIntTensor_desc : Pointer to function : tensor -> THDescBuff
foreign import ccall "THTensor.h &THIntTensor_desc"
  p_THIntTensor_desc :: FunPtr ((Ptr CTHIntTensor) -> CTHDescBuff)

-- |p_THIntTensor_sizeDesc : Pointer to function : tensor -> THDescBuff
foreign import ccall "THTensor.h &THIntTensor_sizeDesc"
  p_THIntTensor_sizeDesc :: FunPtr ((Ptr CTHIntTensor) -> CTHDescBuff)