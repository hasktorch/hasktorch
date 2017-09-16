{-# LANGUAGE ForeignFunctionInterface#-}
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
    c_THDoubleTensor_sizeDesc) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleTensor_storage : self -> THStorage *
foreign import ccall "THTensor.h THDoubleTensor_storage"
  c_THDoubleTensor_storage :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleTensor_storageOffset : self -> THStorage *
foreign import ccall "THTensor.h THDoubleTensor_storageOffset"
  c_THDoubleTensor_storageOffset :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleTensor_nDimension : self -> int
foreign import ccall "THTensor.h THDoubleTensor_nDimension"
  c_THDoubleTensor_nDimension :: (Ptr CTHDoubleTensor) -> CInt

-- |c_THDoubleTensor_size : self dim -> long
foreign import ccall "THTensor.h THDoubleTensor_size"
  c_THDoubleTensor_size :: (Ptr CTHDoubleTensor) -> CInt -> CLong

-- |c_THDoubleTensor_stride : self dim -> long
foreign import ccall "THTensor.h THDoubleTensor_stride"
  c_THDoubleTensor_stride :: (Ptr CTHDoubleTensor) -> CInt -> CLong

-- |c_THDoubleTensor_newSizeOf : self -> THLongStorage *
foreign import ccall "THTensor.h THDoubleTensor_newSizeOf"
  c_THDoubleTensor_newSizeOf :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleLongStorage)

-- |c_THDoubleTensor_newStrideOf : self -> THLongStorage *
foreign import ccall "THTensor.h THDoubleTensor_newStrideOf"
  c_THDoubleTensor_newStrideOf :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleLongStorage)

-- |c_THDoubleTensor_data : self -> real *
foreign import ccall "THTensor.h THDoubleTensor_data"
  c_THDoubleTensor_data :: (Ptr CTHDoubleTensor) -> IO (Ptr CDouble)

-- |c_THDoubleTensor_setFlag : self flag -> void
foreign import ccall "THTensor.h THDoubleTensor_setFlag"
  c_THDoubleTensor_setFlag :: (Ptr CTHDoubleTensor) -> CChar -> IO ()

-- |c_THDoubleTensor_clearFlag : self flag -> void
foreign import ccall "THTensor.h THDoubleTensor_clearFlag"
  c_THDoubleTensor_clearFlag :: (Ptr CTHDoubleTensor) -> CChar -> IO ()

-- |c_THDoubleTensor_new :  -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_new"
  c_THDoubleTensor_new :: IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithTensor : tensor -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newWithTensor"
  c_THDoubleTensor_newWithTensor :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithStorage : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newWithStorage"
  c_THDoubleTensor_newWithStorage :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> Ptr CTHDoubleLongStorage -> Ptr CTHDoubleLongStorage -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newWithStorage1d"
  c_THDoubleTensor_newWithStorage1d :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newWithStorage2d"
  c_THDoubleTensor_newWithStorage2d :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newWithStorage3d"
  c_THDoubleTensor_newWithStorage3d :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newWithStorage4d"
  c_THDoubleTensor_newWithStorage4d :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithSize : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newWithSize"
  c_THDoubleTensor_newWithSize :: Ptr CTHDoubleLongStorage -> Ptr CTHDoubleLongStorage -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithSize1d : size0_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newWithSize1d"
  c_THDoubleTensor_newWithSize1d :: CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newWithSize2d"
  c_THDoubleTensor_newWithSize2d :: CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newWithSize3d"
  c_THDoubleTensor_newWithSize3d :: CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newWithSize4d"
  c_THDoubleTensor_newWithSize4d :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newClone : self -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newClone"
  c_THDoubleTensor_newClone :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newContiguous : tensor -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newContiguous"
  c_THDoubleTensor_newContiguous :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newSelect"
  c_THDoubleTensor_newSelect :: (Ptr CTHDoubleTensor) -> CInt -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newNarrow"
  c_THDoubleTensor_newNarrow :: (Ptr CTHDoubleTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newTranspose"
  c_THDoubleTensor_newTranspose :: (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newUnfold"
  c_THDoubleTensor_newUnfold :: (Ptr CTHDoubleTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newView : tensor size -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newView"
  c_THDoubleTensor_newView :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleLongStorage -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_newExpand : tensor size -> THTensor *
foreign import ccall "THTensor.h THDoubleTensor_newExpand"
  c_THDoubleTensor_newExpand :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleLongStorage -> IO (Ptr CTHDoubleTensor)

-- |c_THDoubleTensor_expand : r tensor size -> void
foreign import ccall "THTensor.h THDoubleTensor_expand"
  c_THDoubleTensor_expand :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CTHDoubleLongStorage -> IO ()

-- |c_THDoubleTensor_expandNd : rets ops count -> void
foreign import ccall "THTensor.h THDoubleTensor_expandNd"
  c_THDoubleTensor_expandNd :: Ptr (Ptr CTHDoubleTensor) -> Ptr (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensor_resize : tensor size stride -> void
foreign import ccall "THTensor.h THDoubleTensor_resize"
  c_THDoubleTensor_resize :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleLongStorage -> Ptr CTHDoubleLongStorage -> IO ()

-- |c_THDoubleTensor_resizeAs : tensor src -> void
foreign import ccall "THTensor.h THDoubleTensor_resizeAs"
  c_THDoubleTensor_resizeAs :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_resizeNd : tensor nDimension size stride -> void
foreign import ccall "THTensor.h THDoubleTensor_resizeNd"
  c_THDoubleTensor_resizeNd :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THDoubleTensor_resize1d : tensor size0_ -> void
foreign import ccall "THTensor.h THDoubleTensor_resize1d"
  c_THDoubleTensor_resize1d :: (Ptr CTHDoubleTensor) -> CLong -> IO ()

-- |c_THDoubleTensor_resize2d : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h THDoubleTensor_resize2d"
  c_THDoubleTensor_resize2d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h THDoubleTensor_resize3d"
  c_THDoubleTensor_resize3d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h THDoubleTensor_resize4d"
  c_THDoubleTensor_resize4d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h THDoubleTensor_resize5d"
  c_THDoubleTensor_resize5d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_set : self src -> void
foreign import ccall "THTensor.h THDoubleTensor_set"
  c_THDoubleTensor_set :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h THDoubleTensor_setStorage"
  c_THDoubleTensor_setStorage :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> Ptr CTHDoubleLongStorage -> Ptr CTHDoubleLongStorage -> IO ()

-- |c_THDoubleTensor_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h THDoubleTensor_setStorageNd"
  c_THDoubleTensor_setStorageNd :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THDoubleTensor_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h THDoubleTensor_setStorage1d"
  c_THDoubleTensor_setStorage1d :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h THDoubleTensor_setStorage2d"
  c_THDoubleTensor_setStorage2d :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h THDoubleTensor_setStorage3d"
  c_THDoubleTensor_setStorage3d :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h THDoubleTensor_setStorage4d"
  c_THDoubleTensor_setStorage4d :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h THDoubleTensor_narrow"
  c_THDoubleTensor_narrow :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_select : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h THDoubleTensor_select"
  c_THDoubleTensor_select :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CLong -> IO ()

-- |c_THDoubleTensor_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h THDoubleTensor_transpose"
  c_THDoubleTensor_transpose :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_unfold : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h THDoubleTensor_unfold"
  c_THDoubleTensor_unfold :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_squeeze : self src -> void
foreign import ccall "THTensor.h THDoubleTensor_squeeze"
  c_THDoubleTensor_squeeze :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_squeeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h THDoubleTensor_squeeze1d"
  c_THDoubleTensor_squeeze1d :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensor_unsqueeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h THDoubleTensor_unsqueeze1d"
  c_THDoubleTensor_unsqueeze1d :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensor_isContiguous : self -> int
foreign import ccall "THTensor.h THDoubleTensor_isContiguous"
  c_THDoubleTensor_isContiguous :: (Ptr CTHDoubleTensor) -> CInt

-- |c_THDoubleTensor_isSameSizeAs : self src -> int
foreign import ccall "THTensor.h THDoubleTensor_isSameSizeAs"
  c_THDoubleTensor_isSameSizeAs :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt

-- |c_THDoubleTensor_isSetTo : self src -> int
foreign import ccall "THTensor.h THDoubleTensor_isSetTo"
  c_THDoubleTensor_isSetTo :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt

-- |c_THDoubleTensor_isSize : self dims -> int
foreign import ccall "THTensor.h THDoubleTensor_isSize"
  c_THDoubleTensor_isSize :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleLongStorage -> CInt

-- |c_THDoubleTensor_nElement : self -> THStorage *
foreign import ccall "THTensor.h THDoubleTensor_nElement"
  c_THDoubleTensor_nElement :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleTensor_retain : self -> void
foreign import ccall "THTensor.h THDoubleTensor_retain"
  c_THDoubleTensor_retain :: (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_free : self -> void
foreign import ccall "THTensor.h THDoubleTensor_free"
  c_THDoubleTensor_free :: (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_freeCopyTo : self dst -> void
foreign import ccall "THTensor.h THDoubleTensor_freeCopyTo"
  c_THDoubleTensor_freeCopyTo :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_set1d : tensor x0 value -> void
foreign import ccall "THTensor.h THDoubleTensor_set1d"
  c_THDoubleTensor_set1d :: (Ptr CTHDoubleTensor) -> CLong -> CDouble -> IO ()

-- |c_THDoubleTensor_set2d : tensor x0 x1 value -> void
foreign import ccall "THTensor.h THDoubleTensor_set2d"
  c_THDoubleTensor_set2d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CDouble -> IO ()

-- |c_THDoubleTensor_set3d : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h THDoubleTensor_set3d"
  c_THDoubleTensor_set3d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CDouble -> IO ()

-- |c_THDoubleTensor_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h THDoubleTensor_set4d"
  c_THDoubleTensor_set4d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CLong -> CDouble -> IO ()

-- |c_THDoubleTensor_get1d : tensor x0 -> real
foreign import ccall "THTensor.h THDoubleTensor_get1d"
  c_THDoubleTensor_get1d :: (Ptr CTHDoubleTensor) -> CLong -> CDouble

-- |c_THDoubleTensor_get2d : tensor x0 x1 -> real
foreign import ccall "THTensor.h THDoubleTensor_get2d"
  c_THDoubleTensor_get2d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CDouble

-- |c_THDoubleTensor_get3d : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h THDoubleTensor_get3d"
  c_THDoubleTensor_get3d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CDouble

-- |c_THDoubleTensor_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h THDoubleTensor_get4d"
  c_THDoubleTensor_get4d :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> CLong -> CDouble

-- |c_THDoubleTensor_desc : tensor -> THDescBuff
foreign import ccall "THTensor.h THDoubleTensor_desc"
  c_THDoubleTensor_desc :: (Ptr CTHDoubleTensor) -> CTHDescBuff

-- |c_THDoubleTensor_sizeDesc : tensor -> THDescBuff
foreign import ccall "THTensor.h THDoubleTensor_sizeDesc"
  c_THDoubleTensor_sizeDesc :: (Ptr CTHDoubleTensor) -> CTHDescBuff