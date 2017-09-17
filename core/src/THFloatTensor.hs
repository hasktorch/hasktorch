{-# LANGUAGE ForeignFunctionInterface#-}

module THFloatTensor (
    c_THFloatTensor_storage,
    c_THFloatTensor_storageOffset,
    c_THFloatTensor_nDimension,
    c_THFloatTensor_size,
    c_THFloatTensor_stride,
    c_THFloatTensor_newSizeOf,
    c_THFloatTensor_newStrideOf,
    c_THFloatTensor_data,
    c_THFloatTensor_setFlag,
    c_THFloatTensor_clearFlag,
    c_THFloatTensor_new,
    c_THFloatTensor_newWithTensor,
    c_THFloatTensor_newWithStorage,
    c_THFloatTensor_newWithStorage1d,
    c_THFloatTensor_newWithStorage2d,
    c_THFloatTensor_newWithStorage3d,
    c_THFloatTensor_newWithStorage4d,
    c_THFloatTensor_newWithSize,
    c_THFloatTensor_newWithSize1d,
    c_THFloatTensor_newWithSize2d,
    c_THFloatTensor_newWithSize3d,
    c_THFloatTensor_newWithSize4d,
    c_THFloatTensor_newClone,
    c_THFloatTensor_newContiguous,
    c_THFloatTensor_newSelect,
    c_THFloatTensor_newNarrow,
    c_THFloatTensor_newTranspose,
    c_THFloatTensor_newUnfold,
    c_THFloatTensor_newView,
    c_THFloatTensor_newExpand,
    c_THFloatTensor_expand,
    c_THFloatTensor_expandNd,
    c_THFloatTensor_resize,
    c_THFloatTensor_resizeAs,
    c_THFloatTensor_resizeNd,
    c_THFloatTensor_resize1d,
    c_THFloatTensor_resize2d,
    c_THFloatTensor_resize3d,
    c_THFloatTensor_resize4d,
    c_THFloatTensor_resize5d,
    c_THFloatTensor_set,
    c_THFloatTensor_setStorage,
    c_THFloatTensor_setStorageNd,
    c_THFloatTensor_setStorage1d,
    c_THFloatTensor_setStorage2d,
    c_THFloatTensor_setStorage3d,
    c_THFloatTensor_setStorage4d,
    c_THFloatTensor_narrow,
    c_THFloatTensor_select,
    c_THFloatTensor_transpose,
    c_THFloatTensor_unfold,
    c_THFloatTensor_squeeze,
    c_THFloatTensor_squeeze1d,
    c_THFloatTensor_unsqueeze1d,
    c_THFloatTensor_isContiguous,
    c_THFloatTensor_isSameSizeAs,
    c_THFloatTensor_isSetTo,
    c_THFloatTensor_isSize,
    c_THFloatTensor_nElement,
    c_THFloatTensor_retain,
    c_THFloatTensor_free,
    c_THFloatTensor_freeCopyTo,
    c_THFloatTensor_set1d,
    c_THFloatTensor_set2d,
    c_THFloatTensor_set3d,
    c_THFloatTensor_set4d,
    c_THFloatTensor_get1d,
    c_THFloatTensor_get2d,
    c_THFloatTensor_get3d,
    c_THFloatTensor_get4d,
    c_THFloatTensor_desc,
    c_THFloatTensor_sizeDesc) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatTensor_storage : self -> THStorage *
foreign import ccall "THTensor.h THFloatTensor_storage"
  c_THFloatTensor_storage :: (Ptr CTHFloatTensor) -> IO (Ptr CTHFloatStorage)

-- |c_THFloatTensor_storageOffset : self -> THStorage *
foreign import ccall "THTensor.h THFloatTensor_storageOffset"
  c_THFloatTensor_storageOffset :: (Ptr CTHFloatTensor) -> IO (Ptr CTHFloatStorage)

-- |c_THFloatTensor_nDimension : self -> int
foreign import ccall "THTensor.h THFloatTensor_nDimension"
  c_THFloatTensor_nDimension :: (Ptr CTHFloatTensor) -> CInt

-- |c_THFloatTensor_size : self dim -> long
foreign import ccall "THTensor.h THFloatTensor_size"
  c_THFloatTensor_size :: (Ptr CTHFloatTensor) -> CInt -> CLong

-- |c_THFloatTensor_stride : self dim -> long
foreign import ccall "THTensor.h THFloatTensor_stride"
  c_THFloatTensor_stride :: (Ptr CTHFloatTensor) -> CInt -> CLong

-- |c_THFloatTensor_newSizeOf : self -> THLongStorage *
foreign import ccall "THTensor.h THFloatTensor_newSizeOf"
  c_THFloatTensor_newSizeOf :: (Ptr CTHFloatTensor) -> IO (Ptr CTHLongStorage)

-- |c_THFloatTensor_newStrideOf : self -> THLongStorage *
foreign import ccall "THTensor.h THFloatTensor_newStrideOf"
  c_THFloatTensor_newStrideOf :: (Ptr CTHFloatTensor) -> IO (Ptr CTHLongStorage)

-- |c_THFloatTensor_data : self -> real *
foreign import ccall "THTensor.h THFloatTensor_data"
  c_THFloatTensor_data :: (Ptr CTHFloatTensor) -> IO (Ptr CFloat)

-- |c_THFloatTensor_setFlag : self flag -> void
foreign import ccall "THTensor.h THFloatTensor_setFlag"
  c_THFloatTensor_setFlag :: (Ptr CTHFloatTensor) -> CChar -> IO ()

-- |c_THFloatTensor_clearFlag : self flag -> void
foreign import ccall "THTensor.h THFloatTensor_clearFlag"
  c_THFloatTensor_clearFlag :: (Ptr CTHFloatTensor) -> CChar -> IO ()

-- |c_THFloatTensor_new :  -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_new"
  c_THFloatTensor_new :: IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newWithTensor : tensor -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newWithTensor"
  c_THFloatTensor_newWithTensor :: (Ptr CTHFloatTensor) -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newWithStorage : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newWithStorage"
  c_THFloatTensor_newWithStorage :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newWithStorage1d"
  c_THFloatTensor_newWithStorage1d :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> CLong -> CLong -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newWithStorage2d"
  c_THFloatTensor_newWithStorage2d :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newWithStorage3d"
  c_THFloatTensor_newWithStorage3d :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newWithStorage4d"
  c_THFloatTensor_newWithStorage4d :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newWithSize : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newWithSize"
  c_THFloatTensor_newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newWithSize1d : size0_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newWithSize1d"
  c_THFloatTensor_newWithSize1d :: CLong -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newWithSize2d"
  c_THFloatTensor_newWithSize2d :: CLong -> CLong -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newWithSize3d"
  c_THFloatTensor_newWithSize3d :: CLong -> CLong -> CLong -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newWithSize4d"
  c_THFloatTensor_newWithSize4d :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newClone : self -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newClone"
  c_THFloatTensor_newClone :: (Ptr CTHFloatTensor) -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newContiguous : tensor -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newContiguous"
  c_THFloatTensor_newContiguous :: (Ptr CTHFloatTensor) -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newSelect"
  c_THFloatTensor_newSelect :: (Ptr CTHFloatTensor) -> CInt -> CLong -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newNarrow"
  c_THFloatTensor_newNarrow :: (Ptr CTHFloatTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newTranspose"
  c_THFloatTensor_newTranspose :: (Ptr CTHFloatTensor) -> CInt -> CInt -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newUnfold"
  c_THFloatTensor_newUnfold :: (Ptr CTHFloatTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newView : tensor size -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newView"
  c_THFloatTensor_newView :: (Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_newExpand : tensor size -> THTensor *
foreign import ccall "THTensor.h THFloatTensor_newExpand"
  c_THFloatTensor_newExpand :: (Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHFloatTensor)

-- |c_THFloatTensor_expand : r tensor size -> void
foreign import ccall "THTensor.h THFloatTensor_expand"
  c_THFloatTensor_expand :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatTensor_expandNd : rets ops count -> void
foreign import ccall "THTensor.h THFloatTensor_expandNd"
  c_THFloatTensor_expandNd :: Ptr (Ptr CTHFloatTensor) -> Ptr (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensor_resize : tensor size stride -> void
foreign import ccall "THTensor.h THFloatTensor_resize"
  c_THFloatTensor_resize :: (Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatTensor_resizeAs : tensor src -> void
foreign import ccall "THTensor.h THFloatTensor_resizeAs"
  c_THFloatTensor_resizeAs :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_resizeNd : tensor nDimension size stride -> void
foreign import ccall "THTensor.h THFloatTensor_resizeNd"
  c_THFloatTensor_resizeNd :: (Ptr CTHFloatTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THFloatTensor_resize1d : tensor size0_ -> void
foreign import ccall "THTensor.h THFloatTensor_resize1d"
  c_THFloatTensor_resize1d :: (Ptr CTHFloatTensor) -> CLong -> IO ()

-- |c_THFloatTensor_resize2d : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h THFloatTensor_resize2d"
  c_THFloatTensor_resize2d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h THFloatTensor_resize3d"
  c_THFloatTensor_resize3d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h THFloatTensor_resize4d"
  c_THFloatTensor_resize4d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h THFloatTensor_resize5d"
  c_THFloatTensor_resize5d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_set : self src -> void
foreign import ccall "THTensor.h THFloatTensor_set"
  c_THFloatTensor_set :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h THFloatTensor_setStorage"
  c_THFloatTensor_setStorage :: (Ptr CTHFloatTensor) -> Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatTensor_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h THFloatTensor_setStorageNd"
  c_THFloatTensor_setStorageNd :: (Ptr CTHFloatTensor) -> Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THFloatTensor_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h THFloatTensor_setStorage1d"
  c_THFloatTensor_setStorage1d :: (Ptr CTHFloatTensor) -> Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h THFloatTensor_setStorage2d"
  c_THFloatTensor_setStorage2d :: (Ptr CTHFloatTensor) -> Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h THFloatTensor_setStorage3d"
  c_THFloatTensor_setStorage3d :: (Ptr CTHFloatTensor) -> Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h THFloatTensor_setStorage4d"
  c_THFloatTensor_setStorage4d :: (Ptr CTHFloatTensor) -> Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h THFloatTensor_narrow"
  c_THFloatTensor_narrow :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_select : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h THFloatTensor_select"
  c_THFloatTensor_select :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CLong -> IO ()

-- |c_THFloatTensor_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h THFloatTensor_transpose"
  c_THFloatTensor_transpose :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_unfold : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h THFloatTensor_unfold"
  c_THFloatTensor_unfold :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_squeeze : self src -> void
foreign import ccall "THTensor.h THFloatTensor_squeeze"
  c_THFloatTensor_squeeze :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_squeeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h THFloatTensor_squeeze1d"
  c_THFloatTensor_squeeze1d :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensor_unsqueeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h THFloatTensor_unsqueeze1d"
  c_THFloatTensor_unsqueeze1d :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensor_isContiguous : self -> int
foreign import ccall "THTensor.h THFloatTensor_isContiguous"
  c_THFloatTensor_isContiguous :: (Ptr CTHFloatTensor) -> CInt

-- |c_THFloatTensor_isSameSizeAs : self src -> int
foreign import ccall "THTensor.h THFloatTensor_isSameSizeAs"
  c_THFloatTensor_isSameSizeAs :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt

-- |c_THFloatTensor_isSetTo : self src -> int
foreign import ccall "THTensor.h THFloatTensor_isSetTo"
  c_THFloatTensor_isSetTo :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt

-- |c_THFloatTensor_isSize : self dims -> int
foreign import ccall "THTensor.h THFloatTensor_isSize"
  c_THFloatTensor_isSize :: (Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> CInt

-- |c_THFloatTensor_nElement : self -> THStorage *
foreign import ccall "THTensor.h THFloatTensor_nElement"
  c_THFloatTensor_nElement :: (Ptr CTHFloatTensor) -> IO (Ptr CTHFloatStorage)

-- |c_THFloatTensor_retain : self -> void
foreign import ccall "THTensor.h THFloatTensor_retain"
  c_THFloatTensor_retain :: (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_free : self -> void
foreign import ccall "THTensor.h THFloatTensor_free"
  c_THFloatTensor_free :: (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_freeCopyTo : self dst -> void
foreign import ccall "THTensor.h THFloatTensor_freeCopyTo"
  c_THFloatTensor_freeCopyTo :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_set1d : tensor x0 value -> void
foreign import ccall "THTensor.h THFloatTensor_set1d"
  c_THFloatTensor_set1d :: (Ptr CTHFloatTensor) -> CLong -> CFloat -> IO ()

-- |c_THFloatTensor_set2d : tensor x0 x1 value -> void
foreign import ccall "THTensor.h THFloatTensor_set2d"
  c_THFloatTensor_set2d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CFloat -> IO ()

-- |c_THFloatTensor_set3d : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h THFloatTensor_set3d"
  c_THFloatTensor_set3d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> CFloat -> IO ()

-- |c_THFloatTensor_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h THFloatTensor_set4d"
  c_THFloatTensor_set4d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> CLong -> CFloat -> IO ()

-- |c_THFloatTensor_get1d : tensor x0 -> real
foreign import ccall "THTensor.h THFloatTensor_get1d"
  c_THFloatTensor_get1d :: (Ptr CTHFloatTensor) -> CLong -> CFloat

-- |c_THFloatTensor_get2d : tensor x0 x1 -> real
foreign import ccall "THTensor.h THFloatTensor_get2d"
  c_THFloatTensor_get2d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CFloat

-- |c_THFloatTensor_get3d : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h THFloatTensor_get3d"
  c_THFloatTensor_get3d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> CFloat

-- |c_THFloatTensor_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h THFloatTensor_get4d"
  c_THFloatTensor_get4d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> CLong -> CFloat

-- |c_THFloatTensor_desc : tensor -> THDescBuff
foreign import ccall "THTensor.h THFloatTensor_desc"
  c_THFloatTensor_desc :: (Ptr CTHFloatTensor) -> CTHDescBuff

-- |c_THFloatTensor_sizeDesc : tensor -> THDescBuff
foreign import ccall "THTensor.h THFloatTensor_sizeDesc"
  c_THFloatTensor_sizeDesc :: (Ptr CTHFloatTensor) -> CTHDescBuff