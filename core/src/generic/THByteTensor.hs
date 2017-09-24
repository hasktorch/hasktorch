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
    c_THByteTensor_sizeDesc) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteTensor_storage : self -> THStorage *
foreign import ccall "THTensor.h THByteTensor_storage"
  c_THByteTensor_storage :: (Ptr CTHByteTensor) -> IO (Ptr CTHByteStorage)

-- |c_THByteTensor_storageOffset : self -> THStorage *
foreign import ccall "THTensor.h THByteTensor_storageOffset"
  c_THByteTensor_storageOffset :: (Ptr CTHByteTensor) -> IO (Ptr CTHByteStorage)

-- |c_THByteTensor_nDimension : self -> int
foreign import ccall "THTensor.h THByteTensor_nDimension"
  c_THByteTensor_nDimension :: (Ptr CTHByteTensor) -> CInt

-- |c_THByteTensor_size : self dim -> long
foreign import ccall "THTensor.h THByteTensor_size"
  c_THByteTensor_size :: (Ptr CTHByteTensor) -> CInt -> CLong

-- |c_THByteTensor_stride : self dim -> long
foreign import ccall "THTensor.h THByteTensor_stride"
  c_THByteTensor_stride :: (Ptr CTHByteTensor) -> CInt -> CLong

-- |c_THByteTensor_newSizeOf : self -> THLongStorage *
foreign import ccall "THTensor.h THByteTensor_newSizeOf"
  c_THByteTensor_newSizeOf :: (Ptr CTHByteTensor) -> IO (Ptr CTHLongStorage)

-- |c_THByteTensor_newStrideOf : self -> THLongStorage *
foreign import ccall "THTensor.h THByteTensor_newStrideOf"
  c_THByteTensor_newStrideOf :: (Ptr CTHByteTensor) -> IO (Ptr CTHLongStorage)

-- |c_THByteTensor_data : self -> real *
foreign import ccall "THTensor.h THByteTensor_data"
  c_THByteTensor_data :: (Ptr CTHByteTensor) -> IO (Ptr CChar)

-- |c_THByteTensor_setFlag : self flag -> void
foreign import ccall "THTensor.h THByteTensor_setFlag"
  c_THByteTensor_setFlag :: (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_clearFlag : self flag -> void
foreign import ccall "THTensor.h THByteTensor_clearFlag"
  c_THByteTensor_clearFlag :: (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_new :  -> THTensor *
foreign import ccall "THTensor.h THByteTensor_new"
  c_THByteTensor_new :: IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithTensor : tensor -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newWithTensor"
  c_THByteTensor_newWithTensor :: (Ptr CTHByteTensor) -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithStorage : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newWithStorage"
  c_THByteTensor_newWithStorage :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newWithStorage1d"
  c_THByteTensor_newWithStorage1d :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newWithStorage2d"
  c_THByteTensor_newWithStorage2d :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newWithStorage3d"
  c_THByteTensor_newWithStorage3d :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newWithStorage4d"
  c_THByteTensor_newWithStorage4d :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithSize : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newWithSize"
  c_THByteTensor_newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithSize1d : size0_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newWithSize1d"
  c_THByteTensor_newWithSize1d :: CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newWithSize2d"
  c_THByteTensor_newWithSize2d :: CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newWithSize3d"
  c_THByteTensor_newWithSize3d :: CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newWithSize4d"
  c_THByteTensor_newWithSize4d :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newClone : self -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newClone"
  c_THByteTensor_newClone :: (Ptr CTHByteTensor) -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newContiguous : tensor -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newContiguous"
  c_THByteTensor_newContiguous :: (Ptr CTHByteTensor) -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newSelect"
  c_THByteTensor_newSelect :: (Ptr CTHByteTensor) -> CInt -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newNarrow"
  c_THByteTensor_newNarrow :: (Ptr CTHByteTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newTranspose"
  c_THByteTensor_newTranspose :: (Ptr CTHByteTensor) -> CInt -> CInt -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newUnfold"
  c_THByteTensor_newUnfold :: (Ptr CTHByteTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newView : tensor size -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newView"
  c_THByteTensor_newView :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_newExpand : tensor size -> THTensor *
foreign import ccall "THTensor.h THByteTensor_newExpand"
  c_THByteTensor_newExpand :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor)

-- |c_THByteTensor_expand : r tensor size -> void
foreign import ccall "THTensor.h THByteTensor_expand"
  c_THByteTensor_expand :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensor_expandNd : rets ops count -> void
foreign import ccall "THTensor.h THByteTensor_expandNd"
  c_THByteTensor_expandNd :: Ptr (Ptr CTHByteTensor) -> Ptr (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_resize : tensor size stride -> void
foreign import ccall "THTensor.h THByteTensor_resize"
  c_THByteTensor_resize :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensor_resizeAs : tensor src -> void
foreign import ccall "THTensor.h THByteTensor_resizeAs"
  c_THByteTensor_resizeAs :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_resizeNd : tensor nDimension size stride -> void
foreign import ccall "THTensor.h THByteTensor_resizeNd"
  c_THByteTensor_resizeNd :: (Ptr CTHByteTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THByteTensor_resize1d : tensor size0_ -> void
foreign import ccall "THTensor.h THByteTensor_resize1d"
  c_THByteTensor_resize1d :: (Ptr CTHByteTensor) -> CLong -> IO ()

-- |c_THByteTensor_resize2d : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h THByteTensor_resize2d"
  c_THByteTensor_resize2d :: (Ptr CTHByteTensor) -> CLong -> CLong -> IO ()

-- |c_THByteTensor_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h THByteTensor_resize3d"
  c_THByteTensor_resize3d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h THByteTensor_resize4d"
  c_THByteTensor_resize4d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h THByteTensor_resize5d"
  c_THByteTensor_resize5d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_set : self src -> void
foreign import ccall "THTensor.h THByteTensor_set"
  c_THByteTensor_set :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h THByteTensor_setStorage"
  c_THByteTensor_setStorage :: (Ptr CTHByteTensor) -> Ptr CTHByteStorage -> Ptr CTHByteStorage -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensor_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h THByteTensor_setStorageNd"
  c_THByteTensor_setStorageNd :: (Ptr CTHByteTensor) -> Ptr CTHByteStorage -> Ptr CTHByteStorage -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THByteTensor_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h THByteTensor_setStorage1d"
  c_THByteTensor_setStorage1d :: (Ptr CTHByteTensor) -> Ptr CTHByteStorage -> Ptr CTHByteStorage -> CLong -> CLong -> IO ()

-- |c_THByteTensor_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h THByteTensor_setStorage2d"
  c_THByteTensor_setStorage2d :: (Ptr CTHByteTensor) -> Ptr CTHByteStorage -> Ptr CTHByteStorage -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h THByteTensor_setStorage3d"
  c_THByteTensor_setStorage3d :: (Ptr CTHByteTensor) -> Ptr CTHByteStorage -> Ptr CTHByteStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h THByteTensor_setStorage4d"
  c_THByteTensor_setStorage4d :: (Ptr CTHByteTensor) -> Ptr CTHByteStorage -> Ptr CTHByteStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h THByteTensor_narrow"
  c_THByteTensor_narrow :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THByteTensor_select : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h THByteTensor_select"
  c_THByteTensor_select :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CLong -> IO ()

-- |c_THByteTensor_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h THByteTensor_transpose"
  c_THByteTensor_transpose :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_unfold : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h THByteTensor_unfold"
  c_THByteTensor_unfold :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THByteTensor_squeeze : self src -> void
foreign import ccall "THTensor.h THByteTensor_squeeze"
  c_THByteTensor_squeeze :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_squeeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h THByteTensor_squeeze1d"
  c_THByteTensor_squeeze1d :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_unsqueeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h THByteTensor_unsqueeze1d"
  c_THByteTensor_unsqueeze1d :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_isContiguous : self -> int
foreign import ccall "THTensor.h THByteTensor_isContiguous"
  c_THByteTensor_isContiguous :: (Ptr CTHByteTensor) -> CInt

-- |c_THByteTensor_isSameSizeAs : self src -> int
foreign import ccall "THTensor.h THByteTensor_isSameSizeAs"
  c_THByteTensor_isSameSizeAs :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt

-- |c_THByteTensor_isSetTo : self src -> int
foreign import ccall "THTensor.h THByteTensor_isSetTo"
  c_THByteTensor_isSetTo :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt

-- |c_THByteTensor_isSize : self dims -> int
foreign import ccall "THTensor.h THByteTensor_isSize"
  c_THByteTensor_isSize :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> CInt

-- |c_THByteTensor_nElement : self -> THStorage *
foreign import ccall "THTensor.h THByteTensor_nElement"
  c_THByteTensor_nElement :: (Ptr CTHByteTensor) -> IO (Ptr CTHByteStorage)

-- |c_THByteTensor_retain : self -> void
foreign import ccall "THTensor.h THByteTensor_retain"
  c_THByteTensor_retain :: (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_free : self -> void
foreign import ccall "THTensor.h THByteTensor_free"
  c_THByteTensor_free :: (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_freeCopyTo : self dst -> void
foreign import ccall "THTensor.h THByteTensor_freeCopyTo"
  c_THByteTensor_freeCopyTo :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_set1d : tensor x0 value -> void
foreign import ccall "THTensor.h THByteTensor_set1d"
  c_THByteTensor_set1d :: (Ptr CTHByteTensor) -> CLong -> CChar -> IO ()

-- |c_THByteTensor_set2d : tensor x0 x1 value -> void
foreign import ccall "THTensor.h THByteTensor_set2d"
  c_THByteTensor_set2d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CChar -> IO ()

-- |c_THByteTensor_set3d : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h THByteTensor_set3d"
  c_THByteTensor_set3d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CChar -> IO ()

-- |c_THByteTensor_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h THByteTensor_set4d"
  c_THByteTensor_set4d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CLong -> CChar -> IO ()

-- |c_THByteTensor_get1d : tensor x0 -> real
foreign import ccall "THTensor.h THByteTensor_get1d"
  c_THByteTensor_get1d :: (Ptr CTHByteTensor) -> CLong -> CChar

-- |c_THByteTensor_get2d : tensor x0 x1 -> real
foreign import ccall "THTensor.h THByteTensor_get2d"
  c_THByteTensor_get2d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CChar

-- |c_THByteTensor_get3d : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h THByteTensor_get3d"
  c_THByteTensor_get3d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CChar

-- |c_THByteTensor_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h THByteTensor_get4d"
  c_THByteTensor_get4d :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> CLong -> CChar

-- |c_THByteTensor_desc : tensor -> THDescBuff
foreign import ccall "THTensor.h THByteTensor_desc"
  c_THByteTensor_desc :: (Ptr CTHByteTensor) -> CTHDescBuff

-- |c_THByteTensor_sizeDesc : tensor -> THDescBuff
foreign import ccall "THTensor.h THByteTensor_sizeDesc"
  c_THByteTensor_sizeDesc :: (Ptr CTHByteTensor) -> CTHDescBuff