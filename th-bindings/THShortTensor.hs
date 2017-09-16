{-# LANGUAGE ForeignFunctionInterface#-}

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
    c_THShortTensor_sizeDesc) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortTensor_storage : self -> THStorage *
foreign import ccall "THTensor.h THShortTensor_storage"
  c_THShortTensor_storage :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortStorage)

-- |c_THShortTensor_storageOffset : self -> THStorage *
foreign import ccall "THTensor.h THShortTensor_storageOffset"
  c_THShortTensor_storageOffset :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortStorage)

-- |c_THShortTensor_nDimension : self -> int
foreign import ccall "THTensor.h THShortTensor_nDimension"
  c_THShortTensor_nDimension :: (Ptr CTHShortTensor) -> CInt

-- |c_THShortTensor_size : self dim -> long
foreign import ccall "THTensor.h THShortTensor_size"
  c_THShortTensor_size :: (Ptr CTHShortTensor) -> CInt -> CLong

-- |c_THShortTensor_stride : self dim -> long
foreign import ccall "THTensor.h THShortTensor_stride"
  c_THShortTensor_stride :: (Ptr CTHShortTensor) -> CInt -> CLong

-- |c_THShortTensor_newSizeOf : self -> THLongStorage *
foreign import ccall "THTensor.h THShortTensor_newSizeOf"
  c_THShortTensor_newSizeOf :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortLongStorage)

-- |c_THShortTensor_newStrideOf : self -> THLongStorage *
foreign import ccall "THTensor.h THShortTensor_newStrideOf"
  c_THShortTensor_newStrideOf :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortLongStorage)

-- |c_THShortTensor_data : self -> real *
foreign import ccall "THTensor.h THShortTensor_data"
  c_THShortTensor_data :: (Ptr CTHShortTensor) -> IO (Ptr CShort)

-- |c_THShortTensor_setFlag : self flag -> void
foreign import ccall "THTensor.h THShortTensor_setFlag"
  c_THShortTensor_setFlag :: (Ptr CTHShortTensor) -> CChar -> IO ()

-- |c_THShortTensor_clearFlag : self flag -> void
foreign import ccall "THTensor.h THShortTensor_clearFlag"
  c_THShortTensor_clearFlag :: (Ptr CTHShortTensor) -> CChar -> IO ()

-- |c_THShortTensor_new :  -> THTensor *
foreign import ccall "THTensor.h THShortTensor_new"
  c_THShortTensor_new :: IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithTensor : tensor -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithTensor"
  c_THShortTensor_newWithTensor :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithStorage : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage"
  c_THShortTensor_newWithStorage :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> Ptr CTHShortLongStorage -> Ptr CTHShortLongStorage -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage1d"
  c_THShortTensor_newWithStorage1d :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage2d"
  c_THShortTensor_newWithStorage2d :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage3d"
  c_THShortTensor_newWithStorage3d :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage4d"
  c_THShortTensor_newWithStorage4d :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithSize : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize"
  c_THShortTensor_newWithSize :: Ptr CTHShortLongStorage -> Ptr CTHShortLongStorage -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithSize1d : size0_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize1d"
  c_THShortTensor_newWithSize1d :: CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize2d"
  c_THShortTensor_newWithSize2d :: CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize3d"
  c_THShortTensor_newWithSize3d :: CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize4d"
  c_THShortTensor_newWithSize4d :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newClone : self -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newClone"
  c_THShortTensor_newClone :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newContiguous : tensor -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newContiguous"
  c_THShortTensor_newContiguous :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newSelect"
  c_THShortTensor_newSelect :: (Ptr CTHShortTensor) -> CInt -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newNarrow"
  c_THShortTensor_newNarrow :: (Ptr CTHShortTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newTranspose"
  c_THShortTensor_newTranspose :: (Ptr CTHShortTensor) -> CInt -> CInt -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newUnfold"
  c_THShortTensor_newUnfold :: (Ptr CTHShortTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newView : tensor size -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newView"
  c_THShortTensor_newView :: (Ptr CTHShortTensor) -> Ptr CTHShortLongStorage -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_newExpand : tensor size -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newExpand"
  c_THShortTensor_newExpand :: (Ptr CTHShortTensor) -> Ptr CTHShortLongStorage -> IO (Ptr CTHShortTensor)

-- |c_THShortTensor_expand : r tensor size -> void
foreign import ccall "THTensor.h THShortTensor_expand"
  c_THShortTensor_expand :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CTHShortLongStorage -> IO ()

-- |c_THShortTensor_expandNd : rets ops count -> void
foreign import ccall "THTensor.h THShortTensor_expandNd"
  c_THShortTensor_expandNd :: Ptr (Ptr CTHShortTensor) -> Ptr (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensor_resize : tensor size stride -> void
foreign import ccall "THTensor.h THShortTensor_resize"
  c_THShortTensor_resize :: (Ptr CTHShortTensor) -> Ptr CTHShortLongStorage -> Ptr CTHShortLongStorage -> IO ()

-- |c_THShortTensor_resizeAs : tensor src -> void
foreign import ccall "THTensor.h THShortTensor_resizeAs"
  c_THShortTensor_resizeAs :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_resizeNd : tensor nDimension size stride -> void
foreign import ccall "THTensor.h THShortTensor_resizeNd"
  c_THShortTensor_resizeNd :: (Ptr CTHShortTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THShortTensor_resize1d : tensor size0_ -> void
foreign import ccall "THTensor.h THShortTensor_resize1d"
  c_THShortTensor_resize1d :: (Ptr CTHShortTensor) -> CLong -> IO ()

-- |c_THShortTensor_resize2d : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h THShortTensor_resize2d"
  c_THShortTensor_resize2d :: (Ptr CTHShortTensor) -> CLong -> CLong -> IO ()

-- |c_THShortTensor_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h THShortTensor_resize3d"
  c_THShortTensor_resize3d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h THShortTensor_resize4d"
  c_THShortTensor_resize4d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h THShortTensor_resize5d"
  c_THShortTensor_resize5d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_set : self src -> void
foreign import ccall "THTensor.h THShortTensor_set"
  c_THShortTensor_set :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage"
  c_THShortTensor_setStorage :: (Ptr CTHShortTensor) -> Ptr CTHShortStorage -> Ptr CTHShortStorage -> Ptr CTHShortLongStorage -> Ptr CTHShortLongStorage -> IO ()

-- |c_THShortTensor_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h THShortTensor_setStorageNd"
  c_THShortTensor_setStorageNd :: (Ptr CTHShortTensor) -> Ptr CTHShortStorage -> Ptr CTHShortStorage -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THShortTensor_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage1d"
  c_THShortTensor_setStorage1d :: (Ptr CTHShortTensor) -> Ptr CTHShortStorage -> Ptr CTHShortStorage -> CLong -> CLong -> IO ()

-- |c_THShortTensor_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage2d"
  c_THShortTensor_setStorage2d :: (Ptr CTHShortTensor) -> Ptr CTHShortStorage -> Ptr CTHShortStorage -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage3d"
  c_THShortTensor_setStorage3d :: (Ptr CTHShortTensor) -> Ptr CTHShortStorage -> Ptr CTHShortStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage4d"
  c_THShortTensor_setStorage4d :: (Ptr CTHShortTensor) -> Ptr CTHShortStorage -> Ptr CTHShortStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h THShortTensor_narrow"
  c_THShortTensor_narrow :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THShortTensor_select : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h THShortTensor_select"
  c_THShortTensor_select :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CLong -> IO ()

-- |c_THShortTensor_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h THShortTensor_transpose"
  c_THShortTensor_transpose :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensor_unfold : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h THShortTensor_unfold"
  c_THShortTensor_unfold :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THShortTensor_squeeze : self src -> void
foreign import ccall "THTensor.h THShortTensor_squeeze"
  c_THShortTensor_squeeze :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_squeeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h THShortTensor_squeeze1d"
  c_THShortTensor_squeeze1d :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensor_unsqueeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h THShortTensor_unsqueeze1d"
  c_THShortTensor_unsqueeze1d :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensor_isContiguous : self -> int
foreign import ccall "THTensor.h THShortTensor_isContiguous"
  c_THShortTensor_isContiguous :: (Ptr CTHShortTensor) -> CInt

-- |c_THShortTensor_isSameSizeAs : self src -> int
foreign import ccall "THTensor.h THShortTensor_isSameSizeAs"
  c_THShortTensor_isSameSizeAs :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt

-- |c_THShortTensor_isSetTo : self src -> int
foreign import ccall "THTensor.h THShortTensor_isSetTo"
  c_THShortTensor_isSetTo :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt

-- |c_THShortTensor_isSize : self dims -> int
foreign import ccall "THTensor.h THShortTensor_isSize"
  c_THShortTensor_isSize :: (Ptr CTHShortTensor) -> Ptr CTHShortLongStorage -> CInt

-- |c_THShortTensor_nElement : self -> THStorage *
foreign import ccall "THTensor.h THShortTensor_nElement"
  c_THShortTensor_nElement :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortStorage)

-- |c_THShortTensor_retain : self -> void
foreign import ccall "THTensor.h THShortTensor_retain"
  c_THShortTensor_retain :: (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_free : self -> void
foreign import ccall "THTensor.h THShortTensor_free"
  c_THShortTensor_free :: (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_freeCopyTo : self dst -> void
foreign import ccall "THTensor.h THShortTensor_freeCopyTo"
  c_THShortTensor_freeCopyTo :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_set1d : tensor x0 value -> void
foreign import ccall "THTensor.h THShortTensor_set1d"
  c_THShortTensor_set1d :: (Ptr CTHShortTensor) -> CLong -> CShort -> IO ()

-- |c_THShortTensor_set2d : tensor x0 x1 value -> void
foreign import ccall "THTensor.h THShortTensor_set2d"
  c_THShortTensor_set2d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CShort -> IO ()

-- |c_THShortTensor_set3d : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h THShortTensor_set3d"
  c_THShortTensor_set3d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CShort -> IO ()

-- |c_THShortTensor_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h THShortTensor_set4d"
  c_THShortTensor_set4d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CLong -> CShort -> IO ()

-- |c_THShortTensor_get1d : tensor x0 -> real
foreign import ccall "THTensor.h THShortTensor_get1d"
  c_THShortTensor_get1d :: (Ptr CTHShortTensor) -> CLong -> CShort

-- |c_THShortTensor_get2d : tensor x0 x1 -> real
foreign import ccall "THTensor.h THShortTensor_get2d"
  c_THShortTensor_get2d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CShort

-- |c_THShortTensor_get3d : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h THShortTensor_get3d"
  c_THShortTensor_get3d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CShort

-- |c_THShortTensor_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h THShortTensor_get4d"
  c_THShortTensor_get4d :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> CLong -> CShort

-- |c_THShortTensor_desc : tensor -> THDescBuff
foreign import ccall "THTensor.h THShortTensor_desc"
  c_THShortTensor_desc :: (Ptr CTHShortTensor) -> CTHDescBuff

-- |c_THShortTensor_sizeDesc : tensor -> THDescBuff
foreign import ccall "THTensor.h THShortTensor_sizeDesc"
  c_THShortTensor_sizeDesc :: (Ptr CTHShortTensor) -> CTHDescBuff