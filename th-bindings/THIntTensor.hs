{-# LANGUAGE ForeignFunctionInterface#-}

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
    c_THIntTensor_sizeDesc) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntTensor_storage : self -> THStorage *
foreign import ccall "THTensor.h THIntTensor_storage"
  c_THIntTensor_storage :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntStorage)

-- |c_THIntTensor_storageOffset : self -> THStorage *
foreign import ccall "THTensor.h THIntTensor_storageOffset"
  c_THIntTensor_storageOffset :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntStorage)

-- |c_THIntTensor_nDimension : self -> int
foreign import ccall "THTensor.h THIntTensor_nDimension"
  c_THIntTensor_nDimension :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_size : self dim -> long
foreign import ccall "THTensor.h THIntTensor_size"
  c_THIntTensor_size :: (Ptr CTHIntTensor) -> CInt -> CLong

-- |c_THIntTensor_stride : self dim -> long
foreign import ccall "THTensor.h THIntTensor_stride"
  c_THIntTensor_stride :: (Ptr CTHIntTensor) -> CInt -> CLong

-- |c_THIntTensor_newSizeOf : self -> THLongStorage *
foreign import ccall "THTensor.h THIntTensor_newSizeOf"
  c_THIntTensor_newSizeOf :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntLongStorage)

-- |c_THIntTensor_newStrideOf : self -> THLongStorage *
foreign import ccall "THTensor.h THIntTensor_newStrideOf"
  c_THIntTensor_newStrideOf :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntLongStorage)

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
  c_THIntTensor_newWithStorage :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> Ptr CTHIntLongStorage -> Ptr CTHIntLongStorage -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage1d"
  c_THIntTensor_newWithStorage1d :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> CLong -> CLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage2d"
  c_THIntTensor_newWithStorage2d :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage3d"
  c_THIntTensor_newWithStorage3d :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage4d"
  c_THIntTensor_newWithStorage4d :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithSize : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize"
  c_THIntTensor_newWithSize :: Ptr CTHIntLongStorage -> Ptr CTHIntLongStorage -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithSize1d : size0_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize1d"
  c_THIntTensor_newWithSize1d :: CLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize2d"
  c_THIntTensor_newWithSize2d :: CLong -> CLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize3d"
  c_THIntTensor_newWithSize3d :: CLong -> CLong -> CLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize4d"
  c_THIntTensor_newWithSize4d :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newClone : self -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newClone"
  c_THIntTensor_newClone :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newContiguous : tensor -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newContiguous"
  c_THIntTensor_newContiguous :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newSelect"
  c_THIntTensor_newSelect :: (Ptr CTHIntTensor) -> CInt -> CLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newNarrow"
  c_THIntTensor_newNarrow :: (Ptr CTHIntTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newTranspose"
  c_THIntTensor_newTranspose :: (Ptr CTHIntTensor) -> CInt -> CInt -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newUnfold"
  c_THIntTensor_newUnfold :: (Ptr CTHIntTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newView : tensor size -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newView"
  c_THIntTensor_newView :: (Ptr CTHIntTensor) -> Ptr CTHIntLongStorage -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_newExpand : tensor size -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newExpand"
  c_THIntTensor_newExpand :: (Ptr CTHIntTensor) -> Ptr CTHIntLongStorage -> IO (Ptr CTHIntTensor)

-- |c_THIntTensor_expand : r tensor size -> void
foreign import ccall "THTensor.h THIntTensor_expand"
  c_THIntTensor_expand :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHIntLongStorage -> IO ()

-- |c_THIntTensor_expandNd : rets ops count -> void
foreign import ccall "THTensor.h THIntTensor_expandNd"
  c_THIntTensor_expandNd :: Ptr (Ptr CTHIntTensor) -> Ptr (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_resize : tensor size stride -> void
foreign import ccall "THTensor.h THIntTensor_resize"
  c_THIntTensor_resize :: (Ptr CTHIntTensor) -> Ptr CTHIntLongStorage -> Ptr CTHIntLongStorage -> IO ()

-- |c_THIntTensor_resizeAs : tensor src -> void
foreign import ccall "THTensor.h THIntTensor_resizeAs"
  c_THIntTensor_resizeAs :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_resizeNd : tensor nDimension size stride -> void
foreign import ccall "THTensor.h THIntTensor_resizeNd"
  c_THIntTensor_resizeNd :: (Ptr CTHIntTensor) -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THIntTensor_resize1d : tensor size0_ -> void
foreign import ccall "THTensor.h THIntTensor_resize1d"
  c_THIntTensor_resize1d :: (Ptr CTHIntTensor) -> CLong -> IO ()

-- |c_THIntTensor_resize2d : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h THIntTensor_resize2d"
  c_THIntTensor_resize2d :: (Ptr CTHIntTensor) -> CLong -> CLong -> IO ()

-- |c_THIntTensor_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h THIntTensor_resize3d"
  c_THIntTensor_resize3d :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h THIntTensor_resize4d"
  c_THIntTensor_resize4d :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h THIntTensor_resize5d"
  c_THIntTensor_resize5d :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_set : self src -> void
foreign import ccall "THTensor.h THIntTensor_set"
  c_THIntTensor_set :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage"
  c_THIntTensor_setStorage :: (Ptr CTHIntTensor) -> Ptr CTHIntStorage -> Ptr CTHIntStorage -> Ptr CTHIntLongStorage -> Ptr CTHIntLongStorage -> IO ()

-- |c_THIntTensor_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h THIntTensor_setStorageNd"
  c_THIntTensor_setStorageNd :: (Ptr CTHIntTensor) -> Ptr CTHIntStorage -> Ptr CTHIntStorage -> CInt -> Ptr CLong -> Ptr CLong -> IO ()

-- |c_THIntTensor_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage1d"
  c_THIntTensor_setStorage1d :: (Ptr CTHIntTensor) -> Ptr CTHIntStorage -> Ptr CTHIntStorage -> CLong -> CLong -> IO ()

-- |c_THIntTensor_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage2d"
  c_THIntTensor_setStorage2d :: (Ptr CTHIntTensor) -> Ptr CTHIntStorage -> Ptr CTHIntStorage -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage3d"
  c_THIntTensor_setStorage3d :: (Ptr CTHIntTensor) -> Ptr CTHIntStorage -> Ptr CTHIntStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage4d"
  c_THIntTensor_setStorage4d :: (Ptr CTHIntTensor) -> Ptr CTHIntStorage -> Ptr CTHIntStorage -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h THIntTensor_narrow"
  c_THIntTensor_narrow :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CLong -> CLong -> IO ()

-- |c_THIntTensor_select : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h THIntTensor_select"
  c_THIntTensor_select :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CLong -> IO ()

-- |c_THIntTensor_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h THIntTensor_transpose"
  c_THIntTensor_transpose :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_unfold : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h THIntTensor_unfold"
  c_THIntTensor_unfold :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CLong -> CLong -> IO ()

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
  c_THIntTensor_isSize :: (Ptr CTHIntTensor) -> Ptr CTHIntLongStorage -> CInt

-- |c_THIntTensor_nElement : self -> THStorage *
foreign import ccall "THTensor.h THIntTensor_nElement"
  c_THIntTensor_nElement :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntStorage)

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
  c_THIntTensor_set1d :: (Ptr CTHIntTensor) -> CLong -> CInt -> IO ()

-- |c_THIntTensor_set2d : tensor x0 x1 value -> void
foreign import ccall "THTensor.h THIntTensor_set2d"
  c_THIntTensor_set2d :: (Ptr CTHIntTensor) -> CLong -> CLong -> CInt -> IO ()

-- |c_THIntTensor_set3d : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h THIntTensor_set3d"
  c_THIntTensor_set3d :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> CInt -> IO ()

-- |c_THIntTensor_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h THIntTensor_set4d"
  c_THIntTensor_set4d :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> CLong -> CInt -> IO ()

-- |c_THIntTensor_get1d : tensor x0 -> real
foreign import ccall "THTensor.h THIntTensor_get1d"
  c_THIntTensor_get1d :: (Ptr CTHIntTensor) -> CLong -> CInt

-- |c_THIntTensor_get2d : tensor x0 x1 -> real
foreign import ccall "THTensor.h THIntTensor_get2d"
  c_THIntTensor_get2d :: (Ptr CTHIntTensor) -> CLong -> CLong -> CInt

-- |c_THIntTensor_get3d : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h THIntTensor_get3d"
  c_THIntTensor_get3d :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> CInt

-- |c_THIntTensor_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h THIntTensor_get4d"
  c_THIntTensor_get4d :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> CLong -> CInt

-- |c_THIntTensor_desc : tensor -> THDescBuff
foreign import ccall "THTensor.h THIntTensor_desc"
  c_THIntTensor_desc :: (Ptr CTHIntTensor) -> CTHDescBuff

-- |c_THIntTensor_sizeDesc : tensor -> THDescBuff
foreign import ccall "THTensor.h THIntTensor_sizeDesc"
  c_THIntTensor_sizeDesc :: (Ptr CTHIntTensor) -> CTHDescBuff