{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Half.Tensor
  ( c_storage
  , c_storageOffset
  , c_nDimension
  , c_size
  , c_stride
  , c_newSizeOf
  , c_newStrideOf
  , c_data
  , c_setFlag
  , c_clearFlag
  , c_new
  , c_newWithTensor
  , c_newWithStorage
  , c_newWithStorage1d
  , c_newWithStorage2d
  , c_newWithStorage3d
  , c_newWithStorage4d
  , c_newWithSize
  , c_newWithSize1d
  , c_newWithSize2d
  , c_newWithSize3d
  , c_newWithSize4d
  , c_newClone
  , c_newContiguous
  , c_newSelect
  , c_newNarrow
  , c_newTranspose
  , c_newUnfold
  , c_newView
  , c_newExpand
  , c_expand
  , c_expandNd
  , c_resize
  , c_resizeAs
  , c_resize1d
  , c_resize2d
  , c_resize3d
  , c_resize4d
  , c_resize5d
  , c_resizeNd
  , c_set
  , c_setStorage
  , c_setStorageNd
  , c_setStorage1d
  , c_setStorage2d
  , c_setStorage3d
  , c_setStorage4d
  , c_narrow
  , c_select
  , c_transpose
  , c_unfold
  , c_squeeze
  , c_squeeze1d
  , c_unsqueeze1d
  , c_isContiguous
  , c_isSameSizeAs
  , c_isSetTo
  , c_isSize
  , c_nElement
  , c_retain
  , c_free
  , c_freeCopyTo
  , c_set1d
  , c_set2d
  , c_set3d
  , c_set4d
  , c_get1d
  , c_get2d
  , c_get3d
  , c_get4d
  , c_getDevice
  , c_sizeDesc
  , p_storage
  , p_storageOffset
  , p_nDimension
  , p_size
  , p_stride
  , p_newSizeOf
  , p_newStrideOf
  , p_data
  , p_setFlag
  , p_clearFlag
  , p_new
  , p_newWithTensor
  , p_newWithStorage
  , p_newWithStorage1d
  , p_newWithStorage2d
  , p_newWithStorage3d
  , p_newWithStorage4d
  , p_newWithSize
  , p_newWithSize1d
  , p_newWithSize2d
  , p_newWithSize3d
  , p_newWithSize4d
  , p_newClone
  , p_newContiguous
  , p_newSelect
  , p_newNarrow
  , p_newTranspose
  , p_newUnfold
  , p_newView
  , p_newExpand
  , p_expand
  , p_expandNd
  , p_resize
  , p_resizeAs
  , p_resize1d
  , p_resize2d
  , p_resize3d
  , p_resize4d
  , p_resize5d
  , p_resizeNd
  , p_set
  , p_setStorage
  , p_setStorageNd
  , p_setStorage1d
  , p_setStorage2d
  , p_setStorage3d
  , p_setStorage4d
  , p_narrow
  , p_select
  , p_transpose
  , p_unfold
  , p_squeeze
  , p_squeeze1d
  , p_unsqueeze1d
  , p_isContiguous
  , p_isSameSizeAs
  , p_isSetTo
  , p_isSize
  , p_nElement
  , p_retain
  , p_free
  , p_freeCopyTo
  , p_set1d
  , p_set2d
  , p_set3d
  , p_set4d
  , p_get1d
  , p_get2d
  , p_get3d
  , p_get4d
  , p_getDevice
  , p_sizeDesc
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_storage :  state self -> THCStorage *
foreign import ccall "THCTensor.h THCHalfTensor_storage"
  c_storage :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHCudaHalfStorage)

-- | c_storageOffset :  state self -> ptrdiff_t
foreign import ccall "THCTensor.h THCHalfTensor_storageOffset"
  c_storageOffset :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CPtrdiff

-- | c_nDimension :  state self -> int
foreign import ccall "THCTensor.h THCHalfTensor_nDimension"
  c_nDimension :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CInt

-- | c_size :  state self dim -> int64_t
foreign import ccall "THCTensor.h THCHalfTensor_size"
  c_size :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> IO CLLong

-- | c_stride :  state self dim -> int64_t
foreign import ccall "THCTensor.h THCHalfTensor_stride"
  c_stride :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> IO CLLong

-- | c_newSizeOf :  state self -> THCLongStorage *
foreign import ccall "THCTensor.h THCHalfTensor_newSizeOf"
  c_newSizeOf :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHCudaLongStorage)

-- | c_newStrideOf :  state self -> THCLongStorage *
foreign import ccall "THCTensor.h THCHalfTensor_newStrideOf"
  c_newStrideOf :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHCudaLongStorage)

-- | c_data :  state self -> real *
foreign import ccall "THCTensor.h THCHalfTensor_data"
  c_data :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHHalf)

-- | c_setFlag :  state self flag -> void
foreign import ccall "THCTensor.h THCHalfTensor_setFlag"
  c_setFlag :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CChar -> IO ()

-- | c_clearFlag :  state self flag -> void
foreign import ccall "THCTensor.h THCHalfTensor_clearFlag"
  c_clearFlag :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CChar -> IO ()

-- | c_new :  state -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_new"
  c_new :: Ptr CTHCudaState -> IO (Ptr CTHCudaHalfTensor)

-- | c_newWithTensor :  state tensor -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newWithTensor"
  c_newWithTensor :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHCudaHalfTensor)

-- | c_newWithStorage :  state storage_ storageOffset_ size_ stride_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newWithStorage"
  c_newWithStorage :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO (Ptr CTHCudaHalfTensor)

-- | c_newWithStorage1d :  state storage_ storageOffset_ size0_ stride0_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newWithStorage1d"
  c_newWithStorage1d :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor)

-- | c_newWithStorage2d :  state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newWithStorage2d"
  c_newWithStorage2d :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor)

-- | c_newWithStorage3d :  state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newWithStorage3d"
  c_newWithStorage3d :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor)

-- | c_newWithStorage4d :  state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newWithStorage4d"
  c_newWithStorage4d :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor)

-- | c_newWithSize :  state size_ stride_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newWithSize"
  c_newWithSize :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO (Ptr CTHCudaHalfTensor)

-- | c_newWithSize1d :  state size0_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newWithSize1d"
  c_newWithSize1d :: Ptr CTHCudaState -> CLLong -> IO (Ptr CTHCudaHalfTensor)

-- | c_newWithSize2d :  state size0_ size1_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newWithSize2d"
  c_newWithSize2d :: Ptr CTHCudaState -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor)

-- | c_newWithSize3d :  state size0_ size1_ size2_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newWithSize3d"
  c_newWithSize3d :: Ptr CTHCudaState -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor)

-- | c_newWithSize4d :  state size0_ size1_ size2_ size3_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newWithSize4d"
  c_newWithSize4d :: Ptr CTHCudaState -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor)

-- | c_newClone :  state self -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newClone"
  c_newClone :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHCudaHalfTensor)

-- | c_newContiguous :  state tensor -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newContiguous"
  c_newContiguous :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHCudaHalfTensor)

-- | c_newSelect :  state tensor dimension_ sliceIndex_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newSelect"
  c_newSelect :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> CLLong -> IO (Ptr CTHCudaHalfTensor)

-- | c_newNarrow :  state tensor dimension_ firstIndex_ size_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newNarrow"
  c_newNarrow :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor)

-- | c_newTranspose :  state tensor dimension1_ dimension2_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newTranspose"
  c_newTranspose :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO (Ptr CTHCudaHalfTensor)

-- | c_newUnfold :  state tensor dimension_ size_ step_ -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newUnfold"
  c_newUnfold :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor)

-- | c_newView :  state tensor size -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newView"
  c_newView :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongStorage -> IO (Ptr CTHCudaHalfTensor)

-- | c_newExpand :  state tensor size -> THCTensor *
foreign import ccall "THCTensor.h THCHalfTensor_newExpand"
  c_newExpand :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongStorage -> IO (Ptr CTHCudaHalfTensor)

-- | c_expand :  state r tensor sizes -> void
foreign import ccall "THCTensor.h THCHalfTensor_expand"
  c_expand :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongStorage -> IO ()

-- | c_expandNd :  state rets ops count -> void
foreign import ccall "THCTensor.h THCHalfTensor_expandNd"
  c_expandNd :: Ptr CTHCudaState -> Ptr (Ptr CTHCudaHalfTensor) -> Ptr (Ptr CTHCudaHalfTensor) -> CInt -> IO ()

-- | c_resize :  state tensor size stride -> void
foreign import ccall "THCTensor.h THCHalfTensor_resize"
  c_resize :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO ()

-- | c_resizeAs :  state tensor src -> void
foreign import ccall "THCTensor.h THCHalfTensor_resizeAs"
  c_resizeAs :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_resize1d :  state tensor size0_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_resize1d"
  c_resize1d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> IO ()

-- | c_resize2d :  state tensor size0_ size1_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_resize2d"
  c_resize2d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> IO ()

-- | c_resize3d :  state tensor size0_ size1_ size2_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_resize3d"
  c_resize3d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_resize4d :  state tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_resize4d"
  c_resize4d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_resize5d :  state tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_resize5d"
  c_resize5d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_resizeNd :  state tensor nDimension size stride -> void
foreign import ccall "THCTensor.h THCHalfTensor_resizeNd"
  c_resizeNd :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- | c_set :  state self src -> void
foreign import ccall "THCTensor.h THCHalfTensor_set"
  c_set :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_setStorage :  state self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_setStorage"
  c_setStorage :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfStorage -> CPtrdiff -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO ()

-- | c_setStorageNd :  state self storage storageOffset nDimension size stride -> void
foreign import ccall "THCTensor.h THCHalfTensor_setStorageNd"
  c_setStorageNd :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- | c_setStorage1d :  state self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_setStorage1d"
  c_setStorage1d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> IO ()

-- | c_setStorage2d :  state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_setStorage2d"
  c_setStorage2d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_setStorage3d :  state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_setStorage3d"
  c_setStorage3d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_setStorage4d :  state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_setStorage4d"
  c_setStorage4d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_narrow :  state self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_narrow"
  c_narrow :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> CLLong -> CLLong -> IO ()

-- | c_select :  state self src dimension_ sliceIndex_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_select"
  c_select :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> CLLong -> IO ()

-- | c_transpose :  state self src dimension1_ dimension2_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_transpose"
  c_transpose :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ()

-- | c_unfold :  state self src dimension_ size_ step_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_unfold"
  c_unfold :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> CLLong -> CLLong -> IO ()

-- | c_squeeze :  state self src -> void
foreign import ccall "THCTensor.h THCHalfTensor_squeeze"
  c_squeeze :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_squeeze1d :  state self src dimension_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_squeeze1d"
  c_squeeze1d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> IO ()

-- | c_unsqueeze1d :  state self src dimension_ -> void
foreign import ccall "THCTensor.h THCHalfTensor_unsqueeze1d"
  c_unsqueeze1d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> IO ()

-- | c_isContiguous :  state self -> int
foreign import ccall "THCTensor.h THCHalfTensor_isContiguous"
  c_isContiguous :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CInt

-- | c_isSameSizeAs :  state self src -> int
foreign import ccall "THCTensor.h THCHalfTensor_isSameSizeAs"
  c_isSameSizeAs :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO CInt

-- | c_isSetTo :  state self src -> int
foreign import ccall "THCTensor.h THCHalfTensor_isSetTo"
  c_isSetTo :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO CInt

-- | c_isSize :  state self dims -> int
foreign import ccall "THCTensor.h THCHalfTensor_isSize"
  c_isSize :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongStorage -> IO CInt

-- | c_nElement :  state self -> ptrdiff_t
foreign import ccall "THCTensor.h THCHalfTensor_nElement"
  c_nElement :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CPtrdiff

-- | c_retain :  state self -> void
foreign import ccall "THCTensor.h THCHalfTensor_retain"
  c_retain :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_free :  state self -> void
foreign import ccall "THCTensor.h THCHalfTensor_free"
  c_free :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_freeCopyTo :  state self dst -> void
foreign import ccall "THCTensor.h THCHalfTensor_freeCopyTo"
  c_freeCopyTo :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_set1d :  state tensor x0 value -> void
foreign import ccall "THCTensor.h THCHalfTensor_set1d"
  c_set1d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CTHHalf -> IO ()

-- | c_set2d :  state tensor x0 x1 value -> void
foreign import ccall "THCTensor.h THCHalfTensor_set2d"
  c_set2d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CTHHalf -> IO ()

-- | c_set3d :  state tensor x0 x1 x2 value -> void
foreign import ccall "THCTensor.h THCHalfTensor_set3d"
  c_set3d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> CTHHalf -> IO ()

-- | c_set4d :  state tensor x0 x1 x2 x3 value -> void
foreign import ccall "THCTensor.h THCHalfTensor_set4d"
  c_set4d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CTHHalf -> IO ()

-- | c_get1d :  state tensor x0 -> real
foreign import ccall "THCTensor.h THCHalfTensor_get1d"
  c_get1d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> IO CTHHalf

-- | c_get2d :  state tensor x0 x1 -> real
foreign import ccall "THCTensor.h THCHalfTensor_get2d"
  c_get2d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> IO CTHHalf

-- | c_get3d :  state tensor x0 x1 x2 -> real
foreign import ccall "THCTensor.h THCHalfTensor_get3d"
  c_get3d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> IO CTHHalf

-- | c_get4d :  state tensor x0 x1 x2 x3 -> real
foreign import ccall "THCTensor.h THCHalfTensor_get4d"
  c_get4d :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO CTHHalf

-- | c_getDevice :  state self -> int
foreign import ccall "THCTensor.h THCHalfTensor_getDevice"
  c_getDevice :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CInt

-- | c_sizeDesc :  state tensor -> THCDescBuff
foreign import ccall "THCTensor.h THCHalfTensor_sizeDesc"
  c_sizeDesc :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CTHCudaDescBuff

-- | p_storage : Pointer to function : state self -> THCStorage *
foreign import ccall "THCTensor.h &THCHalfTensor_storage"
  p_storage :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHCudaHalfStorage))

-- | p_storageOffset : Pointer to function : state self -> ptrdiff_t
foreign import ccall "THCTensor.h &THCHalfTensor_storageOffset"
  p_storageOffset :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CPtrdiff)

-- | p_nDimension : Pointer to function : state self -> int
foreign import ccall "THCTensor.h &THCHalfTensor_nDimension"
  p_nDimension :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CInt)

-- | p_size : Pointer to function : state self dim -> int64_t
foreign import ccall "THCTensor.h &THCHalfTensor_size"
  p_size :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> IO CLLong)

-- | p_stride : Pointer to function : state self dim -> int64_t
foreign import ccall "THCTensor.h &THCHalfTensor_stride"
  p_stride :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> IO CLLong)

-- | p_newSizeOf : Pointer to function : state self -> THCLongStorage *
foreign import ccall "THCTensor.h &THCHalfTensor_newSizeOf"
  p_newSizeOf :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHCudaLongStorage))

-- | p_newStrideOf : Pointer to function : state self -> THCLongStorage *
foreign import ccall "THCTensor.h &THCHalfTensor_newStrideOf"
  p_newStrideOf :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHCudaLongStorage))

-- | p_data : Pointer to function : state self -> real *
foreign import ccall "THCTensor.h &THCHalfTensor_data"
  p_data :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHHalf))

-- | p_setFlag : Pointer to function : state self flag -> void
foreign import ccall "THCTensor.h &THCHalfTensor_setFlag"
  p_setFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state self flag -> void
foreign import ccall "THCTensor.h &THCHalfTensor_clearFlag"
  p_clearFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CChar -> IO ())

-- | p_new : Pointer to function : state -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_new"
  p_new :: FunPtr (Ptr CTHCudaState -> IO (Ptr CTHCudaHalfTensor))

-- | p_newWithTensor : Pointer to function : state tensor -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newWithTensor"
  p_newWithTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHCudaHalfTensor))

-- | p_newWithStorage : Pointer to function : state storage_ storageOffset_ size_ stride_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newWithStorage"
  p_newWithStorage :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO (Ptr CTHCudaHalfTensor))

-- | p_newWithStorage1d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newWithStorage1d"
  p_newWithStorage1d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor))

-- | p_newWithStorage2d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newWithStorage2d"
  p_newWithStorage2d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor))

-- | p_newWithStorage3d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newWithStorage3d"
  p_newWithStorage3d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor))

-- | p_newWithStorage4d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newWithStorage4d"
  p_newWithStorage4d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor))

-- | p_newWithSize : Pointer to function : state size_ stride_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newWithSize"
  p_newWithSize :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO (Ptr CTHCudaHalfTensor))

-- | p_newWithSize1d : Pointer to function : state size0_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newWithSize1d"
  p_newWithSize1d :: FunPtr (Ptr CTHCudaState -> CLLong -> IO (Ptr CTHCudaHalfTensor))

-- | p_newWithSize2d : Pointer to function : state size0_ size1_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newWithSize2d"
  p_newWithSize2d :: FunPtr (Ptr CTHCudaState -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor))

-- | p_newWithSize3d : Pointer to function : state size0_ size1_ size2_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newWithSize3d"
  p_newWithSize3d :: FunPtr (Ptr CTHCudaState -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor))

-- | p_newWithSize4d : Pointer to function : state size0_ size1_ size2_ size3_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newWithSize4d"
  p_newWithSize4d :: FunPtr (Ptr CTHCudaState -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor))

-- | p_newClone : Pointer to function : state self -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newClone"
  p_newClone :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHCudaHalfTensor))

-- | p_newContiguous : Pointer to function : state tensor -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newContiguous"
  p_newContiguous :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO (Ptr CTHCudaHalfTensor))

-- | p_newSelect : Pointer to function : state tensor dimension_ sliceIndex_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newSelect"
  p_newSelect :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> CLLong -> IO (Ptr CTHCudaHalfTensor))

-- | p_newNarrow : Pointer to function : state tensor dimension_ firstIndex_ size_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newNarrow"
  p_newNarrow :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor))

-- | p_newTranspose : Pointer to function : state tensor dimension1_ dimension2_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newTranspose"
  p_newTranspose :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO (Ptr CTHCudaHalfTensor))

-- | p_newUnfold : Pointer to function : state tensor dimension_ size_ step_ -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newUnfold"
  p_newUnfold :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> CLLong -> CLLong -> IO (Ptr CTHCudaHalfTensor))

-- | p_newView : Pointer to function : state tensor size -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newView"
  p_newView :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongStorage -> IO (Ptr CTHCudaHalfTensor))

-- | p_newExpand : Pointer to function : state tensor size -> THCTensor *
foreign import ccall "THCTensor.h &THCHalfTensor_newExpand"
  p_newExpand :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongStorage -> IO (Ptr CTHCudaHalfTensor))

-- | p_expand : Pointer to function : state r tensor sizes -> void
foreign import ccall "THCTensor.h &THCHalfTensor_expand"
  p_expand :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongStorage -> IO ())

-- | p_expandNd : Pointer to function : state rets ops count -> void
foreign import ccall "THCTensor.h &THCHalfTensor_expandNd"
  p_expandNd :: FunPtr (Ptr CTHCudaState -> Ptr (Ptr CTHCudaHalfTensor) -> Ptr (Ptr CTHCudaHalfTensor) -> CInt -> IO ())

-- | p_resize : Pointer to function : state tensor size stride -> void
foreign import ccall "THCTensor.h &THCHalfTensor_resize"
  p_resize :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO ())

-- | p_resizeAs : Pointer to function : state tensor src -> void
foreign import ccall "THCTensor.h &THCHalfTensor_resizeAs"
  p_resizeAs :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_resize1d : Pointer to function : state tensor size0_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_resize1d"
  p_resize1d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> IO ())

-- | p_resize2d : Pointer to function : state tensor size0_ size1_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_resize2d"
  p_resize2d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> IO ())

-- | p_resize3d : Pointer to function : state tensor size0_ size1_ size2_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_resize3d"
  p_resize3d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_resize4d : Pointer to function : state tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_resize4d"
  p_resize4d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_resize5d : Pointer to function : state tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_resize5d"
  p_resize5d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_resizeNd : Pointer to function : state tensor nDimension size stride -> void
foreign import ccall "THCTensor.h &THCHalfTensor_resizeNd"
  p_resizeNd :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- | p_set : Pointer to function : state self src -> void
foreign import ccall "THCTensor.h &THCHalfTensor_set"
  p_set :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_setStorage : Pointer to function : state self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_setStorage"
  p_setStorage :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfStorage -> CPtrdiff -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO ())

-- | p_setStorageNd : Pointer to function : state self storage storageOffset nDimension size stride -> void
foreign import ccall "THCTensor.h &THCHalfTensor_setStorageNd"
  p_setStorageNd :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- | p_setStorage1d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_setStorage1d"
  p_setStorage1d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> IO ())

-- | p_setStorage2d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_setStorage2d"
  p_setStorage2d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_setStorage3d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_setStorage3d"
  p_setStorage3d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_setStorage4d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_setStorage4d"
  p_setStorage4d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_narrow : Pointer to function : state self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_narrow"
  p_narrow :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> CLLong -> CLLong -> IO ())

-- | p_select : Pointer to function : state self src dimension_ sliceIndex_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_select"
  p_select :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> CLLong -> IO ())

-- | p_transpose : Pointer to function : state self src dimension1_ dimension2_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_transpose"
  p_transpose :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ())

-- | p_unfold : Pointer to function : state self src dimension_ size_ step_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_unfold"
  p_unfold :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> CLLong -> CLLong -> IO ())

-- | p_squeeze : Pointer to function : state self src -> void
foreign import ccall "THCTensor.h &THCHalfTensor_squeeze"
  p_squeeze :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_squeeze1d : Pointer to function : state self src dimension_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_squeeze1d"
  p_squeeze1d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> IO ())

-- | p_unsqueeze1d : Pointer to function : state self src dimension_ -> void
foreign import ccall "THCTensor.h &THCHalfTensor_unsqueeze1d"
  p_unsqueeze1d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> IO ())

-- | p_isContiguous : Pointer to function : state self -> int
foreign import ccall "THCTensor.h &THCHalfTensor_isContiguous"
  p_isContiguous :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CInt)

-- | p_isSameSizeAs : Pointer to function : state self src -> int
foreign import ccall "THCTensor.h &THCHalfTensor_isSameSizeAs"
  p_isSameSizeAs :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO CInt)

-- | p_isSetTo : Pointer to function : state self src -> int
foreign import ccall "THCTensor.h &THCHalfTensor_isSetTo"
  p_isSetTo :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO CInt)

-- | p_isSize : Pointer to function : state self dims -> int
foreign import ccall "THCTensor.h &THCHalfTensor_isSize"
  p_isSize :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongStorage -> IO CInt)

-- | p_nElement : Pointer to function : state self -> ptrdiff_t
foreign import ccall "THCTensor.h &THCHalfTensor_nElement"
  p_nElement :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CPtrdiff)

-- | p_retain : Pointer to function : state self -> void
foreign import ccall "THCTensor.h &THCHalfTensor_retain"
  p_retain :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_free : Pointer to function : state self -> void
foreign import ccall "THCTensor.h &THCHalfTensor_free"
  p_free :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_freeCopyTo : Pointer to function : state self dst -> void
foreign import ccall "THCTensor.h &THCHalfTensor_freeCopyTo"
  p_freeCopyTo :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_set1d : Pointer to function : state tensor x0 value -> void
foreign import ccall "THCTensor.h &THCHalfTensor_set1d"
  p_set1d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CTHHalf -> IO ())

-- | p_set2d : Pointer to function : state tensor x0 x1 value -> void
foreign import ccall "THCTensor.h &THCHalfTensor_set2d"
  p_set2d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CTHHalf -> IO ())

-- | p_set3d : Pointer to function : state tensor x0 x1 x2 value -> void
foreign import ccall "THCTensor.h &THCHalfTensor_set3d"
  p_set3d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> CTHHalf -> IO ())

-- | p_set4d : Pointer to function : state tensor x0 x1 x2 x3 value -> void
foreign import ccall "THCTensor.h &THCHalfTensor_set4d"
  p_set4d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CTHHalf -> IO ())

-- | p_get1d : Pointer to function : state tensor x0 -> real
foreign import ccall "THCTensor.h &THCHalfTensor_get1d"
  p_get1d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> IO CTHHalf)

-- | p_get2d : Pointer to function : state tensor x0 x1 -> real
foreign import ccall "THCTensor.h &THCHalfTensor_get2d"
  p_get2d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> IO CTHHalf)

-- | p_get3d : Pointer to function : state tensor x0 x1 x2 -> real
foreign import ccall "THCTensor.h &THCHalfTensor_get3d"
  p_get3d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> IO CTHHalf)

-- | p_get4d : Pointer to function : state tensor x0 x1 x2 x3 -> real
foreign import ccall "THCTensor.h &THCHalfTensor_get4d"
  p_get4d :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO CTHHalf)

-- | p_getDevice : Pointer to function : state self -> int
foreign import ccall "THCTensor.h &THCHalfTensor_getDevice"
  p_getDevice :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CInt)

-- | p_sizeDesc : Pointer to function : state tensor -> THCDescBuff
foreign import ccall "THCTensor.h &THCHalfTensor_sizeDesc"
  p_sizeDesc :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CTHCudaDescBuff)