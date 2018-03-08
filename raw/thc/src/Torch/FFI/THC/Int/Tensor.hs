{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.Tensor
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

-- | c_storage :  state self -> THStorage *
foreign import ccall "THCTensor.h THIntTensor_storage"
  c_storage :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CTHIntStorage))

-- | c_storageOffset :  state self -> ptrdiff_t
foreign import ccall "THCTensor.h THIntTensor_storageOffset"
  c_storageOffset :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (CPtrdiff)

-- | c_nDimension :  state self -> int
foreign import ccall "THCTensor.h THIntTensor_nDimension"
  c_nDimension :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (CInt)

-- | c_size :  state self dim -> int64_t
foreign import ccall "THCTensor.h THIntTensor_size"
  c_size :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> IO (CLLong)

-- | c_stride :  state self dim -> int64_t
foreign import ccall "THCTensor.h THIntTensor_stride"
  c_stride :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> IO (CLLong)

-- | c_newSizeOf :  state self -> THLongStorage *
foreign import ccall "THCTensor.h THIntTensor_newSizeOf"
  c_newSizeOf :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CTHLongStorage))

-- | c_newStrideOf :  state self -> THLongStorage *
foreign import ccall "THCTensor.h THIntTensor_newStrideOf"
  c_newStrideOf :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CTHLongStorage))

-- | c_data :  state self -> real *
foreign import ccall "THCTensor.h THIntTensor_data"
  c_data :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CInt))

-- | c_setFlag :  state self flag -> void
foreign import ccall "THCTensor.h THIntTensor_setFlag"
  c_setFlag :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CChar -> IO (())

-- | c_clearFlag :  state self flag -> void
foreign import ccall "THCTensor.h THIntTensor_clearFlag"
  c_clearFlag :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CChar -> IO (())

-- | c_new :  state -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_new"
  c_new :: Ptr (CTHState) -> IO (Ptr (CTHIntTensor))

-- | c_newWithTensor :  state tensor -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newWithTensor"
  c_newWithTensor :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CTHIntTensor))

-- | c_newWithStorage :  state storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newWithStorage"
  c_newWithStorage :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> CPtrdiff -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHIntTensor))

-- | c_newWithStorage1d :  state storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newWithStorage1d"
  c_newWithStorage1d :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor))

-- | c_newWithStorage2d :  state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newWithStorage2d"
  c_newWithStorage2d :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor))

-- | c_newWithStorage3d :  state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newWithStorage3d"
  c_newWithStorage3d :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor))

-- | c_newWithStorage4d :  state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newWithStorage4d"
  c_newWithStorage4d :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor))

-- | c_newWithSize :  state size_ stride_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newWithSize"
  c_newWithSize :: Ptr (CTHState) -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHIntTensor))

-- | c_newWithSize1d :  state size0_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newWithSize1d"
  c_newWithSize1d :: Ptr (CTHState) -> CLLong -> IO (Ptr (CTHIntTensor))

-- | c_newWithSize2d :  state size0_ size1_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newWithSize2d"
  c_newWithSize2d :: Ptr (CTHState) -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor))

-- | c_newWithSize3d :  state size0_ size1_ size2_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newWithSize3d"
  c_newWithSize3d :: Ptr (CTHState) -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor))

-- | c_newWithSize4d :  state size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newWithSize4d"
  c_newWithSize4d :: Ptr (CTHState) -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor))

-- | c_newClone :  state self -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newClone"
  c_newClone :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CTHIntTensor))

-- | c_newContiguous :  state tensor -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newContiguous"
  c_newContiguous :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CTHIntTensor))

-- | c_newSelect :  state tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newSelect"
  c_newSelect :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> CLLong -> IO (Ptr (CTHIntTensor))

-- | c_newNarrow :  state tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newNarrow"
  c_newNarrow :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor))

-- | c_newTranspose :  state tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newTranspose"
  c_newTranspose :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (Ptr (CTHIntTensor))

-- | c_newUnfold :  state tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newUnfold"
  c_newUnfold :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor))

-- | c_newView :  state tensor size -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newView"
  c_newView :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHIntTensor))

-- | c_newExpand :  state tensor size -> THTensor *
foreign import ccall "THCTensor.h THIntTensor_newExpand"
  c_newExpand :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHIntTensor))

-- | c_expand :  state r tensor sizes -> void
foreign import ccall "THCTensor.h THIntTensor_expand"
  c_expand :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_expandNd :  state rets ops count -> void
foreign import ccall "THCTensor.h THIntTensor_expandNd"
  c_expandNd :: Ptr (CTHState) -> Ptr (Ptr (CTHIntTensor)) -> Ptr (Ptr (CTHIntTensor)) -> CInt -> IO (())

-- | c_resize :  state tensor size stride -> void
foreign import ccall "THCTensor.h THIntTensor_resize"
  c_resize :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_resizeAs :  state tensor src -> void
foreign import ccall "THCTensor.h THIntTensor_resizeAs"
  c_resizeAs :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_resize1d :  state tensor size0_ -> void
foreign import ccall "THCTensor.h THIntTensor_resize1d"
  c_resize1d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> IO (())

-- | c_resize2d :  state tensor size0_ size1_ -> void
foreign import ccall "THCTensor.h THIntTensor_resize2d"
  c_resize2d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> IO (())

-- | c_resize3d :  state tensor size0_ size1_ size2_ -> void
foreign import ccall "THCTensor.h THIntTensor_resize3d"
  c_resize3d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> IO (())

-- | c_resize4d :  state tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THCTensor.h THIntTensor_resize4d"
  c_resize4d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> IO (())

-- | c_resize5d :  state tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THCTensor.h THIntTensor_resize5d"
  c_resize5d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (())

-- | c_resizeNd :  state tensor nDimension size stride -> void
foreign import ccall "THCTensor.h THIntTensor_resizeNd"
  c_resizeNd :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CLLong) -> Ptr (CLLong) -> IO (())

-- | c_set :  state self src -> void
foreign import ccall "THCTensor.h THIntTensor_set"
  c_set :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_setStorage :  state self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THCTensor.h THIntTensor_setStorage"
  c_setStorage :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntStorage) -> CPtrdiff -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_setStorageNd :  state self storage storageOffset nDimension size stride -> void
foreign import ccall "THCTensor.h THIntTensor_setStorageNd"
  c_setStorageNd :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntStorage) -> CPtrdiff -> CInt -> Ptr (CLLong) -> Ptr (CLLong) -> IO (())

-- | c_setStorage1d :  state self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THCTensor.h THIntTensor_setStorage1d"
  c_setStorage1d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> IO (())

-- | c_setStorage2d :  state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THCTensor.h THIntTensor_setStorage2d"
  c_setStorage2d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (())

-- | c_setStorage3d :  state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THCTensor.h THIntTensor_setStorage3d"
  c_setStorage3d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (())

-- | c_setStorage4d :  state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THCTensor.h THIntTensor_setStorage4d"
  c_setStorage4d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (())

-- | c_narrow :  state self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THCTensor.h THIntTensor_narrow"
  c_narrow :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO (())

-- | c_select :  state self src dimension_ sliceIndex_ -> void
foreign import ccall "THCTensor.h THIntTensor_select"
  c_select :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CLLong -> IO (())

-- | c_transpose :  state self src dimension1_ dimension2_ -> void
foreign import ccall "THCTensor.h THIntTensor_transpose"
  c_transpose :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (())

-- | c_unfold :  state self src dimension_ size_ step_ -> void
foreign import ccall "THCTensor.h THIntTensor_unfold"
  c_unfold :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO (())

-- | c_squeeze :  state self src -> void
foreign import ccall "THCTensor.h THIntTensor_squeeze"
  c_squeeze :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_squeeze1d :  state self src dimension_ -> void
foreign import ccall "THCTensor.h THIntTensor_squeeze1d"
  c_squeeze1d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_unsqueeze1d :  state self src dimension_ -> void
foreign import ccall "THCTensor.h THIntTensor_unsqueeze1d"
  c_unsqueeze1d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_isContiguous :  state self -> int
foreign import ccall "THCTensor.h THIntTensor_isContiguous"
  c_isContiguous :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (CInt)

-- | c_isSameSizeAs :  state self src -> int
foreign import ccall "THCTensor.h THIntTensor_isSameSizeAs"
  c_isSameSizeAs :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (CInt)

-- | c_isSetTo :  state self src -> int
foreign import ccall "THCTensor.h THIntTensor_isSetTo"
  c_isSetTo :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (CInt)

-- | c_isSize :  state self dims -> int
foreign import ccall "THCTensor.h THIntTensor_isSize"
  c_isSize :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (CInt)

-- | c_nElement :  state self -> ptrdiff_t
foreign import ccall "THCTensor.h THIntTensor_nElement"
  c_nElement :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (CPtrdiff)

-- | c_retain :  state self -> void
foreign import ccall "THCTensor.h THIntTensor_retain"
  c_retain :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (())

-- | c_free :  state self -> void
foreign import ccall "THCTensor.h THIntTensor_free"
  c_free :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (())

-- | c_freeCopyTo :  state self dst -> void
foreign import ccall "THCTensor.h THIntTensor_freeCopyTo"
  c_freeCopyTo :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_set1d :  state tensor x0 value -> void
foreign import ccall "THCTensor.h THIntTensor_set1d"
  c_set1d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CInt -> IO (())

-- | c_set2d :  state tensor x0 x1 value -> void
foreign import ccall "THCTensor.h THIntTensor_set2d"
  c_set2d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CInt -> IO (())

-- | c_set3d :  state tensor x0 x1 x2 value -> void
foreign import ccall "THCTensor.h THIntTensor_set3d"
  c_set3d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CInt -> IO (())

-- | c_set4d :  state tensor x0 x1 x2 x3 value -> void
foreign import ccall "THCTensor.h THIntTensor_set4d"
  c_set4d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CInt -> IO (())

-- | c_get1d :  state tensor x0 -> real
foreign import ccall "THCTensor.h THIntTensor_get1d"
  c_get1d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> IO (CInt)

-- | c_get2d :  state tensor x0 x1 -> real
foreign import ccall "THCTensor.h THIntTensor_get2d"
  c_get2d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> IO (CInt)

-- | c_get3d :  state tensor x0 x1 x2 -> real
foreign import ccall "THCTensor.h THIntTensor_get3d"
  c_get3d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> IO (CInt)

-- | c_get4d :  state tensor x0 x1 x2 x3 -> real
foreign import ccall "THCTensor.h THIntTensor_get4d"
  c_get4d :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> IO (CInt)

-- | c_getDevice :  state self -> int
foreign import ccall "THCTensor.h THIntTensor_getDevice"
  c_getDevice :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (CInt)

-- | c_sizeDesc :  state tensor -> THDescBuff
foreign import ccall "THCTensor.h THIntTensor_sizeDesc"
  c_sizeDesc :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (CTHDescBuff)

-- | p_storage : Pointer to function : state self -> THStorage *
foreign import ccall "THCTensor.h &THIntTensor_storage"
  p_storage :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CTHIntStorage)))

-- | p_storageOffset : Pointer to function : state self -> ptrdiff_t
foreign import ccall "THCTensor.h &THIntTensor_storageOffset"
  p_storageOffset :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (CPtrdiff))

-- | p_nDimension : Pointer to function : state self -> int
foreign import ccall "THCTensor.h &THIntTensor_nDimension"
  p_nDimension :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (CInt))

-- | p_size : Pointer to function : state self dim -> int64_t
foreign import ccall "THCTensor.h &THIntTensor_size"
  p_size :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> IO (CLLong))

-- | p_stride : Pointer to function : state self dim -> int64_t
foreign import ccall "THCTensor.h &THIntTensor_stride"
  p_stride :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> IO (CLLong))

-- | p_newSizeOf : Pointer to function : state self -> THLongStorage *
foreign import ccall "THCTensor.h &THIntTensor_newSizeOf"
  p_newSizeOf :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CTHLongStorage)))

-- | p_newStrideOf : Pointer to function : state self -> THLongStorage *
foreign import ccall "THCTensor.h &THIntTensor_newStrideOf"
  p_newStrideOf :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CTHLongStorage)))

-- | p_data : Pointer to function : state self -> real *
foreign import ccall "THCTensor.h &THIntTensor_data"
  p_data :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CInt)))

-- | p_setFlag : Pointer to function : state self flag -> void
foreign import ccall "THCTensor.h &THIntTensor_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : state self flag -> void
foreign import ccall "THCTensor.h &THIntTensor_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CChar -> IO (()))

-- | p_new : Pointer to function : state -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_new"
  p_new :: FunPtr (Ptr (CTHState) -> IO (Ptr (CTHIntTensor)))

-- | p_newWithTensor : Pointer to function : state tensor -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newWithTensor"
  p_newWithTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CTHIntTensor)))

-- | p_newWithStorage : Pointer to function : state storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newWithStorage"
  p_newWithStorage :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> CPtrdiff -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHIntTensor)))

-- | p_newWithStorage1d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newWithStorage1d"
  p_newWithStorage1d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor)))

-- | p_newWithStorage2d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newWithStorage2d"
  p_newWithStorage2d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor)))

-- | p_newWithStorage3d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newWithStorage3d"
  p_newWithStorage3d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor)))

-- | p_newWithStorage4d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newWithStorage4d"
  p_newWithStorage4d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor)))

-- | p_newWithSize : Pointer to function : state size_ stride_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newWithSize"
  p_newWithSize :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHIntTensor)))

-- | p_newWithSize1d : Pointer to function : state size0_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newWithSize1d"
  p_newWithSize1d :: FunPtr (Ptr (CTHState) -> CLLong -> IO (Ptr (CTHIntTensor)))

-- | p_newWithSize2d : Pointer to function : state size0_ size1_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newWithSize2d"
  p_newWithSize2d :: FunPtr (Ptr (CTHState) -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor)))

-- | p_newWithSize3d : Pointer to function : state size0_ size1_ size2_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newWithSize3d"
  p_newWithSize3d :: FunPtr (Ptr (CTHState) -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor)))

-- | p_newWithSize4d : Pointer to function : state size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newWithSize4d"
  p_newWithSize4d :: FunPtr (Ptr (CTHState) -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor)))

-- | p_newClone : Pointer to function : state self -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newClone"
  p_newClone :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CTHIntTensor)))

-- | p_newContiguous : Pointer to function : state tensor -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newContiguous"
  p_newContiguous :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (Ptr (CTHIntTensor)))

-- | p_newSelect : Pointer to function : state tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newSelect"
  p_newSelect :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> CLLong -> IO (Ptr (CTHIntTensor)))

-- | p_newNarrow : Pointer to function : state tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newNarrow"
  p_newNarrow :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor)))

-- | p_newTranspose : Pointer to function : state tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newTranspose"
  p_newTranspose :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (Ptr (CTHIntTensor)))

-- | p_newUnfold : Pointer to function : state tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newUnfold"
  p_newUnfold :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr (CTHIntTensor)))

-- | p_newView : Pointer to function : state tensor size -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newView"
  p_newView :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHIntTensor)))

-- | p_newExpand : Pointer to function : state tensor size -> THTensor *
foreign import ccall "THCTensor.h &THIntTensor_newExpand"
  p_newExpand :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHIntTensor)))

-- | p_expand : Pointer to function : state r tensor sizes -> void
foreign import ccall "THCTensor.h &THIntTensor_expand"
  p_expand :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_expandNd : Pointer to function : state rets ops count -> void
foreign import ccall "THCTensor.h &THIntTensor_expandNd"
  p_expandNd :: FunPtr (Ptr (CTHState) -> Ptr (Ptr (CTHIntTensor)) -> Ptr (Ptr (CTHIntTensor)) -> CInt -> IO (()))

-- | p_resize : Pointer to function : state tensor size stride -> void
foreign import ccall "THCTensor.h &THIntTensor_resize"
  p_resize :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_resizeAs : Pointer to function : state tensor src -> void
foreign import ccall "THCTensor.h &THIntTensor_resizeAs"
  p_resizeAs :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_resize1d : Pointer to function : state tensor size0_ -> void
foreign import ccall "THCTensor.h &THIntTensor_resize1d"
  p_resize1d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> IO (()))

-- | p_resize2d : Pointer to function : state tensor size0_ size1_ -> void
foreign import ccall "THCTensor.h &THIntTensor_resize2d"
  p_resize2d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> IO (()))

-- | p_resize3d : Pointer to function : state tensor size0_ size1_ size2_ -> void
foreign import ccall "THCTensor.h &THIntTensor_resize3d"
  p_resize3d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> IO (()))

-- | p_resize4d : Pointer to function : state tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THCTensor.h &THIntTensor_resize4d"
  p_resize4d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> IO (()))

-- | p_resize5d : Pointer to function : state tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THCTensor.h &THIntTensor_resize5d"
  p_resize5d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (()))

-- | p_resizeNd : Pointer to function : state tensor nDimension size stride -> void
foreign import ccall "THCTensor.h &THIntTensor_resizeNd"
  p_resizeNd :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CLLong) -> Ptr (CLLong) -> IO (()))

-- | p_set : Pointer to function : state self src -> void
foreign import ccall "THCTensor.h &THIntTensor_set"
  p_set :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_setStorage : Pointer to function : state self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THCTensor.h &THIntTensor_setStorage"
  p_setStorage :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntStorage) -> CPtrdiff -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_setStorageNd : Pointer to function : state self storage storageOffset nDimension size stride -> void
foreign import ccall "THCTensor.h &THIntTensor_setStorageNd"
  p_setStorageNd :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntStorage) -> CPtrdiff -> CInt -> Ptr (CLLong) -> Ptr (CLLong) -> IO (()))

-- | p_setStorage1d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THCTensor.h &THIntTensor_setStorage1d"
  p_setStorage1d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> IO (()))

-- | p_setStorage2d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THCTensor.h &THIntTensor_setStorage2d"
  p_setStorage2d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (()))

-- | p_setStorage3d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THCTensor.h &THIntTensor_setStorage3d"
  p_setStorage3d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (()))

-- | p_setStorage4d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THCTensor.h &THIntTensor_setStorage4d"
  p_setStorage4d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (()))

-- | p_narrow : Pointer to function : state self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THCTensor.h &THIntTensor_narrow"
  p_narrow :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO (()))

-- | p_select : Pointer to function : state self src dimension_ sliceIndex_ -> void
foreign import ccall "THCTensor.h &THIntTensor_select"
  p_select :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CLLong -> IO (()))

-- | p_transpose : Pointer to function : state self src dimension1_ dimension2_ -> void
foreign import ccall "THCTensor.h &THIntTensor_transpose"
  p_transpose :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (()))

-- | p_unfold : Pointer to function : state self src dimension_ size_ step_ -> void
foreign import ccall "THCTensor.h &THIntTensor_unfold"
  p_unfold :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CLLong -> CLLong -> IO (()))

-- | p_squeeze : Pointer to function : state self src -> void
foreign import ccall "THCTensor.h &THIntTensor_squeeze"
  p_squeeze :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_squeeze1d : Pointer to function : state self src dimension_ -> void
foreign import ccall "THCTensor.h &THIntTensor_squeeze1d"
  p_squeeze1d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_unsqueeze1d : Pointer to function : state self src dimension_ -> void
foreign import ccall "THCTensor.h &THIntTensor_unsqueeze1d"
  p_unsqueeze1d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_isContiguous : Pointer to function : state self -> int
foreign import ccall "THCTensor.h &THIntTensor_isContiguous"
  p_isContiguous :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (CInt))

-- | p_isSameSizeAs : Pointer to function : state self src -> int
foreign import ccall "THCTensor.h &THIntTensor_isSameSizeAs"
  p_isSameSizeAs :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (CInt))

-- | p_isSetTo : Pointer to function : state self src -> int
foreign import ccall "THCTensor.h &THIntTensor_isSetTo"
  p_isSetTo :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (CInt))

-- | p_isSize : Pointer to function : state self dims -> int
foreign import ccall "THCTensor.h &THIntTensor_isSize"
  p_isSize :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (CInt))

-- | p_nElement : Pointer to function : state self -> ptrdiff_t
foreign import ccall "THCTensor.h &THIntTensor_nElement"
  p_nElement :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (CPtrdiff))

-- | p_retain : Pointer to function : state self -> void
foreign import ccall "THCTensor.h &THIntTensor_retain"
  p_retain :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_free : Pointer to function : state self -> void
foreign import ccall "THCTensor.h &THIntTensor_free"
  p_free :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_freeCopyTo : Pointer to function : state self dst -> void
foreign import ccall "THCTensor.h &THIntTensor_freeCopyTo"
  p_freeCopyTo :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_set1d : Pointer to function : state tensor x0 value -> void
foreign import ccall "THCTensor.h &THIntTensor_set1d"
  p_set1d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CInt -> IO (()))

-- | p_set2d : Pointer to function : state tensor x0 x1 value -> void
foreign import ccall "THCTensor.h &THIntTensor_set2d"
  p_set2d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CInt -> IO (()))

-- | p_set3d : Pointer to function : state tensor x0 x1 x2 value -> void
foreign import ccall "THCTensor.h &THIntTensor_set3d"
  p_set3d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CInt -> IO (()))

-- | p_set4d : Pointer to function : state tensor x0 x1 x2 x3 value -> void
foreign import ccall "THCTensor.h &THIntTensor_set4d"
  p_set4d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CInt -> IO (()))

-- | p_get1d : Pointer to function : state tensor x0 -> real
foreign import ccall "THCTensor.h &THIntTensor_get1d"
  p_get1d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> IO (CInt))

-- | p_get2d : Pointer to function : state tensor x0 x1 -> real
foreign import ccall "THCTensor.h &THIntTensor_get2d"
  p_get2d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> IO (CInt))

-- | p_get3d : Pointer to function : state tensor x0 x1 x2 -> real
foreign import ccall "THCTensor.h &THIntTensor_get3d"
  p_get3d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> IO (CInt))

-- | p_get4d : Pointer to function : state tensor x0 x1 x2 x3 -> real
foreign import ccall "THCTensor.h &THIntTensor_get4d"
  p_get4d :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> IO (CInt))

-- | p_getDevice : Pointer to function : state self -> int
foreign import ccall "THCTensor.h &THIntTensor_getDevice"
  p_getDevice :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (CInt))

-- | p_sizeDesc : Pointer to function : state tensor -> THDescBuff
foreign import ccall "THCTensor.h &THIntTensor_sizeDesc"
  p_sizeDesc :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> IO (CTHDescBuff))