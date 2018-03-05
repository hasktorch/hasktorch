{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.Tensor
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
  , c_resizeNd
  , c_resize1d
  , c_resize2d
  , c_resize3d
  , c_resize4d
  , c_resize5d
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
  , c_desc
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
  , p_resizeNd
  , p_resize1d
  , p_resize2d
  , p_resize3d
  , p_resize4d
  , p_resize5d
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
  , p_desc
  , p_sizeDesc
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_storage :  self -> THStorage *
foreign import ccall "THTensor.h THShortTensor_storage"
  c_storage :: Ptr (CTHShortTensor) -> IO (Ptr (CTHShortStorage))

-- | c_storageOffset :  self -> ptrdiff_t
foreign import ccall "THTensor.h THShortTensor_storageOffset"
  c_storageOffset :: Ptr (CTHShortTensor) -> IO (CPtrdiff)

-- | c_nDimension :  self -> int
foreign import ccall "THTensor.h THShortTensor_nDimension"
  c_nDimension :: Ptr (CTHShortTensor) -> IO (CInt)

-- | c_size :  self dim -> int64_t
foreign import ccall "THTensor.h THShortTensor_size"
  c_size :: Ptr (CTHShortTensor) -> CInt -> IO (CLLong)

-- | c_stride :  self dim -> int64_t
foreign import ccall "THTensor.h THShortTensor_stride"
  c_stride :: Ptr (CTHShortTensor) -> CInt -> IO (CLLong)

-- | c_newSizeOf :  self -> THLongStorage *
foreign import ccall "THTensor.h THShortTensor_newSizeOf"
  c_newSizeOf :: Ptr (CTHShortTensor) -> IO (Ptr (CTHLongStorage))

-- | c_newStrideOf :  self -> THLongStorage *
foreign import ccall "THTensor.h THShortTensor_newStrideOf"
  c_newStrideOf :: Ptr (CTHShortTensor) -> IO (Ptr (CTHLongStorage))

-- | c_data :  self -> real *
foreign import ccall "THTensor.h THShortTensor_data"
  c_data :: Ptr (CTHShortTensor) -> IO (Ptr (CShort))

-- | c_setFlag :  self flag -> void
foreign import ccall "THTensor.h THShortTensor_setFlag"
  c_setFlag :: Ptr (CTHShortTensor) -> CChar -> IO (())

-- | c_clearFlag :  self flag -> void
foreign import ccall "THTensor.h THShortTensor_clearFlag"
  c_clearFlag :: Ptr (CTHShortTensor) -> CChar -> IO (())

-- | c_new :   -> THTensor *
foreign import ccall "THTensor.h THShortTensor_new"
  c_new :: IO (Ptr (CTHShortTensor))

-- | c_newWithTensor :  tensor -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithTensor"
  c_newWithTensor :: Ptr (CTHShortTensor) -> IO (Ptr (CTHShortTensor))

-- | c_newWithStorage :  storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage"
  c_newWithStorage :: Ptr (CTHShortStorage) -> CPtrdiff -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHShortTensor))

-- | c_newWithStorage1d :  storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage1d"
  c_newWithStorage1d :: Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor))

-- | c_newWithStorage2d :  storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage2d"
  c_newWithStorage2d :: Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor))

-- | c_newWithStorage3d :  storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage3d"
  c_newWithStorage3d :: Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor))

-- | c_newWithStorage4d :  storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithStorage4d"
  c_newWithStorage4d :: Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor))

-- | c_newWithSize :  size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize"
  c_newWithSize :: Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHShortTensor))

-- | c_newWithSize1d :  size0_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize1d"
  c_newWithSize1d :: CLLong -> IO (Ptr (CTHShortTensor))

-- | c_newWithSize2d :  size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize2d"
  c_newWithSize2d :: CLLong -> CLLong -> IO (Ptr (CTHShortTensor))

-- | c_newWithSize3d :  size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize3d"
  c_newWithSize3d :: CLLong -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor))

-- | c_newWithSize4d :  size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newWithSize4d"
  c_newWithSize4d :: CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor))

-- | c_newClone :  self -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newClone"
  c_newClone :: Ptr (CTHShortTensor) -> IO (Ptr (CTHShortTensor))

-- | c_newContiguous :  tensor -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newContiguous"
  c_newContiguous :: Ptr (CTHShortTensor) -> IO (Ptr (CTHShortTensor))

-- | c_newSelect :  tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newSelect"
  c_newSelect :: Ptr (CTHShortTensor) -> CInt -> CLLong -> IO (Ptr (CTHShortTensor))

-- | c_newNarrow :  tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newNarrow"
  c_newNarrow :: Ptr (CTHShortTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor))

-- | c_newTranspose :  tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newTranspose"
  c_newTranspose :: Ptr (CTHShortTensor) -> CInt -> CInt -> IO (Ptr (CTHShortTensor))

-- | c_newUnfold :  tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newUnfold"
  c_newUnfold :: Ptr (CTHShortTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor))

-- | c_newView :  tensor size -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newView"
  c_newView :: Ptr (CTHShortTensor) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHShortTensor))

-- | c_newExpand :  tensor size -> THTensor *
foreign import ccall "THTensor.h THShortTensor_newExpand"
  c_newExpand :: Ptr (CTHShortTensor) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHShortTensor))

-- | c_expand :  r tensor size -> void
foreign import ccall "THTensor.h THShortTensor_expand"
  c_expand :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_expandNd :  rets ops count -> void
foreign import ccall "THTensor.h THShortTensor_expandNd"
  c_expandNd :: Ptr (Ptr (CTHShortTensor)) -> Ptr (Ptr (CTHShortTensor)) -> CInt -> IO (())

-- | c_resize :  tensor size stride -> void
foreign import ccall "THTensor.h THShortTensor_resize"
  c_resize :: Ptr (CTHShortTensor) -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_resizeAs :  tensor src -> void
foreign import ccall "THTensor.h THShortTensor_resizeAs"
  c_resizeAs :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_resizeNd :  tensor nDimension size stride -> void
foreign import ccall "THTensor.h THShortTensor_resizeNd"
  c_resizeNd :: Ptr (CTHShortTensor) -> CInt -> Ptr (CLLong) -> Ptr (CLLong) -> IO (())

-- | c_resize1d :  tensor size0_ -> void
foreign import ccall "THTensor.h THShortTensor_resize1d"
  c_resize1d :: Ptr (CTHShortTensor) -> CLLong -> IO (())

-- | c_resize2d :  tensor size0_ size1_ -> void
foreign import ccall "THTensor.h THShortTensor_resize2d"
  c_resize2d :: Ptr (CTHShortTensor) -> CLLong -> CLLong -> IO (())

-- | c_resize3d :  tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h THShortTensor_resize3d"
  c_resize3d :: Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> IO (())

-- | c_resize4d :  tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h THShortTensor_resize4d"
  c_resize4d :: Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> IO (())

-- | c_resize5d :  tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h THShortTensor_resize5d"
  c_resize5d :: Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (())

-- | c_set :  self src -> void
foreign import ccall "THTensor.h THShortTensor_set"
  c_set :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_setStorage :  self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage"
  c_setStorage :: Ptr (CTHShortTensor) -> Ptr (CTHShortStorage) -> CPtrdiff -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_setStorageNd :  self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h THShortTensor_setStorageNd"
  c_setStorageNd :: Ptr (CTHShortTensor) -> Ptr (CTHShortStorage) -> CPtrdiff -> CInt -> Ptr (CLLong) -> Ptr (CLLong) -> IO (())

-- | c_setStorage1d :  self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage1d"
  c_setStorage1d :: Ptr (CTHShortTensor) -> Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> IO (())

-- | c_setStorage2d :  self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage2d"
  c_setStorage2d :: Ptr (CTHShortTensor) -> Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (())

-- | c_setStorage3d :  self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage3d"
  c_setStorage3d :: Ptr (CTHShortTensor) -> Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (())

-- | c_setStorage4d :  self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h THShortTensor_setStorage4d"
  c_setStorage4d :: Ptr (CTHShortTensor) -> Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (())

-- | c_narrow :  self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h THShortTensor_narrow"
  c_narrow :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> CLLong -> CLLong -> IO (())

-- | c_select :  self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h THShortTensor_select"
  c_select :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> CLLong -> IO (())

-- | c_transpose :  self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h THShortTensor_transpose"
  c_transpose :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (())

-- | c_unfold :  self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h THShortTensor_unfold"
  c_unfold :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> CLLong -> CLLong -> IO (())

-- | c_squeeze :  self src -> void
foreign import ccall "THTensor.h THShortTensor_squeeze"
  c_squeeze :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_squeeze1d :  self src dimension_ -> void
foreign import ccall "THTensor.h THShortTensor_squeeze1d"
  c_squeeze1d :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> IO (())

-- | c_unsqueeze1d :  self src dimension_ -> void
foreign import ccall "THTensor.h THShortTensor_unsqueeze1d"
  c_unsqueeze1d :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> IO (())

-- | c_isContiguous :  self -> int
foreign import ccall "THTensor.h THShortTensor_isContiguous"
  c_isContiguous :: Ptr (CTHShortTensor) -> IO (CInt)

-- | c_isSameSizeAs :  self src -> int
foreign import ccall "THTensor.h THShortTensor_isSameSizeAs"
  c_isSameSizeAs :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (CInt)

-- | c_isSetTo :  self src -> int
foreign import ccall "THTensor.h THShortTensor_isSetTo"
  c_isSetTo :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (CInt)

-- | c_isSize :  self dims -> int
foreign import ccall "THTensor.h THShortTensor_isSize"
  c_isSize :: Ptr (CTHShortTensor) -> Ptr (CTHLongStorage) -> IO (CInt)

-- | c_nElement :  self -> ptrdiff_t
foreign import ccall "THTensor.h THShortTensor_nElement"
  c_nElement :: Ptr (CTHShortTensor) -> IO (CPtrdiff)

-- | c_retain :  self -> void
foreign import ccall "THTensor.h THShortTensor_retain"
  c_retain :: Ptr (CTHShortTensor) -> IO (())

-- | c_free :  self -> void
foreign import ccall "THTensor.h THShortTensor_free"
  c_free :: Ptr (CTHShortTensor) -> IO (())

-- | c_freeCopyTo :  self dst -> void
foreign import ccall "THTensor.h THShortTensor_freeCopyTo"
  c_freeCopyTo :: Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_set1d :  tensor x0 value -> void
foreign import ccall "THTensor.h THShortTensor_set1d"
  c_set1d :: Ptr (CTHShortTensor) -> CLLong -> CShort -> IO (())

-- | c_set2d :  tensor x0 x1 value -> void
foreign import ccall "THTensor.h THShortTensor_set2d"
  c_set2d :: Ptr (CTHShortTensor) -> CLLong -> CLLong -> CShort -> IO (())

-- | c_set3d :  tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h THShortTensor_set3d"
  c_set3d :: Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> CShort -> IO (())

-- | c_set4d :  tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h THShortTensor_set4d"
  c_set4d :: Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CShort -> IO (())

-- | c_get1d :  tensor x0 -> real
foreign import ccall "THTensor.h THShortTensor_get1d"
  c_get1d :: Ptr (CTHShortTensor) -> CLLong -> IO (CShort)

-- | c_get2d :  tensor x0 x1 -> real
foreign import ccall "THTensor.h THShortTensor_get2d"
  c_get2d :: Ptr (CTHShortTensor) -> CLLong -> CLLong -> IO (CShort)

-- | c_get3d :  tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h THShortTensor_get3d"
  c_get3d :: Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> IO (CShort)

-- | c_get4d :  tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h THShortTensor_get4d"
  c_get4d :: Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> IO (CShort)

-- | c_desc :  tensor -> THDescBuff
foreign import ccall "THTensor.h THShortTensor_desc"
  c_desc :: Ptr (CTHShortTensor) -> IO (CTHDescBuff)

-- | c_sizeDesc :  tensor -> THDescBuff
foreign import ccall "THTensor.h THShortTensor_sizeDesc"
  c_sizeDesc :: Ptr (CTHShortTensor) -> IO (CTHDescBuff)

-- | p_storage : Pointer to function : self -> THStorage *
foreign import ccall "THTensor.h &THShortTensor_storage"
  p_storage :: FunPtr (Ptr (CTHShortTensor) -> IO (Ptr (CTHShortStorage)))

-- | p_storageOffset : Pointer to function : self -> ptrdiff_t
foreign import ccall "THTensor.h &THShortTensor_storageOffset"
  p_storageOffset :: FunPtr (Ptr (CTHShortTensor) -> IO (CPtrdiff))

-- | p_nDimension : Pointer to function : self -> int
foreign import ccall "THTensor.h &THShortTensor_nDimension"
  p_nDimension :: FunPtr (Ptr (CTHShortTensor) -> IO (CInt))

-- | p_size : Pointer to function : self dim -> int64_t
foreign import ccall "THTensor.h &THShortTensor_size"
  p_size :: FunPtr (Ptr (CTHShortTensor) -> CInt -> IO (CLLong))

-- | p_stride : Pointer to function : self dim -> int64_t
foreign import ccall "THTensor.h &THShortTensor_stride"
  p_stride :: FunPtr (Ptr (CTHShortTensor) -> CInt -> IO (CLLong))

-- | p_newSizeOf : Pointer to function : self -> THLongStorage *
foreign import ccall "THTensor.h &THShortTensor_newSizeOf"
  p_newSizeOf :: FunPtr (Ptr (CTHShortTensor) -> IO (Ptr (CTHLongStorage)))

-- | p_newStrideOf : Pointer to function : self -> THLongStorage *
foreign import ccall "THTensor.h &THShortTensor_newStrideOf"
  p_newStrideOf :: FunPtr (Ptr (CTHShortTensor) -> IO (Ptr (CTHLongStorage)))

-- | p_data : Pointer to function : self -> real *
foreign import ccall "THTensor.h &THShortTensor_data"
  p_data :: FunPtr (Ptr (CTHShortTensor) -> IO (Ptr (CShort)))

-- | p_setFlag : Pointer to function : self flag -> void
foreign import ccall "THTensor.h &THShortTensor_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHShortTensor) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : self flag -> void
foreign import ccall "THTensor.h &THShortTensor_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHShortTensor) -> CChar -> IO (()))

-- | p_new : Pointer to function :  -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_new"
  p_new :: FunPtr (IO (Ptr (CTHShortTensor)))

-- | p_newWithTensor : Pointer to function : tensor -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithTensor"
  p_newWithTensor :: FunPtr (Ptr (CTHShortTensor) -> IO (Ptr (CTHShortTensor)))

-- | p_newWithStorage : Pointer to function : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithStorage"
  p_newWithStorage :: FunPtr (Ptr (CTHShortStorage) -> CPtrdiff -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHShortTensor)))

-- | p_newWithStorage1d : Pointer to function : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithStorage1d"
  p_newWithStorage1d :: FunPtr (Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor)))

-- | p_newWithStorage2d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithStorage2d"
  p_newWithStorage2d :: FunPtr (Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor)))

-- | p_newWithStorage3d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithStorage3d"
  p_newWithStorage3d :: FunPtr (Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor)))

-- | p_newWithStorage4d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithStorage4d"
  p_newWithStorage4d :: FunPtr (Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor)))

-- | p_newWithSize : Pointer to function : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithSize"
  p_newWithSize :: FunPtr (Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHShortTensor)))

-- | p_newWithSize1d : Pointer to function : size0_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithSize1d"
  p_newWithSize1d :: FunPtr (CLLong -> IO (Ptr (CTHShortTensor)))

-- | p_newWithSize2d : Pointer to function : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithSize2d"
  p_newWithSize2d :: FunPtr (CLLong -> CLLong -> IO (Ptr (CTHShortTensor)))

-- | p_newWithSize3d : Pointer to function : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithSize3d"
  p_newWithSize3d :: FunPtr (CLLong -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor)))

-- | p_newWithSize4d : Pointer to function : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newWithSize4d"
  p_newWithSize4d :: FunPtr (CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor)))

-- | p_newClone : Pointer to function : self -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newClone"
  p_newClone :: FunPtr (Ptr (CTHShortTensor) -> IO (Ptr (CTHShortTensor)))

-- | p_newContiguous : Pointer to function : tensor -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newContiguous"
  p_newContiguous :: FunPtr (Ptr (CTHShortTensor) -> IO (Ptr (CTHShortTensor)))

-- | p_newSelect : Pointer to function : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newSelect"
  p_newSelect :: FunPtr (Ptr (CTHShortTensor) -> CInt -> CLLong -> IO (Ptr (CTHShortTensor)))

-- | p_newNarrow : Pointer to function : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newNarrow"
  p_newNarrow :: FunPtr (Ptr (CTHShortTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor)))

-- | p_newTranspose : Pointer to function : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newTranspose"
  p_newTranspose :: FunPtr (Ptr (CTHShortTensor) -> CInt -> CInt -> IO (Ptr (CTHShortTensor)))

-- | p_newUnfold : Pointer to function : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newUnfold"
  p_newUnfold :: FunPtr (Ptr (CTHShortTensor) -> CInt -> CLLong -> CLLong -> IO (Ptr (CTHShortTensor)))

-- | p_newView : Pointer to function : tensor size -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newView"
  p_newView :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHShortTensor)))

-- | p_newExpand : Pointer to function : tensor size -> THTensor *
foreign import ccall "THTensor.h &THShortTensor_newExpand"
  p_newExpand :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHLongStorage) -> IO (Ptr (CTHShortTensor)))

-- | p_expand : Pointer to function : r tensor size -> void
foreign import ccall "THTensor.h &THShortTensor_expand"
  p_expand :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_expandNd : Pointer to function : rets ops count -> void
foreign import ccall "THTensor.h &THShortTensor_expandNd"
  p_expandNd :: FunPtr (Ptr (Ptr (CTHShortTensor)) -> Ptr (Ptr (CTHShortTensor)) -> CInt -> IO (()))

-- | p_resize : Pointer to function : tensor size stride -> void
foreign import ccall "THTensor.h &THShortTensor_resize"
  p_resize :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_resizeAs : Pointer to function : tensor src -> void
foreign import ccall "THTensor.h &THShortTensor_resizeAs"
  p_resizeAs :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_resizeNd : Pointer to function : tensor nDimension size stride -> void
foreign import ccall "THTensor.h &THShortTensor_resizeNd"
  p_resizeNd :: FunPtr (Ptr (CTHShortTensor) -> CInt -> Ptr (CLLong) -> Ptr (CLLong) -> IO (()))

-- | p_resize1d : Pointer to function : tensor size0_ -> void
foreign import ccall "THTensor.h &THShortTensor_resize1d"
  p_resize1d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> IO (()))

-- | p_resize2d : Pointer to function : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h &THShortTensor_resize2d"
  p_resize2d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> CLLong -> IO (()))

-- | p_resize3d : Pointer to function : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h &THShortTensor_resize3d"
  p_resize3d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> IO (()))

-- | p_resize4d : Pointer to function : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h &THShortTensor_resize4d"
  p_resize4d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> IO (()))

-- | p_resize5d : Pointer to function : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h &THShortTensor_resize5d"
  p_resize5d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (()))

-- | p_set : Pointer to function : self src -> void
foreign import ccall "THTensor.h &THShortTensor_set"
  p_set :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_setStorage : Pointer to function : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h &THShortTensor_setStorage"
  p_setStorage :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortStorage) -> CPtrdiff -> Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_setStorageNd : Pointer to function : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h &THShortTensor_setStorageNd"
  p_setStorageNd :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortStorage) -> CPtrdiff -> CInt -> Ptr (CLLong) -> Ptr (CLLong) -> IO (()))

-- | p_setStorage1d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h &THShortTensor_setStorage1d"
  p_setStorage1d :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> IO (()))

-- | p_setStorage2d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h &THShortTensor_setStorage2d"
  p_setStorage2d :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (()))

-- | p_setStorage3d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h &THShortTensor_setStorage3d"
  p_setStorage3d :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (()))

-- | p_setStorage4d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h &THShortTensor_setStorage4d"
  p_setStorage4d :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortStorage) -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (()))

-- | p_narrow : Pointer to function : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h &THShortTensor_narrow"
  p_narrow :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> CLLong -> CLLong -> IO (()))

-- | p_select : Pointer to function : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h &THShortTensor_select"
  p_select :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> CLLong -> IO (()))

-- | p_transpose : Pointer to function : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h &THShortTensor_transpose"
  p_transpose :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (()))

-- | p_unfold : Pointer to function : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h &THShortTensor_unfold"
  p_unfold :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> CLLong -> CLLong -> IO (()))

-- | p_squeeze : Pointer to function : self src -> void
foreign import ccall "THTensor.h &THShortTensor_squeeze"
  p_squeeze :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_squeeze1d : Pointer to function : self src dimension_ -> void
foreign import ccall "THTensor.h &THShortTensor_squeeze1d"
  p_squeeze1d :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> IO (()))

-- | p_unsqueeze1d : Pointer to function : self src dimension_ -> void
foreign import ccall "THTensor.h &THShortTensor_unsqueeze1d"
  p_unsqueeze1d :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> IO (()))

-- | p_isContiguous : Pointer to function : self -> int
foreign import ccall "THTensor.h &THShortTensor_isContiguous"
  p_isContiguous :: FunPtr (Ptr (CTHShortTensor) -> IO (CInt))

-- | p_isSameSizeAs : Pointer to function : self src -> int
foreign import ccall "THTensor.h &THShortTensor_isSameSizeAs"
  p_isSameSizeAs :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (CInt))

-- | p_isSetTo : Pointer to function : self src -> int
foreign import ccall "THTensor.h &THShortTensor_isSetTo"
  p_isSetTo :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (CInt))

-- | p_isSize : Pointer to function : self dims -> int
foreign import ccall "THTensor.h &THShortTensor_isSize"
  p_isSize :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHLongStorage) -> IO (CInt))

-- | p_nElement : Pointer to function : self -> ptrdiff_t
foreign import ccall "THTensor.h &THShortTensor_nElement"
  p_nElement :: FunPtr (Ptr (CTHShortTensor) -> IO (CPtrdiff))

-- | p_retain : Pointer to function : self -> void
foreign import ccall "THTensor.h &THShortTensor_retain"
  p_retain :: FunPtr (Ptr (CTHShortTensor) -> IO (()))

-- | p_free : Pointer to function : self -> void
foreign import ccall "THTensor.h &THShortTensor_free"
  p_free :: FunPtr (Ptr (CTHShortTensor) -> IO (()))

-- | p_freeCopyTo : Pointer to function : self dst -> void
foreign import ccall "THTensor.h &THShortTensor_freeCopyTo"
  p_freeCopyTo :: FunPtr (Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_set1d : Pointer to function : tensor x0 value -> void
foreign import ccall "THTensor.h &THShortTensor_set1d"
  p_set1d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> CShort -> IO (()))

-- | p_set2d : Pointer to function : tensor x0 x1 value -> void
foreign import ccall "THTensor.h &THShortTensor_set2d"
  p_set2d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> CLLong -> CShort -> IO (()))

-- | p_set3d : Pointer to function : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h &THShortTensor_set3d"
  p_set3d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> CShort -> IO (()))

-- | p_set4d : Pointer to function : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h &THShortTensor_set4d"
  p_set4d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> CShort -> IO (()))

-- | p_get1d : Pointer to function : tensor x0 -> real
foreign import ccall "THTensor.h &THShortTensor_get1d"
  p_get1d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> IO (CShort))

-- | p_get2d : Pointer to function : tensor x0 x1 -> real
foreign import ccall "THTensor.h &THShortTensor_get2d"
  p_get2d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> CLLong -> IO (CShort))

-- | p_get3d : Pointer to function : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h &THShortTensor_get3d"
  p_get3d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> IO (CShort))

-- | p_get4d : Pointer to function : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h &THShortTensor_get4d"
  p_get4d :: FunPtr (Ptr (CTHShortTensor) -> CLLong -> CLLong -> CLLong -> CLLong -> IO (CShort))

-- | p_desc : Pointer to function : tensor -> THDescBuff
foreign import ccall "THTensor.h &THShortTensor_desc"
  p_desc :: FunPtr (Ptr (CTHShortTensor) -> IO (CTHDescBuff))

-- | p_sizeDesc : Pointer to function : tensor -> THDescBuff
foreign import ccall "THTensor.h &THShortTensor_sizeDesc"
  p_sizeDesc :: FunPtr (Ptr (CTHShortTensor) -> IO (CTHDescBuff))