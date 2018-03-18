{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.Tensor where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_storage :  state self -> THCStorage *
foreign import ccall "THCTensor.h THCIntTensor_storage"
  c_storage :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCIntStorage)

-- | c_storageOffset :  state self -> ptrdiff_t
foreign import ccall "THCTensor.h THCIntTensor_storageOffset"
  c_storageOffset :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO CPtrdiff

-- | c_nDimension :  state self -> int
foreign import ccall "THCTensor.h THCIntTensor_nDimension"
  c_nDimension :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO CInt

-- | c_size :  state self dim -> int64_t
foreign import ccall "THCTensor.h THCIntTensor_size"
  c_size :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> IO CLLong

-- | c_stride :  state self dim -> int64_t
foreign import ccall "THCTensor.h THCIntTensor_stride"
  c_stride :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> IO CLLong

-- | c_newSizeOf :  state self -> THCLongStorage *
foreign import ccall "THCTensor.h THCIntTensor_newSizeOf"
  c_newSizeOf :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCLongStorage)

-- | c_newStrideOf :  state self -> THCLongStorage *
foreign import ccall "THCTensor.h THCIntTensor_newStrideOf"
  c_newStrideOf :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCLongStorage)

-- | c_data :  state self -> real *
foreign import ccall "THCTensor.h THCIntTensor_data"
  c_data :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr CInt)

-- | c_setFlag :  state self flag -> void
foreign import ccall "THCTensor.h THCIntTensor_setFlag"
  c_setFlag :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CChar -> IO ()

-- | c_clearFlag :  state self flag -> void
foreign import ccall "THCTensor.h THCIntTensor_clearFlag"
  c_clearFlag :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CChar -> IO ()

-- | c_new :  state -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_new"
  c_new :: Ptr C'THCState -> IO (Ptr C'THCudaIntTensor)

-- | c_newWithTensor :  state tensor -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newWithTensor"
  c_newWithTensor :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCudaIntTensor)

-- | c_newWithStorage :  state storage_ storageOffset_ size_ stride_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newWithStorage"
  c_newWithStorage :: Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> Ptr C'THCLongStorage -> Ptr C'THCLongStorage -> IO (Ptr C'THCudaIntTensor)

-- | c_newWithStorage1d :  state storage_ storageOffset_ size0_ stride0_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newWithStorage1d"
  c_newWithStorage1d :: Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor)

-- | c_newWithStorage2d :  state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newWithStorage2d"
  c_newWithStorage2d :: Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor)

-- | c_newWithStorage3d :  state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newWithStorage3d"
  c_newWithStorage3d :: Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor)

-- | c_newWithStorage4d :  state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newWithStorage4d"
  c_newWithStorage4d :: Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor)

-- | c_newWithSize :  state size_ stride_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newWithSize"
  c_newWithSize :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THCLongStorage -> IO (Ptr C'THCudaIntTensor)

-- | c_newWithSize1d :  state size0_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newWithSize1d"
  c_newWithSize1d :: Ptr C'THCState -> CLLong -> IO (Ptr C'THCudaIntTensor)

-- | c_newWithSize2d :  state size0_ size1_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newWithSize2d"
  c_newWithSize2d :: Ptr C'THCState -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor)

-- | c_newWithSize3d :  state size0_ size1_ size2_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newWithSize3d"
  c_newWithSize3d :: Ptr C'THCState -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor)

-- | c_newWithSize4d :  state size0_ size1_ size2_ size3_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newWithSize4d"
  c_newWithSize4d :: Ptr C'THCState -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor)

-- | c_newClone :  state self -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newClone"
  c_newClone :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCudaIntTensor)

-- | c_newContiguous :  state tensor -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newContiguous"
  c_newContiguous :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCudaIntTensor)

-- | c_newSelect :  state tensor dimension_ sliceIndex_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newSelect"
  c_newSelect :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> CLLong -> IO (Ptr C'THCudaIntTensor)

-- | c_newNarrow :  state tensor dimension_ firstIndex_ size_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newNarrow"
  c_newNarrow :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor)

-- | c_newTranspose :  state tensor dimension1_ dimension2_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newTranspose"
  c_newTranspose :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> CInt -> IO (Ptr C'THCudaIntTensor)

-- | c_newUnfold :  state tensor dimension_ size_ step_ -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newUnfold"
  c_newUnfold :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor)

-- | c_newView :  state tensor size -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newView"
  c_newView :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCLongStorage -> IO (Ptr C'THCudaIntTensor)

-- | c_newExpand :  state tensor size -> THCTensor *
foreign import ccall "THCTensor.h THCIntTensor_newExpand"
  c_newExpand :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCLongStorage -> IO (Ptr C'THCudaIntTensor)

-- | c_expand :  state r tensor sizes -> void
foreign import ccall "THCTensor.h THCIntTensor_expand"
  c_expand :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCLongStorage -> IO ()

-- | c_expandNd :  state rets ops count -> void
foreign import ccall "THCTensor.h THCIntTensor_expandNd"
  c_expandNd :: Ptr C'THCState -> Ptr (Ptr C'THCudaIntTensor) -> Ptr (Ptr C'THCudaIntTensor) -> CInt -> IO ()

-- | c_resize :  state tensor size stride -> void
foreign import ccall "THCTensor.h THCIntTensor_resize"
  c_resize :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCLongStorage -> Ptr C'THCLongStorage -> IO ()

-- | c_resizeAs :  state tensor src -> void
foreign import ccall "THCTensor.h THCIntTensor_resizeAs"
  c_resizeAs :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_resize1d :  state tensor size0_ -> void
foreign import ccall "THCTensor.h THCIntTensor_resize1d"
  c_resize1d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> IO ()

-- | c_resize2d :  state tensor size0_ size1_ -> void
foreign import ccall "THCTensor.h THCIntTensor_resize2d"
  c_resize2d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> IO ()

-- | c_resize3d :  state tensor size0_ size1_ size2_ -> void
foreign import ccall "THCTensor.h THCIntTensor_resize3d"
  c_resize3d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_resize4d :  state tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THCTensor.h THCIntTensor_resize4d"
  c_resize4d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_resize5d :  state tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THCTensor.h THCIntTensor_resize5d"
  c_resize5d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_resizeNd :  state tensor nDimension size stride -> void
foreign import ccall "THCTensor.h THCIntTensor_resizeNd"
  c_resizeNd :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- | c_set :  state self src -> void
foreign import ccall "THCTensor.h THCIntTensor_set"
  c_set :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_setStorage :  state self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THCTensor.h THCIntTensor_setStorage"
  c_setStorage :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCIntStorage -> CPtrdiff -> Ptr C'THCLongStorage -> Ptr C'THCLongStorage -> IO ()

-- | c_setStorageNd :  state self storage storageOffset nDimension size stride -> void
foreign import ccall "THCTensor.h THCIntTensor_setStorageNd"
  c_setStorageNd :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCIntStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- | c_setStorage1d :  state self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THCTensor.h THCIntTensor_setStorage1d"
  c_setStorage1d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> IO ()

-- | c_setStorage2d :  state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THCTensor.h THCIntTensor_setStorage2d"
  c_setStorage2d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_setStorage3d :  state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THCTensor.h THCIntTensor_setStorage3d"
  c_setStorage3d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_setStorage4d :  state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THCTensor.h THCIntTensor_setStorage4d"
  c_setStorage4d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_narrow :  state self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THCTensor.h THCIntTensor_narrow"
  c_narrow :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> CLLong -> CLLong -> IO ()

-- | c_select :  state self src dimension_ sliceIndex_ -> void
foreign import ccall "THCTensor.h THCIntTensor_select"
  c_select :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> CLLong -> IO ()

-- | c_transpose :  state self src dimension1_ dimension2_ -> void
foreign import ccall "THCTensor.h THCIntTensor_transpose"
  c_transpose :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> CInt -> IO ()

-- | c_unfold :  state self src dimension_ size_ step_ -> void
foreign import ccall "THCTensor.h THCIntTensor_unfold"
  c_unfold :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> CLLong -> CLLong -> IO ()

-- | c_squeeze :  state self src -> void
foreign import ccall "THCTensor.h THCIntTensor_squeeze"
  c_squeeze :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_squeeze1d :  state self src dimension_ -> void
foreign import ccall "THCTensor.h THCIntTensor_squeeze1d"
  c_squeeze1d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> IO ()

-- | c_unsqueeze1d :  state self src dimension_ -> void
foreign import ccall "THCTensor.h THCIntTensor_unsqueeze1d"
  c_unsqueeze1d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> IO ()

-- | c_isContiguous :  state self -> int
foreign import ccall "THCTensor.h THCIntTensor_isContiguous"
  c_isContiguous :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO CInt

-- | c_isSameSizeAs :  state self src -> int
foreign import ccall "THCTensor.h THCIntTensor_isSameSizeAs"
  c_isSameSizeAs :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO CInt

-- | c_isSetTo :  state self src -> int
foreign import ccall "THCTensor.h THCIntTensor_isSetTo"
  c_isSetTo :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO CInt

-- | c_isSize :  state self dims -> int
foreign import ccall "THCTensor.h THCIntTensor_isSize"
  c_isSize :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCLongStorage -> IO CInt

-- | c_nElement :  state self -> ptrdiff_t
foreign import ccall "THCTensor.h THCIntTensor_nElement"
  c_nElement :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO CPtrdiff

-- | c_retain :  state self -> void
foreign import ccall "THCTensor.h THCIntTensor_retain"
  c_retain :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO ()

-- | c_free :  state self -> void
foreign import ccall "THCTensor.h THCIntTensor_free"
  c_free :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO ()

-- | c_freeCopyTo :  state self dst -> void
foreign import ccall "THCTensor.h THCIntTensor_freeCopyTo"
  c_freeCopyTo :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_set1d :  state tensor x0 value -> void
foreign import ccall "THCTensor.h THCIntTensor_set1d"
  c_set1d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CInt -> IO ()

-- | c_set2d :  state tensor x0 x1 value -> void
foreign import ccall "THCTensor.h THCIntTensor_set2d"
  c_set2d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CInt -> IO ()

-- | c_set3d :  state tensor x0 x1 x2 value -> void
foreign import ccall "THCTensor.h THCIntTensor_set3d"
  c_set3d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> CInt -> IO ()

-- | c_set4d :  state tensor x0 x1 x2 x3 value -> void
foreign import ccall "THCTensor.h THCIntTensor_set4d"
  c_set4d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CInt -> IO ()

-- | c_get1d :  state tensor x0 -> real
foreign import ccall "THCTensor.h THCIntTensor_get1d"
  c_get1d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> IO CInt

-- | c_get2d :  state tensor x0 x1 -> real
foreign import ccall "THCTensor.h THCIntTensor_get2d"
  c_get2d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> IO CInt

-- | c_get3d :  state tensor x0 x1 x2 -> real
foreign import ccall "THCTensor.h THCIntTensor_get3d"
  c_get3d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> IO CInt

-- | c_get4d :  state tensor x0 x1 x2 x3 -> real
foreign import ccall "THCTensor.h THCIntTensor_get4d"
  c_get4d :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO CInt

-- | c_getDevice :  state self -> int
foreign import ccall "THCTensor.h THCIntTensor_getDevice"
  c_getDevice :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO CInt

-- | c_sizeDesc :  state tensor -> THCDescBuff
foreign import ccall "THCTensor.h THCIntTensor_sizeDesc"
  c_sizeDesc :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCDescBuff)

-- | p_storage : Pointer to function : state self -> THCStorage *
foreign import ccall "THCTensor.h &THCIntTensor_storage"
  p_storage :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCIntStorage))

-- | p_storageOffset : Pointer to function : state self -> ptrdiff_t
foreign import ccall "THCTensor.h &THCIntTensor_storageOffset"
  p_storageOffset :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO CPtrdiff)

-- | p_nDimension : Pointer to function : state self -> int
foreign import ccall "THCTensor.h &THCIntTensor_nDimension"
  p_nDimension :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO CInt)

-- | p_size : Pointer to function : state self dim -> int64_t
foreign import ccall "THCTensor.h &THCIntTensor_size"
  p_size :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> IO CLLong)

-- | p_stride : Pointer to function : state self dim -> int64_t
foreign import ccall "THCTensor.h &THCIntTensor_stride"
  p_stride :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> IO CLLong)

-- | p_newSizeOf : Pointer to function : state self -> THCLongStorage *
foreign import ccall "THCTensor.h &THCIntTensor_newSizeOf"
  p_newSizeOf :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCLongStorage))

-- | p_newStrideOf : Pointer to function : state self -> THCLongStorage *
foreign import ccall "THCTensor.h &THCIntTensor_newStrideOf"
  p_newStrideOf :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCLongStorage))

-- | p_data : Pointer to function : state self -> real *
foreign import ccall "THCTensor.h &THCIntTensor_data"
  p_data :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr CInt))

-- | p_setFlag : Pointer to function : state self flag -> void
foreign import ccall "THCTensor.h &THCIntTensor_setFlag"
  p_setFlag :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state self flag -> void
foreign import ccall "THCTensor.h &THCIntTensor_clearFlag"
  p_clearFlag :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CChar -> IO ())

-- | p_new : Pointer to function : state -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_new"
  p_new :: FunPtr (Ptr C'THCState -> IO (Ptr C'THCudaIntTensor))

-- | p_newWithTensor : Pointer to function : state tensor -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newWithTensor"
  p_newWithTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCudaIntTensor))

-- | p_newWithStorage : Pointer to function : state storage_ storageOffset_ size_ stride_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newWithStorage"
  p_newWithStorage :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> Ptr C'THCLongStorage -> Ptr C'THCLongStorage -> IO (Ptr C'THCudaIntTensor))

-- | p_newWithStorage1d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newWithStorage1d"
  p_newWithStorage1d :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor))

-- | p_newWithStorage2d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newWithStorage2d"
  p_newWithStorage2d :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor))

-- | p_newWithStorage3d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newWithStorage3d"
  p_newWithStorage3d :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor))

-- | p_newWithStorage4d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newWithStorage4d"
  p_newWithStorage4d :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor))

-- | p_newWithSize : Pointer to function : state size_ stride_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newWithSize"
  p_newWithSize :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THCLongStorage -> IO (Ptr C'THCudaIntTensor))

-- | p_newWithSize1d : Pointer to function : state size0_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newWithSize1d"
  p_newWithSize1d :: FunPtr (Ptr C'THCState -> CLLong -> IO (Ptr C'THCudaIntTensor))

-- | p_newWithSize2d : Pointer to function : state size0_ size1_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newWithSize2d"
  p_newWithSize2d :: FunPtr (Ptr C'THCState -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor))

-- | p_newWithSize3d : Pointer to function : state size0_ size1_ size2_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newWithSize3d"
  p_newWithSize3d :: FunPtr (Ptr C'THCState -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor))

-- | p_newWithSize4d : Pointer to function : state size0_ size1_ size2_ size3_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newWithSize4d"
  p_newWithSize4d :: FunPtr (Ptr C'THCState -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor))

-- | p_newClone : Pointer to function : state self -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newClone"
  p_newClone :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCudaIntTensor))

-- | p_newContiguous : Pointer to function : state tensor -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newContiguous"
  p_newContiguous :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCudaIntTensor))

-- | p_newSelect : Pointer to function : state tensor dimension_ sliceIndex_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newSelect"
  p_newSelect :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> CLLong -> IO (Ptr C'THCudaIntTensor))

-- | p_newNarrow : Pointer to function : state tensor dimension_ firstIndex_ size_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newNarrow"
  p_newNarrow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor))

-- | p_newTranspose : Pointer to function : state tensor dimension1_ dimension2_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newTranspose"
  p_newTranspose :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> CInt -> IO (Ptr C'THCudaIntTensor))

-- | p_newUnfold : Pointer to function : state tensor dimension_ size_ step_ -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newUnfold"
  p_newUnfold :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THCudaIntTensor))

-- | p_newView : Pointer to function : state tensor size -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newView"
  p_newView :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCLongStorage -> IO (Ptr C'THCudaIntTensor))

-- | p_newExpand : Pointer to function : state tensor size -> THCTensor *
foreign import ccall "THCTensor.h &THCIntTensor_newExpand"
  p_newExpand :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCLongStorage -> IO (Ptr C'THCudaIntTensor))

-- | p_expand : Pointer to function : state r tensor sizes -> void
foreign import ccall "THCTensor.h &THCIntTensor_expand"
  p_expand :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCLongStorage -> IO ())

-- | p_expandNd : Pointer to function : state rets ops count -> void
foreign import ccall "THCTensor.h &THCIntTensor_expandNd"
  p_expandNd :: FunPtr (Ptr C'THCState -> Ptr (Ptr C'THCudaIntTensor) -> Ptr (Ptr C'THCudaIntTensor) -> CInt -> IO ())

-- | p_resize : Pointer to function : state tensor size stride -> void
foreign import ccall "THCTensor.h &THCIntTensor_resize"
  p_resize :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCLongStorage -> Ptr C'THCLongStorage -> IO ())

-- | p_resizeAs : Pointer to function : state tensor src -> void
foreign import ccall "THCTensor.h &THCIntTensor_resizeAs"
  p_resizeAs :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_resize1d : Pointer to function : state tensor size0_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_resize1d"
  p_resize1d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> IO ())

-- | p_resize2d : Pointer to function : state tensor size0_ size1_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_resize2d"
  p_resize2d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> IO ())

-- | p_resize3d : Pointer to function : state tensor size0_ size1_ size2_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_resize3d"
  p_resize3d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_resize4d : Pointer to function : state tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_resize4d"
  p_resize4d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_resize5d : Pointer to function : state tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_resize5d"
  p_resize5d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_resizeNd : Pointer to function : state tensor nDimension size stride -> void
foreign import ccall "THCTensor.h &THCIntTensor_resizeNd"
  p_resizeNd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- | p_set : Pointer to function : state self src -> void
foreign import ccall "THCTensor.h &THCIntTensor_set"
  p_set :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_setStorage : Pointer to function : state self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_setStorage"
  p_setStorage :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCIntStorage -> CPtrdiff -> Ptr C'THCLongStorage -> Ptr C'THCLongStorage -> IO ())

-- | p_setStorageNd : Pointer to function : state self storage storageOffset nDimension size stride -> void
foreign import ccall "THCTensor.h &THCIntTensor_setStorageNd"
  p_setStorageNd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCIntStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- | p_setStorage1d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_setStorage1d"
  p_setStorage1d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> IO ())

-- | p_setStorage2d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_setStorage2d"
  p_setStorage2d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_setStorage3d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_setStorage3d"
  p_setStorage3d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_setStorage4d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_setStorage4d"
  p_setStorage4d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_narrow : Pointer to function : state self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_narrow"
  p_narrow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> CLLong -> CLLong -> IO ())

-- | p_select : Pointer to function : state self src dimension_ sliceIndex_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_select"
  p_select :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> CLLong -> IO ())

-- | p_transpose : Pointer to function : state self src dimension1_ dimension2_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_transpose"
  p_transpose :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> CInt -> IO ())

-- | p_unfold : Pointer to function : state self src dimension_ size_ step_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_unfold"
  p_unfold :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> CLLong -> CLLong -> IO ())

-- | p_squeeze : Pointer to function : state self src -> void
foreign import ccall "THCTensor.h &THCIntTensor_squeeze"
  p_squeeze :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_squeeze1d : Pointer to function : state self src dimension_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_squeeze1d"
  p_squeeze1d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> IO ())

-- | p_unsqueeze1d : Pointer to function : state self src dimension_ -> void
foreign import ccall "THCTensor.h &THCIntTensor_unsqueeze1d"
  p_unsqueeze1d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> IO ())

-- | p_isContiguous : Pointer to function : state self -> int
foreign import ccall "THCTensor.h &THCIntTensor_isContiguous"
  p_isContiguous :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO CInt)

-- | p_isSameSizeAs : Pointer to function : state self src -> int
foreign import ccall "THCTensor.h &THCIntTensor_isSameSizeAs"
  p_isSameSizeAs :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO CInt)

-- | p_isSetTo : Pointer to function : state self src -> int
foreign import ccall "THCTensor.h &THCIntTensor_isSetTo"
  p_isSetTo :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO CInt)

-- | p_isSize : Pointer to function : state self dims -> int
foreign import ccall "THCTensor.h &THCIntTensor_isSize"
  p_isSize :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCLongStorage -> IO CInt)

-- | p_nElement : Pointer to function : state self -> ptrdiff_t
foreign import ccall "THCTensor.h &THCIntTensor_nElement"
  p_nElement :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO CPtrdiff)

-- | p_retain : Pointer to function : state self -> void
foreign import ccall "THCTensor.h &THCIntTensor_retain"
  p_retain :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO ())

-- | p_free : Pointer to function : state self -> void
foreign import ccall "THCTensor.h &THCIntTensor_free"
  p_free :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO ())

-- | p_freeCopyTo : Pointer to function : state self dst -> void
foreign import ccall "THCTensor.h &THCIntTensor_freeCopyTo"
  p_freeCopyTo :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_set1d : Pointer to function : state tensor x0 value -> void
foreign import ccall "THCTensor.h &THCIntTensor_set1d"
  p_set1d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CInt -> IO ())

-- | p_set2d : Pointer to function : state tensor x0 x1 value -> void
foreign import ccall "THCTensor.h &THCIntTensor_set2d"
  p_set2d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CInt -> IO ())

-- | p_set3d : Pointer to function : state tensor x0 x1 x2 value -> void
foreign import ccall "THCTensor.h &THCIntTensor_set3d"
  p_set3d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> CInt -> IO ())

-- | p_set4d : Pointer to function : state tensor x0 x1 x2 x3 value -> void
foreign import ccall "THCTensor.h &THCIntTensor_set4d"
  p_set4d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CInt -> IO ())

-- | p_get1d : Pointer to function : state tensor x0 -> real
foreign import ccall "THCTensor.h &THCIntTensor_get1d"
  p_get1d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> IO CInt)

-- | p_get2d : Pointer to function : state tensor x0 x1 -> real
foreign import ccall "THCTensor.h &THCIntTensor_get2d"
  p_get2d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> IO CInt)

-- | p_get3d : Pointer to function : state tensor x0 x1 x2 -> real
foreign import ccall "THCTensor.h &THCIntTensor_get3d"
  p_get3d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> IO CInt)

-- | p_get4d : Pointer to function : state tensor x0 x1 x2 x3 -> real
foreign import ccall "THCTensor.h &THCIntTensor_get4d"
  p_get4d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO CInt)

-- | p_getDevice : Pointer to function : state self -> int
foreign import ccall "THCTensor.h &THCIntTensor_getDevice"
  p_getDevice :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO CInt)

-- | p_sizeDesc : Pointer to function : state tensor -> THCDescBuff
foreign import ccall "THCTensor.h &THCIntTensor_sizeDesc"
  p_sizeDesc :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO (Ptr C'THCDescBuff))