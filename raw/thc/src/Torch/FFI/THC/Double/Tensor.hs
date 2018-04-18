{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.Tensor where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_storage :  state self -> THCStorage *
foreign import ccall "THCTensor.h THCudaDoubleTensor_storage"
  c_storage :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THCDoubleStorage)

-- | c_storageOffset :  state self -> ptrdiff_t
foreign import ccall "THCTensor.h THCudaDoubleTensor_storageOffset"
  c_storageOffset :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CPtrdiff

-- | c_nDimension :  state self -> int
foreign import ccall "THCTensor.h THCudaDoubleTensor_nDimension"
  c_nDimension :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CInt

-- | c_size :  state self dim -> int64_t
foreign import ccall "THCTensor.h THCudaDoubleTensor_size"
  c_size :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> IO CLLong

-- | c_stride :  state self dim -> int64_t
foreign import ccall "THCTensor.h THCudaDoubleTensor_stride"
  c_stride :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> IO CLLong

-- | c_newSizeOf :  state self -> THLongStorage *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newSizeOf"
  c_newSizeOf :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THLongStorage)

-- | c_newStrideOf :  state self -> THLongStorage *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newStrideOf"
  c_newStrideOf :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THLongStorage)

-- | c_data :  state self -> real *
foreign import ccall "THCTensor.h THCudaDoubleTensor_data"
  c_data :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr CDouble)

-- | c_setFlag :  state self flag -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_setFlag"
  c_setFlag :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CChar -> IO ()

-- | c_clearFlag :  state self flag -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_clearFlag"
  c_clearFlag :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CChar -> IO ()

-- | c_new :  state -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_new"
  c_new :: Ptr C'THCState -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newWithTensor :  state tensor -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newWithTensor"
  c_newWithTensor :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newWithStorage :  state storage_ storageOffset_ size_ stride_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newWithStorage"
  c_newWithStorage :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newWithStorage1d :  state storage_ storageOffset_ size0_ stride0_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newWithStorage1d"
  c_newWithStorage1d :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newWithStorage2d :  state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newWithStorage2d"
  c_newWithStorage2d :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newWithStorage3d :  state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newWithStorage3d"
  c_newWithStorage3d :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newWithStorage4d :  state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newWithStorage4d"
  c_newWithStorage4d :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newWithSize :  state size_ stride_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newWithSize"
  c_newWithSize :: Ptr C'THCState -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newWithSize1d :  state size0_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newWithSize1d"
  c_newWithSize1d :: Ptr C'THCState -> CLLong -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newWithSize2d :  state size0_ size1_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newWithSize2d"
  c_newWithSize2d :: Ptr C'THCState -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newWithSize3d :  state size0_ size1_ size2_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newWithSize3d"
  c_newWithSize3d :: Ptr C'THCState -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newWithSize4d :  state size0_ size1_ size2_ size3_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newWithSize4d"
  c_newWithSize4d :: Ptr C'THCState -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newClone :  state self -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newClone"
  c_newClone :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newContiguous :  state tensor -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newContiguous"
  c_newContiguous :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newSelect :  state tensor dimension_ sliceIndex_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newSelect"
  c_newSelect :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> CLLong -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newNarrow :  state tensor dimension_ firstIndex_ size_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newNarrow"
  c_newNarrow :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newTranspose :  state tensor dimension1_ dimension2_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newTranspose"
  c_newTranspose :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newUnfold :  state tensor dimension_ size_ step_ -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newUnfold"
  c_newUnfold :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newView :  state tensor size -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newView"
  c_newView :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newFoldBatchDim :  state input -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newFoldBatchDim"
  c_newFoldBatchDim :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THCudaDoubleTensor)

-- | c_newExpand :  state tensor size -> THCTensor *
foreign import ccall "THCTensor.h THCudaDoubleTensor_newExpand"
  c_newExpand :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO (Ptr C'THCudaDoubleTensor)

-- | c_expand :  state r tensor sizes -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_expand"
  c_expand :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO ()

-- | c_expandNd :  state rets ops count -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_expandNd"
  c_expandNd :: Ptr C'THCState -> Ptr (Ptr C'THCudaDoubleTensor) -> Ptr (Ptr C'THCudaDoubleTensor) -> CInt -> IO ()

-- | c_resize :  state tensor size stride -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_resize"
  c_resize :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ()

-- | c_resizeAs :  state tensor src -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_resizeAs"
  c_resizeAs :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_resize1d :  state tensor size0_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_resize1d"
  c_resize1d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> IO ()

-- | c_resize2d :  state tensor size0_ size1_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_resize2d"
  c_resize2d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> IO ()

-- | c_resize3d :  state tensor size0_ size1_ size2_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_resize3d"
  c_resize3d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_resize4d :  state tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_resize4d"
  c_resize4d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_resize5d :  state tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_resize5d"
  c_resize5d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_resizeNd :  state tensor nDimension size stride -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_resizeNd"
  c_resizeNd :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- | c_set :  state self src -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_set"
  c_set :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_setStorage :  state self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_setStorage"
  c_setStorage :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCDoubleStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ()

-- | c_setStorageNd :  state self storage storageOffset nDimension size stride -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_setStorageNd"
  c_setStorageNd :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCDoubleStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- | c_setStorage1d :  state self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_setStorage1d"
  c_setStorage1d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> IO ()

-- | c_setStorage2d :  state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_setStorage2d"
  c_setStorage2d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_setStorage3d :  state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_setStorage3d"
  c_setStorage3d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_setStorage4d :  state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_setStorage4d"
  c_setStorage4d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_narrow :  state self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_narrow"
  c_narrow :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CLLong -> CLLong -> IO ()

-- | c_select :  state self src dimension_ sliceIndex_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_select"
  c_select :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CLLong -> IO ()

-- | c_transpose :  state self src dimension1_ dimension2_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_transpose"
  c_transpose :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ()

-- | c_unfold :  state self src dimension_ size_ step_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_unfold"
  c_unfold :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CLLong -> CLLong -> IO ()

-- | c_squeeze :  state self src -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_squeeze"
  c_squeeze :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_squeeze1d :  state self src dimension_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_squeeze1d"
  c_squeeze1d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_unsqueeze1d :  state self src dimension_ -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_unsqueeze1d"
  c_unsqueeze1d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_isContiguous :  state self -> int
foreign import ccall "THCTensor.h THCudaDoubleTensor_isContiguous"
  c_isContiguous :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CInt

-- | c_isSameSizeAs :  state self src -> int
foreign import ccall "THCTensor.h THCudaDoubleTensor_isSameSizeAs"
  c_isSameSizeAs :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO CInt

-- | c_isSetTo :  state self src -> int
foreign import ccall "THCTensor.h THCudaDoubleTensor_isSetTo"
  c_isSetTo :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO CInt

-- | c_isSize :  state self dims -> int
foreign import ccall "THCTensor.h THCudaDoubleTensor_isSize"
  c_isSize :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO CInt

-- | c_nElement :  state self -> ptrdiff_t
foreign import ccall "THCTensor.h THCudaDoubleTensor_nElement"
  c_nElement :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CPtrdiff

-- | c_retain :  state self -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_retain"
  c_retain :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_free :  state self -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_free"
  c_free :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_freeCopyTo :  state self dst -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_freeCopyTo"
  c_freeCopyTo :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_set1d :  state tensor x0 value -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_set1d"
  c_set1d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CDouble -> IO ()

-- | c_set2d :  state tensor x0 x1 value -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_set2d"
  c_set2d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CDouble -> IO ()

-- | c_set3d :  state tensor x0 x1 x2 value -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_set3d"
  c_set3d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> CDouble -> IO ()

-- | c_set4d :  state tensor x0 x1 x2 x3 value -> void
foreign import ccall "THCTensor.h THCudaDoubleTensor_set4d"
  c_set4d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ()

-- | c_get1d :  state tensor x0 -> real
foreign import ccall "THCTensor.h THCudaDoubleTensor_get1d"
  c_get1d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> IO CDouble

-- | c_get2d :  state tensor x0 x1 -> real
foreign import ccall "THCTensor.h THCudaDoubleTensor_get2d"
  c_get2d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> IO CDouble

-- | c_get3d :  state tensor x0 x1 x2 -> real
foreign import ccall "THCTensor.h THCudaDoubleTensor_get3d"
  c_get3d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> IO CDouble

-- | c_get4d :  state tensor x0 x1 x2 x3 -> real
foreign import ccall "THCTensor.h THCudaDoubleTensor_get4d"
  c_get4d :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO CDouble

-- | c_getDevice :  state self -> int
foreign import ccall "THCTensor.h THCudaDoubleTensor_getDevice"
  c_getDevice :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CInt

-- | c_sizeDesc :  state tensor -> THCDescBuff
foreign import ccall "THCTensor.h THCudaDoubleTensor_sizeDesc"
  c_sizeDesc :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THCDescBuff)

-- | p_storage : Pointer to function : state self -> THCStorage *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_storage"
  p_storage :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THCDoubleStorage))

-- | p_storageOffset : Pointer to function : state self -> ptrdiff_t
foreign import ccall "THCTensor.h &THCudaDoubleTensor_storageOffset"
  p_storageOffset :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CPtrdiff)

-- | p_nDimension : Pointer to function : state self -> int
foreign import ccall "THCTensor.h &THCudaDoubleTensor_nDimension"
  p_nDimension :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CInt)

-- | p_size : Pointer to function : state self dim -> int64_t
foreign import ccall "THCTensor.h &THCudaDoubleTensor_size"
  p_size :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> IO CLLong)

-- | p_stride : Pointer to function : state self dim -> int64_t
foreign import ccall "THCTensor.h &THCudaDoubleTensor_stride"
  p_stride :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> IO CLLong)

-- | p_newSizeOf : Pointer to function : state self -> THLongStorage *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newSizeOf"
  p_newSizeOf :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THLongStorage))

-- | p_newStrideOf : Pointer to function : state self -> THLongStorage *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newStrideOf"
  p_newStrideOf :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THLongStorage))

-- | p_data : Pointer to function : state self -> real *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_data"
  p_data :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr CDouble))

-- | p_setFlag : Pointer to function : state self flag -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_setFlag"
  p_setFlag :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state self flag -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_clearFlag"
  p_clearFlag :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CChar -> IO ())

-- | p_new : Pointer to function : state -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_new"
  p_new :: FunPtr (Ptr C'THCState -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newWithTensor : Pointer to function : state tensor -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newWithTensor"
  p_newWithTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newWithStorage : Pointer to function : state storage_ storageOffset_ size_ stride_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newWithStorage"
  p_newWithStorage :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newWithStorage1d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newWithStorage1d"
  p_newWithStorage1d :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newWithStorage2d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newWithStorage2d"
  p_newWithStorage2d :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newWithStorage3d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newWithStorage3d"
  p_newWithStorage3d :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newWithStorage4d : Pointer to function : state storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newWithStorage4d"
  p_newWithStorage4d :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newWithSize : Pointer to function : state size_ stride_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newWithSize"
  p_newWithSize :: FunPtr (Ptr C'THCState -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newWithSize1d : Pointer to function : state size0_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newWithSize1d"
  p_newWithSize1d :: FunPtr (Ptr C'THCState -> CLLong -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newWithSize2d : Pointer to function : state size0_ size1_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newWithSize2d"
  p_newWithSize2d :: FunPtr (Ptr C'THCState -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newWithSize3d : Pointer to function : state size0_ size1_ size2_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newWithSize3d"
  p_newWithSize3d :: FunPtr (Ptr C'THCState -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newWithSize4d : Pointer to function : state size0_ size1_ size2_ size3_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newWithSize4d"
  p_newWithSize4d :: FunPtr (Ptr C'THCState -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newClone : Pointer to function : state self -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newClone"
  p_newClone :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newContiguous : Pointer to function : state tensor -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newContiguous"
  p_newContiguous :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newSelect : Pointer to function : state tensor dimension_ sliceIndex_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newSelect"
  p_newSelect :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> CLLong -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newNarrow : Pointer to function : state tensor dimension_ firstIndex_ size_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newNarrow"
  p_newNarrow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newTranspose : Pointer to function : state tensor dimension1_ dimension2_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newTranspose"
  p_newTranspose :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newUnfold : Pointer to function : state tensor dimension_ size_ step_ -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newUnfold"
  p_newUnfold :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newView : Pointer to function : state tensor size -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newView"
  p_newView :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newFoldBatchDim : Pointer to function : state input -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newFoldBatchDim"
  p_newFoldBatchDim :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THCudaDoubleTensor))

-- | p_newExpand : Pointer to function : state tensor size -> THCTensor *
foreign import ccall "THCTensor.h &THCudaDoubleTensor_newExpand"
  p_newExpand :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO (Ptr C'THCudaDoubleTensor))

-- | p_expand : Pointer to function : state r tensor sizes -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_expand"
  p_expand :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO ())

-- | p_expandNd : Pointer to function : state rets ops count -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_expandNd"
  p_expandNd :: FunPtr (Ptr C'THCState -> Ptr (Ptr C'THCudaDoubleTensor) -> Ptr (Ptr C'THCudaDoubleTensor) -> CInt -> IO ())

-- | p_resize : Pointer to function : state tensor size stride -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_resize"
  p_resize :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ())

-- | p_resizeAs : Pointer to function : state tensor src -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_resizeAs"
  p_resizeAs :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_resize1d : Pointer to function : state tensor size0_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_resize1d"
  p_resize1d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> IO ())

-- | p_resize2d : Pointer to function : state tensor size0_ size1_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_resize2d"
  p_resize2d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> IO ())

-- | p_resize3d : Pointer to function : state tensor size0_ size1_ size2_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_resize3d"
  p_resize3d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_resize4d : Pointer to function : state tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_resize4d"
  p_resize4d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_resize5d : Pointer to function : state tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_resize5d"
  p_resize5d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_resizeNd : Pointer to function : state tensor nDimension size stride -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_resizeNd"
  p_resizeNd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- | p_set : Pointer to function : state self src -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_set"
  p_set :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_setStorage : Pointer to function : state self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_setStorage"
  p_setStorage :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCDoubleStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ())

-- | p_setStorageNd : Pointer to function : state self storage storageOffset nDimension size stride -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_setStorageNd"
  p_setStorageNd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCDoubleStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- | p_setStorage1d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_setStorage1d"
  p_setStorage1d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> IO ())

-- | p_setStorage2d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_setStorage2d"
  p_setStorage2d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_setStorage3d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_setStorage3d"
  p_setStorage3d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_setStorage4d : Pointer to function : state self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_setStorage4d"
  p_setStorage4d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCDoubleStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_narrow : Pointer to function : state self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_narrow"
  p_narrow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CLLong -> CLLong -> IO ())

-- | p_select : Pointer to function : state self src dimension_ sliceIndex_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_select"
  p_select :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CLLong -> IO ())

-- | p_transpose : Pointer to function : state self src dimension1_ dimension2_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_transpose"
  p_transpose :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ())

-- | p_unfold : Pointer to function : state self src dimension_ size_ step_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_unfold"
  p_unfold :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CLLong -> CLLong -> IO ())

-- | p_squeeze : Pointer to function : state self src -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_squeeze"
  p_squeeze :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_squeeze1d : Pointer to function : state self src dimension_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_squeeze1d"
  p_squeeze1d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_unsqueeze1d : Pointer to function : state self src dimension_ -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_unsqueeze1d"
  p_unsqueeze1d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_isContiguous : Pointer to function : state self -> int
foreign import ccall "THCTensor.h &THCudaDoubleTensor_isContiguous"
  p_isContiguous :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CInt)

-- | p_isSameSizeAs : Pointer to function : state self src -> int
foreign import ccall "THCTensor.h &THCudaDoubleTensor_isSameSizeAs"
  p_isSameSizeAs :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO CInt)

-- | p_isSetTo : Pointer to function : state self src -> int
foreign import ccall "THCTensor.h &THCudaDoubleTensor_isSetTo"
  p_isSetTo :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO CInt)

-- | p_isSize : Pointer to function : state self dims -> int
foreign import ccall "THCTensor.h &THCudaDoubleTensor_isSize"
  p_isSize :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO CInt)

-- | p_nElement : Pointer to function : state self -> ptrdiff_t
foreign import ccall "THCTensor.h &THCudaDoubleTensor_nElement"
  p_nElement :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CPtrdiff)

-- | p_retain : Pointer to function : state self -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_retain"
  p_retain :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_free : Pointer to function : state self -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_free"
  p_free :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_freeCopyTo : Pointer to function : state self dst -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_freeCopyTo"
  p_freeCopyTo :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_set1d : Pointer to function : state tensor x0 value -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_set1d"
  p_set1d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CDouble -> IO ())

-- | p_set2d : Pointer to function : state tensor x0 x1 value -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_set2d"
  p_set2d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CDouble -> IO ())

-- | p_set3d : Pointer to function : state tensor x0 x1 x2 value -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_set3d"
  p_set3d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> CDouble -> IO ())

-- | p_set4d : Pointer to function : state tensor x0 x1 x2 x3 value -> void
foreign import ccall "THCTensor.h &THCudaDoubleTensor_set4d"
  p_set4d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CDouble -> IO ())

-- | p_get1d : Pointer to function : state tensor x0 -> real
foreign import ccall "THCTensor.h &THCudaDoubleTensor_get1d"
  p_get1d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> IO CDouble)

-- | p_get2d : Pointer to function : state tensor x0 x1 -> real
foreign import ccall "THCTensor.h &THCudaDoubleTensor_get2d"
  p_get2d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> IO CDouble)

-- | p_get3d : Pointer to function : state tensor x0 x1 x2 -> real
foreign import ccall "THCTensor.h &THCudaDoubleTensor_get3d"
  p_get3d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> IO CDouble)

-- | p_get4d : Pointer to function : state tensor x0 x1 x2 x3 -> real
foreign import ccall "THCTensor.h &THCudaDoubleTensor_get4d"
  p_get4d :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO CDouble)

-- | p_getDevice : Pointer to function : state self -> int
foreign import ccall "THCTensor.h &THCudaDoubleTensor_getDevice"
  p_getDevice :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CInt)

-- | p_sizeDesc : Pointer to function : state tensor -> THCDescBuff
foreign import ccall "THCTensor.h &THCudaDoubleTensor_sizeDesc"
  p_sizeDesc :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO (Ptr C'THCDescBuff))