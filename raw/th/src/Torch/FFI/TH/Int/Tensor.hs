{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.Tensor where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_storage :  self -> THStorage *
foreign import ccall "THTensor.h THIntTensor_storage"
  c_storage_ :: Ptr C'THIntTensor -> IO (Ptr C'THIntStorage)

-- | alias of c_storage_ with unused argument (for CTHState) to unify backpack signatures.
c_storage = const c_storage_

-- | c_storageOffset :  self -> ptrdiff_t
foreign import ccall "THTensor.h THIntTensor_storageOffset"
  c_storageOffset_ :: Ptr C'THIntTensor -> IO CPtrdiff

-- | alias of c_storageOffset_ with unused argument (for CTHState) to unify backpack signatures.
c_storageOffset = const c_storageOffset_

-- | c_nDimension :  self -> int
foreign import ccall "THTensor.h THIntTensor_nDimension"
  c_nDimension_ :: Ptr C'THIntTensor -> IO CInt

-- | alias of c_nDimension_ with unused argument (for CTHState) to unify backpack signatures.
c_nDimension = const c_nDimension_

-- | c_size :  self dim -> int64_t
foreign import ccall "THTensor.h THIntTensor_size"
  c_size_ :: Ptr C'THIntTensor -> CInt -> IO CLLong

-- | alias of c_size_ with unused argument (for CTHState) to unify backpack signatures.
c_size = const c_size_

-- | c_stride :  self dim -> int64_t
foreign import ccall "THTensor.h THIntTensor_stride"
  c_stride_ :: Ptr C'THIntTensor -> CInt -> IO CLLong

-- | alias of c_stride_ with unused argument (for CTHState) to unify backpack signatures.
c_stride = const c_stride_

-- | c_newSizeOf :  self -> THLongStorage *
foreign import ccall "THTensor.h THIntTensor_newSizeOf"
  c_newSizeOf_ :: Ptr C'THIntTensor -> IO (Ptr C'THLongStorage)

-- | alias of c_newSizeOf_ with unused argument (for CTHState) to unify backpack signatures.
c_newSizeOf = const c_newSizeOf_

-- | c_newStrideOf :  self -> THLongStorage *
foreign import ccall "THTensor.h THIntTensor_newStrideOf"
  c_newStrideOf_ :: Ptr C'THIntTensor -> IO (Ptr C'THLongStorage)

-- | alias of c_newStrideOf_ with unused argument (for CTHState) to unify backpack signatures.
c_newStrideOf = const c_newStrideOf_

-- | c_data :  self -> real *
foreign import ccall "THTensor.h THIntTensor_data"
  c_data_ :: Ptr C'THIntTensor -> IO (Ptr CInt)

-- | alias of c_data_ with unused argument (for CTHState) to unify backpack signatures.
c_data = const c_data_

-- | c_setFlag :  self flag -> void
foreign import ccall "THTensor.h THIntTensor_setFlag"
  c_setFlag_ :: Ptr C'THIntTensor -> CChar -> IO ()

-- | alias of c_setFlag_ with unused argument (for CTHState) to unify backpack signatures.
c_setFlag = const c_setFlag_

-- | c_clearFlag :  self flag -> void
foreign import ccall "THTensor.h THIntTensor_clearFlag"
  c_clearFlag_ :: Ptr C'THIntTensor -> CChar -> IO ()

-- | alias of c_clearFlag_ with unused argument (for CTHState) to unify backpack signatures.
c_clearFlag = const c_clearFlag_

-- | c_new :   -> THTensor *
foreign import ccall "THTensor.h THIntTensor_new"
  c_new_ :: IO (Ptr C'THIntTensor)

-- | alias of c_new_ with unused argument (for CTHState) to unify backpack signatures.
c_new = const c_new_

-- | c_newWithTensor :  tensor -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithTensor"
  c_newWithTensor_ :: Ptr C'THIntTensor -> IO (Ptr C'THIntTensor)

-- | alias of c_newWithTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithTensor = const c_newWithTensor_

-- | c_newWithStorage :  storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage"
  c_newWithStorage_ :: Ptr C'THIntStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THIntTensor)

-- | alias of c_newWithStorage_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithStorage = const c_newWithStorage_

-- | c_newWithStorage1d :  storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage1d"
  c_newWithStorage1d_ :: Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr C'THIntTensor)

-- | alias of c_newWithStorage1d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithStorage1d = const c_newWithStorage1d_

-- | c_newWithStorage2d :  storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage2d"
  c_newWithStorage2d_ :: Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THIntTensor)

-- | alias of c_newWithStorage2d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithStorage2d = const c_newWithStorage2d_

-- | c_newWithStorage3d :  storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage3d"
  c_newWithStorage3d_ :: Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THIntTensor)

-- | alias of c_newWithStorage3d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithStorage3d = const c_newWithStorage3d_

-- | c_newWithStorage4d :  storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithStorage4d"
  c_newWithStorage4d_ :: Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THIntTensor)

-- | alias of c_newWithStorage4d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithStorage4d = const c_newWithStorage4d_

-- | c_newWithSize :  size_ stride_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize"
  c_newWithSize_ :: Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THIntTensor)

-- | alias of c_newWithSize_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize = const c_newWithSize_

-- | c_newWithSize1d :  size0_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize1d"
  c_newWithSize1d_ :: CLLong -> IO (Ptr C'THIntTensor)

-- | alias of c_newWithSize1d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize1d = const c_newWithSize1d_

-- | c_newWithSize2d :  size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize2d"
  c_newWithSize2d_ :: CLLong -> CLLong -> IO (Ptr C'THIntTensor)

-- | alias of c_newWithSize2d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize2d = const c_newWithSize2d_

-- | c_newWithSize3d :  size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize3d"
  c_newWithSize3d_ :: CLLong -> CLLong -> CLLong -> IO (Ptr C'THIntTensor)

-- | alias of c_newWithSize3d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize3d = const c_newWithSize3d_

-- | c_newWithSize4d :  size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newWithSize4d"
  c_newWithSize4d_ :: CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THIntTensor)

-- | alias of c_newWithSize4d_ with unused argument (for CTHState) to unify backpack signatures.
c_newWithSize4d = const c_newWithSize4d_

-- | c_newClone :  self -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newClone"
  c_newClone_ :: Ptr C'THIntTensor -> IO (Ptr C'THIntTensor)

-- | alias of c_newClone_ with unused argument (for CTHState) to unify backpack signatures.
c_newClone = const c_newClone_

-- | c_newContiguous :  tensor -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newContiguous"
  c_newContiguous_ :: Ptr C'THIntTensor -> IO (Ptr C'THIntTensor)

-- | alias of c_newContiguous_ with unused argument (for CTHState) to unify backpack signatures.
c_newContiguous = const c_newContiguous_

-- | c_newSelect :  tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newSelect"
  c_newSelect_ :: Ptr C'THIntTensor -> CInt -> CLLong -> IO (Ptr C'THIntTensor)

-- | alias of c_newSelect_ with unused argument (for CTHState) to unify backpack signatures.
c_newSelect = const c_newSelect_

-- | c_newNarrow :  tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newNarrow"
  c_newNarrow_ :: Ptr C'THIntTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THIntTensor)

-- | alias of c_newNarrow_ with unused argument (for CTHState) to unify backpack signatures.
c_newNarrow = const c_newNarrow_

-- | c_newTranspose :  tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newTranspose"
  c_newTranspose_ :: Ptr C'THIntTensor -> CInt -> CInt -> IO (Ptr C'THIntTensor)

-- | alias of c_newTranspose_ with unused argument (for CTHState) to unify backpack signatures.
c_newTranspose = const c_newTranspose_

-- | c_newUnfold :  tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newUnfold"
  c_newUnfold_ :: Ptr C'THIntTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THIntTensor)

-- | alias of c_newUnfold_ with unused argument (for CTHState) to unify backpack signatures.
c_newUnfold = const c_newUnfold_

-- | c_newView :  tensor size -> THTensor *
foreign import ccall "THTensor.h THIntTensor_newView"
  c_newView_ :: Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO (Ptr C'THIntTensor)

-- | alias of c_newView_ with unused argument (for CTHState) to unify backpack signatures.
c_newView = const c_newView_

-- | c_resize :  tensor size stride -> void
foreign import ccall "THTensor.h THIntTensor_resize"
  c_resize_ :: Ptr C'THIntTensor -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ()

-- | alias of c_resize_ with unused argument (for CTHState) to unify backpack signatures.
c_resize = const c_resize_

-- | c_resizeAs :  tensor src -> void
foreign import ccall "THTensor.h THIntTensor_resizeAs"
  c_resizeAs_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_resizeAs_ with unused argument (for CTHState) to unify backpack signatures.
c_resizeAs = const c_resizeAs_

-- | c_resizeNd :  tensor nDimension size stride -> void
foreign import ccall "THTensor.h THIntTensor_resizeNd"
  c_resizeNd_ :: Ptr C'THIntTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- | alias of c_resizeNd_ with unused argument (for CTHState) to unify backpack signatures.
c_resizeNd = const c_resizeNd_

-- | c_resize1d :  tensor size0_ -> void
foreign import ccall "THTensor.h THIntTensor_resize1d"
  c_resize1d_ :: Ptr C'THIntTensor -> CLLong -> IO ()

-- | alias of c_resize1d_ with unused argument (for CTHState) to unify backpack signatures.
c_resize1d = const c_resize1d_

-- | c_resize2d :  tensor size0_ size1_ -> void
foreign import ccall "THTensor.h THIntTensor_resize2d"
  c_resize2d_ :: Ptr C'THIntTensor -> CLLong -> CLLong -> IO ()

-- | alias of c_resize2d_ with unused argument (for CTHState) to unify backpack signatures.
c_resize2d = const c_resize2d_

-- | c_resize3d :  tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h THIntTensor_resize3d"
  c_resize3d_ :: Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> IO ()

-- | alias of c_resize3d_ with unused argument (for CTHState) to unify backpack signatures.
c_resize3d = const c_resize3d_

-- | c_resize4d :  tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h THIntTensor_resize4d"
  c_resize4d_ :: Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | alias of c_resize4d_ with unused argument (for CTHState) to unify backpack signatures.
c_resize4d = const c_resize4d_

-- | c_resize5d :  tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h THIntTensor_resize5d"
  c_resize5d_ :: Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | alias of c_resize5d_ with unused argument (for CTHState) to unify backpack signatures.
c_resize5d = const c_resize5d_

-- | c_set :  self src -> void
foreign import ccall "THTensor.h THIntTensor_set"
  c_set_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_set_ with unused argument (for CTHState) to unify backpack signatures.
c_set = const c_set_

-- | c_setStorage :  self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage"
  c_setStorage_ :: Ptr C'THIntTensor -> Ptr C'THIntStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ()

-- | alias of c_setStorage_ with unused argument (for CTHState) to unify backpack signatures.
c_setStorage = const c_setStorage_

-- | c_setStorageNd :  self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h THIntTensor_setStorageNd"
  c_setStorageNd_ :: Ptr C'THIntTensor -> Ptr C'THIntStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- | alias of c_setStorageNd_ with unused argument (for CTHState) to unify backpack signatures.
c_setStorageNd = const c_setStorageNd_

-- | c_setStorage1d :  self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage1d"
  c_setStorage1d_ :: Ptr C'THIntTensor -> Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> IO ()

-- | alias of c_setStorage1d_ with unused argument (for CTHState) to unify backpack signatures.
c_setStorage1d = const c_setStorage1d_

-- | c_setStorage2d :  self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage2d"
  c_setStorage2d_ :: Ptr C'THIntTensor -> Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | alias of c_setStorage2d_ with unused argument (for CTHState) to unify backpack signatures.
c_setStorage2d = const c_setStorage2d_

-- | c_setStorage3d :  self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage3d"
  c_setStorage3d_ :: Ptr C'THIntTensor -> Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | alias of c_setStorage3d_ with unused argument (for CTHState) to unify backpack signatures.
c_setStorage3d = const c_setStorage3d_

-- | c_setStorage4d :  self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h THIntTensor_setStorage4d"
  c_setStorage4d_ :: Ptr C'THIntTensor -> Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | alias of c_setStorage4d_ with unused argument (for CTHState) to unify backpack signatures.
c_setStorage4d = const c_setStorage4d_

-- | c_narrow :  self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h THIntTensor_narrow"
  c_narrow_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CLLong -> CLLong -> IO ()

-- | alias of c_narrow_ with unused argument (for CTHState) to unify backpack signatures.
c_narrow = const c_narrow_

-- | c_select :  self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h THIntTensor_select"
  c_select_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CLLong -> IO ()

-- | alias of c_select_ with unused argument (for CTHState) to unify backpack signatures.
c_select = const c_select_

-- | c_transpose :  self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h THIntTensor_transpose"
  c_transpose_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()

-- | alias of c_transpose_ with unused argument (for CTHState) to unify backpack signatures.
c_transpose = const c_transpose_

-- | c_unfold :  self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h THIntTensor_unfold"
  c_unfold_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CLLong -> CLLong -> IO ()

-- | alias of c_unfold_ with unused argument (for CTHState) to unify backpack signatures.
c_unfold = const c_unfold_

-- | c_squeeze :  self src -> void
foreign import ccall "THTensor.h THIntTensor_squeeze"
  c_squeeze_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_squeeze_ with unused argument (for CTHState) to unify backpack signatures.
c_squeeze = const c_squeeze_

-- | c_squeeze1d :  self src dimension_ -> void
foreign import ccall "THTensor.h THIntTensor_squeeze1d"
  c_squeeze1d_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_squeeze1d_ with unused argument (for CTHState) to unify backpack signatures.
c_squeeze1d = const c_squeeze1d_

-- | c_unsqueeze1d :  self src dimension_ -> void
foreign import ccall "THTensor.h THIntTensor_unsqueeze1d"
  c_unsqueeze1d_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_unsqueeze1d_ with unused argument (for CTHState) to unify backpack signatures.
c_unsqueeze1d = const c_unsqueeze1d_

-- | c_isContiguous :  self -> int
foreign import ccall "THTensor.h THIntTensor_isContiguous"
  c_isContiguous_ :: Ptr C'THIntTensor -> IO CInt

-- | alias of c_isContiguous_ with unused argument (for CTHState) to unify backpack signatures.
c_isContiguous = const c_isContiguous_

-- | c_isSameSizeAs :  self src -> int
foreign import ccall "THTensor.h THIntTensor_isSameSizeAs"
  c_isSameSizeAs_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO CInt

-- | alias of c_isSameSizeAs_ with unused argument (for CTHState) to unify backpack signatures.
c_isSameSizeAs = const c_isSameSizeAs_

-- | c_isSetTo :  self src -> int
foreign import ccall "THTensor.h THIntTensor_isSetTo"
  c_isSetTo_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO CInt

-- | alias of c_isSetTo_ with unused argument (for CTHState) to unify backpack signatures.
c_isSetTo = const c_isSetTo_

-- | c_isSize :  self dims -> int
foreign import ccall "THTensor.h THIntTensor_isSize"
  c_isSize_ :: Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO CInt

-- | alias of c_isSize_ with unused argument (for CTHState) to unify backpack signatures.
c_isSize = const c_isSize_

-- | c_nElement :  self -> ptrdiff_t
foreign import ccall "THTensor.h THIntTensor_nElement"
  c_nElement_ :: Ptr C'THIntTensor -> IO CPtrdiff

-- | alias of c_nElement_ with unused argument (for CTHState) to unify backpack signatures.
c_nElement = const c_nElement_

-- | c_retain :  self -> void
foreign import ccall "THTensor.h THIntTensor_retain"
  c_retain_ :: Ptr C'THIntTensor -> IO ()

-- | alias of c_retain_ with unused argument (for CTHState) to unify backpack signatures.
c_retain = const c_retain_

-- | c_free :  self -> void
foreign import ccall "THTensor.h THIntTensor_free"
  c_free_ :: Ptr C'THIntTensor -> IO ()

-- | alias of c_free_ with unused argument (for CTHState) to unify backpack signatures.
c_free = const c_free_

-- | c_freeCopyTo :  self dst -> void
foreign import ccall "THTensor.h THIntTensor_freeCopyTo"
  c_freeCopyTo_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_freeCopyTo_ with unused argument (for CTHState) to unify backpack signatures.
c_freeCopyTo = const c_freeCopyTo_

-- | c_set1d :  tensor x0 value -> void
foreign import ccall "THTensor.h THIntTensor_set1d"
  c_set1d_ :: Ptr C'THIntTensor -> CLLong -> CInt -> IO ()

-- | alias of c_set1d_ with unused argument (for CTHState) to unify backpack signatures.
c_set1d = const c_set1d_

-- | c_set2d :  tensor x0 x1 value -> void
foreign import ccall "THTensor.h THIntTensor_set2d"
  c_set2d_ :: Ptr C'THIntTensor -> CLLong -> CLLong -> CInt -> IO ()

-- | alias of c_set2d_ with unused argument (for CTHState) to unify backpack signatures.
c_set2d = const c_set2d_

-- | c_set3d :  tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h THIntTensor_set3d"
  c_set3d_ :: Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> CInt -> IO ()

-- | alias of c_set3d_ with unused argument (for CTHState) to unify backpack signatures.
c_set3d = const c_set3d_

-- | c_set4d :  tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h THIntTensor_set4d"
  c_set4d_ :: Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CInt -> IO ()

-- | alias of c_set4d_ with unused argument (for CTHState) to unify backpack signatures.
c_set4d = const c_set4d_

-- | c_get1d :  tensor x0 -> real
foreign import ccall "THTensor.h THIntTensor_get1d"
  c_get1d_ :: Ptr C'THIntTensor -> CLLong -> IO CInt

-- | alias of c_get1d_ with unused argument (for CTHState) to unify backpack signatures.
c_get1d = const c_get1d_

-- | c_get2d :  tensor x0 x1 -> real
foreign import ccall "THTensor.h THIntTensor_get2d"
  c_get2d_ :: Ptr C'THIntTensor -> CLLong -> CLLong -> IO CInt

-- | alias of c_get2d_ with unused argument (for CTHState) to unify backpack signatures.
c_get2d = const c_get2d_

-- | c_get3d :  tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h THIntTensor_get3d"
  c_get3d_ :: Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> IO CInt

-- | alias of c_get3d_ with unused argument (for CTHState) to unify backpack signatures.
c_get3d = const c_get3d_

-- | c_get4d :  tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h THIntTensor_get4d"
  c_get4d_ :: Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO CInt

-- | alias of c_get4d_ with unused argument (for CTHState) to unify backpack signatures.
c_get4d = const c_get4d_

-- | c_desc :  tensor -> THDescBuff
foreign import ccall "THTensor.h THIntTensor_desc"
  c_desc_ :: Ptr C'THIntTensor -> IO (Ptr C'THDescBuff)

-- | alias of c_desc_ with unused argument (for CTHState) to unify backpack signatures.
c_desc = const c_desc_

-- | c_sizeDesc :  tensor -> THDescBuff
foreign import ccall "THTensor.h THIntTensor_sizeDesc"
  c_sizeDesc_ :: Ptr C'THIntTensor -> IO (Ptr C'THDescBuff)

-- | alias of c_sizeDesc_ with unused argument (for CTHState) to unify backpack signatures.
c_sizeDesc = const c_sizeDesc_

-- | p_storage : Pointer to function : self -> THStorage *
foreign import ccall "THTensor.h &THIntTensor_storage"
  p_storage_ :: FunPtr (Ptr C'THIntTensor -> IO (Ptr C'THIntStorage))

-- | alias of p_storage_ with unused argument (for CTHState) to unify backpack signatures.
p_storage = const p_storage_

-- | p_storageOffset : Pointer to function : self -> ptrdiff_t
foreign import ccall "THTensor.h &THIntTensor_storageOffset"
  p_storageOffset_ :: FunPtr (Ptr C'THIntTensor -> IO CPtrdiff)

-- | alias of p_storageOffset_ with unused argument (for CTHState) to unify backpack signatures.
p_storageOffset = const p_storageOffset_

-- | p_nDimension : Pointer to function : self -> int
foreign import ccall "THTensor.h &THIntTensor_nDimension"
  p_nDimension_ :: FunPtr (Ptr C'THIntTensor -> IO CInt)

-- | alias of p_nDimension_ with unused argument (for CTHState) to unify backpack signatures.
p_nDimension = const p_nDimension_

-- | p_size : Pointer to function : self dim -> int64_t
foreign import ccall "THTensor.h &THIntTensor_size"
  p_size_ :: FunPtr (Ptr C'THIntTensor -> CInt -> IO CLLong)

-- | alias of p_size_ with unused argument (for CTHState) to unify backpack signatures.
p_size = const p_size_

-- | p_stride : Pointer to function : self dim -> int64_t
foreign import ccall "THTensor.h &THIntTensor_stride"
  p_stride_ :: FunPtr (Ptr C'THIntTensor -> CInt -> IO CLLong)

-- | alias of p_stride_ with unused argument (for CTHState) to unify backpack signatures.
p_stride = const p_stride_

-- | p_newSizeOf : Pointer to function : self -> THLongStorage *
foreign import ccall "THTensor.h &THIntTensor_newSizeOf"
  p_newSizeOf_ :: FunPtr (Ptr C'THIntTensor -> IO (Ptr C'THLongStorage))

-- | alias of p_newSizeOf_ with unused argument (for CTHState) to unify backpack signatures.
p_newSizeOf = const p_newSizeOf_

-- | p_newStrideOf : Pointer to function : self -> THLongStorage *
foreign import ccall "THTensor.h &THIntTensor_newStrideOf"
  p_newStrideOf_ :: FunPtr (Ptr C'THIntTensor -> IO (Ptr C'THLongStorage))

-- | alias of p_newStrideOf_ with unused argument (for CTHState) to unify backpack signatures.
p_newStrideOf = const p_newStrideOf_

-- | p_data : Pointer to function : self -> real *
foreign import ccall "THTensor.h &THIntTensor_data"
  p_data_ :: FunPtr (Ptr C'THIntTensor -> IO (Ptr CInt))

-- | alias of p_data_ with unused argument (for CTHState) to unify backpack signatures.
p_data = const p_data_

-- | p_setFlag : Pointer to function : self flag -> void
foreign import ccall "THTensor.h &THIntTensor_setFlag"
  p_setFlag_ :: FunPtr (Ptr C'THIntTensor -> CChar -> IO ())

-- | alias of p_setFlag_ with unused argument (for CTHState) to unify backpack signatures.
p_setFlag = const p_setFlag_

-- | p_clearFlag : Pointer to function : self flag -> void
foreign import ccall "THTensor.h &THIntTensor_clearFlag"
  p_clearFlag_ :: FunPtr (Ptr C'THIntTensor -> CChar -> IO ())

-- | alias of p_clearFlag_ with unused argument (for CTHState) to unify backpack signatures.
p_clearFlag = const p_clearFlag_

-- | p_new : Pointer to function :  -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_new"
  p_new_ :: FunPtr (IO (Ptr C'THIntTensor))

-- | alias of p_new_ with unused argument (for CTHState) to unify backpack signatures.
p_new = const p_new_

-- | p_newWithTensor : Pointer to function : tensor -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithTensor"
  p_newWithTensor_ :: FunPtr (Ptr C'THIntTensor -> IO (Ptr C'THIntTensor))

-- | alias of p_newWithTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithTensor = const p_newWithTensor_

-- | p_newWithStorage : Pointer to function : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithStorage"
  p_newWithStorage_ :: FunPtr (Ptr C'THIntStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THIntTensor))

-- | alias of p_newWithStorage_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithStorage = const p_newWithStorage_

-- | p_newWithStorage1d : Pointer to function : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithStorage1d"
  p_newWithStorage1d_ :: FunPtr (Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr C'THIntTensor))

-- | alias of p_newWithStorage1d_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithStorage1d = const p_newWithStorage1d_

-- | p_newWithStorage2d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithStorage2d"
  p_newWithStorage2d_ :: FunPtr (Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THIntTensor))

-- | alias of p_newWithStorage2d_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithStorage2d = const p_newWithStorage2d_

-- | p_newWithStorage3d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithStorage3d"
  p_newWithStorage3d_ :: FunPtr (Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THIntTensor))

-- | alias of p_newWithStorage3d_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithStorage3d = const p_newWithStorage3d_

-- | p_newWithStorage4d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithStorage4d"
  p_newWithStorage4d_ :: FunPtr (Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THIntTensor))

-- | alias of p_newWithStorage4d_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithStorage4d = const p_newWithStorage4d_

-- | p_newWithSize : Pointer to function : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithSize"
  p_newWithSize_ :: FunPtr (Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO (Ptr C'THIntTensor))

-- | alias of p_newWithSize_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithSize = const p_newWithSize_

-- | p_newWithSize1d : Pointer to function : size0_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithSize1d"
  p_newWithSize1d_ :: FunPtr (CLLong -> IO (Ptr C'THIntTensor))

-- | alias of p_newWithSize1d_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithSize1d = const p_newWithSize1d_

-- | p_newWithSize2d : Pointer to function : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithSize2d"
  p_newWithSize2d_ :: FunPtr (CLLong -> CLLong -> IO (Ptr C'THIntTensor))

-- | alias of p_newWithSize2d_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithSize2d = const p_newWithSize2d_

-- | p_newWithSize3d : Pointer to function : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithSize3d"
  p_newWithSize3d_ :: FunPtr (CLLong -> CLLong -> CLLong -> IO (Ptr C'THIntTensor))

-- | alias of p_newWithSize3d_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithSize3d = const p_newWithSize3d_

-- | p_newWithSize4d : Pointer to function : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newWithSize4d"
  p_newWithSize4d_ :: FunPtr (CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr C'THIntTensor))

-- | alias of p_newWithSize4d_ with unused argument (for CTHState) to unify backpack signatures.
p_newWithSize4d = const p_newWithSize4d_

-- | p_newClone : Pointer to function : self -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newClone"
  p_newClone_ :: FunPtr (Ptr C'THIntTensor -> IO (Ptr C'THIntTensor))

-- | alias of p_newClone_ with unused argument (for CTHState) to unify backpack signatures.
p_newClone = const p_newClone_

-- | p_newContiguous : Pointer to function : tensor -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newContiguous"
  p_newContiguous_ :: FunPtr (Ptr C'THIntTensor -> IO (Ptr C'THIntTensor))

-- | alias of p_newContiguous_ with unused argument (for CTHState) to unify backpack signatures.
p_newContiguous = const p_newContiguous_

-- | p_newSelect : Pointer to function : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newSelect"
  p_newSelect_ :: FunPtr (Ptr C'THIntTensor -> CInt -> CLLong -> IO (Ptr C'THIntTensor))

-- | alias of p_newSelect_ with unused argument (for CTHState) to unify backpack signatures.
p_newSelect = const p_newSelect_

-- | p_newNarrow : Pointer to function : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newNarrow"
  p_newNarrow_ :: FunPtr (Ptr C'THIntTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THIntTensor))

-- | alias of p_newNarrow_ with unused argument (for CTHState) to unify backpack signatures.
p_newNarrow = const p_newNarrow_

-- | p_newTranspose : Pointer to function : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newTranspose"
  p_newTranspose_ :: FunPtr (Ptr C'THIntTensor -> CInt -> CInt -> IO (Ptr C'THIntTensor))

-- | alias of p_newTranspose_ with unused argument (for CTHState) to unify backpack signatures.
p_newTranspose = const p_newTranspose_

-- | p_newUnfold : Pointer to function : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newUnfold"
  p_newUnfold_ :: FunPtr (Ptr C'THIntTensor -> CInt -> CLLong -> CLLong -> IO (Ptr C'THIntTensor))

-- | alias of p_newUnfold_ with unused argument (for CTHState) to unify backpack signatures.
p_newUnfold = const p_newUnfold_

-- | p_newView : Pointer to function : tensor size -> THTensor *
foreign import ccall "THTensor.h &THIntTensor_newView"
  p_newView_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO (Ptr C'THIntTensor))

-- | alias of p_newView_ with unused argument (for CTHState) to unify backpack signatures.
p_newView = const p_newView_

-- | p_resize : Pointer to function : tensor size stride -> void
foreign import ccall "THTensor.h &THIntTensor_resize"
  p_resize_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ())

-- | alias of p_resize_ with unused argument (for CTHState) to unify backpack signatures.
p_resize = const p_resize_

-- | p_resizeAs : Pointer to function : tensor src -> void
foreign import ccall "THTensor.h &THIntTensor_resizeAs"
  p_resizeAs_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | alias of p_resizeAs_ with unused argument (for CTHState) to unify backpack signatures.
p_resizeAs = const p_resizeAs_

-- | p_resizeNd : Pointer to function : tensor nDimension size stride -> void
foreign import ccall "THTensor.h &THIntTensor_resizeNd"
  p_resizeNd_ :: FunPtr (Ptr C'THIntTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- | alias of p_resizeNd_ with unused argument (for CTHState) to unify backpack signatures.
p_resizeNd = const p_resizeNd_

-- | p_resize1d : Pointer to function : tensor size0_ -> void
foreign import ccall "THTensor.h &THIntTensor_resize1d"
  p_resize1d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> IO ())

-- | alias of p_resize1d_ with unused argument (for CTHState) to unify backpack signatures.
p_resize1d = const p_resize1d_

-- | p_resize2d : Pointer to function : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h &THIntTensor_resize2d"
  p_resize2d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> CLLong -> IO ())

-- | alias of p_resize2d_ with unused argument (for CTHState) to unify backpack signatures.
p_resize2d = const p_resize2d_

-- | p_resize3d : Pointer to function : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h &THIntTensor_resize3d"
  p_resize3d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> IO ())

-- | alias of p_resize3d_ with unused argument (for CTHState) to unify backpack signatures.
p_resize3d = const p_resize3d_

-- | p_resize4d : Pointer to function : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h &THIntTensor_resize4d"
  p_resize4d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | alias of p_resize4d_ with unused argument (for CTHState) to unify backpack signatures.
p_resize4d = const p_resize4d_

-- | p_resize5d : Pointer to function : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h &THIntTensor_resize5d"
  p_resize5d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | alias of p_resize5d_ with unused argument (for CTHState) to unify backpack signatures.
p_resize5d = const p_resize5d_

-- | p_set : Pointer to function : self src -> void
foreign import ccall "THTensor.h &THIntTensor_set"
  p_set_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | alias of p_set_ with unused argument (for CTHState) to unify backpack signatures.
p_set = const p_set_

-- | p_setStorage : Pointer to function : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h &THIntTensor_setStorage"
  p_setStorage_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntStorage -> CPtrdiff -> Ptr C'THLongStorage -> Ptr C'THLongStorage -> IO ())

-- | alias of p_setStorage_ with unused argument (for CTHState) to unify backpack signatures.
p_setStorage = const p_setStorage_

-- | p_setStorageNd : Pointer to function : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h &THIntTensor_setStorageNd"
  p_setStorageNd_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- | alias of p_setStorageNd_ with unused argument (for CTHState) to unify backpack signatures.
p_setStorageNd = const p_setStorageNd_

-- | p_setStorage1d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h &THIntTensor_setStorage1d"
  p_setStorage1d_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> IO ())

-- | alias of p_setStorage1d_ with unused argument (for CTHState) to unify backpack signatures.
p_setStorage1d = const p_setStorage1d_

-- | p_setStorage2d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h &THIntTensor_setStorage2d"
  p_setStorage2d_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | alias of p_setStorage2d_ with unused argument (for CTHState) to unify backpack signatures.
p_setStorage2d = const p_setStorage2d_

-- | p_setStorage3d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h &THIntTensor_setStorage3d"
  p_setStorage3d_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | alias of p_setStorage3d_ with unused argument (for CTHState) to unify backpack signatures.
p_setStorage3d = const p_setStorage3d_

-- | p_setStorage4d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h &THIntTensor_setStorage4d"
  p_setStorage4d_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- | alias of p_setStorage4d_ with unused argument (for CTHState) to unify backpack signatures.
p_setStorage4d = const p_setStorage4d_

-- | p_narrow : Pointer to function : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h &THIntTensor_narrow"
  p_narrow_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CLLong -> CLLong -> IO ())

-- | alias of p_narrow_ with unused argument (for CTHState) to unify backpack signatures.
p_narrow = const p_narrow_

-- | p_select : Pointer to function : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h &THIntTensor_select"
  p_select_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CLLong -> IO ())

-- | alias of p_select_ with unused argument (for CTHState) to unify backpack signatures.
p_select = const p_select_

-- | p_transpose : Pointer to function : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h &THIntTensor_transpose"
  p_transpose_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ())

-- | alias of p_transpose_ with unused argument (for CTHState) to unify backpack signatures.
p_transpose = const p_transpose_

-- | p_unfold : Pointer to function : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h &THIntTensor_unfold"
  p_unfold_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CLLong -> CLLong -> IO ())

-- | alias of p_unfold_ with unused argument (for CTHState) to unify backpack signatures.
p_unfold = const p_unfold_

-- | p_squeeze : Pointer to function : self src -> void
foreign import ccall "THTensor.h &THIntTensor_squeeze"
  p_squeeze_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | alias of p_squeeze_ with unused argument (for CTHState) to unify backpack signatures.
p_squeeze = const p_squeeze_

-- | p_squeeze1d : Pointer to function : self src dimension_ -> void
foreign import ccall "THTensor.h &THIntTensor_squeeze1d"
  p_squeeze1d_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | alias of p_squeeze1d_ with unused argument (for CTHState) to unify backpack signatures.
p_squeeze1d = const p_squeeze1d_

-- | p_unsqueeze1d : Pointer to function : self src dimension_ -> void
foreign import ccall "THTensor.h &THIntTensor_unsqueeze1d"
  p_unsqueeze1d_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | alias of p_unsqueeze1d_ with unused argument (for CTHState) to unify backpack signatures.
p_unsqueeze1d = const p_unsqueeze1d_

-- | p_isContiguous : Pointer to function : self -> int
foreign import ccall "THTensor.h &THIntTensor_isContiguous"
  p_isContiguous_ :: FunPtr (Ptr C'THIntTensor -> IO CInt)

-- | alias of p_isContiguous_ with unused argument (for CTHState) to unify backpack signatures.
p_isContiguous = const p_isContiguous_

-- | p_isSameSizeAs : Pointer to function : self src -> int
foreign import ccall "THTensor.h &THIntTensor_isSameSizeAs"
  p_isSameSizeAs_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO CInt)

-- | alias of p_isSameSizeAs_ with unused argument (for CTHState) to unify backpack signatures.
p_isSameSizeAs = const p_isSameSizeAs_

-- | p_isSetTo : Pointer to function : self src -> int
foreign import ccall "THTensor.h &THIntTensor_isSetTo"
  p_isSetTo_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO CInt)

-- | alias of p_isSetTo_ with unused argument (for CTHState) to unify backpack signatures.
p_isSetTo = const p_isSetTo_

-- | p_isSize : Pointer to function : self dims -> int
foreign import ccall "THTensor.h &THIntTensor_isSize"
  p_isSize_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO CInt)

-- | alias of p_isSize_ with unused argument (for CTHState) to unify backpack signatures.
p_isSize = const p_isSize_

-- | p_nElement : Pointer to function : self -> ptrdiff_t
foreign import ccall "THTensor.h &THIntTensor_nElement"
  p_nElement_ :: FunPtr (Ptr C'THIntTensor -> IO CPtrdiff)

-- | alias of p_nElement_ with unused argument (for CTHState) to unify backpack signatures.
p_nElement = const p_nElement_

-- | p_retain : Pointer to function : self -> void
foreign import ccall "THTensor.h &THIntTensor_retain"
  p_retain_ :: FunPtr (Ptr C'THIntTensor -> IO ())

-- | alias of p_retain_ with unused argument (for CTHState) to unify backpack signatures.
p_retain = const p_retain_

-- | p_free : Pointer to function : self -> void
foreign import ccall "THTensor.h &THIntTensor_free"
  p_free_ :: FunPtr (Ptr C'THIntTensor -> IO ())

-- | alias of p_free_ with unused argument (for CTHState) to unify backpack signatures.
p_free = const p_free_

-- | p_freeCopyTo : Pointer to function : self dst -> void
foreign import ccall "THTensor.h &THIntTensor_freeCopyTo"
  p_freeCopyTo_ :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | alias of p_freeCopyTo_ with unused argument (for CTHState) to unify backpack signatures.
p_freeCopyTo = const p_freeCopyTo_

-- | p_set1d : Pointer to function : tensor x0 value -> void
foreign import ccall "THTensor.h &THIntTensor_set1d"
  p_set1d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> CInt -> IO ())

-- | alias of p_set1d_ with unused argument (for CTHState) to unify backpack signatures.
p_set1d = const p_set1d_

-- | p_set2d : Pointer to function : tensor x0 x1 value -> void
foreign import ccall "THTensor.h &THIntTensor_set2d"
  p_set2d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> CLLong -> CInt -> IO ())

-- | alias of p_set2d_ with unused argument (for CTHState) to unify backpack signatures.
p_set2d = const p_set2d_

-- | p_set3d : Pointer to function : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h &THIntTensor_set3d"
  p_set3d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> CInt -> IO ())

-- | alias of p_set3d_ with unused argument (for CTHState) to unify backpack signatures.
p_set3d = const p_set3d_

-- | p_set4d : Pointer to function : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h &THIntTensor_set4d"
  p_set4d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CInt -> IO ())

-- | alias of p_set4d_ with unused argument (for CTHState) to unify backpack signatures.
p_set4d = const p_set4d_

-- | p_get1d : Pointer to function : tensor x0 -> real
foreign import ccall "THTensor.h &THIntTensor_get1d"
  p_get1d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> IO CInt)

-- | alias of p_get1d_ with unused argument (for CTHState) to unify backpack signatures.
p_get1d = const p_get1d_

-- | p_get2d : Pointer to function : tensor x0 x1 -> real
foreign import ccall "THTensor.h &THIntTensor_get2d"
  p_get2d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> CLLong -> IO CInt)

-- | alias of p_get2d_ with unused argument (for CTHState) to unify backpack signatures.
p_get2d = const p_get2d_

-- | p_get3d : Pointer to function : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h &THIntTensor_get3d"
  p_get3d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> IO CInt)

-- | alias of p_get3d_ with unused argument (for CTHState) to unify backpack signatures.
p_get3d = const p_get3d_

-- | p_get4d : Pointer to function : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h &THIntTensor_get4d"
  p_get4d_ :: FunPtr (Ptr C'THIntTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO CInt)

-- | alias of p_get4d_ with unused argument (for CTHState) to unify backpack signatures.
p_get4d = const p_get4d_

-- | p_desc : Pointer to function : tensor -> THDescBuff
foreign import ccall "THTensor.h &THIntTensor_desc"
  p_desc_ :: FunPtr (Ptr C'THIntTensor -> IO (Ptr C'THDescBuff))

-- | alias of p_desc_ with unused argument (for CTHState) to unify backpack signatures.
p_desc = const p_desc_

-- | p_sizeDesc : Pointer to function : tensor -> THDescBuff
foreign import ccall "THTensor.h &THIntTensor_sizeDesc"
  p_sizeDesc_ :: FunPtr (Ptr C'THIntTensor -> IO (Ptr C'THDescBuff))

-- | alias of p_sizeDesc_ with unused argument (for CTHState) to unify backpack signatures.
p_sizeDesc = const p_sizeDesc_