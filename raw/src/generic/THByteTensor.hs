{-# LANGUAGE ForeignFunctionInterface #-}

module THByteTensor
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
import THTypes
import Data.Word
import Data.Int

-- | c_storage : self -> THStorage *
foreign import ccall "THTensor.h storage"
  c_storage :: Ptr CTHByteTensor -> IO (Ptr CTHByteStorage)

-- | c_storageOffset : self -> ptrdiff_t
foreign import ccall "THTensor.h storageOffset"
  c_storageOffset :: Ptr CTHByteTensor -> CPtrdiff

-- | c_nDimension : self -> int
foreign import ccall "THTensor.h nDimension"
  c_nDimension :: Ptr CTHByteTensor -> CInt

-- | c_size : self dim -> int64_t
foreign import ccall "THTensor.h size"
  c_size :: Ptr CTHByteTensor -> CInt -> CLLong

-- | c_stride : self dim -> int64_t
foreign import ccall "THTensor.h stride"
  c_stride :: Ptr CTHByteTensor -> CInt -> CLLong

-- | c_newSizeOf : self -> THLongStorage *
foreign import ccall "THTensor.h newSizeOf"
  c_newSizeOf :: Ptr CTHByteTensor -> IO (Ptr CTHLongStorage)

-- | c_newStrideOf : self -> THLongStorage *
foreign import ccall "THTensor.h newStrideOf"
  c_newStrideOf :: Ptr CTHByteTensor -> IO (Ptr CTHLongStorage)

-- | c_data : self -> real *
foreign import ccall "THTensor.h data"
  c_data :: Ptr CTHByteTensor -> IO (Ptr CChar)

-- | c_setFlag : self flag -> void
foreign import ccall "THTensor.h setFlag"
  c_setFlag :: Ptr CTHByteTensor -> CChar -> IO ()

-- | c_clearFlag : self flag -> void
foreign import ccall "THTensor.h clearFlag"
  c_clearFlag :: Ptr CTHByteTensor -> CChar -> IO ()

-- | c_new :  -> THTensor *
foreign import ccall "THTensor.h new"
  c_new :: IO (Ptr CTHByteTensor)

-- | c_newWithTensor : tensor -> THTensor *
foreign import ccall "THTensor.h newWithTensor"
  c_newWithTensor :: Ptr CTHByteTensor -> IO (Ptr CTHByteTensor)

-- | c_newWithStorage : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h newWithStorage"
  c_newWithStorage :: Ptr CTHByteStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor)

-- | c_newWithStorage1d : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h newWithStorage1d"
  c_newWithStorage1d :: Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr CTHByteTensor)

-- | c_newWithStorage2d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h newWithStorage2d"
  c_newWithStorage2d :: Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHByteTensor)

-- | c_newWithStorage3d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h newWithStorage3d"
  c_newWithStorage3d :: Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHByteTensor)

-- | c_newWithStorage4d : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h newWithStorage4d"
  c_newWithStorage4d :: Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHByteTensor)

-- | c_newWithSize : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h newWithSize"
  c_newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor)

-- | c_newWithSize1d : size0_ -> THTensor *
foreign import ccall "THTensor.h newWithSize1d"
  c_newWithSize1d :: CLLong -> IO (Ptr CTHByteTensor)

-- | c_newWithSize2d : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h newWithSize2d"
  c_newWithSize2d :: CLLong -> CLLong -> IO (Ptr CTHByteTensor)

-- | c_newWithSize3d : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h newWithSize3d"
  c_newWithSize3d :: CLLong -> CLLong -> CLLong -> IO (Ptr CTHByteTensor)

-- | c_newWithSize4d : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h newWithSize4d"
  c_newWithSize4d :: CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHByteTensor)

-- | c_newClone : self -> THTensor *
foreign import ccall "THTensor.h newClone"
  c_newClone :: Ptr CTHByteTensor -> IO (Ptr CTHByteTensor)

-- | c_newContiguous : tensor -> THTensor *
foreign import ccall "THTensor.h newContiguous"
  c_newContiguous :: Ptr CTHByteTensor -> IO (Ptr CTHByteTensor)

-- | c_newSelect : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h newSelect"
  c_newSelect :: Ptr CTHByteTensor -> CInt -> CLLong -> IO (Ptr CTHByteTensor)

-- | c_newNarrow : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h newNarrow"
  c_newNarrow :: Ptr CTHByteTensor -> CInt -> CLLong -> CLLong -> IO (Ptr CTHByteTensor)

-- | c_newTranspose : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h newTranspose"
  c_newTranspose :: Ptr CTHByteTensor -> CInt -> CInt -> IO (Ptr CTHByteTensor)

-- | c_newUnfold : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h newUnfold"
  c_newUnfold :: Ptr CTHByteTensor -> CInt -> CLLong -> CLLong -> IO (Ptr CTHByteTensor)

-- | c_newView : tensor size -> THTensor *
foreign import ccall "THTensor.h newView"
  c_newView :: Ptr CTHByteTensor -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor)

-- | c_newExpand : tensor size -> THTensor *
foreign import ccall "THTensor.h newExpand"
  c_newExpand :: Ptr CTHByteTensor -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor)

-- | c_expand : r tensor size -> void
foreign import ccall "THTensor.h expand"
  c_expand :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHLongStorage -> IO ()

-- | c_expandNd : rets ops count -> void
foreign import ccall "THTensor.h expandNd"
  c_expandNd :: Ptr (Ptr CTHByteTensor) -> Ptr (Ptr CTHByteTensor) -> CInt -> IO ()

-- | c_resize : tensor size stride -> void
foreign import ccall "THTensor.h resize"
  c_resize :: Ptr CTHByteTensor -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- | c_resizeAs : tensor src -> void
foreign import ccall "THTensor.h resizeAs"
  c_resizeAs :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ()

-- | c_resizeNd : tensor nDimension size stride -> void
foreign import ccall "THTensor.h resizeNd"
  c_resizeNd :: Ptr CTHByteTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- | c_resize1d : tensor size0_ -> void
foreign import ccall "THTensor.h resize1d"
  c_resize1d :: Ptr CTHByteTensor -> CLLong -> IO ()

-- | c_resize2d : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h resize2d"
  c_resize2d :: Ptr CTHByteTensor -> CLLong -> CLLong -> IO ()

-- | c_resize3d : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h resize3d"
  c_resize3d :: Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_resize4d : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h resize4d"
  c_resize4d :: Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_resize5d : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h resize5d"
  c_resize5d :: Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_set : self src -> void
foreign import ccall "THTensor.h set"
  c_set :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ()

-- | c_setStorage : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h setStorage"
  c_setStorage :: Ptr CTHByteTensor -> Ptr CTHByteStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- | c_setStorageNd : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h setStorageNd"
  c_setStorageNd :: Ptr CTHByteTensor -> Ptr CTHByteStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()

-- | c_setStorage1d : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h setStorage1d"
  c_setStorage1d :: Ptr CTHByteTensor -> Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> IO ()

-- | c_setStorage2d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h setStorage2d"
  c_setStorage2d :: Ptr CTHByteTensor -> Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_setStorage3d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h setStorage3d"
  c_setStorage3d :: Ptr CTHByteTensor -> Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_setStorage4d : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h setStorage4d"
  c_setStorage4d :: Ptr CTHByteTensor -> Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_narrow : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h narrow"
  c_narrow :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt -> CLLong -> CLLong -> IO ()

-- | c_select : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h select"
  c_select :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt -> CLLong -> IO ()

-- | c_transpose : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h transpose"
  c_transpose :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt -> CInt -> IO ()

-- | c_unfold : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h unfold"
  c_unfold :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt -> CLLong -> CLLong -> IO ()

-- | c_squeeze : self src -> void
foreign import ccall "THTensor.h squeeze"
  c_squeeze :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ()

-- | c_squeeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h squeeze1d"
  c_squeeze1d :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt -> IO ()

-- | c_unsqueeze1d : self src dimension_ -> void
foreign import ccall "THTensor.h unsqueeze1d"
  c_unsqueeze1d :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt -> IO ()

-- | c_isContiguous : self -> int
foreign import ccall "THTensor.h isContiguous"
  c_isContiguous :: Ptr CTHByteTensor -> CInt

-- | c_isSameSizeAs : self src -> int
foreign import ccall "THTensor.h isSameSizeAs"
  c_isSameSizeAs :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt

-- | c_isSetTo : self src -> int
foreign import ccall "THTensor.h isSetTo"
  c_isSetTo :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt

-- | c_isSize : self dims -> int
foreign import ccall "THTensor.h isSize"
  c_isSize :: Ptr CTHByteTensor -> Ptr CTHLongStorage -> CInt

-- | c_nElement : self -> ptrdiff_t
foreign import ccall "THTensor.h nElement"
  c_nElement :: Ptr CTHByteTensor -> CPtrdiff

-- | c_retain : self -> void
foreign import ccall "THTensor.h retain"
  c_retain :: Ptr CTHByteTensor -> IO ()

-- | c_free : self -> void
foreign import ccall "THTensor.h free"
  c_free :: Ptr CTHByteTensor -> IO ()

-- | c_freeCopyTo : self dst -> void
foreign import ccall "THTensor.h freeCopyTo"
  c_freeCopyTo :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ()

-- | c_set1d : tensor x0 value -> void
foreign import ccall "THTensor.h set1d"
  c_set1d :: Ptr CTHByteTensor -> CLLong -> CChar -> IO ()

-- | c_set2d : tensor x0 x1 value -> void
foreign import ccall "THTensor.h set2d"
  c_set2d :: Ptr CTHByteTensor -> CLLong -> CLLong -> CChar -> IO ()

-- | c_set3d : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h set3d"
  c_set3d :: Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> CChar -> IO ()

-- | c_set4d : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h set4d"
  c_set4d :: Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CChar -> IO ()

-- | c_get1d : tensor x0 -> real
foreign import ccall "THTensor.h get1d"
  c_get1d :: Ptr CTHByteTensor -> CLLong -> CChar

-- | c_get2d : tensor x0 x1 -> real
foreign import ccall "THTensor.h get2d"
  c_get2d :: Ptr CTHByteTensor -> CLLong -> CLLong -> CChar

-- | c_get3d : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h get3d"
  c_get3d :: Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> CChar

-- | c_get4d : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h get4d"
  c_get4d :: Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CChar

-- | c_desc : tensor -> THDescBuff
foreign import ccall "THTensor.h desc"
  c_desc :: Ptr CTHByteTensor -> CTHDescBuff

-- | c_sizeDesc : tensor -> THDescBuff
foreign import ccall "THTensor.h sizeDesc"
  c_sizeDesc :: Ptr CTHByteTensor -> CTHDescBuff

-- |p_storage : Pointer to function : self -> THStorage *
foreign import ccall "THTensor.h &storage"
  p_storage :: FunPtr (Ptr CTHByteTensor -> IO (Ptr CTHByteStorage))

-- |p_storageOffset : Pointer to function : self -> ptrdiff_t
foreign import ccall "THTensor.h &storageOffset"
  p_storageOffset :: FunPtr (Ptr CTHByteTensor -> CPtrdiff)

-- |p_nDimension : Pointer to function : self -> int
foreign import ccall "THTensor.h &nDimension"
  p_nDimension :: FunPtr (Ptr CTHByteTensor -> CInt)

-- |p_size : Pointer to function : self dim -> int64_t
foreign import ccall "THTensor.h &size"
  p_size :: FunPtr (Ptr CTHByteTensor -> CInt -> CLLong)

-- |p_stride : Pointer to function : self dim -> int64_t
foreign import ccall "THTensor.h &stride"
  p_stride :: FunPtr (Ptr CTHByteTensor -> CInt -> CLLong)

-- |p_newSizeOf : Pointer to function : self -> THLongStorage *
foreign import ccall "THTensor.h &newSizeOf"
  p_newSizeOf :: FunPtr (Ptr CTHByteTensor -> IO (Ptr CTHLongStorage))

-- |p_newStrideOf : Pointer to function : self -> THLongStorage *
foreign import ccall "THTensor.h &newStrideOf"
  p_newStrideOf :: FunPtr (Ptr CTHByteTensor -> IO (Ptr CTHLongStorage))

-- |p_data : Pointer to function : self -> real *
foreign import ccall "THTensor.h &data"
  p_data :: FunPtr (Ptr CTHByteTensor -> IO (Ptr CChar))

-- |p_setFlag : Pointer to function : self flag -> void
foreign import ccall "THTensor.h &setFlag"
  p_setFlag :: FunPtr (Ptr CTHByteTensor -> CChar -> IO ())

-- |p_clearFlag : Pointer to function : self flag -> void
foreign import ccall "THTensor.h &clearFlag"
  p_clearFlag :: FunPtr (Ptr CTHByteTensor -> CChar -> IO ())

-- |p_new : Pointer to function :  -> THTensor *
foreign import ccall "THTensor.h &new"
  p_new :: FunPtr (IO (Ptr CTHByteTensor))

-- |p_newWithTensor : Pointer to function : tensor -> THTensor *
foreign import ccall "THTensor.h &newWithTensor"
  p_newWithTensor :: FunPtr (Ptr CTHByteTensor -> IO (Ptr CTHByteTensor))

-- |p_newWithStorage : Pointer to function : storage_ storageOffset_ size_ stride_ -> THTensor *
foreign import ccall "THTensor.h &newWithStorage"
  p_newWithStorage :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor))

-- |p_newWithStorage1d : Pointer to function : storage_ storageOffset_ size0_ stride0_ -> THTensor *
foreign import ccall "THTensor.h &newWithStorage1d"
  p_newWithStorage1d :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> IO (Ptr CTHByteTensor))

-- |p_newWithStorage2d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> THTensor *
foreign import ccall "THTensor.h &newWithStorage2d"
  p_newWithStorage2d :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHByteTensor))

-- |p_newWithStorage3d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> THTensor *
foreign import ccall "THTensor.h &newWithStorage3d"
  p_newWithStorage3d :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHByteTensor))

-- |p_newWithStorage4d : Pointer to function : storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> THTensor *
foreign import ccall "THTensor.h &newWithStorage4d"
  p_newWithStorage4d :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHByteTensor))

-- |p_newWithSize : Pointer to function : size_ stride_ -> THTensor *
foreign import ccall "THTensor.h &newWithSize"
  p_newWithSize :: FunPtr (Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor))

-- |p_newWithSize1d : Pointer to function : size0_ -> THTensor *
foreign import ccall "THTensor.h &newWithSize1d"
  p_newWithSize1d :: FunPtr (CLLong -> IO (Ptr CTHByteTensor))

-- |p_newWithSize2d : Pointer to function : size0_ size1_ -> THTensor *
foreign import ccall "THTensor.h &newWithSize2d"
  p_newWithSize2d :: FunPtr (CLLong -> CLLong -> IO (Ptr CTHByteTensor))

-- |p_newWithSize3d : Pointer to function : size0_ size1_ size2_ -> THTensor *
foreign import ccall "THTensor.h &newWithSize3d"
  p_newWithSize3d :: FunPtr (CLLong -> CLLong -> CLLong -> IO (Ptr CTHByteTensor))

-- |p_newWithSize4d : Pointer to function : size0_ size1_ size2_ size3_ -> THTensor *
foreign import ccall "THTensor.h &newWithSize4d"
  p_newWithSize4d :: FunPtr (CLLong -> CLLong -> CLLong -> CLLong -> IO (Ptr CTHByteTensor))

-- |p_newClone : Pointer to function : self -> THTensor *
foreign import ccall "THTensor.h &newClone"
  p_newClone :: FunPtr (Ptr CTHByteTensor -> IO (Ptr CTHByteTensor))

-- |p_newContiguous : Pointer to function : tensor -> THTensor *
foreign import ccall "THTensor.h &newContiguous"
  p_newContiguous :: FunPtr (Ptr CTHByteTensor -> IO (Ptr CTHByteTensor))

-- |p_newSelect : Pointer to function : tensor dimension_ sliceIndex_ -> THTensor *
foreign import ccall "THTensor.h &newSelect"
  p_newSelect :: FunPtr (Ptr CTHByteTensor -> CInt -> CLLong -> IO (Ptr CTHByteTensor))

-- |p_newNarrow : Pointer to function : tensor dimension_ firstIndex_ size_ -> THTensor *
foreign import ccall "THTensor.h &newNarrow"
  p_newNarrow :: FunPtr (Ptr CTHByteTensor -> CInt -> CLLong -> CLLong -> IO (Ptr CTHByteTensor))

-- |p_newTranspose : Pointer to function : tensor dimension1_ dimension2_ -> THTensor *
foreign import ccall "THTensor.h &newTranspose"
  p_newTranspose :: FunPtr (Ptr CTHByteTensor -> CInt -> CInt -> IO (Ptr CTHByteTensor))

-- |p_newUnfold : Pointer to function : tensor dimension_ size_ step_ -> THTensor *
foreign import ccall "THTensor.h &newUnfold"
  p_newUnfold :: FunPtr (Ptr CTHByteTensor -> CInt -> CLLong -> CLLong -> IO (Ptr CTHByteTensor))

-- |p_newView : Pointer to function : tensor size -> THTensor *
foreign import ccall "THTensor.h &newView"
  p_newView :: FunPtr (Ptr CTHByteTensor -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor))

-- |p_newExpand : Pointer to function : tensor size -> THTensor *
foreign import ccall "THTensor.h &newExpand"
  p_newExpand :: FunPtr (Ptr CTHByteTensor -> Ptr CTHLongStorage -> IO (Ptr CTHByteTensor))

-- |p_expand : Pointer to function : r tensor size -> void
foreign import ccall "THTensor.h &expand"
  p_expand :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHLongStorage -> IO ())

-- |p_expandNd : Pointer to function : rets ops count -> void
foreign import ccall "THTensor.h &expandNd"
  p_expandNd :: FunPtr (Ptr (Ptr CTHByteTensor) -> Ptr (Ptr CTHByteTensor) -> CInt -> IO ())

-- |p_resize : Pointer to function : tensor size stride -> void
foreign import ccall "THTensor.h &resize"
  p_resize :: FunPtr (Ptr CTHByteTensor -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_resizeAs : Pointer to function : tensor src -> void
foreign import ccall "THTensor.h &resizeAs"
  p_resizeAs :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ())

-- |p_resizeNd : Pointer to function : tensor nDimension size stride -> void
foreign import ccall "THTensor.h &resizeNd"
  p_resizeNd :: FunPtr (Ptr CTHByteTensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- |p_resize1d : Pointer to function : tensor size0_ -> void
foreign import ccall "THTensor.h &resize1d"
  p_resize1d :: FunPtr (Ptr CTHByteTensor -> CLLong -> IO ())

-- |p_resize2d : Pointer to function : tensor size0_ size1_ -> void
foreign import ccall "THTensor.h &resize2d"
  p_resize2d :: FunPtr (Ptr CTHByteTensor -> CLLong -> CLLong -> IO ())

-- |p_resize3d : Pointer to function : tensor size0_ size1_ size2_ -> void
foreign import ccall "THTensor.h &resize3d"
  p_resize3d :: FunPtr (Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_resize4d : Pointer to function : tensor size0_ size1_ size2_ size3_ -> void
foreign import ccall "THTensor.h &resize4d"
  p_resize4d :: FunPtr (Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_resize5d : Pointer to function : tensor size0_ size1_ size2_ size3_ size4_ -> void
foreign import ccall "THTensor.h &resize5d"
  p_resize5d :: FunPtr (Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_set : Pointer to function : self src -> void
foreign import ccall "THTensor.h &set"
  p_set :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ())

-- |p_setStorage : Pointer to function : self storage_ storageOffset_ size_ stride_ -> void
foreign import ccall "THTensor.h &setStorage"
  p_setStorage :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteStorage -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_setStorageNd : Pointer to function : self storage_ storageOffset_ nDimension size stride -> void
foreign import ccall "THTensor.h &setStorageNd"
  p_setStorageNd :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteStorage -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ())

-- |p_setStorage1d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ -> void
foreign import ccall "THTensor.h &setStorage1d"
  p_setStorage1d :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> IO ())

-- |p_setStorage2d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ -> void
foreign import ccall "THTensor.h &setStorage2d"
  p_setStorage2d :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_setStorage3d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ -> void
foreign import ccall "THTensor.h &setStorage3d"
  p_setStorage3d :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_setStorage4d : Pointer to function : self storage_ storageOffset_ size0_ stride0_ size1_ stride1_ size2_ stride2_ size3_ stride3_ -> void
foreign import ccall "THTensor.h &setStorage4d"
  p_setStorage4d :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteStorage -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_narrow : Pointer to function : self src dimension_ firstIndex_ size_ -> void
foreign import ccall "THTensor.h &narrow"
  p_narrow :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt -> CLLong -> CLLong -> IO ())

-- |p_select : Pointer to function : self src dimension_ sliceIndex_ -> void
foreign import ccall "THTensor.h &select"
  p_select :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt -> CLLong -> IO ())

-- |p_transpose : Pointer to function : self src dimension1_ dimension2_ -> void
foreign import ccall "THTensor.h &transpose"
  p_transpose :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt -> CInt -> IO ())

-- |p_unfold : Pointer to function : self src dimension_ size_ step_ -> void
foreign import ccall "THTensor.h &unfold"
  p_unfold :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt -> CLLong -> CLLong -> IO ())

-- |p_squeeze : Pointer to function : self src -> void
foreign import ccall "THTensor.h &squeeze"
  p_squeeze :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ())

-- |p_squeeze1d : Pointer to function : self src dimension_ -> void
foreign import ccall "THTensor.h &squeeze1d"
  p_squeeze1d :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt -> IO ())

-- |p_unsqueeze1d : Pointer to function : self src dimension_ -> void
foreign import ccall "THTensor.h &unsqueeze1d"
  p_unsqueeze1d :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt -> IO ())

-- |p_isContiguous : Pointer to function : self -> int
foreign import ccall "THTensor.h &isContiguous"
  p_isContiguous :: FunPtr (Ptr CTHByteTensor -> CInt)

-- |p_isSameSizeAs : Pointer to function : self src -> int
foreign import ccall "THTensor.h &isSameSizeAs"
  p_isSameSizeAs :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt)

-- |p_isSetTo : Pointer to function : self src -> int
foreign import ccall "THTensor.h &isSetTo"
  p_isSetTo :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> CInt)

-- |p_isSize : Pointer to function : self dims -> int
foreign import ccall "THTensor.h &isSize"
  p_isSize :: FunPtr (Ptr CTHByteTensor -> Ptr CTHLongStorage -> CInt)

-- |p_nElement : Pointer to function : self -> ptrdiff_t
foreign import ccall "THTensor.h &nElement"
  p_nElement :: FunPtr (Ptr CTHByteTensor -> CPtrdiff)

-- |p_retain : Pointer to function : self -> void
foreign import ccall "THTensor.h &retain"
  p_retain :: FunPtr (Ptr CTHByteTensor -> IO ())

-- |p_free : Pointer to function : self -> void
foreign import ccall "THTensor.h &free"
  p_free :: FunPtr (Ptr CTHByteTensor -> IO ())

-- |p_freeCopyTo : Pointer to function : self dst -> void
foreign import ccall "THTensor.h &freeCopyTo"
  p_freeCopyTo :: FunPtr (Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ())

-- |p_set1d : Pointer to function : tensor x0 value -> void
foreign import ccall "THTensor.h &set1d"
  p_set1d :: FunPtr (Ptr CTHByteTensor -> CLLong -> CChar -> IO ())

-- |p_set2d : Pointer to function : tensor x0 x1 value -> void
foreign import ccall "THTensor.h &set2d"
  p_set2d :: FunPtr (Ptr CTHByteTensor -> CLLong -> CLLong -> CChar -> IO ())

-- |p_set3d : Pointer to function : tensor x0 x1 x2 value -> void
foreign import ccall "THTensor.h &set3d"
  p_set3d :: FunPtr (Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> CChar -> IO ())

-- |p_set4d : Pointer to function : tensor x0 x1 x2 x3 value -> void
foreign import ccall "THTensor.h &set4d"
  p_set4d :: FunPtr (Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CChar -> IO ())

-- |p_get1d : Pointer to function : tensor x0 -> real
foreign import ccall "THTensor.h &get1d"
  p_get1d :: FunPtr (Ptr CTHByteTensor -> CLLong -> CChar)

-- |p_get2d : Pointer to function : tensor x0 x1 -> real
foreign import ccall "THTensor.h &get2d"
  p_get2d :: FunPtr (Ptr CTHByteTensor -> CLLong -> CLLong -> CChar)

-- |p_get3d : Pointer to function : tensor x0 x1 x2 -> real
foreign import ccall "THTensor.h &get3d"
  p_get3d :: FunPtr (Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> CChar)

-- |p_get4d : Pointer to function : tensor x0 x1 x2 x3 -> real
foreign import ccall "THTensor.h &get4d"
  p_get4d :: FunPtr (Ptr CTHByteTensor -> CLLong -> CLLong -> CLLong -> CLLong -> CChar)

-- |p_desc : Pointer to function : tensor -> THDescBuff
foreign import ccall "THTensor.h &desc"
  p_desc :: FunPtr (Ptr CTHByteTensor -> CTHDescBuff)

-- |p_sizeDesc : Pointer to function : tensor -> THDescBuff
foreign import ccall "THTensor.h &sizeDesc"
  p_sizeDesc :: FunPtr (Ptr CTHByteTensor -> CTHDescBuff)