{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatTensorLegacy
    (
     -- access methods
     c_THFloatTensor_storage,
     c_THFloatTensor_storageOffset,
     c_THFloatTensor_nDimension,
     c_THFloatTensor_size,
     c_THFloatTensor_stride,
     c_THFloatTensor_newSizeOf,
     c_THFloatTensor_newStrideOf,
     c_THFloatTEnsor_data,

     -- creation methods
     c_THFloatTensor_new,
     c_THFloatTensor_newWithTensor,
     c_THFloatTensor_newWithStorage,
     c_THFloatTensor_newWithStorage1d,
     c_THFloatTensor_newWithStorage2d,
     c_THFloatTensor_newWithStorage3d,
     c_THFloatTensor_newWithStorage4d,
     c_THFloatTensor_newWithSize,
     c_THFloatTensor_newWithSize1d,
     c_THFloatTensor_newWithSize2d,
     c_THFloatTensor_newWithSize3d,
     c_THFloatTensor_newWithSize4d,

     c_THFloatTensor_newClone,
     c_THFloatTensor_newNewContiguous,
     c_THFloatTensor_newNewSelect,
     c_THFloatTensor_newNarrow,
     c_THFloatTensor_newTranspose,
     c_THFloatTensor_newUnfold,
     c_THFloatTensor_newView,
     c_THFloatTensor_newExpand,

     -- mutation methods
     c_THFloatTensor_expand,
     c_THFloatTensor_expandNd,
     c_THFloatTensor_resize,
     c_THFloatTensor_resizeAs,
     c_THFloatTensor_resizeNd,
     c_THFloatTensor_resize1d,
     c_THFloatTensor_resize2d,
     c_THFloatTensor_resize3d,
     c_THFloatTensor_resize4d,
     c_THFloatTensor_resize5d,
     c_THFloatTensor_set,
     c_THFloatTensor_setStorage,
     c_THFloatTensor_setStorageNd,
     c_THFloatTensor_setStorage1d,
     c_THFloatTensor_setStorage2d,
     c_THFloatTensor_setStorage3d,
     c_THFloatTensor_setStorage4d,
     c_THFloatTensor_narrow,
     c_THFloatTensor_select,
     c_THFloatTensor_transpose,
     c_THFloatTensor_unfold,
     c_THFloatTensor_squeeze,
     c_THFloatTensor_squeeze1d,
     c_THFloatTensor_unsqueeze1d,

     -- check methods
     c_THFloatTensor_isContiguous,
     c_THFloatTensor_isSameSizeAs,
     c_THFloatTensor_isSetTo,
     c_THFloatTensor_isSize,
     c_THFloatTensor_nElement,

     -- allocation methods
     c_THFloatTensor_retain,
     c_THFloatTensor_free,
     c_THFloatTensor_freeCopyTo,

     -- Slow access methods [check everything]
     c_THFloatTensor_set1d,
     c_THFloatTensor_set2d,
     c_THFloatTensor_set3d,
     c_THFloatTensor_set4d,
     c_THFloatTensor_get1d,
     c_THFloatTensor_get2d,
     c_THFloatTensor_get3d,
     c_THFloatTensor_get4d,

     -- debug methods
     c_THFloatTensor_desc,
     c_THFloatTensor_sizeDesc

    ) where

import Foreign
import Foreign.C.Types
import THTypes

-- ----------------------------------------
-- access methods
-- ----------------------------------------

foreign import ccall "THTensor.h THFloatTensor_storage"
  c_THFloatTensor_storage :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatStorage)
  -- THStorage* THTensor_(storage)(const THTensor *self);

foreign import ccall "THTensor.h THFloatTensor_storageOffset"
  c_THFloatTensor_storageOffset :: (Ptr CTHFloatTensor) -> CTHFloatPtrDiff
  -- ptrdiff_t THTensor_(storageOffset)(const THTensor *self);

foreign import ccall "THTensor.h THFloatTensor_nDimension"
  c_THFloatTensor_nDimension :: (Ptr CTHFloatTensor) -> CInt
  -- int THTensor_(nDimension)(const THTensor *self);

foreign import ccall "THTensor.h THFloatTensor_size"
  c_THFloatTensor_size :: (Ptr CTHFloatTensor) -> CInt -> CLong
  -- long THTensor_(size)(const THTensor *self, int dim);

foreign import ccall "THTensor.h THFloatTensor_stride"
  c_THFloatTensor_stride :: (Ptr CTHFloatTensor) -> CInt -> CLong
  -- long THTensor_(stride)(const THTensor *self, int dim);

foreign import ccall "THTensor.h THFloatTensor_newSizeOf"
  c_THFloatTensor_newSizeOf :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatLongStorage)
  -- THLongStorage *THTensor_(newSizeOf)(THTensor *self);

foreign import ccall "THTensor THFloatTensor_newStrideOf"
  c_THFloatTensor_newStrideOf :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatLongStorage)

foreign import ccall "THTensor THFloatTensor_data"
  c_THFloatTEnsor_data :: (Ptr CTHFloatTensor) -> (Ptr CFloat)

-- ----------------------------------------
-- creation methods
-- ----------------------------------------

-- empty init
foreign import ccall "THTensor.h THFloatTensor_new"
  c_THFloatTensor_new :: IO (Ptr CTHFloatStorage)

-- pointer copy init
foreign import ccall "THTensor.h THFloatTensor_newWithTensor"
  c_THFloatTensor_newWithTensor :: (Ptr CTHFloatTensor) -> IO (Ptr CTHFloatTensor)

-- storage init

foreign import ccall "THTensor.h THFloatTensor_newWithStorage"
  c_THFloatTensor_newWithStorage ::
     (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
     (Ptr CTHFloatLongStorage) -> IO (Ptr CTHFloatLongStorage)
  -- (THStorage *storage, ptrdiff_t storageOffset, THLongStorage *size, THLongStorage *stride)

foreign import ccall "THTensor.h THFloatTensor_newWithStorage1d"
  c_THFloatTensor_newWithStorage1d ::
    (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
     CLong -> CLong ->
    IO (Ptr CTHFloatTensor)
  -- THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, ptrdiff_t storageOffset_, long size0_, long stride0_);

foreign import ccall "THTensor.h THFloatTensor_newWithStorage2d"
  c_THFloatTensor_newWithStorage2d ::
    (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
    CLong -> CLong ->
    CLong -> CLong ->
    IO (Ptr CTHFloatTensor)
  -- THTensor *THTensor_(newWithStorage2d)(THStorage *storage_, ptrdiff_t storageOffset_,
  --                                 long size0_, long stride0_,
  --                                 long size1_, long stride1_);

foreign import ccall "THTensor.h THFloatTensor_newWithStorage3d"
  c_THFloatTensor_newWithStorage3d ::
    (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
    CLong -> CLong ->
    CLong -> CLong ->
    CLong -> CLong ->
    IO (Ptr CTHFloatTensor)
  -- THTensor *THTensor_(newWithStorage3d)(THStorage *storage_, ptrdiff_t storageOffset_,
  --                                 long size0_, long stride0_,
  --                                 long size1_, long stride1_,
  --                                 long size2_, long stride2_);

foreign import ccall "THTensor.h THFloatTensor_newWithStorage4d"
  c_THFloatTensor_newWithStorage4d ::
    (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
    CLong -> CLong ->
    CLong -> CLong ->
    CLong -> CLong ->
    CLong -> CLong ->
    IO (Ptr CTHFloatTensor)
  -- THTensor *THTensor_(newWithStorage4d)(THStorage *storage_, ptrdiff_t storageOffset_,
  --                                 long size0_, long stride0_,
  --                                 long size1_, long stride1_,
  --                                 long size2_, long stride2_,
  --                                 long size3_, long stride3_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize"
  c_THFloatTensor_newWithSize :: (Ptr CTHFloatLongStorage) -> (Ptr CTHFloatLongStorage) ->
    IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize)(THLongStorage *size_, THLongStorage *stride_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize1d"
  c_THFloatTensor_newWithSize1d :: CLong -> IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize1d)(long size0_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize2d"
  c_THFloatTensor_newWithSize2d :: CLong -> CLong -> IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize2d)(long size0_, long size1_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize3d"
  c_THFloatTensor_newWithSize3d :: CLong -> CLong -> CLong -> IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize3d)(long size0_, long size1_, long size2_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize4d"
  c_THFloatTensor_newWithSize4d :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize4d)(long size0_, long size1_, long size2_, long size3_);

foreign import ccall "THTensor.h THFloatTensor_newClone"
  c_THFloatTensor_newClone :: (Ptr CTHFloatTensor) -> IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newClone)(THTensor *self);

foreign import ccall "THTensor.h THFloatTensor_newContiguous"
  c_THFloatTensor_newNewContiguous :: (Ptr CTHFloatTensor) -> IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newContiguous)(THTensor *tensor);

foreign import ccall "THTensor.h THFloatTensor_newSelect"
  c_THFloatTensor_newNewSelect :: (Ptr CTHFloatTensor) -> CInt -> CLong -> IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newSelect)(THTensor *tensor, int dimension_, long sliceIndex_);

foreign import ccall "THTensor.h THFloatTensor_newNarrow"
  c_THFloatTensor_newNarrow ::
    (Ptr CTHFloatTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newNarrow)(THTensor *tensor, int dimension_, long firstIndex_, long size_);

foreign import ccall "THTensor.h THFloatTensor_newTranspose"
  c_THFloatTensor_newTranspose :: (Ptr CTHFloatTensor) -> CInt -> CInt -> IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newTranspose)(THTensor *tensor, int dimension1_, int dimension2_);

foreign import ccall "THTensor.h THFloatTensor_newUnfold"
  c_THFloatTensor_newUnfold :: (Ptr CTHFloatTensor) -> CInt -> CLong -> CLong -> IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newUnfold)(THTensor *tensor, int dimension_, long size_, long step_);

foreign import ccall "THTensor.h THFloatTensor_newView"
  c_THFloatTensor_newView ::
    (Ptr CTHFloatTensor) -> (Ptr CTHFloatLongStorage) -> IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newView)(THTensor *tensor, THLongStorage *size);

foreign import ccall "THTensor.h THFloatTensor_newExpand"
  c_THFloatTensor_newExpand ::
    (Ptr CTHFloatTensor) -> (Ptr CTHFloatLongStorage) -> IO (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newExpand)(THTensor *tensor, THLongStorage *size);

-- ----------------------------------------
-- mutation methods
-- ----------------------------------------

foreign import ccall "THTensor.h THFloatTensor_expand"
  c_THFloatTensor_expand ::
    (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatLongStorage) -> IO ()
-- void THTensor_(expand)(THTensor *r, THTensor *tensor, THLongStorage *size);

foreign import ccall "THTensor.h THFloatTensor_expandNd"
  c_THFloatTensor_expandNd ::
    (Ptr (Ptr CTHFloatTensor)) -> (Ptr (Ptr CTHFloatTensor)) -> CInt -> IO ()
-- void THTensor_(expandNd)(THTensor **rets, THTensor **ops, int count);

foreign import ccall "THTensor.h THFloatTensor_resize"
  c_THFloatTensor_resize ::
    (Ptr CTHFloatTensor) -> (Ptr CTHFloatLongStorage) -> IO ()
-- void THTensor_(resize)(THTensor *tensor, THLongStorage *size, THLongStorage *stride);

foreign import ccall "THTensor.h THFloatTensor_resizeAs"
  c_THFloatTensor_resizeAs ::
     (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()
-- void THTensor_(resizeAs)(THTensor *tensor, THTensor *src);

foreign import ccall "THTensor.h THFloatTensor_resizeNd"
  c_THFloatTensor_resizeNd ::
    (Ptr CTHFloatTensor) -> CInt -> (Ptr CLong) -> (Ptr CLong) -> IO ()
-- void THTensor_(resizeNd)(THTensor *tensor, int nDimension, long *size, long *stride);

foreign import ccall "THTensor.h THFloatTensor_resize1d"
  c_THFloatTensor_resize1d :: (Ptr CTHFloatTensor) -> CLong -> IO ()
-- void THTensor_(resize1d)(THTensor *tensor, long size0_);

foreign import ccall "THTensor.h THFloatTensor_resize2d"
  c_THFloatTensor_resize2d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> IO ()
-- void THTensor_(resize2d)(THTensor *tensor, long size0_, long size1_);

foreign import ccall "THTensor.h THFloatTensor_resize3d"
  c_THFloatTensor_resize3d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> IO ()
-- void THTensor_(resize3d)(THTensor *tensor, long size0_, long size1_, long size2_);

foreign import ccall "THTensor.h THFloatTensor_resize4d"
  c_THFloatTensor_resize4d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> CLong -> IO ()
-- void THTensor_(resize4d)(THTensor *tensor, long size0_, long size1_, long size2_, long size3_);

foreign import ccall "THTensor.h THFloatTensor_resize5d"
  c_THFloatTensor_resize5d ::
    (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()
-- void THTensor_(resize5d)(THTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

foreign import ccall "THTensor.h THFloatTensor_set"
  c_THFloatTensor_set :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()
-- void THTensor_(set)(THTensor *self, THTensor *src);

foreign import ccall "THTensor.h THFloatTensor_setStorage"
  c_THFloatTensor_setStorage ::
    (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (CTHFloatPtrDiff) ->
    (Ptr CTHFloatLongStorage) -> (Ptr CTHFloatLongStorage) -> IO ()
-- void THTensor_(setStorage)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_);

foreign import ccall "THTensor.h THFloatTensor_setStorageNd"
  c_THFloatTensor_setStorageNd ::
    (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CTHFloatPtrDiff -> CInt -> (Ptr CLong) -> (Ptr CLong) -> IO ()
-- void THTensor_(setStorageNd)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, int nDimension, long *size, long *stride);

foreign import ccall "THTensor.h THFloatTensor_setStorage1d"
  c_THFloatTensor_setStorage1d ::
    (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CTHFloatPtrDiff -> CLong -> CLong -> IO ()
-- void THTensor_(setStorage1d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
--                                     long size0_, long stride0_);

foreign import ccall "THTensor.h THFloatTensor_setStorage2d"
  c_THFloatTensor_setStorage2d ::
    (Ptr CTHFloatTensor) -> (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
    CLong -> CLong -> CLong -> CLong -> IO ()
-- void THTensor_(setStorage2d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
--                                     long size0_, long stride0_,
--                                     long size1_, long stride1_);

foreign import ccall "THTensor.h THFloatTensor_setStorage3d"
  c_THFloatTensor_setStorage3d ::
    (Ptr CTHFloatTensor) -> (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
    CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()
-- void THTensor_(setStorage3d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
--                                     long size0_, long stride0_,
--                                     long size1_, long stride1_,
--                                     long size2_, long stride2_);

foreign import ccall "THTensor.h THFloatTensor_setStorage4d"
  c_THFloatTensor_setStorage4d ::
    (Ptr CTHFloatTensor) -> (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
    CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()
-- void THTensor_(setStorage4d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
--                                     long size0_, long stride0_,
--                                     long size1_, long stride1_,
--                                     long size2_, long stride2_,
--                                     long size3_, long stride3_);

foreign import ccall "THTensor.h THFloatTensor_narrow"
  c_THFloatTensor_narrow :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CLong -> CLong -> IO ()
-- void THTensor_(narrow)(THTensor *self, THTensor *src, int dimension_, long firstIndex_, long size_);

foreign import ccall "THTensor.h THFloatTensor_select"
  c_THFloatTensor_select :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CLong -> IO ()
-- void THTensor_(select)(THTensor *self, THTensor *src, int dimension_, long sliceIndex_);

foreign import ccall "THTensor.h THFloatTensor_transpose"
  c_THFloatTensor_transpose :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()
-- void THTensor_(transpose)(THTensor *self, THTensor *src, int dimension1_, int dimension2_);

foreign import ccall "THTensor.h THFloatTensor_unfold"
  c_THFloatTensor_unfold :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CLong -> CLong -> IO ()
-- void THTensor_(unfold)(THTensor *self, THTensor *src, int dimension_, long size_, long step_);

foreign import ccall "THTensor.h THFloatTensor_squeeze"
  c_THFloatTensor_squeeze :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()
-- void THTensor_(squeeze)(THTensor *self, THTensor *src);

foreign import ccall "THTensor.h THFloatTensor_squeeze1d"
  c_THFloatTensor_squeeze1d :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()
-- void THTensor_(squeeze1d)(THTensor *self, THTensor *src, int dimension_);

foreign import ccall "THTensor.h THFloatTensor_unsqueeze1d"
  c_THFloatTensor_unsqueeze1d :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()
-- void THTensor_(unsqueeze1d)(THTensor *self, THTensor *src, int dimension_);

-- ----------------------------------------
-- check methods
-- ----------------------------------------

foreign import ccall "THTensor.h THFloatTensor_isContiguous"
  c_THFloatTensor_isContiguous :: (Ptr CTHFloatTensor) -> CInt
-- int THTensor_(isContiguous)(const THTensor *self);

foreign import ccall "THTensor.h THFloatTensor_isSameSizeAs"
  c_THFloatTensor_isSameSizeAs :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt
-- int THTensor_(isSameSizeAs)(const THTensor *self, const THTensor *src);

foreign import ccall "THTensor.h THFloatTensor_isSetTo"
  c_THFloatTensor_isSetTo :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt
-- int THTensor_(isSetTo)(const THTensor *self, const THTensor *src);

foreign import ccall "THTensor.h THFloatTensor_isSize"
  c_THFloatTensor_isSize :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatLongStorage)
-- int THTensor_(isSize)(const THTensor *self, const THLongStorage *dims);

foreign import ccall "THTensor.h THFloatTensor_nElement"
  c_THFloatTensor_nElement :: (Ptr CTHFloatTensor) -> CTHFloatPtrDiff
-- ptrdiff_t THTensor_(nElement)(const THTensor *self);

-- ----------------------------------------
-- allocation methods
-- ----------------------------------------

foreign import ccall "THTensor.h THFloatTensor_retain"
  c_THFloatTensor_retain :: (Ptr CTHFloatTensor) -> IO ()
-- void THTensor_(retain)(THTensor *self);

foreign import ccall "THTensor.h THFloatTensor_free"
  c_THFloatTensor_free :: (Ptr CTHFloatTensor) -> IO ()
-- void THTensor_(free)(THTensor *self);

foreign import ccall "THTensor.h THFloatTensor_freeCopyTo"
  c_THFloatTensor_freeCopyTo :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()
-- void THTensor_(freeCopyTo)(THTensor *self, THTensor *dst);

-- ----------------------------------------
-- Slow access methods [check everything]
-- ----------------------------------------

foreign import ccall "THTensor.h THFloatTensor_set1d"
  c_THFloatTensor_set1d :: (Ptr CTHFloatTensor) -> CLong -> CFloat -> IO ()
-- void THTensor_(set1d)(THTensor *tensor, long x0, real value);

foreign import ccall "THTensor.h THFloatTensor_set2d"
  c_THFloatTensor_set2d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CFloat -> IO ()
-- void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value);

foreign import ccall "THTensor.h THFloatTensor_set3d"
  c_THFloatTensor_set3d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> CFloat -> IO ()
-- void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value);

foreign import ccall "THTensor.h THFloatTensor_set4d"
  c_THFloatTensor_set4d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> CLong -> CFloat -> IO ()
-- void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value);

foreign import ccall "THTensor.h THFloatTensor_get1d"
  c_THFloatTensor_get1d :: (Ptr CTHFloatTensor) -> CLong -> CFloat
-- real THTensor_(get1d)(const THTensor *tensor, long x0);

foreign import ccall "THTensor.h THFloatTensor_get2d"
  c_THFloatTensor_get2d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CFloat
-- real THTensor_(get2d)(const THTensor *tensor, long x0, long x1);

foreign import ccall "THTensor.h THFloatTensor_get3d"
  c_THFloatTensor_get3d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> CFloat
-- real THTensor_(get3d)(const THTensor *tensor, long x0, long x1, long x2);

foreign import ccall "THTensor.h THFloatTensor_get4d"
  c_THFloatTensor_get4d :: (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> CLong -> CFloat
-- real THTensor_(get4d)(const THTensor *tensor, long x0, long x1, long x2, long x3);

-- ----------------------------------------
-- debug methods
-- ----------------------------------------

foreign import ccall "THTensor.h THFloatTensor_desc"
  c_THFloatTensor_desc :: (Ptr CTHFloatTensor) -> CTHDescBuff
-- THDescBuff THTensor_(desc)(const THTensor *tensor);

foreign import ccall "THTensor.h THFloatTensor_sizeDesc"
  c_THFloatTensor_sizeDesc :: (Ptr CTHFloatTensor) -> CTHDescBuff
-- THDescBuff THTensor_(sizeDesc)(const THTensor *tensor);
