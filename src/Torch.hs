{-# LANGUAGE ForeignFunctionInterface #-}

module Torch
    (tests,

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
     c_THFloatTensor_newWithSize1d,
     c_THFloatTensor_newWithSize2d,
     c_THFloatTensor_newWithSize3d,
     c_THFloatTensor_newWithSize4d

    ) where

import Foreign
import Foreign.C.Types
import Foreign.C.String
import Foreign.ForeignPtr
import Foreign.Marshal.Array

{- preprocess-generated float types -}

type CTHFloatTensor = ()      -- THTensor / THTensor.h
type CTHFloatStorage = ()     -- THStorage / THStorag
type CTHFloatLongStorage = () -- THLongStorage / THStorage.h
type CTHFloatPtrDiff = CInt     -- ptrdiff_t / THStorage.h TODO: det appropriate type

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
     (Ptr CTHFloatLongStorage) -> (Ptr CTHFloatLongStorage)
  -- (THStorage *storage, ptrdiff_t storageOffset, THLongStorage *size, THLongStorage *stride)

foreign import ccall "THTensor.h THFloatTensor_newWithStorage1d"
  c_THFloatTensor_newWithStorage1d ::
    (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
     CLong -> CLong ->
    (Ptr CTHFloatTensor)
  -- THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, ptrdiff_t storageOffset_, long size0_, long stride0_);

foreign import ccall "THTensor.h THFloatTensor_newWithStorage2d"
  c_THFloatTensor_newWithStorage2d ::
    (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
    CLong -> CLong ->
    CLong -> CLong ->
    (Ptr CTHFloatTensor)
  -- THTensor *THTensor_(newWithStorage2d)(THStorage *storage_, ptrdiff_t storageOffset_,
  --                                 long size0_, long stride0_,
  --                                 long size1_, long stride1_);

foreign import ccall "THTensor.h THFloatTensor_newWithStorage3d"
  c_THFloatTensor_newWithStorage3d ::
    (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
    CLong -> CLong ->
    CLong -> CLong ->
    CLong -> CLong ->
    (Ptr CTHFloatTensor)
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
    (Ptr CTHFloatTensor)
  -- THTensor *THTensor_(newWithStorage4d)(THStorage *storage_, ptrdiff_t storageOffset_,
  --                                 long size0_, long stride0_,
  --                                 long size1_, long stride1_,
  --                                 long size2_, long stride2_,
  --                                 long size3_, long stride3_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize"
  c_THFloatTensor_newWithSize :: (Ptr CTHFloatLongStorage) -> (Ptr CTHFloatLongStorage) ->
    (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize)(THLongStorage *size_, THLongStorage *stride_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize1d"
  c_THFloatTensor_newWithSize1d :: CLong -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize1d)(long size0_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize1d"
  c_THFloatTensor_newWithSize2d :: CLong -> CLong -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize2d)(long size0_, long size1_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize1d"
  c_THFloatTensor_newWithSize3d :: CLong -> CLong -> CLong -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize3d)(long size0_, long size1_, long size2_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize1d"
  c_THFloatTensor_newWithSize4d :: CLong -> CLong -> CLong -> CLong -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize4d)(long size0_, long size1_, long size2_, long size3_);

foreign import ccall "THTensor.h THFloatTensor_newClone"
  c_THFloatTensor_newClone :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newClone)(THTensor *self);

foreign import ccall "THTensor.h THFloatTensor_newContiguous"
  c_THFloatTensor_newNewContiguous :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newContiguous)(THTensor *tensor);

foreign import ccall "THTensor.h THFloatTensor_newSelect"
  c_THFloatTensor_newNewSelect :: (Ptr CTHFloatTensor) -> CInt -> CLong -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newSelect)(THTensor *tensor, int dimension_, long sliceIndex_);

foreign import ccall "THTensor.h THFloatTensor_newNarrow"
  c_THFloatTensor_newNarrow ::
    (Ptr CTHFloatTensor) -> CInt -> CLong -> CLong -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newNarrow)(THTensor *tensor, int dimension_, long firstIndex_, long size_);

foreign import ccall "THTensor.h THFloatTensor_newTranspose"
  c_THFloatTensor_newTranspose :: (Ptr CTHFloatTensor) -> CInt -> CInt -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newTranspose)(THTensor *tensor, int dimension1_, int dimension2_);

foreign import ccall "THTensor.h THFloatTensor_newUnfold"
  c_THFloatTensor_newUnfold :: (Ptr CTHFloatTensor) -> CInt -> CLong -> CLong -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newUnfold)(THTensor *tensor, int dimension_, long size_, long step_);

-- THTensor *THTensor_(newView)(THTensor *tensor, THLongStorage *size);
-- THTensor *THTensor_(newExpand)(THTensor *tensor, THLongStorage *size);

-- void THTensor_(expand)(THTensor *r, THTensor *tensor, THLongStorage *size);
-- void THTensor_(expandNd)(THTensor **rets, THTensor **ops, int count);

-- void THTensor_(resize)(THTensor *tensor, THLongStorage *size, THLongStorage *stride);
-- void THTensor_(resizeAs)(THTensor *tensor, THTensor *src);
-- void THTensor_(resizeNd)(THTensor *tensor, int nDimension, long *size, long *stride);
-- void THTensor_(resize1d)(THTensor *tensor, long size0_);
-- void THTensor_(resize2d)(THTensor *tensor, long size0_, long size1_);
-- void THTensor_(resize3d)(THTensor *tensor, long size0_, long size1_, long size2_);
-- void THTensor_(resize4d)(THTensor *tensor, long size0_, long size1_, long size2_, long size3_);
-- void THTensor_(resize5d)(THTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

-- void THTensor_(set)(THTensor *self, THTensor *src);
-- void THTensor_(setStorage)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_);
-- void THTensor_(setStorageNd)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, int nDimension, long *size, long *stride);
-- void THTensor_(setStorage1d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
--                                     long size0_, long stride0_);
-- void THTensor_(setStorage2d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
--                                     long size0_, long stride0_,
--                                     long size1_, long stride1_);
-- void THTensor_(setStorage3d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
--                                     long size0_, long stride0_,
--                                     long size1_, long stride1_,
--                                     long size2_, long stride2_);
-- void THTensor_(setStorage4d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
--                                     long size0_, long stride0_,
--                                     long size1_, long stride1_,
--                                     long size2_, long stride2_,
--                                     long size3_, long stride3_);

-- void THTensor_(narrow)(THTensor *self, THTensor *src, int dimension_, long firstIndex_, long size_);
-- void THTensor_(select)(THTensor *self, THTensor *src, int dimension_, long sliceIndex_);
-- void THTensor_(transpose)(THTensor *self, THTensor *src, int dimension1_, int dimension2_);
-- void THTensor_(unfold)(THTensor *self, THTensor *src, int dimension_, long size_, long step_);

-- void THTensor_(squeeze)(THTensor *self, THTensor *src);
-- void THTensor_(squeeze1d)(THTensor *self, THTensor *src, int dimension_);
-- void THTensor_(unsqueeze1d)(THTensor *self, THTensor *src, int dimension_);

-- int THTensor_(isContiguous)(const THTensor *self);
-- int THTensor_(isSameSizeAs)(const THTensor *self, const THTensor *src);
-- int THTensor_(isSetTo)(const THTensor *self, const THTensor *src);
-- int THTensor_(isSize)(const THTensor *self, const THLongStorage *dims);
-- ptrdiff_t THTensor_(nElement)(const THTensor *self);

-- void THTensor_(retain)(THTensor *self);
-- void THTensor_(free)(THTensor *self);
-- void THTensor_(freeCopyTo)(THTensor *self, THTensor *dst);

-- /* Slow access methods [check everything] */
-- void THTensor_(set1d)(THTensor *tensor, long x0, real value);
-- void THTensor_(set2d)(THTensor *tensor, long x0, long x1, real value);
-- void THTensor_(set3d)(THTensor *tensor, long x0, long x1, long x2, real value);
-- void THTensor_(set4d)(THTensor *tensor, long x0, long x1, long x2, long x3, real value);

-- real THTensor_(get1d)(const THTensor *tensor, long x0);
-- real THTensor_(get2d)(const THTensor *tensor, long x0, long x1);
-- real THTensor_(get3d)(const THTensor *tensor, long x0, long x1, long x2);
-- real THTensor_(get4d)(const THTensor *tensor, long x0, long x1, long x2, long x3);

-- /* Debug methods */
-- THDescBuff THTensor_(desc)(const THTensor *tensor);
-- THDescBuff THTensor_(sizeDesc)(const THTensor *tensor);

{- test -}

tests = do
  t1 <- c_THFloatTensor_new
  t2 <- c_THFloatTensor_newWithTensor t1
  print "size of tensor 1"
  print $ c_THFloatTensor_nDimension t1
  print "size of tensor 2"
  print $ c_THFloatTensor_nDimension t2
