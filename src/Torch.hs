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
  c_THFloatTensor_size :: (Ptr CTHFloatTensor) -> CInt -> CDouble
  -- long THTensor_(size)(const THTensor *self, int dim);

foreign import ccall "THTensor.h THFloatTensor_stride"
  c_THFloatTensor_stride :: (Ptr CTHFloatTensor) -> CInt -> CDouble
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
     CDouble -> CDouble ->
    (Ptr CTHFloatTensor)
  -- THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, ptrdiff_t storageOffset_, long size0_, long stride0_);

foreign import ccall "THTensor.h THFloatTensor_newWithStorage2d"
  c_THFloatTensor_newWithStorage2d ::
    (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
    CDouble -> CDouble ->
    CDouble -> CDouble ->
    (Ptr CTHFloatTensor)
  -- THTensor *THTensor_(newWithStorage2d)(THStorage *storage_, ptrdiff_t storageOffset_,
  --                                 long size0_, long stride0_,
  --                                 long size1_, long stride1_);

foreign import ccall "THTensor.h THFloatTensor_newWithStorage3d"
  c_THFloatTensor_newWithStorage3d ::
    (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
    CDouble -> CDouble ->
    CDouble -> CDouble ->
    CDouble -> CDouble ->
    (Ptr CTHFloatTensor)
  -- THTensor *THTensor_(newWithStorage3d)(THStorage *storage_, ptrdiff_t storageOffset_,
  --                                 long size0_, long stride0_,
  --                                 long size1_, long stride1_,
  --                                 long size2_, long stride2_);

foreign import ccall "THTensor.h THFloatTensor_newWithStorage4d"
  c_THFloatTensor_newWithStorage4d ::
    (Ptr CTHFloatStorage) -> CTHFloatPtrDiff ->
    CDouble -> CDouble ->
    CDouble -> CDouble ->
    CDouble -> CDouble ->
    CDouble -> CDouble ->
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
  c_THFloatTensor_newWithSize1d :: CDouble -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize1d)(long size0_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize1d"
  c_THFloatTensor_newWithSize2d :: CDouble -> CDouble -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize2d)(long size0_, long size1_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize1d"
  c_THFloatTensor_newWithSize3d :: CDouble -> CDouble -> CDouble -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize3d)(long size0_, long size1_, long size2_);

foreign import ccall "THTensor.h THFloatTensor_newWithSize1d"
  c_THFloatTensor_newWithSize4d :: CDouble -> CDouble -> CDouble -> CDouble -> (Ptr CTHFloatTensor)
-- THTensor *THTensor_(newWithSize4d)(long size0_, long size1_, long size2_, long size3_);

{- test -}

tests = do
  t1 <- c_THFloatTensor_new
  t2 <- c_THFloatTensor_newWithTensor t1
  print "size of tensor 1"
  print $ c_THFloatTensor_nDimension t1
  print "size of tensor 2"
  print $ c_THFloatTensor_nDimension t2
