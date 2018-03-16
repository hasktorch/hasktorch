{-# LANGUAGE ConstraintKinds #-}
module Torch.Types.THC
  ( module X
  , CTHCDescBuff
  -- , CTHCAllocator
  , CTHCGenerator
  , CTHCFile
  , CTHCHalf
  , CTHCState
  , CTHCudaByteTensor
  , CTHCByteStorage
  , CTHCByteStorageCopy
  , CTHCudaCharTensor
  , CTHCCharStorage
  , CTHCCharStorageCopy
  , CTHCudaDoubleTensor
  , CTHCDoubleStorage
  , CTHCDoubleStorageCopy
  , CTHCudaFloatTensor
  , CTHCFloatStorage
  , CTHCFloatStorageCopy
  , CTHCudaHalfTensor
  , CTHCHalfStorage
  , CTHCHalfStorageCopy
  , CTHCudaIntTensor
  , CTHCIntStorage
  , CTHCIntStorageCopy
  , CTHCudaLongTensor
  , CTHCLongStorage
  , CTHCLongStorageCopy
  , CTHCudaShortTensor
  , CTHCShortStorage
  , CTHCShortStorageCopy
  ) where

import Torch.Types.THC.Types as X

import Foreign

type CTHCDescBuff = Ptr C'THCDescBuff
type CTHCGenerator = C'_Generator
type CTHCFile = ()
type CTHCHalf = ()
type CTHCState = C'THCState

-- ----------------------------------------
-- Templated types
-- ----------------------------------------

{- Byte -}

type CTHCudaByteTensor = C'THCudaByteTensor
type CTHCByteStorage = C'THCByteStorage
type CTHCByteStorageCopy = CTHCByteStorage


{- Char -}

type CTHCudaCharTensor = C'THCudaCharTensor
type CTHCCharStorage = C'THCCharStorage
type CTHCCharStorageCopy = CTHCCharStorage

{- Double -}

type CTHCudaDoubleTensor = C'THCudaDoubleTensor
type CTHCDoubleStorage = C'THCDoubleStorage
type CTHCDoubleStorageCopy = CTHCDoubleStorage

{- Float -}

type CTHCudaFloatTensor = C'THCudaFloatTensor
type CTHCFloatStorage = C'THCFloatStorage
type CTHCFloatStorageCopy = CTHCFloatStorage

{- Half -}

type CTHCudaHalfTensor = ()
type CTHCHalfStorage = ()
type CTHCHalfStorageCopy = ()

{- Int -}

type CTHCudaIntTensor = C'THCudaIntTensor
type CTHCIntStorage = C'THCIntStorage
type CTHCIntStorageCopy = CTHCIntStorage

{- Long -}

type CTHCudaLongTensor = C'THCudaLongTensor
type CTHCLongStorage = C'THCLongStorage
type CTHCLongStorageCopy = CTHCLongStorage

{- Short -}

type CTHCudaShortTensor = C'THCudaShortTensor
type CTHCShortStorage = C'THCShortStorage
type CTHCShortStorageCopy = CTHCShortStorage
