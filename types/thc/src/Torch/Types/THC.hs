{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Types.THC
  ( module X
  , CudaState, CudaGenerator
  , State(..), asState

  , ByteStorage(..)
  , ByteDynTensor(..)
  , ByteTensor(..)

  , LongStorage(..)
  , LongDynTensor(..)
  , LongTensor(..)

  , CTHCDescBuff
  , CTHCAllocator, C'THCAllocator
  , CTHCGenerator, C'THCGenerator
  , CTHCFile
  , CTHCHalf, C'THCHalf
  , CTHCState
  , CTHCStream
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
  , CTHCudaHalfTensor, C'THCudaHalfTensor
  , CTHCHalfStorage, C'THCHalfStorage
  , CTHCHalfStorageCopy, C'THCHalfStorageCopy
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

import Torch.Types.THC.Structs as X

import Foreign
import GHC.TypeLits

type CTHCDescBuff = C'THCDescBuff
type CTHCGenerator = C'_Generator
type C'THCGenerator = CTHCGenerator
type CTHCStream = C'THCStream
type CTHCState = C'THCState

type CudaState = State

newtype State = State { asForeign :: ForeignPtr C'THCState }
  deriving (Show, Eq)

asState = State

newtype CudaGenerator = CudaGenerator { unCudaGenerator :: ForeignPtr C'THCGenerator }
  deriving (Show, Eq)

-- * Memory-managed mask and index types for TH to break the dependency cycle

newtype LongStorage = LongStorage { longStorage :: ForeignPtr C'THCLongStorage }
  deriving (Eq, Show)

newtype LongDynTensor = LongDynTensor { longTensor :: ForeignPtr C'THCudaLongTensor }
  deriving (Show, Eq)

newtype LongTensor (ds :: [Nat]) = LongTensor { longDynamic :: LongDynTensor }
  deriving (Show, Eq)

newtype ByteStorage = ByteStorage { byteStorage :: ForeignPtr C'THCByteStorage }
  deriving (Eq, Show)

newtype ByteDynTensor = ByteDynTensor { byteTensor :: ForeignPtr C'THCudaByteTensor }
  deriving (Show, Eq)

newtype ByteTensor (ds :: [Nat]) = ByteTensor { byteDynamic :: ByteDynTensor }
  deriving (Show, Eq)

-- Types we haven't figured out what to do with, yet
type CTHCFile = ()
type CTHCAllocator = ()
type C'THCAllocator = CTHCAllocator

type C'THCHalf = ()
type CTHCHalf = ()
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

type C'THCudaHalfTensor = ()
type C'THCHalfStorage = ()
type C'THCHalfStorageCopy = ()


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
