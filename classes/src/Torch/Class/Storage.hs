{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Torch.Class.Storage where

import Foreign (Ptr, Int64, Int8, Int32)
import Torch.Class.Types

-- should be CPtrdiff
newtype StorageSize = StorageSize Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

newtype AllocatorContext = AllocatorContext (Ptr ())

newtype Index = Index Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

class Storage t where
  tensordata  :: t -> io [HsReal t]
  size        :: t -> io StorageSize
  set         :: t -> Index -> HsReal t -> io ()
  get         :: t -> Index -> io (HsReal t)
  empty        :: io t
  newWithSize  :: StorageSize -> io t
  newWithSize1 :: HsReal t -> io t
  newWithSize2 :: HsReal t -> HsReal t -> io t
  newWithSize3 :: HsReal t -> HsReal t -> HsReal t -> io t
  newWithSize4 :: HsReal t -> HsReal t -> HsReal t -> HsReal t -> io t
  newWithMapping :: [Int8] -> StorageSize -> Int32 -> io t
  newWithData    :: [HsReal t] -> StorageSize -> io t
  setFlag   :: t -> Int8 -> io ()
  clearFlag :: t -> Int8 -> io ()
  retain    :: t -> io ()
  resize    :: t -> StorageSize -> io ()
  fill      :: t -> HsReal t -> io ()

class CPUStorage t where
  newWithAllocator :: StorageSize -> (Allocator t, AllocatorContext) -> io t
  newWithDataAndAllocator :: [HsReal t] -> StorageSize -> (Allocator t, AllocatorContext) -> io t
  swap :: t -> t -> io ()

class GPUStorage t where
  c_getDevice :: t -> io Int

