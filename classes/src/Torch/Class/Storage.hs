{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Torch.Class.Storage where

import Foreign (Ptr, Int64, Int8, Int32)
import Torch.Class.Types
import Control.Monad.IO.Class
import Control.Monad.Reader.Class

-- should be CPtrdiff
newtype StorageSize = StorageSize Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

newtype AllocatorContext = AllocatorContext (Ptr ())

newtype Index = Index Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

class Storage t where
  tensordata     :: t -> IO [HsReal t]
  size           :: t -> IO StorageSize
  set            :: t -> Index -> HsReal t -> IO ()
  get            :: t -> Index -> IO (HsReal t)
  empty          :: IO t
  newWithSize    :: StorageSize -> IO t
  newWithSize1   :: HsReal t -> IO t
  newWithSize2   :: HsReal t -> HsReal t -> IO t
  newWithSize3   :: HsReal t -> HsReal t -> HsReal t -> IO t
  newWithSize4   :: HsReal t -> HsReal t -> HsReal t -> HsReal t -> IO t
  newWithMapping :: [Int8] -> StorageSize -> Int32 -> IO t
  newWithData    :: [HsReal t] -> StorageSize -> IO t
  setFlag        :: t -> Int8 -> IO ()
  clearFlag      :: t -> Int8 -> IO ()
  retain         :: t -> IO ()
  resize         :: t -> StorageSize -> IO ()
  fill           :: t -> HsReal t -> IO ()

class CPUStorage t where
  newWithAllocator :: StorageSize -> (Allocator t, AllocatorContext) -> IO t
  newWithDataAndAllocator :: [HsReal t] -> StorageSize -> (Allocator t, AllocatorContext) -> IO t
  swap :: t -> t -> IO ()

class GPUStorage t where
  c_getDevice :: t -> IO Int
