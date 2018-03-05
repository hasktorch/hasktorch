{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Torch.Class.Storage where

import Foreign (Ptr, Int64, Int8, Int32)
import Torch.Types.TH
import Torch.Class.Internal

-- should be CPtrdiff
newtype StorageSize = StorageSize Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

newtype AllocatorContext = AllocatorContext (Ptr ())

newtype Index = Index Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

class IsStorage t where
  tensordata  :: t -> IO [HsReal t]
  size        :: t -> IO StorageSize
  -- c_elementSize :: CSize -- CSize = CSize Word64
  set         :: t -> Index -> HsReal t -> IO ()
  get         :: t -> Index -> IO (HsReal t)
  empty        :: IO t
  newWithSize  :: StorageSize -> IO t
  newWithSize1 :: HsReal t -> IO t
  newWithSize2 :: HsReal t -> HsReal t -> IO t
  newWithSize3 :: HsReal t -> HsReal t -> HsReal t -> IO t
  newWithSize4 :: HsReal t -> HsReal t -> HsReal t -> HsReal t -> IO t
  newWithMapping :: [Int8] -> StorageSize -> Int32 -> IO t
  newWithData    :: [HsReal t] -> StorageSize -> IO t
  newWithAllocator  :: StorageSize -> (CTHAllocatorPtr, AllocatorContext) -> IO t
  newWithDataAndAllocator :: [HsReal t] -> StorageSize -> (CTHAllocatorPtr, AllocatorContext) -> IO t
  setFlag   :: t -> Int8 -> IO ()
  clearFlag :: t -> Int8 -> IO ()
  retain    :: t -> IO ()
  swap      :: t -> t -> IO ()
  resize    :: t -> StorageSize -> IO ()
  fill      :: t -> HsReal t -> IO ()
