module Torch.Class.Storage where

import Foreign
import Foreign.C.Types
import THTypes
import Torch.Class.Internal

class IsStorage t where
  tensordata  :: t -> IO [HsReal t]
  size        :: t -> IO Int64
  -- c_elementSize :: CSize -- CSize = CSize Word64
  set         :: t -> Int64 -> HsReal t -> IO ()
  get         :: t -> Int64 -> IO (HsReal t)
  new         :: IO t
  newWithSize  :: Int64 -> IO t
  newWithSize1 :: HsReal t -> IO t
  newWithSize2 :: HsReal t -> HsReal t -> IO t
  newWithSize3 :: HsReal t -> HsReal t -> HsReal t -> IO t
  newWithSize4 :: HsReal t -> HsReal t -> HsReal t -> HsReal t -> IO t
  newWithMapping :: [Int8] -> Int64 -> Int32 -> IO t
  newWithData    :: [HsReal t] -> Int64 -> IO t
  newWithAllocator  :: Int64 -> CTHAllocatorPtr -> Ptr () -> IO t
  newWithDataAndAllocator :: [HsReal t] -> Int64 -> CTHAllocatorPtr -> Ptr () -> IO t
  setFlag   :: t -> Int8 -> IO ()
  clearFlag :: t -> Int8 -> IO ()
  retain    :: t -> IO ()
  swap      :: t -> t -> IO ()
  resize    :: t -> Int64 -> IO ()
  fill      :: t -> HsReal t -> IO ()
