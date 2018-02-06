module Torch.Class.Storage where

import Foreign
import Foreign.C.Types
import THTypes
import Torch.Class.Internal

class Storage t where
  tensordata  :: t -> IO (Ptr (HsReal t))
  size        :: t -> CPtrdiff
  -- c_elementSize :: CSize
  set         :: t -> CPtrdiff -> HsReal t -> IO ()
  get         :: t -> CPtrdiff -> HsReal t
  new         :: IO t
  newWithSize  :: CPtrdiff -> IO t
  newWithSize1 :: HsReal t -> IO t
  newWithSize2 :: HsReal t -> HsReal t -> IO t
  newWithSize3 :: HsReal t -> HsReal t -> HsReal t -> IO t
  newWithSize4 :: HsReal t -> HsReal t -> HsReal t -> HsReal t -> IO t
  newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO t
  newWithData    :: Ptr (HsReal t) -> CPtrdiff -> IO t
  newWithAllocator  :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO t
  newWithDataAndAllocator :: Ptr (HsReal t) -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO t
  setFlag   :: t -> CChar -> IO ()
  clearFlag :: t -> CChar -> IO ()
  retain    :: t -> IO ()
  swap      :: t -> t -> IO ()
  resize    :: t -> CPtrdiff -> IO ()
  fill      :: t -> HsReal t -> IO ()
