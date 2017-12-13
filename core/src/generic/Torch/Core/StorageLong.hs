{-# LANGUAGE ForeignFunctionInterface #-}

module Torch.Core.StorageLong
  ( newStorageLong
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.StorageTypes
import THLongStorage

newStorageLong :: StorageSize Int -> StorageLong
newStorageLong size = unsafePerformIO $ do
  newPtr <- go size
  fPtr <- newForeignPtr p_THLongStorage_free newPtr
  pure $ StorageLong fPtr size
  where
    w2cl = fromIntegral
    go S0 = c_THLongStorage_new
    go (S1 s1) = c_THLongStorage_newWithSize1 $ w2cl s1
    go (S2 (s1, s2)) = c_THLongStorage_newWithSize2
                       (w2cl s1) (w2cl s2)
    go (S3 (s1, s2, s3)) = c_THLongStorage_newWithSize3
                           (w2cl s1) (w2cl s2) (w2cl s3)
    go (S4 (s1, s2, s3, s4)) = c_THLongStorage_newWithSize4
                               (w2cl s1) (w2cl s2) (w2cl s3) (w2cl s4)
{-# NOINLINE newStorageLong #-}
