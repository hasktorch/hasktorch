{-# LANGUAGE ForeignFunctionInterface #-}

module StorageDouble (
  newStorageDouble
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import StorageTypes
import THDoubleStorage

newStorageDouble :: StorageSize Double -> StorageDouble
newStorageDouble size = unsafePerformIO $ do
  newPtr <- go size
  fPtr <- newForeignPtr p_THDoubleStorage_free newPtr
  pure $ StorageDouble fPtr size
  where
    d2cd = realToFrac -- Double to CDouble
    go S0 = c_THDoubleStorage_new
    go (S1 s1) = c_THDoubleStorage_newWithSize1 $ d2cd s1
    go (S2 s1 s2) = c_THDoubleStorage_newWithSize2
                    (d2cd s1) (d2cd s2)
    go (S3 s1 s2 s3) = c_THDoubleStorage_newWithSize3
                       (d2cd s1) (d2cd s2) (d2cd s3)
    go (S4 s1 s2 s3 s4) = c_THDoubleStorage_newWithSize4
                          (d2cd s1) (d2cd s2) (d2cd s3) (d2cd s4)

