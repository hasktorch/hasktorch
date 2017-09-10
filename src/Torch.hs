{-# LANGUAGE ForeignFunctionInterface #-}

module Torch
    (
      test
    ) where

import Foreign
import Foreign.C.Types
import Foreign.C.String
import Foreign.ForeignPtr
import Foreign.Marshal.Array

type CTHFile = ()

foreign import ccall "THDiskFile.h THDiskFile_new"
  c_THDiskFile_new :: CString -> CString -> CInt -> IO (Ptr CTHFile)

type CTHFloatTensor = ()

foreign import ccall "THTensor.h THFloatTensor_newWithSize1d"
  c_THFloatTensor_newWithSize1d :: CLong -> IO (Ptr CTHFloatTensor)

type CTHFloatStorage = ()

foreign import ccall "THFile.h THFile_readFloat"
  c_THFile_readFloat :: (Ptr CTHFile) -> (Ptr CTHFloatStorage) -> IO CSize

foreign import ccall "THTensorMath.h THFloatTensor_dot"
  c_THFloatTensor_dot :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO CDouble

foreign import ccall "THTensorMath.h THFloatTensor_sumall"
  c_THFloatTensor_sumall :: (Ptr CTHFloatTensor) -> IO CDouble

foreign import ccall "THTensor.h THFloatTensor_free"
  c_THFloatTensor_free :: (Ptr CTHFloatTensor) -> IO ()

foreign import ccall "THTensor.h &THFloatTensor_free"
  p_THFloatTensor_free :: FunPtr ((Ptr CTHFloatTensor) -> IO ())

foreign import ccall "THFile.h THFile_free"
  c_THFile_free :: (Ptr CTHFile) -> IO CSize

foreign import ccall "THTensor.h THFloatTensor_storage"
  c_THFloatTensor_storage :: (Ptr CTHFloatTensor) -> IO (Ptr CTHFloatStorage)

data FloatTensor = FT (ForeignPtr CTHFloatTensor) deriving Show

type Tensor = FloatTensor
type AccReal = Double

read1dFile :: FilePath -> Int64 -> IO Tensor
read1dFile f n = withCString f $ \x ->
  withCString "r" $ \r -> do
    xfile <- c_THDiskFile_new x r 0
    t <- c_THFloatTensor_newWithSize1d $ fromIntegral n
    ft <- newForeignPtr p_THFloatTensor_free t
    storaget <- withForeignPtr ft c_THFloatTensor_storage
    c_THFile_readFloat xfile storaget
    c_THFile_free xfile
    return (FT ft)

dot :: Tensor -> Tensor -> IO AccReal
dot (FT f) (FT g) = withForeignPtr f $ \x ->
  withForeignPtr g $ \y -> do
    d <- c_THFloatTensor_dot x y
    return (realToFrac d)

sumall :: Tensor -> IO AccReal
sumall (FT f) = withForeignPtr f $ \x -> do
  d <- c_THFloatTensor_sumall x
  return (realToFrac d)

test :: IO Tensor
test = do
  t <- c_THFloatTensor_newWithSize1d $ fromIntegral 10
  ft <- newForeignPtr p_THFloatTensor_free t
  return (FT ft)
