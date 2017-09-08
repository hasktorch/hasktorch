{-# LANGUAGE ForeignFunctionInterface #-}

module Torch
    ( test
    ) where

import Foreign
import Foreign.C.String
import Foreign.C.Types
import Foreign.ForeignPtr

type CTHFloatTensor = ()
type CTHFloatStorage = ()

data FloatTensor = FT (ForeignPtr CTHFloatTensor)

type Tensor = FloatTensor
type AccReal = Double

foreign import ccall "THTensor.h THFloatTensor_newWithSize1d"
  t_THFloatTensor_newWithSize1d :: CLong -> IO (Ptr CTHFloatTensor)

foreign import ccall "THTensor.h &THFloatTensor_free"
  t_THFloatTensor_free :: FunPtr ((Ptr CTHFloatTensor) -> IO ())

foreign import ccall "THTensor.h THFloatTensor_storage"
  t_THFloatTensor_storage :: (Ptr CTHFloatTensor) -> IO (Ptr CTHFloatStorage)

test2 :: IO Tensor
test2 = do
  t <- t_THFloatTensor_newWithSize1d $ fromIntegral 10
  ft <- newForeignPtr t_THFloatTensor_free t
  return (FT ft)

test :: IO ()
test = do
  test2
  putStrLn "Test"
