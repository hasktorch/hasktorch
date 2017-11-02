{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

module TensorLong (
  tb_new
  )
where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import TensorRaw hiding (fillRaw, fillRaw0)
import TensorTypes
import THTypes
import THByteTensor
import THByteTensorMath
import THByteLapack


w2cl = fromIntegral

-- |Fill a raw Byte tensor with 0.0
fillRaw0 :: TensorByteRaw -> IO (TensorByteRaw)
fillRaw0 tensor = fillRaw 0 tensor >> pure tensor

-- |Create a new (byte) tensor of specified dimensions and fill it with 0
tl_new :: TensorDim Word -> TensorByte
tl_new dims = unsafePerformIO $ do
  newPtr <- go dims
  fPtr <- newForeignPtr p_THByteTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ TensorByte fPtr dims
  where
    wrap ptr = newForeignPtr p_THByteTensor_free ptr
    go D0 = c_THByteTensor_new
    go (D1 d1) = c_THByteTensor_newWithSize1d $ w2cl d1
    go (D2 (d1, d2)) = c_THByteTensor_newWithSize2d
                    (w2cl d1) (w2cl d2)
    go (D3 (d1, d2, d3)) = c_THByteTensor_newWithSize3d
                       (w2cl d1) (w2cl d2) (w2cl d3)
    go (D4 (d1, d2, d3, d4)) = c_THByteTensor_newWithSize4d
                          (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)

