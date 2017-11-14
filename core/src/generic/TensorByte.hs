{-# LANGUAGE OverloadedStrings #-}
module TensorByte
  ( tb_new
  , fillRaw
  , fillRaw0
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Internal (w2cl, onDims)
import TensorRaw hiding (fillRaw, fillRaw0)
import TensorTypes
import THTypes
import THByteTensor
import THByteTensorMath
import THByteLapack


-- | Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor.
fillRaw :: Int8 -> TensorByteRaw -> IO ()
fillRaw v = flip c_THByteTensor_fill (CChar v)


-- | Fill a raw Byte tensor with 0.0
fillRaw0 :: TensorByteRaw -> IO TensorByteRaw
fillRaw0 t = fillRaw 0 t >> pure t


-- | Create a new (byte) tensor of specified dimensions and fill it with 0
tb_new :: TensorDim Word -> TensorByte
tb_new dims = unsafePerformIO $ do
  newPtr <- go dims
  fPtr <- newForeignPtr p_THByteTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  pure (TensorByte fPtr dims)
  where
    wrap :: Ptr CTHByteTensor -> IO (ForeignPtr CTHByteTensor)
    wrap ptr = newForeignPtr p_THByteTensor_free ptr

    go :: TensorDim Word -> IO (Ptr CTHByteTensor)
    go = onDims w2cl
      c_THByteTensor_new
      c_THByteTensor_newWithSize1d
      c_THByteTensor_newWithSize2d
      c_THByteTensor_newWithSize3d
      c_THByteTensor_newWithSize4d

