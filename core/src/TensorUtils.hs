module TensorUtils (
  disp,
  w2cl
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import TensorTypes

import THDoubleTensor
import THDoubleTensorMath
import THTypes

import TensorDouble
import TensorDoubleMath
import TensorRaw (dispRaw)

-- |Display memory managed tensor
disp tensor =
  (withForeignPtr(tdTensor tensor) dispRaw)

-- |Dimensions of a tensor as a list
size :: (Ptr CTHDoubleTensor) -> [Int]
size t =
  fmap f [0..maxdim]
  where
    maxdim = (c_THDoubleTensor_nDimension t) - 1
    f x = fromIntegral (c_THDoubleTensor_size t x) :: Int

-- |Show a real value with limited precision (convenience function)
showLim :: RealFloat a => a -> String
showLim x = showGFloat (Just 2) x ""

-- |Word to CLong conversion
w2cl :: Word -> CLong
w2cl = fromIntegral
