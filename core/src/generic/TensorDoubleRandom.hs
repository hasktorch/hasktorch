module TensorDoubleRandom (
  randomT
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import GHC.Ptr (FunPtr)
import System.IO.Unsafe (unsafePerformIO)

import TensorDouble
import TensorRaw
import TensorTypes
import TensorUtils
import Random

import THTypes
import THRandom
import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorRandom

randomT :: TensorDouble_ -> RandGen -> IO TensorDouble_
randomT self gen = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_random s g
         )
    )
  pure self

test = do
  let foo = tensorNew_ (D1 3)
  disp_ foo
  gen <- newRNG
  randomT foo gen
  disp_ foo
  pure ()
