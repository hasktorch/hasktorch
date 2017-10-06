module RandomDouble (
                    ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import GHC.Ptr (FunPtr)
import System.IO.Unsafe (unsafePerformIO)

import TensorRaw
import TensorTypes
import TensorUtils
import Random
import THTypes
import THRandom
import THDoubleTensor
import THDoubleTensorMath

-- randomT :: TensorDouble_ -> RandGen
-- randomT self gen = 
--   withForeignPtr (tdTensor self)
--     (\s ->
--        withForeignPtr (rng gen)
--          (\g ->
--              pure THDoubleTensor_random s g
--              pure $ operation g)
