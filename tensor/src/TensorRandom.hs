module TensorRandom (
  newRNG,
  seed,
  manualSeed,
  initialSeed,
  random,
  uniform,
  normal,
  exponential,
  cauchy,
  logNormal,
  geometric,
  bernoulli
  ) where

-- TODO - refactor core to raw/memory-managed. This should probably go into
-- core/memory-managed.

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import TensorRaw
import TensorTypes
import TensorUtils
import THTypes
import THRandom
import THDoubleTensor
import THDoubleTensorMath

data RandGen = RandGen {
  rng :: !(ForeignPtr CTHGenerator)
  } deriving (Eq, Show)

newRNG :: RandGen
newRNG = unsafePerformIO $ do
  newPtr <- c_THGenerator_new
  fPtr <- newForeignPtr p_THGenerator_free newPtr
  pure $ RandGen fPtr

seed :: RandGen -> Int
seed = undefined

manualSeed :: RandGen -> Int -> IO ()
manualSeed = undefined

initialSeed :: RandGen -> Int
initialSeed = undefined

random :: RandGen -> Int
random = undefined

uniform :: RandGen -> Double -> Double -> Double
uniform = undefined

normal :: RandGen -> Double -> Double -> Double
normal = undefined

exponential :: RandGen -> Double -> Double
exponential = undefined

cauchy :: RandGen -> Double -> Double -> Double
cauchy = undefined

logNormal :: RandGen -> Double -> Double -> Double
logNormal = undefined

geometric :: RandGen -> Double -> Int
geometric = undefined

bernoulli :: RandGen -> Double -> Int
bernoulli = undefined
