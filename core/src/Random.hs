module Random (
  RandGen(..),
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

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import GHC.Ptr (FunPtr)
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

applyGen :: (Ptr CTHGenerator -> a) -> RandGen -> IO (a)
applyGen operation gen = do
  withForeignPtr (rng gen) (\g -> pure $ operation g)

newRNG :: IO RandGen
newRNG = do
  newPtr <- c_THGenerator_new
  fPtr <- newForeignPtr p_THGenerator_free newPtr
  pure $ RandGen fPtr

seed :: RandGen -> IO Int
seed gen = do
  value <- applyGen c_THRandom_seed gen
  pure (fromIntegral value)

-- |TODO - this doesn't seem to set the seed as intended based on output from
-- seed/initialSeed
manualSeed :: RandGen -> Int -> IO ()
manualSeed gen seedVal = do
  newContext <- applyGen ((flip c_THRandom_manualSeed) valC) gen
  newContext
  where
    valC = (fromIntegral seedVal) :: CLong

initialSeed :: RandGen -> Int
initialSeed gen = unsafePerformIO $ do
  initial <- applyGen c_THRandom_initialSeed gen
  pure (fromIntegral initial)

type Arg2DoubleFun = Ptr CTHGenerator -> CDouble -> CDouble -> CDouble
type Arg1DoubleFun = Ptr CTHGenerator -> CDouble -> CDouble
type Arg1IntFun = Ptr CTHGenerator -> CDouble -> CInt

apply2Double :: RandGen -> Double -> Double -> Arg2DoubleFun
             -> IO Double
apply2Double gen arg1 arg2 cFun = do
  value <- applyGen fun gen
  pure (realToFrac value)
  where
    arg1C = realToFrac arg1
    arg2C = realToFrac arg2
    fun = (flip . flip cFun) arg1C arg2C

apply1Double :: RandGen -> Double -> Arg1DoubleFun
             -> IO Double
apply1Double gen arg1 cFun = do
  value <- applyGen fun gen
  pure (realToFrac value)
  where
    arg1C = realToFrac arg1
    fun = (flip cFun) arg1C

apply1Int :: RandGen -> Double -> Arg1IntFun
          -> IO Int
apply1Int gen arg1 cFun = do
  value <- applyGen fun gen
  pure (fromIntegral value)
  where
    arg1C = realToFrac arg1
    fun = (flip cFun) arg1C

random :: RandGen -> IO Int
random gen = do
  value <- applyGen c_THRandom_random gen
  pure ((fromIntegral value) :: Int)

uniform :: RandGen -> Double -> Double -> IO Double
uniform gen lower upper = apply2Double gen lower upper c_THRandom_uniform

normal :: RandGen -> Double -> Double -> IO Double
normal gen mean stdev = apply2Double gen mean stdev c_THRandom_normal

exponential :: RandGen -> Double -> IO Double
exponential gen lambda = apply1Double gen lambda c_THRandom_exponential

cauchy :: RandGen -> Double -> Double -> IO Double
cauchy gen med sigma = apply2Double gen med sigma c_THRandom_cauchy

logNormal :: RandGen -> Double -> Double -> IO Double
logNormal gen mean stdev = apply2Double gen mean stdev c_THRandom_logNormal

geometric :: RandGen -> Double -> IO Int
geometric gen p = apply1Int gen p c_THRandom_geometric

bernoulli :: RandGen -> Double -> IO Int
bernoulli gen p = apply1Int gen p c_THRandom_bernoulli

-- |Check that seeds work as intended
test = do
  rng <- newRNG
  manualSeed rng 332323401
  val1 <- normal rng 0.0 1000.0
  val2 <- normal rng 0.0 1000.0
  print val1
  print val2
  print (val1 /= val2)
  manualSeed rng 332323401
  manualSeed rng 332323401
  val3 <- normal rng 0.0 1000.0
  print val3
  print (val1 == val3)
