module Torch.Core.Random (
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

import Control.Monad (replicateM)
import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import GHC.Ptr (FunPtr)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Internal
import Torch.Core.Tensor.Types
import THDoubleTensor
import THDoubleTensorMath

import THTypes
import THRandom (CTHGenerator)
import qualified THRandom as TH

newtype Generator = Generator
  { rng :: ForeignPtr CTHGenerator
  } deriving (Eq, Show)

new :: IO Generator
new = Generator <$> newForeignPtr TH.c_THGenerator_new

copy :: Generator -> Generator -> IO Generator
copy g0 g1 = TH.c_THGenerator_new

{-
free
isValid

seed
manualSeed
initialSeed
random
random64
uniform
uniformFloat
normal
exponential
standard_gamma
cauchy
logNormal
geometric
bernoulli



withGen :: (Ptr CTHGenerator -> a) -> Generator -> IO a
withGen operation gen = do
  withForeignPtr (rng gen) (\g -> pure $ operation g)

newRNG :: IO Generator
newRNG = do
  newPtr <- c_THGenerator_new
  fPtr <- newForeignPtr p_THGenerator_free newPtr
  pure $ Generator fPtr

seed :: Generator -> IO Int
seed gen = do
  value <- withGen c_THRandom_seed gen
  pure (fromIntegral value)

-- |TODO - this doesn't seem to set the seed as intended based on output from
-- seed/initialSeed
manualSeed :: Generator -> Int -> IO ()
manualSeed gen seedVal = do
  newContext <- withGen ((flip c_THRandom_manualSeed) valC) gen
  newContext
  where
    valC = (fromIntegral seedVal) :: CULong

initialSeed :: Generator -> Int
initialSeed gen = unsafePerformIO $ do
  initial <- withGen c_THRandom_initialSeed gen
  pure (fromIntegral initial)

type Arg2DoubleFun = Ptr CTHGenerator -> CDouble -> CDouble -> CDouble
type Arg1DoubleFun = Ptr CTHGenerator -> CDouble -> CDouble
type Arg1IntFun = Ptr CTHGenerator -> CDouble -> CInt

apply2Double :: Generator -> Double -> Double -> Arg2DoubleFun
             -> IO Double
apply2Double gen arg1 arg2 cFun = do
  value <- withGen fun gen
  pure (realToFrac value)
  where
    arg1C = realToFrac arg1
    arg2C = realToFrac arg2
    fun = (flip . flip cFun) arg1C arg2C

apply1Double :: Generator -> Double -> Arg1DoubleFun
             -> IO Double
apply1Double gen arg1 cFun = do
  value <- withGen fun gen
  pure (realToFrac value)
  where
    arg1C = realToFrac arg1
    fun = (flip cFun) arg1C

apply1Int :: Generator -> Double -> Arg1IntFun -> IO Int
apply1Int gen arg1 cFun = do
  value <- withGen fun gen
  pure (fromIntegral value)
  where
    arg1C = realToFrac arg1
    fun = (flip cFun) arg1C

random :: Generator -> IO Int
random gen = fromIntegral <$> withGen c_THRandom_random gen

uniform :: Generator -> Double -> Double -> IO Double
uniform gen lower upper = apply2Double gen lower upper c_THRandom_uniform

normal :: Generator -> Double -> Positive Double -> IO Double
normal gen mean stdev = apply2Double gen mean (fromPositive stdev) c_THRandom_normal

exponential :: Generator -> Double -> IO Double
exponential gen lambda = apply1Double gen lambda c_THRandom_exponential

cauchy :: Generator -> Double -> Double -> IO Double
cauchy gen med sigma = apply2Double gen med sigma c_THRandom_cauchy

logNormal :: Generator -> Double -> Double -> IO Double
logNormal gen mean stdev = apply2Double gen mean stdev c_THRandom_logNormal

geometric :: Generator -> Double -> IO Int
geometric gen p = apply1Int gen p c_THRandom_geometric

bernoulli :: Generator -> Double -> IO Int
bernoulli gen p = apply1Int gen p c_THRandom_bernoulli
-}
