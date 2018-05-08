-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Core.Random
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Random number generation for single values. FFI over TH/THRandom.h
-------------------------------------------------------------------------------
module Torch.Core.Random
  ( Generator
  , Seed
  , newRNG
  , copy
  , seed
  , manualSeed
  , initialSeed
  , random
  , random64
  , uniform
  , uniformFloat
  , normal
  , exponential
  , standard_gamma
  , cauchy
  , logNormal
  , geometric
  , bernoulli
  ) where

import Foreign (Ptr)
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr, newForeignPtr)
import Data.Word

import Torch.Types.TH
import qualified Torch.FFI.TH.Random as TH

-- ========================================================================= --
-- helpers
-- ========================================================================= --

asRNG :: Ptr C'THGenerator -> IO Generator
asRNG = fmap Generator . newForeignPtr TH.p_THGenerator_free

with2RNGs :: (Ptr C'THGenerator -> Ptr C'THGenerator -> IO x) -> Generator -> Generator -> IO x
with2RNGs fn g0 g1 = _with2RNGs g0 g1 fn

_with2RNGs :: Generator -> Generator -> (Ptr C'THGenerator -> Ptr C'THGenerator -> IO x) -> IO x
_with2RNGs g0 g1 fn = _withRNG g0 (\g0' -> _withRNG g1 (\g1' -> fn g0' g1'))

withRNG :: (Ptr C'THGenerator -> IO x) -> Generator -> IO x
withRNG fn g = withForeignPtr (rng g) fn

_withRNG :: Generator -> (Ptr C'THGenerator -> IO x) -> IO x
_withRNG = flip withRNG

-- ========================================================================= --

-- | Construct a new 'Generator'
newRNG :: IO Generator
newRNG = TH.c_THGenerator_new >>= asRNG

-- | Copy a 'Generator' state to a new generator
copy :: Generator -> Generator -> IO Generator
copy g0 g1 = (with2RNGs TH.c_THGenerator_copy g0 g1) >>= asRNG

-- | Get the current 'Seed' of a 'Generator'
seed :: Generator -> IO Seed
seed = withRNG (fmap fromIntegral . TH.c_THRandom_seed)

-- | Manually set the seed 'Seed' of a 'Generator'
manualSeed :: Generator -> Seed -> IO ()
manualSeed g s = _withRNG g $ \p -> TH.c_THRandom_manualSeed p (fromIntegral s)

-- | Get the first 'Seed' that initialized a given 'Generator'
initialSeed :: Generator -> IO Seed
initialSeed = withRNG (fmap fromIntegral . TH.c_THRandom_initialSeed)

random :: Generator -> IO Seed
random = withRNG (fmap fromIntegral . TH.c_THRandom_random)

random64 :: Generator -> IO Seed
random64 = withRNG (fmap fromIntegral . TH.c_THRandom_random64)

-- | Returns a random double according to uniform distribution on [a,b).
uniform
  :: Generator
  -> Double -- ^ lower bound
  -> Double -- ^ upper bound
  -> IO Double
uniform g a b = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_uniform p (realToFrac a) (realToFrac b)

-- | Returns a random float according to uniform distribution on [a,b).
uniformFloat
  :: Generator
  -> Float -- ^ lower bound
  -> Float -- ^ upper bound
  -> IO Float
uniformFloat g a b = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_uniformFloat p (realToFrac a) (realToFrac b)

-- | Returns a random real number according to a normal distribution with the given mean and standard deviation stdv. stdv must be positive, but this is not yet enforced.
--
-- TODO: add a @newtype Pos a = Pos { getPos :: a }@ package with a smart constructor export
normal
  :: Generator
  -> Double -- ^ mean
  -> Double -- ^ stddev (must be positive)
  -> IO Double
normal g a b = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_normal p (realToFrac a) (realToFrac b)

-- | Returns a random real number according to the exponential distribution @p(x) = lambda * exp(-lambda * x)@
exponential :: Generator -> Double -> IO Double
exponential g a = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_exponential p (realToFrac a)

standard_gamma :: Generator -> Double -> IO Double
standard_gamma g a = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_standard_gamma p (realToFrac a)

-- | Returns a random real number according to the Cauchy distribution @p(x) = sigma/(pi*(sigma^2 + (x-median)^2))@
cauchy
  :: Generator
  -> Double -- ^ median
  -> Double -- ^ sigma
  -> IO Double
cauchy g a b = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_cauchy p (realToFrac a) (realToFrac b)

-- | Returns a random real number according to the log-normal distribution, with the given mean and
-- standard deviation stdv. mean and stdv are the corresponding mean and standard deviation of the
-- underlying normal distribution, and not of the returned distribution.
--
-- stdv must be positive.
logNormal
  :: Generator
  -> Double -- ^ mean
  -> Double -- ^ stddev (must be positive)
  -> IO Double
logNormal g a b = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_logNormal p (realToFrac a) (realToFrac b)

-- | Returns a random integer number according to a geometric distribution
-- @p(i) = (1-p) * p^(i-1)@. p must satisfy 0 < p < 1.
geometric :: Generator -> Double -> IO Int
geometric g a = _withRNG g $ \p -> fromIntegral <$> TH.c_THRandom_geometric p (realToFrac a)

-- | Returns 1 with probability p and 0 with probability 1-p. p must satisfy 0 <= p <= 1.
--
-- TODO: By default p is equal to 0.5 -- this isn't encoded in the API
bernoulli :: Generator -> Double -> IO Int
bernoulli g a = _withRNG g $ \p -> fromIntegral <$> TH.c_THRandom_bernoulli p (realToFrac a)


