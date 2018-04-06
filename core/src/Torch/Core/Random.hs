{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Torch.Core.Random
  ( Generator
  , Seed
  , new
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

newtype Seed = Seed { unSeed :: Word64 }
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

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
-- Memory managed versions of THRandom
new :: IO Generator
new = TH.c_THGenerator_new >>= asRNG

copy :: Generator -> Generator -> IO Generator
copy g0 g1 = (with2RNGs TH.c_THGenerator_copy g0 g1) >>= asRNG

seed :: Generator -> IO Seed
seed = withRNG (fmap fromIntegral . TH.c_THRandom_seed)

manualSeed :: Generator -> Seed -> IO ()
manualSeed g s = _withRNG g $ \p -> TH.c_THRandom_manualSeed p (fromIntegral s)

initialSeed :: Generator -> IO Seed
initialSeed = withRNG (fmap fromIntegral . TH.c_THRandom_initialSeed)

random :: Generator -> IO Seed
random = withRNG (fmap fromIntegral . TH.c_THRandom_random)

random64 :: Generator -> IO Seed
random64 = withRNG (fmap fromIntegral . TH.c_THRandom_random64)

uniform :: Generator -> Double -> Double -> IO Double
uniform g a b = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_uniform p (realToFrac a) (realToFrac b)

uniformFloat :: Generator -> Float -> Float -> IO Float
uniformFloat g a b = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_uniformFloat p (realToFrac a) (realToFrac b)

normal :: Generator -> Double -> Double -> IO Double
normal g a b = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_normal p (realToFrac a) (realToFrac b)

exponential :: Generator -> Double -> IO Double
exponential g a = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_exponential p (realToFrac a)

standard_gamma :: Generator -> Double -> IO Double
standard_gamma g a = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_standard_gamma p (realToFrac a)

cauchy :: Generator -> Double -> Double -> IO Double
cauchy g a b = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_cauchy p (realToFrac a) (realToFrac b)

logNormal :: Generator -> Double -> Double -> IO Double
logNormal g a b = _withRNG g $ \p -> realToFrac <$> TH.c_THRandom_logNormal p (realToFrac a) (realToFrac b)

geometric :: Generator -> Double -> IO Int
geometric g a = _withRNG g $ \p -> fromIntegral <$> TH.c_THRandom_geometric p (realToFrac a)

bernoulli :: Generator -> Double -> IO Int
bernoulli g a = _withRNG g $ \p -> fromIntegral <$> TH.c_THRandom_bernoulli p (realToFrac a)


