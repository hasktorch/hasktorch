module Torch.Core.Random
  ( Generator
  , new
  , copy
  , isValid
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

import THTypes (CTHGenerator)
import THRandomTypes
import qualified THRandom as TH

-- ========================================================================= --
-- helpers
-- ========================================================================= --

asRNG :: Ptr CTHGenerator -> IO Generator
asRNG = fmap Generator . newForeignPtr TH.p_THGenerator_free

with2RNGs :: (Ptr CTHGenerator -> Ptr CTHGenerator -> IO x) -> Generator -> Generator -> IO x
with2RNGs fn g0 g1 = _with2RNGs g0 g1 fn

_with2RNGs :: Generator -> Generator -> (Ptr CTHGenerator -> Ptr CTHGenerator -> IO x) -> IO x
_with2RNGs g0 g1 fn = _withRNG g0 (\g0' -> _withRNG g1 (\g1' -> fn g0' g1'))

withRNG :: (Ptr CTHGenerator -> IO x) -> Generator -> IO x
withRNG fn g = withForeignPtr (rng g) fn

_withRNG :: Generator -> (Ptr CTHGenerator -> IO x) -> IO x
_withRNG = flip withRNG

-- ========================================================================= --
-- Memory managed versions of THRandom
new :: IO Generator
new = TH.c_THGenerator_new >>= asRNG

copy :: Generator -> Generator -> IO Generator
copy g0 g1 = (with2RNGs TH.c_THGenerator_copy g0 g1) >>= asRNG

isValid :: Generator -> IO Bool
isValid = withRNG (\p -> pure $ TH.c_THGenerator_isValid p == 1)

seed :: Generator -> IO Word64
seed = withRNG (pure . fromIntegral . TH.c_THRandom_seed)

manualSeed :: Generator -> Word64 -> IO ()
manualSeed g s = _withRNG g $ \p -> TH.c_THRandom_manualSeed p (fromIntegral s)

initialSeed :: Generator -> IO Word64
initialSeed = withRNG (pure . fromIntegral . TH.c_THRandom_initialSeed)

random :: Generator -> IO Word64
random = withRNG (pure . fromIntegral . TH.c_THRandom_random)

random64 :: Generator -> IO Word64
random64 = withRNG (pure . fromIntegral . TH.c_THRandom_random64)

uniform :: Generator -> Double -> Double -> IO Double
uniform g a b = _withRNG g $ \p -> pure . realToFrac $ TH.c_THRandom_uniform p (realToFrac a) (realToFrac b)

uniformFloat :: Generator -> Float -> Float -> IO Float
uniformFloat g a b = _withRNG g $ \p -> pure . realToFrac $ TH.c_THRandom_uniformFloat p (realToFrac a) (realToFrac b)

normal :: Generator -> Double -> Double -> IO Double
normal g a b = _withRNG g $ \p -> pure . realToFrac $ TH.c_THRandom_normal p (realToFrac a) (realToFrac b)

exponential :: Generator -> Double -> IO Double
exponential g a = _withRNG g $ \p -> pure . realToFrac $ TH.c_THRandom_exponential p (realToFrac a)

standard_gamma :: Generator -> Double -> IO Double
standard_gamma g a = _withRNG g $ \p -> pure . realToFrac $ TH.c_THRandom_standard_gamma p (realToFrac a)

cauchy :: Generator -> Double -> Double -> IO Double
cauchy g a b = _withRNG g $ \p -> pure . realToFrac $ TH.c_THRandom_cauchy p (realToFrac a) (realToFrac b)

logNormal :: Generator -> Double -> Double -> IO Double
logNormal g a b = _withRNG g $ \p -> pure . realToFrac $ TH.c_THRandom_logNormal p (realToFrac a) (realToFrac b)

geometric :: Generator -> Double -> IO Int
geometric g a = _withRNG g $ \p -> pure . fromIntegral $ TH.c_THRandom_geometric p (realToFrac a)

bernoulli :: Generator -> Double -> IO Int
bernoulli g a = _withRNG g $ \p -> pure . fromIntegral $ TH.c_THRandom_bernoulli p (realToFrac a)


