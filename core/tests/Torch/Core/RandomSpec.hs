{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

module Torch.Core.RandomSpec (spec) where

import Control.Monad (replicateM)
import Foreign (Ptr)

import qualified Control.Exception as E
import Torch.Core.Random as R
import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "newRNG" newRNGSpec
  describe "seed" seedSpec
  describe "manualSeed" manualSeedSpec
  describe "initialSeed" initialSeedSpec
  describe "random" randomSpec
  describe "uniform" uniformSpec
  describe "normal" normalSpec
  describe "exponential" exponentialSpec
  describe "cauchy" cauchySpec
  describe "logNormal" logNormalSpec
  describe "geometric" geometricSpec
  describe "bernoulli" bernoulliSpec
  describe "scenario" $ do
    it "runs this scenario as expected" $ testScenario

newRNGSpec :: Spec
newRNGSpec = do
  rngs <- runIO (replicateM 10 R.new)
  it "always creates a new random number" $
    zipWith (==) (tail rngs) (init rngs) `shouldNotContain` [True]

seedSpec :: Spec
seedSpec = do
  beforeAll
    (do
        rngs <- (replicateM 10 R.new)
        rng1 <- mapM seed rngs
        rng2 <- mapM seed rngs
        pure (rngs, rng1, rng2)
    )
    (describe "seedSpec" $ do
      it "generates different values, given the same starting generators" $
        \(rngs, rng1, rng2) -> do
          zipWith (==) rng1 rng2 `shouldNotContain` [True]
    )

manualSeedSpec :: Spec
manualSeedSpec = do
  rngs <- runIO (replicateM 10 R.new)
  rng1 <- runIO $ mapM (`manualSeed` 1) rngs
  rng2 <- runIO $ mapM (`manualSeed` 1) rngs

  it "generates the same value, given the same seed values" $
    zipWith (==) rng1 rng2 `shouldNotContain` [False]

initialSeedSpec :: Spec
initialSeedSpec = do
  it "doesn't crash" $
    pending

randomSpec :: Spec
randomSpec = do
  rngs <- runIO (replicateM 10 R.new)
  rs <- runIO $ mapM random rngs
  it "generates numbers and doesn't crash" $
    rs `shouldSatisfy` doesn'tCrash

uniformSpec :: Spec
uniformSpec = do
  rng <- runIO R.new
  distributed2BoundsCheck rng uniform $ \a b x ->
    case compare a b of
      LT -> x <= b && x >= a
      _  -> x <= a && x >= b

normalSpec :: Spec
normalSpec = do
  rng <- runIO R.new
  distributed2BoundsCheck rng normal (\a b x -> doesn'tCrash ())

exponentialSpec :: Spec
exponentialSpec = do
  rng <- runIO R.new
  distributed1BoundsCheck rng exponential (\a x -> doesn'tCrash ())

cauchySpec :: Spec
cauchySpec = do
  rng <- runIO R.new
  distributed2BoundsCheck rng cauchy (\a b x -> doesn'tCrash ())

logNormalSpec :: Spec
logNormalSpec = do
  rng <- runIO R.new
  distributed2BoundsCheck rng logNormal (\a b x -> doesn'tCrash ())

geometricSpec :: Spec
geometricSpec = do
  rng <- runIO R.new
  distributed1BoundsCheck rng geometric (\a x -> doesn'tCrash ())

bernoulliSpec :: Spec
bernoulliSpec = do
  rng <- runIO R.new
  distributed1BoundsCheck rng bernoulli (\a x -> doesn'tCrash ())

-- |Check that seeds work as intended
testScenario :: IO ()
testScenario = do
  rng <- R.new
  manualSeed rng 332323401
  val1 <- normal rng 0.0 1000
  val2 <- normal rng 0.0 1000
  E.assert (val1 /= val2) pure ()
  manualSeed rng 332323401
  manualSeed rng 332323401
  val3 <- normal rng 0.0 1000.0
  E.assert (val1 == val3) pure ()


-- ========================================================================= --

distributed2BoundsCheck :: (Show a, Show b, Arbitrary a, Arbitrary b) => Generator -> (Generator -> a -> b -> IO Double) -> (a -> b -> Double -> Bool) -> Spec
distributed2BoundsCheck g fun check = do
  it "should generate random numbers in the correct bounds" . property $ \(a, b) ->
    monadicIO $ do
      x <- run (fun g a b)
      assert (check a b x)

distributed1BoundsCheck :: (Show a, Arbitrary a) => Generator -> (Generator -> a -> IO b) -> (a -> b -> Bool) -> Spec
distributed1BoundsCheck g fun check = do
  it "should generate random numbers in the correct bounds" . property $ \a -> monadicIO $ do
    x <- run (fun g a)
    assert (check a x)



