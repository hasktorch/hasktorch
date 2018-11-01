{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP #-}

#if MIN_VERSION_base(4,12,0)
{-# LANGUAGE NoStarIsType #-}
#endif

{-# OPTIONS_GHC -fno-cse #-}
module Torch.Static.NN.LinearSpec where

import GHC.TypeLits
import Control.Monad (join, void)
import Data.Function ((&))
import Data.Foldable
import Debug.Trace
import Data.Maybe
import GHC.Generics (Generic)
import Test.Hspec
import Lens.Micro.Platform
import Numeric.Backprop
import System.IO.Unsafe
import Data.Generics.Product
import qualified Numeric.Backprop as B

import Debug.Trace

import Torch.Double as Torch
import Torch.Double.NN.Linear
import qualified Torch.Long as Long

reasonablyUnsafeVector :: (KnownDim m, KnownNat m) => [HsReal] -> Tensor '[m]
reasonablyUnsafeVector = unsafePerformIO . unsafeVector
{-# NOINLINE reasonablyUnsafeVector #-}

reasonablyUnsafeLongVector :: (KnownDim m, KnownNat m) => [Long.HsReal] -> Long.Tensor '[m]
reasonablyUnsafeLongVector = unsafePerformIO . Long.unsafeVector
{-# NOINLINE reasonablyUnsafeLongVector #-}

reasonablyUnsafeMatrix
  :: All KnownDim '[m, n, n*m]
  => All KnownNat '[m, n, n*m]
  => [[HsReal]]
  -> Tensor '[n,m]
reasonablyUnsafeMatrix = unsafePerformIO . unsafeMatrix
{-# NOINLINE reasonablyUnsafeMatrix #-}

xavier :: forall d . Dimensions d => IO (Tensor d)
xavier = case (fromIntegral <$> listDims (dims :: Dims d)) of
  [] -> pure empty
  a:_ -> pure $ constant (1 / realToFrac (fromIntegral a))

data FF2Network i h o = FF2Network
  { layer1 :: Linear i h
  , layer2 :: Linear h o
  } deriving (Generic, Show)

instance (KnownDim i, KnownDim h, KnownDim o) => Pairwise (FF2Network i h o) HsReal where
  (FF2Network l0 l1) ^+ v = FF2Network (l0 ^+ v) (l1 ^+ v)
  (FF2Network l0 l1) ^- v = FF2Network (l0 ^+ v) (l1 ^+ v)
  (FF2Network l0 l1) ^* v = FF2Network (l0 ^+ v) (l1 ^+ v)
  (FF2Network l0 l1) ^/ v = FF2Network (l0 ^+ v) (l1 ^+ v)

weightsL :: Lens' (Linear i o) (Tensor '[i, o])
weightsL = field @"getTensors" . _1

biasL :: Lens' (Linear i o) (Tensor '[o])
biasL = field @"getTensors" . _2

specupdate
  :: forall i h o
  .  All KnownDim '[i, h, o]
  => FF2Network i h o
  -> FF2Network i h o
  -> FF2Network i h o
specupdate i g = FF2Network
  { layer1 = B.add (layer1 i) (layer1 g)
  , layer2 = B.add (layer2 i) (layer2 g)
  }

instance (KnownDim i, KnownDim h, KnownDim o) => Backprop (FF2Network i h o)

-- ========================================================================= --

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "a single linear layer" singleLayer
  describe "a two-layer feed forward network" $ do
    describe "with xavier initialization" twoLayerXavier
    describe "forcing ReLU activity"      twoLayerForceReLU
    describe "overfitting to [0, 1]" $ do
      describe "with one layer and binary cross-entropy"   oneLayerOverfit
      describe "with two layers and binary cross-entropy"   twoLayerOverfit
      -- describe "with logSoftmax and multiclass log-loss" $
      --   twoLayerOverfit logSoftMax (classNLLCriterion (reasonablyUnsafeLongVector [1])) Torch.exp

-- ========================================================================= --

singleLayer :: Spec
singleLayer = do
  ll :: Linear 3 2 <- runIO $ mkLinear xavier
  xavierPurityCheck ll $ do
    describe "the forward pass" $ do
      let y = constant 5 :: Tensor '[3]
          o = evalBP2 (linear) ll y

      it "performs matrix multipication as you would expect" $ do
        o =##= ((5/3)*3) + 1/2
        o `elementsSatisfy` ((== 2) . length)

    describe "the backward pass" $ do
      let y = constant 1 :: Tensor '[3]
          (_, (ll', o)) = backprop2 linear ll y

      it "returns plain gradient of weights" $ weights ll' =##= 1   -- 1/2
      it "returns plain gradient of bias"    $ bias    ll' =##= 3/2 -- 2/3
      it "returns plain gradient of output tensor" $    o  =##= 2/3 -- 1/3

-- ========================================================================= --

mkXavierNetwork :: All KnownDim '[i,h,o] => IO (FF2Network i h o)
mkXavierNetwork =
  FF2Network
    <$> mkLinear xavier
    <*> mkLinear xavier


mkUniform :: All KnownDim '[i,h,o] => IO (FF2Network i h o)
mkUniform = do
  g <- newRNG
  manualSeed g 1
  let Just rg = ord2Tuple (-1, 1)
  w0 <- uniform g rg
  w1 <- uniform g rg
  pure $ FF2Network
    (Linear (w0, constant 1))
    (Linear (w1, constant 1))


mkGaussianNetwork :: All KnownDim '[i,h,o] => IO (FF2Network i h o)
mkGaussianNetwork = do
  g <- newRNG
  manualSeed g 1
  let Just std = positive 2

  FF2Network
    <$> mkLinear (normal g 0 std)
    <*> mkLinear (normal g 0 std)


ff2network
  :: forall s i h o
  .  Reifies s W
  => All KnownDim '[i,h,o]
  => (forall s . Reifies s W => BVar s (Tensor '[o]) -> BVar s (Tensor '[o]))
  -> Double
  -> BVar s (FF2Network i h o)     -- ^ ff2network architecture
  -> BVar s (Tensor '[i])          -- ^ input
  -> BVar s (Tensor '[1, o])       -- ^ output
ff2network final lr arch inp
  = linear {-lr-} (arch ^^. field @"layer1") inp
  & relu
  & linear {-lr-} (arch ^^. field @"layer2")
  & final
  & foo
  where
    foo t = unsqueeze1dBP (dim :: Dim 0) t

twoLayerXavier :: Spec
twoLayerXavier = do
  ll :: FF2Network 4 6 2 <- runIO mkXavierNetwork
  describe "the forward pass" $ do
    describe "with xavier instantiation" $ do
      xavierPurityCheck (ll ^. field @"layer1") $
        xavierPurityCheck (ll ^. field @"layer2") $
          describe "with input all positive input" $ do
            let y = constant 4 :: Tensor '[4]
                (o, _) = backprop2 (ff2network softmax undefined) ll y
            it "performs matrix multipication as you would expect" $ o `approx` [1/2, 1/2]

      describe "with input that drops all values via ReLU" $ do
        let y = constant (-1) :: Tensor '[4]
            (o, _) = backprop2 (ff2network softmax undefined) ll y

        it "performs matrix multipication as you would expect" $ o `approx` [1/2, 1/2]
  where
    approx = lapproximately 0.0001


twoLayerForceReLU :: Spec
twoLayerForceReLU = do
  describe "operations that force relu activity" $ do
    let o1 = evalBP2 (relu    .: linear {-1-}) (ff2 ^. field @"layer1") oneInput
        o2 = evalBP2 (           linear {-1-}) (ff2 ^. field @"layer2") o1
        gin :: Tensor '[4]
        (out, (gff2, gin)) = backprop2 (ff2network logSoftMax 1) ff2 oneInput

    describe "dropping half the gradient during ReLU" $ do
      describe "the forward pass" $ do
        it "returns [0,0,0,4,4,4] after the first layer" $
          tensordata o1 `shouldBe` [0,0,0,4,4,4]

        it "returns [-12,0] after the second layer" $
          tensordata o2 `shouldBe` [-12, 0]

        it "returns [0, 1] as the output" $ do
          Torch.exp out `lapprox` [0, 1]

      describe "the backward pass" $ do
        it "returns a half zero-d out layer 1 gradient" $ do
          (gff2 ^. field @"layer1" . weightsL) `approx` l1weightgrad

        it "returns a quarter zero-d out layer 2 gradient" $ do
          (gff2 ^. field @"layer2" . weightsL) `approx` l2weightgrad

        it "returns a [3,3,3] input gradient" $ do
          gin `lapprox` replicate 4 (-3)

 where
  eps = 0.0001

  approx :: Tensor d -> Tensor d -> IO ()
  approx = approximately eps

  lapprox :: Tensor d -> [HsReal] -> IO ()
  lapprox = lapproximately eps

  ff2 :: FF2Network 4 6 2
  ff2 = FF2Network
    (Linear (reasonablyUnsafeMatrix $ replicate 4 [ -1, -1, -1, 1, 1, 1], constant 0))
    (Linear (reasonablyUnsafeMatrix $ replicate 6                [-1, 0], constant 0))

  oneInput :: Tensor '[4]
  oneInput = constant 1

  l1weightgrad :: (Tensor '[4, 6])
  l1weightgrad = reasonablyUnsafeMatrix $ replicate 4 [ 0, 0, 0,-1,-1,-1]

  l2weightgrad :: (Tensor '[6, 2])
  l2weightgrad = reasonablyUnsafeMatrix $ replicateN 3
    [ [ 0, 0]
    , [ 4,-4]
    ]

twoLayerOverfit :: Spec
twoLayerOverfit = do
  net0 <- runIO $ do
    g <- newRNG
    manualSeed g 1
    l0 <- (Linear . (,constant 1)) <$> uniform g rg
    l1 <- (Linear . (,constant 1)) <$> uniform g rg
    pure (l0, l1)

  it "returns around 50-50 on uniform random initialization" . void $ do
    let [l0, r0] = tensordata $ infer net0
    let pointapprox pred truth = Prelude.abs (pred - truth) < 0.3
    (l0, r0) `shouldSatisfy` (\(px, py) -> pointapprox px 0.5 && pointapprox py 0.5)

  it "backprops to yield a loss smaller than its prior" . void $ do
    let [l0, r0] = tensordata $ infer net0
    let lr = (-0.001) :: HsReal
    let (o, _) = bprop net0
    (fnet, (fo, fl, fr)) <-
      foldlM (\(net, (o, l, r)) i -> do
        let (o, (Linear (gw0, gb0), Linear (gw1, gb1))) = bprop net
        let net' = B.add net (Linear (gw0 ^* lr, gb0 ^* lr), Linear (gw1 ^* lr, gb1 ^* lr))
        let (o', grad') = bprop net'
        let [l', r'] = tensordata $ infer net'
        o `shouldSatisfy` (> o')
        pure (net', (o', l', r'))
        ) (net0, (o, l0, r0)) [1..100]
    let pointapprox pred truth = Prelude.abs (pred - truth) < 0.01
    (fl, fr) `shouldSatisfy` (\(px, py) -> pointapprox px 0.0 && pointapprox py 1.0)

  where
    Just rg = ord2Tuple (-1, 1)

    x :: Tensor '[4]
    x = constant 1

    answer :: Tensor '[2]
    answer = reasonablyUnsafeVector [0,1]

    arch :: Reifies s W => BVar s (Linear 4 6, Linear 6 2) -> BVar s (Tensor '[4]) -> BVar s (Tensor '[2])
    arch arch inp
      = linear (arch ^^. _1) inp
      & relu
      & linear (arch ^^. _2)
      & softmax

    infer :: (Linear 4 6, Linear 6 2) -> Tensor '[2]
    infer net = evalBP2 arch net x

    bprop net = (fromJust $ get1d o 0, g)
     where
      (o, (g, _)) = backprop2 (bCECriterion answer .: arch) net x



oneLayerOverfit :: Spec
oneLayerOverfit = do
  net0 <- runIO (newRNG >>= \g -> manualSeed g 1 >> (Linear . (,constant 1)) <$> uniform g rg)
  it "returns around 50-50 on uniform random initialization" . void $ do
    let [l0, r0] = tensordata $ infer net0
    let pointapprox pred truth = Prelude.abs (pred - truth) < 0.3
    (l0, r0) `shouldSatisfy` (\(px, py) -> pointapprox px 0.5 && pointapprox py 0.5)

  it "backprops to yield a loss smaller than its prior" . void $ do
    let [l0, r0] = tensordata $ infer net0
    let lr = (-0.1) :: HsReal
    let (o, _) = bprop net0
    (fnet, (fo, fl, fr)) <-
      foldlM (\(net, (o, l, r)) i -> do
        let (o, Linear (gw, gb)) = bprop net
        let net' = B.add net (Linear (gw ^* lr, gb ^* lr))
        let (o', grad') = bprop net'
        let [l', r'] = tensordata $ infer net'
        o `shouldSatisfy` (> o')
        pure (net', (o', l', r'))
        ) (net0, (o, l0, r0)) [1..100]
    let pointapprox pred truth = Prelude.abs (pred - truth) < 0.01
    (fl, fr) `shouldSatisfy` (\(px, py) -> pointapprox px 0.0 && pointapprox py 1.0)

  where
    Just rg = ord2Tuple (-1, 1)

    x :: Tensor '[6]
    x = constant 1

    answer :: Tensor '[2]
    answer = (reasonablyUnsafeVector [0,1])

    arch :: Reifies s W => BVar s (Linear 6 2) -> BVar s (Tensor '[6]) -> BVar s (Tensor '[2])
    arch a b = softmax $ linear a b

    infer :: Linear 6 2 -> Tensor '[2]
    infer net = evalBP2 arch net x

    bprop net = (fromJust $ get1d o 0, g)
     where
      (o, (g, _)) = backprop2 (bCECriterion answer .: arch) net x


xavierPurityCheck :: forall i o . (KnownDim i, KnownDim o) => Linear i o -> Spec -> Spec
xavierPurityCheck ll tests =
  it (header ++ "initializes with xavier correctly")
    (  weights ll =##= 1/i
    >> bias    ll =##= 1/o)
  >> tests
  >> it (header ++ "leaves weights unchanged") (weights ll =##= 1/i)
  >> it (header ++ "leaves bias unchanged")    (bias    ll =##= 1/o)
 where
  header :: String
  header = "[ref-check] " ++ unwords ["Linear", show (truncate i), show (truncate o)] ++ ": "

  i, o :: Double
  i = fromIntegral (dimVal (dim :: Dim i))
  o = fromIntegral (dimVal (dim :: Dim o))


_lapproximately :: ([Double] -> Bool) -> Double -> Tensor d -> [Double] -> IO ()
_lapproximately pred e o dist = let os = tensordata o in
  zipWith (Prelude.abs .: subtract) os dist `shouldSatisfy` pred

_approximately :: ([Double] -> Bool) -> Double -> Tensor d -> Tensor d -> IO ()
_approximately pred e o dist = _lapproximately pred e o (tensordata dist)

approximately  e = _approximately  (all (< e)) e
notCloseTo     e = _approximately  (all (> e)) e
lapproximately e = _lapproximately (all (< e)) e
lnotCloseTo    e = _lapproximately (all (> e)) e

elementsSatisfy :: Tensor d -> ([Double] -> Bool) -> IO ()
elementsSatisfy o pred = tensordata o `shouldSatisfy` pred

replicateN :: Int -> [a] -> [a]
replicateN inner = concatMap (replicate inner)

(=##=) :: Tensor d -> Double -> IO ()
(=##=) o v = elementsSatisfy o (all (== v))

infixl 2 =##=


