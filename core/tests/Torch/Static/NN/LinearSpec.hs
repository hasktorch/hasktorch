{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
module Torch.Static.NN.LinearSpec where

import GHC.TypeLits
import Control.Monad (join)
import Data.Function ((&))
import Data.Foldable
import Debug.Trace
import GHC.Generics (Generic)
import Test.Hspec
import Lens.Micro.Platform
import Numeric.Backprop
import qualified Numeric.Backprop as B

import Torch.Double
import Torch.Double.NN.Linear


data FF2Network i h o = FF2Network
  { _layer1 :: Linear i h
  , _layer2 :: Linear h o
  } deriving (Generic, Show)

weightsL :: Lens' (Linear i o) (Tensor '[i, o])
weightsL = lens weights $ \(Linear (_, b)) w' -> Linear (w', b)

biasL :: Lens' (Linear i o) (Tensor '[o])
biasL = lens bias $ \(Linear (w, _)) b' -> Linear (w, b')

update
  :: forall i h o
  .  All KnownDim '[i, h, o]
  => FF2Network i h o
  -> FF2Network i h o
  -> FF2Network i h o
update i g = FF2Network
  { _layer1 = B.add (_layer1 i) (_layer1 g)
  , _layer2 = B.add (_layer2 i) (_layer2 g)
  }

makeLenses ''FF2Network
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
    describe "overfitting to [-500, 338]" twoLayerOverfit

-- ========================================================================= --

singleLayer :: Spec
singleLayer = do
  ll :: Linear 3 2 <- runIO $ mkLinear xavier
  xavierPurityCheck ll $ do
    describe "the forward pass" $ do
      let y = constant 5 :: Tensor '[3]
          o = evalBP2 (linear undefined) ll y

      it "performs matrix multipication as you would expect" $ do
        o =##= ((5/3)*3) + 1/2
        o `elementsSatisfy` ((== 2) . length)

    describe "the backward pass" $ do
      let y = constant 1 :: Tensor '[3]
          lr = 1.0
          (_, (ll', o)) = backprop2 (linear lr) ll y

      it "returns updated weights" $ weights ll' =##= 1/2
      it "returns updated bias"    $ bias    ll' =##= 1/2

      it "returns the updated output tensor" $ o =##= 1/3

-- ========================================================================= --

mkFF2Network :: All KnownDim '[i,h,o] => IO (FF2Network i h o)
mkFF2Network = FF2Network
  <$> mkLinear xavier
  <*> mkLinear xavier

ff2network
  :: forall s i h o
  .  Reifies s W
  => All KnownDim '[i,h,o]
  => Double
  -> BVar s (FF2Network i h o)  -- ^ ff2network architecture
  -> BVar s (Tensor '[i])       -- ^ input
  -> BVar s (Tensor '[o])       -- ^ output
ff2network lr arch inp
  = linear lr (arch ^^. layer1) inp
  & relu
  & linear lr (arch ^^. layer2)
  & softmax

train'
  :: forall i h o
  .  All KnownDim '[i,h,o]
  => All KnownNat '[i,h,o]
  => Double
  -> Tensor '[o]
  -> FF2Network i h o  -- ^ ff2network architecture
  -> Tensor '[i]       -- ^ input
  -> FF2Network i h o  -- ^ new ff2network architecture
train' lr tar arch inp = update arch grad
  where
    (grad, _) = gradBP2 (bCECriterion True True Nothing tar .: ff2network lr) arch inp


twoLayerXavier :: Spec
twoLayerXavier = do
  ll :: FF2Network 4 6 2 <- runIO mkFF2Network
  describe "the forward pass" $ do
    describe "with xavier instantiation" $ do
      xavierPurityCheck (ll ^. layer1) $
        xavierPurityCheck (ll ^. layer2) $
          describe "with input all positive input" $ do
            let y = constant 4 :: Tensor '[4]
                (o, _) = backprop2 (ff2network undefined) ll y

            it "performs matrix multipication as you would expect" $ o =##= 1/2

      describe "with input that drops all values via ReLU" $ do
        let y = constant (-1) :: Tensor '[4]
            (o, _) = backprop2 (ff2network undefined) ll y

        it "performs matrix multipication as you would expect" $ o =##= 1/2

twoLayerForceReLU :: Spec
twoLayerForceReLU = do
  describe "operations that force relu activity" $ do
    let o1 = evalBP2 (relu    .: linear 1) (ff2 ^. layer1) oneInput
        o2 = evalBP2 (           linear 1) (ff2 ^. layer2) o1
        (out, (gff2, gin)) = backprop2 (ff2network 1) ff2 oneInput
    describe "dropping half the gradient during ReLU" $ do
      describe "the forward pass" $ do
        it "returns [0,0,0,4,4,4] after the first layer" $
          tensordata o1 >>= (`shouldBe` [0,0,0,4,4,4])

        it "returns [0,0,0,4,4,4] after the second layer" $
          tensordata o2 >>= (`shouldBe` [12, 0])

        it "returns [1, 0] as the output" $
          tensordata out >>= (`shouldBe` [1, 0])

      describe "the backward pass" $ do
        it "returns a half zero-d out layer 1 gradient" $ do
          join (shouldBe
            <$> tensordata (gff2 ^. layer1 . weightsL)
            <*> tensordata l1weightgrad)

        it "returns a quarter zero-d out layer 2 gradient" $ do
          join (shouldBe
            <$> tensordata (gff2 ^. layer2 . weightsL)
            <*> tensordata l2weightgrad)

        it "returns a [-33,-33,-33,-33] input gradient" $ do
          tensordata gin >>= (`shouldBe` replicate 4 (-33))

 where
  ff2 :: FF2Network 4 6 2
  ff2 = FF2Network
    (Linear (unsafeMatrix $ replicate 4 [ -1, -1, -1, 1, 1, 1], constant 0))
    (Linear (unsafeMatrix $ replicate 6                 [1, 0], constant 0))

  oneInput :: Tensor '[4]
  oneInput = constant 1

  l1weightgrad :: Tensor '[4, 6]
  l1weightgrad = unsafeMatrix $ replicate 4 [ 0, 0, 0, -11, -11, -11]

  l2weightgrad :: Tensor '[6, 2]
  l2weightgrad = unsafeMatrix
    [ [   0, 0]
    , [   0, 0]
    , [   0, 0]
    , [ -44, 0]
    , [ -44, 0]
    , [ -44, 0]
    ]

twoLayerOverfit :: Spec
twoLayerOverfit = do
  net0 :: FF2Network 4 6 2 <- runIO mkFF2Network
  describe "with xavier instantiation and binary cross-entropy error" $ do
    it "returns a balanced distribution without training" $
      tensordata (evalBP2 (ff2network undefined) net0  y) >>= (`shouldBe` [0.5, 0.5])

    it "overfits on a single input with lr=1.0 and 100 steps" $ do
      let net' = foldl' (train t 1.0) net0 (replicate 1 y)
      tensordata (evalBP2 (ff2network undefined) net' y) >>= (`shouldBe` [])

      -- tensordata (evalBP2 (ff2network undefined) ll  y) >>= (`shouldBe` [0.0, 1.0])
      -- tensordata (ll' ^. layer2 . weightsL) >>= (`shouldBe` [])
      -- tensordata (ll' ^. layer2 . weightsL) >>= (`shouldBe` [])


  where
    y = constant 1          :: Tensor '[4]
    t = unsafeVector [0, 0.99] :: Tensor '[2]

    train
      :: forall i h o
      .  All KnownDim '[i,h,o]
      => All KnownNat '[i,h,o]
      => Tensor '[o]
      -> Double
      -> FF2Network i h o  -- ^ ff2network architecture
      -> Tensor '[i]       -- ^ input
      -> FF2Network i h o  -- ^ new ff2network architecture
    train tar lr arch inp = update arch
      (fst (gradBP2 (bCECriterion True True Nothing tar .: ff2network lr) arch inp))


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
_lapproximately pred e o dist = tensordata o >>= \os ->
  zipWith (Prelude.abs .: (-)) os dist `shouldSatisfy` pred

_approximately :: ([Double] -> Bool) -> Double -> Tensor d -> Tensor d -> IO ()
_approximately pred e o dist = tensordata dist >>= _lapproximately pred e o

approximately  e = _approximately  (all (< e)) e
notCloseTo     e = _approximately  (all (> e)) e
lapproximately e = _lapproximately (all (< e)) e
lnotCloseTo    e = _lapproximately (all (> e)) e

elementsSatisfy :: Tensor d -> ([Double] -> Bool) -> IO ()
elementsSatisfy o pred = tensordata o >>= (`shouldSatisfy` pred)

replicateN :: Int -> [a] -> [a]
replicateN inner = concatMap (replicate inner)

(=##=) :: Tensor d -> Double -> IO ()
(=##=) o v = elementsSatisfy o (all (== v))

infixl 2 =##=


