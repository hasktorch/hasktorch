{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
module Torch.Static.NN.LinearSpec where

import Data.Function ((&))
import GHC.Generics
import Test.Hspec
import Lens.Micro.Platform
import Numeric.Backprop

import Torch.Double
import Torch.Double.NN.Linear


data FF2Network i h o = FF2Network
  { _layer1 :: Linear i h
  , _layer2 :: Linear h o
  } deriving Generic

weightsL :: Lens' (Linear i o) (Tensor '[i, o])
weightsL = lens weights $ \(Linear (_, b)) w' -> Linear (w', b)

biasL :: Lens' (Linear i o) (Tensor '[o])
biasL = lens bias $ \(Linear (w, _)) b' -> Linear (w, b')

makeLenses ''FF2Network
instance (KnownDim i, KnownDim h, KnownDim o) => Backprop (FF2Network i h o)

-- ========================================================================= --

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "a single linear layer" singleLayer
  describe "a two-layer feed forward network" twoLayer

-- ========================================================================= --

singleLayer :: Spec
singleLayer = do
  ll :: Linear 3 2 <- runIO $ mkLinear xavier
  xavierPurityCheck ll $ do
    describe "the forward pass" $ do
      let y = constant 5 :: Tensor '[3]
          (o, _) = backprop2 (linear undefined) ll y

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

twoLayer :: Spec
twoLayer = do
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

  describe "the backward pass" $ do
    describe "dropping half the gradient during ReLU" $ do
      let y = constant 1 :: Tensor '[4]
          ll' = ll & (layer1 . weightsL) .~ (cat2d0
                                               (stack1d0
                                                 (constant (-40) :: Tensor '[6])
                                                 (constant (-40) :: Tensor '[6]))
                                               (stack1d0
                                                 (constant   40  :: Tensor '[6])
                                                 (constant   40  :: Tensor '[6])))
                   & (layer2 . weightsL) .~ constant 0
                   -- & (layer2 . weightsL) .~ (transpose2d
                   --                             (stack1d0
                   --                               (constant (-40) :: Tensor '[6])
                   --                               (constant   40  :: Tensor '[6])))
                   & (layer1 . biasL) .~ constant 0
                   & (layer2 . biasL) .~ constant 0

          (o, (gll, o')) = backprop2 (ff2network 0.1) ll y
      it "should be" $ tensordata (gll ^. layer1 . weightsL) >>= (`shouldBe` (replicate 12 0 ++ replicate 12 1))
      -- it "should be" $ tensordata (gll ^. layer1 . weightsL) >>= (`shouldBe` (replicate 12 0 ++ replicate 12 1))
      -- it "should be" $ tensordata (gll ^. layer1 . biasL   ) >>= (`shouldBe` (replicate 12 0 ++ replicate 12 1))

xavierPurityCheck :: forall i o . (KnownDim i, KnownDim o) => Linear i o -> Spec -> Spec
xavierPurityCheck ll tests = do
  it (header ++ "initializes with xavier correctly") $ do
    weights ll =##= 1/i
    bias    ll =##= 1/o

  tests

  it (header ++ "leaves weights unchanged") $ weights ll =##= 1/i
  it (header ++ "leaves bias unchanged")    $ bias    ll =##= 1/o
 where
  header :: String
  header = "[ref-check] " ++ unwords ["Linear", show (truncate i), show (truncate o)] ++ ": "

  i, o :: Double
  i = fromIntegral (dimVal (dim :: Dim i))
  o = fromIntegral (dimVal (dim :: Dim o))


elementsSatisfy :: Tensor d -> ([Double] -> Bool) -> IO ()
elementsSatisfy o pred = tensordata o >>= (`shouldSatisfy` pred)

(=##=) :: Tensor d -> Double -> IO ()
(=##=) o v = elementsSatisfy o (all (== v))

infixl 2 =##=

--   describe "the backward pass" $ do
--     let y = constant 1 :: Tensor '[3]
--         lr = 1.0
--         (_, (ll', o)) = backprop2 (linear lr) ll y

--     it "returns updated weights" $ do
--       tensordata (weights ll') >>= (`shouldSatisfy` all (== 1/2))

--     it "returns updated bias" $ do
--       tensordata (bias ll') >>= (`shouldSatisfy` all (== 1/2))

--     it "returns the updated output tensor" $ do
--       -- let x = (weights ll) `mv` (constant lr - bias ll)
--       tensordata o >>= (`shouldSatisfy` all (== 1/3))


