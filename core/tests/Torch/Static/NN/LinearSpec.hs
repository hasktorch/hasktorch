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
  it "initializes with xavier correctly" $ do
    w' <- tensordata (weights ll)
    b' <- tensordata (bias ll)
    w' `shouldSatisfy` all (== 1/3)
    b' `shouldSatisfy` all (== 1/2)

  describe "the forward pass" $ do
    let y = constant 1 :: Tensor '[3]
        (o, _) = backprop2 (linear undefined) ll y

    it "performs matrix multipication as you would expect" $
      tensordata o >>= (`shouldSatisfy` all (== 3/2))

    it "leaves weights unchanged" $
      tensordata (weights ll) >>= (`shouldSatisfy` all (== 1/3))

    it "leaves bias unchanged" $
      tensordata (bias ll) >>= (`shouldSatisfy` all (== 1/2))

  describe "the backward pass" $ do
    let y = constant 1 :: Tensor '[3]
        lr = 1.0
        (_, (ll', o)) = backprop2 (linear lr) ll y

    it "returns updated weights" $ do
      tensordata (weights ll') >>= (`shouldSatisfy` all (== 1/2))

    it "returns updated bias" $ do
      tensordata (bias ll') >>= (`shouldSatisfy` all (== 1/2))

    it "returns the updated output tensor" $ do
      -- let x = (weights ll) `mv` (constant lr - bias ll)
      tensordata o >>= (`shouldSatisfy` all (== 1/3))

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
  ll :: FF2Network 5 10 2 <- runIO mkFF2Network
  it "initializes with xavier correctly" $ do
    tensordata (weights $ _layer1 ll) >>= (`shouldSatisfy` all (== 1 /  5))
    tensordata (bias    $ _layer1 ll) >>= (`shouldSatisfy` all (== 1 / 10))
    tensordata (weights $ _layer2 ll) >>= (`shouldSatisfy` all (== 1 / 10))
    tensordata (bias    $ _layer2 ll) >>= (`shouldSatisfy` all (== 1 /  2))

--   describe "the forward pass" $ do
--     let y = constant 1 :: Tensor '[3]
--         (o, _) = backprop2 (linear undefined) ll y

--     it "performs matrix multipication as you would expect" $
--       tensordata o >>= (`shouldSatisfy` all (== 3/2))

--     it "leaves weights unchanged" $
--       tensordata (weights ll) >>= (`shouldSatisfy` all (== 1/3))

--     it "leaves bias unchanged" $
--       tensordata (bias ll) >>= (`shouldSatisfy` all (== 1/2))

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


