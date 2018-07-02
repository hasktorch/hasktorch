{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Static.NN.LinearSpec where

import Test.Hspec
import Torch.Double
import Numeric.Backprop
import Torch.Double.NN.Linear

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "a single linear layer" $ do
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


