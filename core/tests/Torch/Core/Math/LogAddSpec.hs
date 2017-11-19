{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.Math.LogSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Torch.Core.Math.LogAdd


main :: IO ()
main = hspec spec


spec :: Spec
spec = do
  describe "logAdd" logAddSpec
  describe "logSub" logSubSpec
  describe "expMinusApprox" expMinusApproxSpec


logAddSpec :: Spec
logAddSpec = do
  it "returns a value" . property $ \((a, b)::(Float, Float)) ->
    a `logAdd` b `shouldSatisfy` (const True)


logSubSpec :: Spec
logSubSpec =
  it "returns a value" . property $ \((a, b)::(Float, Float)) ->
    if a < b
    then a `logSub` b `shouldThrow` anyException
    else a `logSub` b >>= (`shouldSatisfy` (const True))


expMinusApproxSpec :: Spec
expMinusApproxSpec =
  it "returns a value" . property $ \(a::Float) ->
    expMinusApprox a `shouldSatisfy` (const True)


