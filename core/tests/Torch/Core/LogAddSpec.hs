{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.LogAddSpec (spec) where

import Torch.Core.LogAdd
import Torch.Prelude.Extras

main :: IO ()
main = hspec spec


spec :: Spec
spec = do
  describe "logAdd" logAddSpec
  describe "logSub" logSubSpec
  describe "expMinusApprox" expMinusApproxSpec


logAddSpec :: Spec
logAddSpec = do
  it "returns a value" . property $ \((a, b)::(Double, Double)) ->
    a `logAdd` b >>= (`shouldSatisfy` doesn'tCrash)


logSubSpec :: Spec
logSubSpec =
  it "returns a value" . property $ \((a, b)::(Double, Double)) ->
    if a < b
    then a `logSub` b `shouldThrow` anyException
    else a `logSub` b >>= (`shouldSatisfy` doesn'tCrash)


expMinusApproxSpec :: Spec
expMinusApproxSpec =
  it "returns a value" . property $ \(a::Double) ->
    expMinusApprox a >>= (`shouldSatisfy` doesn'tCrash)


