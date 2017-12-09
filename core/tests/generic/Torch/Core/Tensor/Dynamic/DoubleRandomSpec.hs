{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.Tensor.Dynamic.DoubleRandomSpec (spec) where

import Torch.Core.Tensor.Types (TensorDim(D1))
import Torch.Core.Tensor.Dynamic.Double
import Torch.Core.Tensor.Dynamic.DoubleRandom (newRNG, td_random)

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario

testScenario :: Property
testScenario = monadicIO $ do
  let t = td_new (D1 3)
  run $ td_p t
  gen <- run newRNG
  run $ td_random t gen
  run $ td_p t
