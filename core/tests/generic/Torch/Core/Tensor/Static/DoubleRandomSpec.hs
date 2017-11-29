{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.Tensor.Static.DoubleRandomSpec (spec) where

import Torch.Core.Tensor.Types (TensorDim(D1))
import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleRandom (newRNG, tds_random)

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario


testScenario :: Property
testScenario = monadicIO $ do
  gen <- run newRNG
  t :: TDS '[5] <- run $ tds_random gen
  run $ tds_p t

