{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.Tensor.Static.DoubleRandomSpec (spec) where

import Torch.Core.Tensor.Types (TensorDim(D1))
import Torch.Core.Tensor.Static.Double
-- import Torch.Core.Tensor.Static.DoubleRandom (newRNG, tds_random)
import Torch.Core.Tensor.Static.DoubleRandom

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

testDistributions :: IO ()
testDistributions = do
  gen <- newRNG
  t_unif:: TDS '[10] <- tds_uniform gen (-5.0) 5.0
  t_norm :: TDS '[10] <- tds_normal gen 100.0 10.0
  t_exp :: TDS '[10] <- tds_exponential gen 2.0
  tds_p t_unif
  tds_p t_norm
  tds_p t_exp

