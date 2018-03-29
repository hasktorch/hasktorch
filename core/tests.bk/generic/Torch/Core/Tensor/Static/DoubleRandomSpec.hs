{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Core.Tensor.Static.DoubleRandomSpec (spec) where

import Torch.Core.Tensor.Dim
import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleRandom

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario
  describe "distributions" $ do
    it "samples from a variety of distributions without crashing" $
      testTensorDistributions

testScenario :: Property
testScenario = monadicIO $ do
  gen <- run newRNG
  t :: TDS '[5] <- run $ tds_random gen
  run $ tds_p t

testTensorDistributions :: IO ()
testTensorDistributions = do
  gen <- newRNG
  t_bern   :: TDS '[10] <- tds_bernoulli gen 0.8
  t_unif   :: TDS '[10] <- tds_uniform gen (-5.0) 5.0
  t_norm   :: TDS '[10] <- tds_normal gen 100.0 10.0
  t_exp    :: TDS '[10] <- tds_exponential gen 2.0
  t_cauchy :: TDS '[10] <- tds_cauchy gen 1.0 1.0
  tds_p t_bern
  tds_p t_unif
  tds_p t_norm
  tds_p t_exp
  tds_p t_cauchy

