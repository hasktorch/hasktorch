{-# LANGUAGE DataKinds #-}
module Torch.Core.Tensor.Dynamic.GenericLapackSpec where

import Torch.Core.Tensor.Dynamic.Double (td_new, td_init, printTensor)
import Torch.Core.Tensor.Dynamic.DoubleRandom (td_uniform)
import Torch.Core.Tensor.Dynamic.GenericLapack
import Torch.Core.Random (newRNG)

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario


testScenario :: Property
testScenario = monadicIO . run $ do
  rng <- newRNG
  let rnd = td_new (dim :: Dim '[2, 2])
  t <- td_uniform rnd rng (-1.0) 1.0
  let b = td_init (dim :: Dim '[2]) 1
  let (resA, resB) = td_gesv t b
  printTensor resA
  printTensor resB

  let (resQ, resR) = td_qr t
  printTensor resQ
  printTensor resR
