{-# LANGUAGE DataKinds #-}
module Torch.Core.Tensor.Dynamic.DoubleLapackSpec where

import Torch.Core.Tensor.Dynamic.Double (td_new, td_init, td_p)
import Torch.Core.Tensor.Dynamic.DoubleRandom (td_uniform)
import Torch.Core.Tensor.Dynamic.DoubleLapack
import Torch.Core.Tensor.Types (TensorDim(D1, D2))
import Torch.Core.Random (newRNG)

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario


testScenario :: Property
testScenario = monadicIO $ do
  rng <- run newRNG
  let rnd = td_new $ D2 (2, 2)
  t <- run $ td_uniform rnd rng (-1.0) 1.0
  let b = td_init (D1 2) 1.0
  let (resA, resB) = td_gesv t b
  run $ td_p resA
  run $ td_p resB

  let (resQ, resR) = td_qr t
  run $ td_p resQ
  run $ td_p resR
