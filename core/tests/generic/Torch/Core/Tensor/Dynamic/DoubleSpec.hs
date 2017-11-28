{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.Tensor.Dynamic.DoubleSpec (spec) where

import Torch.Prelude.Extras

import Torch.Core.Tensor.Types
import Torch.Core.Tensor.Dynamic.Double

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario

testScenario :: Property
testScenario = monadicIO $ do
  let foo = td_new (D1 5)
  let t = td_init (D2 (5, 2)) 3.0
  run (td_p (td_transpose 1 0 (td_transpose 1 0 t)))
