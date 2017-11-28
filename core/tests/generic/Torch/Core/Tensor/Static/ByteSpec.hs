{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.Tensor.Static.ByteSpec (spec) where

import Torch.Core.Tensor.Types (TensorDim(D1))
import Torch.Core.Tensor.Static.Byte

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario


testScenario :: Property
testScenario = monadicIO $
  run $ tbs_p (tbs_init 3 :: TBS '[4,2])
