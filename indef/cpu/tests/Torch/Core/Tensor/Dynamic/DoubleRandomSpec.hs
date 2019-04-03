{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
module Torch.Core.Tensor.Dynamic.DoubleRandomSpec (spec) where

import Torch.Core.Tensor.Types
import Torch.Core.Tensor.Dynamic.Double
import qualified Torch.Core.Tensor.Dynamic.Double as DynamicClass
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
  let t = DynamicClass.new (SomeDims (dim :: Dim '[3])) :: TensorDouble
  run $ DynamicClass.printTensor t
  gen <- run newRNG
  run $ td_random t gen
  run $ DynamicClass.printTensor t
