{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.Tensor.Dynamic.DoubleSpec (spec) where

import Torch.Prelude.Extras

import Torch.Core.Tensor.Types
import Torch.Core.Tensor.Dim
import Torch.Core.Tensor.Dynamic.Double
import qualified Torch.Core.Tensor.Dynamic.Double as DynamicClass

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario

testScenario :: Property
testScenario = monadicIO $ do
  let foo = DynamicClass.new (SomeDims (dim :: Dim '[5])) :: TensorDouble
      t = DynamicClass.init (SomeDims (dim :: Dim '[5, 2])) 3.0  :: TensorDouble
  run (printTensor (DynamicClass.transpose 1 0 (DynamicClass.transpose 1 0 t)))
