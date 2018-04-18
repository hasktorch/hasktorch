{-# LANGUAGE DataKinds #-}
module Torch.Core.Tensor.Dynamic.DoubleMathSpec (spec) where

import Torch.Dimensions
import Torch.Core.Tensor.Types
import Torch.Core.Tensor.Dynamic.Double as DynamicClass
import Torch.Core.Tensor.Dynamic.DoubleMath

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario

testScenario :: Property
testScenario = monadicIO . run $ do
  -- check exception case
  let
    m, v :: TensorDouble
    (m, v) = (DynamicClass.init dim3x2 3, DynamicClass.init dim2 2)

  DynamicClass.printTensor m
  DynamicClass.printTensor v
  DynamicClass.printTensor (td_mv m v)

  let
    m, v :: TensorDouble
    (m, v) = (DynamicClass.init dim1x3x2 3, DynamicClass.init dim2 2)

  DynamicClass.printTensor $ td_addr
    0.0 (DynamicClass.init dim3x2 0 :: TensorDouble)
    1.0 (DynamicClass.init dim3   2 :: TensorDouble) (DynamicClass.init dim2 3 :: TensorDouble)

  DynamicClass.printTensor $ td_outer
    (DynamicClass.init dim3 2 :: TensorDouble)
    (DynamicClass.init dim2 3 :: TensorDouble)

 where
  dim3, dim2, dim3x2 :: SomeDims
  dim3     = SomeDims (dim :: Dim '[3])
  dim2     = SomeDims (dim :: Dim '[2])
  dim3x2   = SomeDims (dim :: Dim '[3, 2])
  dim1x3x2 = SomeDims (dim :: Dim '[1, 3, 2])
