{-# LANGUAGE ScopedTypeVariables #-}
module RandomSpec (spec) where

import Test.Hspec
import Control.Exception.Safe

import Torch.Tensor
import Torch.Device
import Torch.Random
import Torch.TensorOptions

spec :: Spec
spec = do
  it "pure functional random with seed" $ do
    let (t,next) = randn' [4] (cpuGenerator 0)
        (_,next') = randn' [4] next
        (t2,next'') = randn' [4] next'
        (t3,_) = randn' [5] (cpuGenerator 0)
    shape t2 `shouldBe` [4]
    ((asValue t) :: [Float]) `shouldBe` take 4 (asValue t3)
