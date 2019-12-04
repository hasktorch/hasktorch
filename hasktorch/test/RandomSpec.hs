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
    generator <- mkGenerator (Device CPU 0) 0
    let (t,next) = randn' [4] generator
        (_,next') = randn' [4] next
        (t2,next'') = randn' [4] next'
        (t3,_) = randn' [5] generator
    shape t2 `shouldBe` [4]
    ((asValue t) :: [Float]) `shouldBe` take 4 (asValue t3)
