{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.NN.Static.AbsSpec where

import Test.Hspec
import Torch

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "abs_updateOutput" $ do
    it "runs the absolute function" $ do
      x :: DoubleTensor '[2, 4] <- fromList [-4, 4]
      y <- abs_updateOutput x
      y' :: [Double] <- tensordata y
      y' `shouldSatisfy` (all (>0))

