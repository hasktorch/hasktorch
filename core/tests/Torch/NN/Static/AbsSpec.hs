{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.NN.Static.AbsSpec where

import Test.Hspec
import Torch.Double

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  let Just (x :: DoubleTensor '[2, 4]) = fromList [-4..4-1]
  describe "abs_updateOutput" $ do
    it "runs the absolute function" $ do
      y <- tensordata =<< abs_updateOutput x
      y `shouldSatisfy` all (>= 0)

  describe "abs_updateGradInput" $ do
    it "returns the input gradient" $ do
      let go :: DoubleTensor '[2, 4] = constant 1
      rs <- tensordata (signum x)
      ys <- tensordata =<< abs_updateGradInput x go
      zip ys rs `shouldSatisfy` all eqSigns
  where
    eqSigns :: (Double, Double) -> Bool
    eqSigns (y, r) = y == r || (r == 0 && y == 1)

