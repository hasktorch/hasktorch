module Main where

import Test.Hspec (hspec)
import qualified Torch.GraduallyTyped.NN.TransformerSpec

main :: IO ()
main = hspec $ do
  Torch.GraduallyTyped.NN.TransformerSpec.spec
