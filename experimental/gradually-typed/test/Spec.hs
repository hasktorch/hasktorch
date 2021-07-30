module Main where

import Test.Hspec (hspec)
import qualified Torch.GraduallyTyped.NN.TransformerSpec (spec)
import qualified Torch.GraduallyTyped.TensorSpec (spec)

main :: IO ()
main = hspec $ do
  Torch.GraduallyTyped.TensorSpec.spec
  Torch.GraduallyTyped.NN.TransformerSpec.spec