module Main where

import Test.Hspec (hspec)
import qualified Torch.GraduallyTyped.IndexingSpec
import qualified Torch.GraduallyTyped.TensorSpec

main :: IO ()
main = hspec $ do
  Torch.GraduallyTyped.TensorSpec.spec
  Torch.GraduallyTyped.IndexingSpec.spec
