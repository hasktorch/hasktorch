module Main where

import Test.Hspec (hspec)
import qualified BasicSpec
import qualified CudaSpec
import qualified GeneratorSpec
import qualified MemorySpec

main :: IO ()
main = hspec $ do
  BasicSpec.spec
  CudaSpec.spec
  GeneratorSpec.spec
  MemorySpec.spec
