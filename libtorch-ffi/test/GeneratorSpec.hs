module GeneratorSpec (main, spec) where

import Test.Hspec
import Control.Exception (bracket)
import Control.Monad (forM_,forM)
import Data.Int
import Foreign
import LibTorch.ATen.Const
import LibTorch.ATen.Type
import LibTorch.ATen.Managed.Type.Generator
import qualified LibTorch.ATen.Unmanaged.Type.Generator as U
import LibTorch.ATen.Managed.Type.TensorOptions
import LibTorch.ATen.Managed.Type.Tensor
import LibTorch.ATen.Managed.Type.IntArray
import LibTorch.ATen.Managed.Type.Context
import LibTorch.ATen.Managed.Native

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "GeneratorSpec" $ do
    it "create generator" $ do
      g <- newCPUGenerator 123
      generator_current_seed g `shouldReturn` 123
    it "get default generator" $ do
      g <- U.getDefaultCPUGenerator
      manual_seed_L 321
      U.generator_current_seed g `shouldReturn` 321
