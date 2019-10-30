module GeneratorSpec (main, spec) where

import Test.Hspec
import Control.Exception (bracket)
import Control.Monad (forM_,forM)
import Data.Int
import Foreign
import ATen.Const
import ATen.Type
import ATen.Managed.Type.Generator
import qualified ATen.Unmanaged.Type.Generator as U
import ATen.Managed.Type.TensorOptions
import ATen.Managed.Type.Tensor
import ATen.Managed.Type.IntArray
import ATen.Managed.Type.Context
import ATen.Managed.Native

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
