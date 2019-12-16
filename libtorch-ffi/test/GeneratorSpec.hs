module GeneratorSpec (main, spec) where

import Test.Hspec
import Control.Exception (bracket)
import Control.Monad (forM_,forM)
import Data.Int
import Foreign
import Torch.Internal.Const
import Torch.Internal.Type
import Torch.Internal.Managed.Type.Generator
import qualified Torch.Internal.Unmanaged.Type.Generator as U
import Torch.Internal.Managed.Type.TensorOptions
import Torch.Internal.Managed.Type.Tensor
import Torch.Internal.Managed.Type.IntArray
import Torch.Internal.Managed.Type.Context
import Torch.Internal.Managed.Native

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
