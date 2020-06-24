import Test.Hspec

import qualified BasicSpec
import qualified CudaSpec
import qualified GeneratorSpec
import qualified MemorySpec

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "Basic"      BasicSpec.spec
  describe "Cuda"       CudaSpec.spec
  describe "Generator"  GeneratorSpec.spec
  describe "Memory"     MemorySpec.spec
