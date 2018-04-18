module CodeGen.RenderSpec where

import Test.Hspec
import Test.Hspec.QuickCheck
import Test.QuickCheck hiding (Function)
import CodeGen.Render
import CodeGen.Prelude
import CodeGen.Types
import CodeGen.Instances

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  xdescribe "renderFunctions" renderFunctionsSpec

renderFunctionsSpec :: Spec
renderFunctionsSpec = do
  it "renders functions with their prefix if there is a name collision" $ do
    pendingWith "not sure this should be tested here. It looks like it will be placed in a helper function"

