module CodeGen.Render.CSpec where

import Test.Hspec
import Test.Hspec.QuickCheck
import Test.QuickCheck
import CodeGen.Render.C
import CodeGen.Prelude
import CodeGen.Types
import CodeGen.Instances

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "renderTenType" renderTenTypeSpec

renderTenTypeSpec :: Spec
renderTenTypeSpec = do
  prop "renders accreal and real the same regardless of Libtype" $ \(lt1, lt2) -> do
    renderTenType (Pair (Real, lt1)) `shouldBe` renderTenType (Pair (Real, lt2))
    renderTenType (Pair (AccReal, lt1)) `shouldBe` renderTenType (Pair (AccReal, lt2))

  it "renders all non-THC prefixes as the library appended to the raw tensor type"
    . propertyST (\p -> not . or . fmap ($ p) $ [isTHC, isReal, isTHCUNN]) $
    \tt@(Pair (raw, l)) ->
      renderTenType tt `shouldBe` (tshow l <> tshow raw)

  it "renders THC as THCuda for concrete tensor types"
    . propertyST (\p -> isTHC p && not (isReal p)) $
    \tt@(Pair (raw, _)) ->
      -- (if isConcreteCudaPrefixed p then "THCuda" else tshow lib)
      if isConcreteCudaPrefixed tt
      then renderTenType tt `shouldBe` ("THCuda" <> tshow raw)
      else renderTenType tt `shouldBe` ("THC" <> tshow raw)

 where
  propertyST
    :: Arbitrary a
    => Show a
    => Testable prop
    => (a -> Bool)
    -> (a -> prop)
    -> Property
  propertyST cond = forAll (suchThat arbitrary cond)

  isTHC (Pair(_, l)) = l == THC
  isTHCUNN (Pair(_, l)) = l == THCUNN
  isReal (Pair(r, _)) = r == AccReal || r == Real

