module CodeGen.Render.FunctionSpec where

import Test.Hspec
import CodeGen.Render.Function
import CodeGen.Types hiding (describe)
import CodeGen.Prelude


main :: IO ()
main = hspec spec


spec :: Spec
spec = do
  describe "rendering the debugging function `sizeDesc`" $
    describe "returning `Ptr C'THDescBuff` instead of CTHDescBuff for generic/THCTensor.h#sizeDesc" $ do

    let sizeDescSig = "sizeDesc :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> IO (Ptr C'THCDescBuff)"
    it "works as expected in haskellSig" $ do
      haskellSig THC "sizeDesc" IsFun GenFloat
        [ Arg (Ptr (TenType (Pair (State, THC)))) "state"
        , Arg (Ptr (TenType (Pair (Tensor, THC)))) "tensor"
        ] (TenType (Pair (DescBuff, THC)))
        `shouldBe` sizeDescSig

    it "works as expected in renderSig" $ do
      renderSig IsFun THC GenericFiles "test" GenFloat "Tensor" "Tensor"
        ("sizeDesc", TenType (Pair (DescBuff, THC)),
          [ Arg (Ptr (TenType (Pair (State,  THC)))) "state"
          , Arg (Ptr (TenType (Pair (Tensor, THC)))) "tensor"])
        `shouldBe` ("-- | c_sizeDesc :  state tensor -> THCDescBuff\n"
          <> "foreign import ccall \"test THCudaFloatTensor_sizeDesc\""
          <> "\n  c_" <> sizeDescSig)

