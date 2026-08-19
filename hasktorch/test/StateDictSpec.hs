{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module StateDictSpec (spec) where

import Control.Exception (SomeException, try)
import qualified Data.Map.Strict as Map
import GHC.Generics
import Test.Hspec
import Torch
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Typed as T
import Torch.Typed.StateDict ()

-- a nested model: field names become state-dict paths
data Block = Block
  { attn :: Linear,
    mlp :: Linear
  }
  deriving (Generic, FromStateDict, ToStateDict)

data GPT = GPT
  { wte :: Parameter,
    h :: [Block]
  }
  deriving (Generic, FromStateDict, ToStateDict)

mkLinear :: Int -> Int -> IO Linear
mkLinear i o = sample (LinearSpec i o)

spec :: Spec
spec = describe "StateDict" $ do
  it "derives dotted paths from record structure, lists as indices" $ do
    b0 <- Block <$> mkLinear 2 3 <*> mkLinear 3 2
    b1 <- Block <$> mkLinear 2 3 <*> mkLinear 3 2
    w <- makeIndependent =<< randIO' [4, 2]
    let sd = toStateDict (GPT w [b0, b1]) ""
    mapM_
      (\k -> Map.keys sd `shouldContain` [k])
      ["h.0.attn.weight", "h.0.attn.bias", "h.1.mlp.weight", "wte"]
  it "round-trips a model through a state dict" $ do
    b0 <- Block <$> mkLinear 2 3 <*> mkLinear 3 2
    w <- makeIndependent =<< randIO' [4, 2]
    let model = GPT w [b0]
        sd = toStateDict model ""
    model' <- fromStateDict sd ""
    let probe = asTensor [[1, 2 :: Float]]
    (asValue (linear (attn (head (h model'))) probe) :: [[Float]])
      `shouldBe` (asValue (linear (attn (head (h model))) probe) :: [[Float]])
    length (h model') `shouldBe` 1
  it "round-trips through the pickle file format" $ do
    b0 <- Block <$> mkLinear 2 3 <*> mkLinear 3 2
    w <- makeIndependent =<< randIO' [4, 2]
    let sd = toStateDict (GPT w [b0]) ""
    saveStateDict sd "/tmp/statedict-spec.pth"
    sd' <- loadStateDict "/tmp/statedict-spec.pth"
    stateDictKeys sd' `shouldBe` stateDictKeys sd
  it "fails with the offending path on a missing key" $ do
    r <- try (fromStateDict Map.empty "" :: IO GPT)
    case r of
      Left (e :: SomeException) -> show e `shouldContain` "wte"
      Right _ -> expectationFailure "expected a missing-key failure"
  it "typed load checks the stored shape against the static one" $ do
    let sd = Map.singleton "w" (zeros' [2, 3])
    (t :: T.Tensor '(D.CPU, 0) 'D.Float '[2, 3]) <- fromStateDict sd "w"
    T.shape t `shouldBe` [2, 3]
    r <- try (fromStateDict sd "w" :: IO (T.Tensor '(D.CPU, 0) 'D.Float '[3, 2]))
    case r of
      Left (e :: SomeException) -> do
        show e `shouldContain` "[2,3]"
        show e `shouldContain` "[3,2]"
      Right _ -> expectationFailure "expected a shape-mismatch failure"
