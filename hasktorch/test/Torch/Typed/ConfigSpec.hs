{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.ConfigSpec (spec) where

import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as I
import qualified Torch.Tensor as UT
import Torch.Typed
import Prelude hiding (exp, sum)

type Dev = '(D.CPU, 0)

-- Two variants of one architecture: each is one line, and every signature
-- below takes the whole configuration as a single type parameter.
type Tiny = 'TransformerConfig 8 2 4 16 11 5

type Small = 'TransformerConfig 16 4 4 32 11 6

-- One polymorphic body serves every variant; the config is the only
-- type-level plumbing.
sdpaMatchesManual ::
  forall cfg.
  ( KnownConfig cfg,
    TensorOptions '[2, Heads cfg, 3, HeadDim cfg] 'D.Float Dev,
    RandDTypeIsValid Dev 'D.Float
  ) =>
  Expectation
sdpaMatchesManual = do
  q <- randn @'[2, Heads cfg, 3, HeadDim cfg] @'D.Float @Dev
  k <- randn @'[2, Heads cfg, 3, HeadDim cfg] @'D.Float @Dev
  v <- randn @'[2, Heads cfg, 3, HeadDim cfg] @'D.Float @Dev
  let out = scaledDotProductAttention q k v Nothing False
      scale = 1 / Prelude.sqrt (fromIntegral (headDim @cfg)) :: Double
      ref =
        F.matmul
          ( F.softmax
              (F.Dim 3)
              (F.mulScalar scale (F.matmul (toDynamic q) (I.transpose (toDynamic k) 2 3)))
          )
          (toDynamic v)
      flat t = UT.asValue (UT.reshape [-1] t) :: [Float]
      diff = maximum (zipWith (\a b -> Prelude.abs (a - b)) (flat (toDynamic out)) (flat ref))
  UT.shape (toDynamic out) `shouldBe` [2, heads @cfg, 3, headDim @cfg]
  diff `shouldSatisfy` (< 1e-5)

spec :: Spec
spec = describe "TransformerConfig (a type-level hyperparameter record)" $ do
  it "reifies fields per variant" $ do
    modelDim @Tiny `shouldBe` 8
    heads @Tiny `shouldBe` 2
    modelDim @Small `shouldBe` 16
    heads @Small `shouldBe` 4
    maxSeqLen @Small `shouldBe` 6
  it "scaledDotProductAttention matches the manual formula (Tiny)" $
    sdpaMatchesManual @Tiny
  it "scaledDotProductAttention matches the manual formula (Small)" $
    sdpaMatchesManual @Small
