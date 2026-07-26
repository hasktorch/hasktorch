{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoStarIsType #-}

-- | A decoder block, equation by equation.
--
-- This is the pedagogical counterpart of "Torch.Typed.NN.Transformer": the
-- same mathematics, written so that each equation of the paper is one
-- definition, with the paper's shape annotations as checked types.  The test
-- at the bottom trains it on a next-token task and checks that it learns.
module Torch.Typed.AttentionSpec (spec) where

import Control.Monad (foldM)
import Data.List (elemIndices)
import Data.Vector.Sized (Vector)
import GHC.Generics (Generic)
import Test.Hspec
import Prelude
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Tensor as UT
import Torch.Typed hiding (length)
import Torch.Typed.Representable (tabulateList)

-- vocabulary, context length, model width, feed-forward width
type V = 4

type S = 8

type E = 16

type H = 32

type Dev = '(D.CPU, 0)

type T shape = Tensor Dev 'D.Float shape

--------------------------------------------------------------------------------
-- The equations
--------------------------------------------------------------------------------

-- M[i,j] = 0 if j <= i, -inf otherwise: a tensor defined by its formula
causalMask :: T '[S, S]
causalMask = toUnnamed (tabulateList @'[Vector S, Vector S] @'D.Float @Dev mask)
  where
    mask [i, j] = if j <= i then 0 else -1 / 0
    mask _ = 0

-- Attention(Q, K, V) = softmax(Q Kᵀ / √d + M) V
attend :: T '[S, E] -> T '[S, E] -> T '[S, E] -> T '[S, E]
attend q k v = softmax @1 (divScalar sqrtD (q `matmul` transpose2D k) + causalMask) `matmul` v
  where
    sqrtD = Prelude.sqrt (fromIntegral (natValI @E)) :: Float

--------------------------------------------------------------------------------
-- A one-block decoder
--------------------------------------------------------------------------------

data GPT = GPT
  { embed :: Parameter Dev 'D.Float '[V, E],
    wq :: Linear E E 'D.Float Dev,
    wk :: Linear E E 'D.Float Dev,
    wv :: Linear E E 'D.Float Dev,
    wo :: Linear E E 'D.Float Dev,
    norm1 :: LayerNorm '[E] 'D.Float Dev,
    norm2 :: LayerNorm '[E] 'D.Float Dev,
    ff1 :: Linear E H 'D.Float Dev,
    ff2 :: Linear H E 'D.Float Dev,
    unembed :: Linear E V 'D.Float Dev
  }
  deriving (Generic, Parameterized)

data GPTSpec = GPTSpec

instance Randomizable GPTSpec GPT where
  sample GPTSpec =
    GPT
      <$> (makeIndependent . mulScalar (0.1 :: Float) =<< randn)
      <*> sample LinearSpec
      <*> sample LinearSpec
      <*> sample LinearSpec
      <*> sample LinearSpec
      <*> sample (LayerNormSpec 1e-5)
      <*> sample (LayerNormSpec 1e-5)
      <*> sample LinearSpec
      <*> sample LinearSpec
      <*> sample LinearSpec

-- x   = Embed[tokens]                (token embedding)
-- x'  = x  + Wo · Attend(Wq xn, Wk xn, Wv xn)   where xn = LN1 x
-- x'' = x' + FF(LN2 x')                          (residual + feed-forward)
-- out = Unembed x''                              (logits over the vocabulary)
gptForward :: GPT -> Tensor Dev 'D.Int64 '[S] -> T '[S, V]
gptForward GPT {..} tokens = forward unembed x''
  where
    x = embedding @'Nothing False False (toDependent embed) tokens
    xn = forward norm1 x
    x' = x + forward wo (attend (forward wq xn) (forward wk xn) (forward wv xn))
    x'' = x' + forward ff2 (relu (forward ff1 (forward norm2 x')))

--------------------------------------------------------------------------------
-- It learns: next-token prediction on a repeating sequence
--------------------------------------------------------------------------------

tokens, targets :: Tensor Dev 'D.Int64 '[S]
tokens = UnsafeMkTensor (UT.asTensor ([0, 1, 2, 3, 0, 1, 2, 3] :: [Int]))
targets = UnsafeMkTensor (UT.asTensor ([1, 2, 3, 0, 1, 2, 3, 0] :: [Int]))

lossOf :: GPT -> T '[]
lossOf m = nllLoss @ReduceMean ones (-100) (logSoftmax @1 (gptForward m tokens)) targets

predictions :: GPT -> [Int]
predictions m = map argmaxRow (UT.asValue (toDynamic (gptForward m tokens)) :: [[Float]])
  where
    argmaxRow row = head (elemIndices (maximum row) row)

spec :: Spec
spec = describe "a decoder block, equation by equation" $ do
  it "the causal mask never lets position i attend to j > i" $ do
    let m = UT.asValue (toDynamic causalMask) :: [[Float]]
    and [m !! i !! j == 0 | i <- [0 .. 7], j <- [0 .. i]] `shouldBe` True
    and [m !! i !! j < -1e30 | i <- [0 .. 7], j <- [i + 1 .. 7]] `shouldBe` True
  it "learns next-token prediction on a repeating sequence" $ do
    model <- sample GPTSpec
    let optim = mkAdam 0 0.9 0.999 (flattenParameters model)
        loss0 = toFloat (lossOf model)
    (trained, _) <-
      foldM
        (\(m, o) _ -> runStep m o (lossOf m) 1e-2)
        (model, optim)
        [1 .. 1000 :: Int]
    let loss1 = toFloat (lossOf trained)
    loss0 `shouldSatisfy` (> 0.5) -- untrained: near ln 4 ≈ 1.39
    loss1 `shouldSatisfy` (< 0.1) -- trained: near zero
    predictions trained `shouldBe` [1, 2, 3, 0, 1, 2, 3, 0]
