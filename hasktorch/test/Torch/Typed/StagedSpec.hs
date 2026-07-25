{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.StagedSpec (spec) where

import Data.Vector.Sized (Vector)
import GHC.Generics
import GHC.TypeLits
import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import qualified Torch.Tensor as D
import Torch.Typed.Graded (GradedMonad (..), TensorMonad, fromNamed, toNamed)
import Torch.Typed.Representable
import Torch.Typed.Staged
import Torch.Typed.Tensor

newtype Batch (n :: Nat) a = Batch (Vector n a) deriving (Show, Eq, Generic)

data RGB a = RGB
  { r :: a,
    g :: a,
    b :: a
  }
  deriving (Show, Eq, Generic)

type M = TensorMonad '(D.CPU, 0) 'D.Float

type Img = NamedTensor '(D.CPU, 0) 'D.Float '[Batch 2, RGB]

-- All element functions below are written ONCE, polymorphically.  The tests
-- run each at a = Float (per-element reference) and at a = Tensor (vectorized)
-- and compare the results.

fPoly :: (Floating a, Cond a) => a -> a
fPoly x = x * x + 1

fRelu :: (Floating a, Cond a) => a -> a
fRelu x = whereE (gtE x 0) x 0

fTrans :: (Floating a, Cond a) => a -> a
fTrans x = sin x * 10 + cos x

fMax :: (Floating a, Cond a) => a -> a -> a
fMax x y = whereE (gtE x y) x y

kAffine :: (Floating a, Cond a) => a -> HList (ToFinites '[RGB]) -> a
kAffine x (j :. HNil) = x * 10 + fromIntegral (fromEnum j)

kClip :: (Floating a, Cond a) => a -> HList (ToFinites '[RGB]) -> a
kClip x (j :. HNil) = whereE (gtE x 0) x (negate x) + fromIntegral (fromEnum j)

elemsOf :: NamedTensor '(D.CPU, 0) 'D.Float s -> [Float]
elemsOf = D.asValue . D.reshape [-1] . toDynamic

-- a test tensor holding [-3, -2, -1, 0, 1, 2]
timg :: Img
timg = tabulateList @'[Batch 2, RGB] @'D.Float @'(D.CPU, 0) (\[i, j] -> fromIntegral (i * 3 + j) - 3)

-- the per-element reference for emap: same f, instantiated at Float
emapRef :: (forall a. (Floating a, Cond a) => a -> a) -> Img -> Img
emapRef f t = tabulateList @'[Batch 2, RGB] @'D.Float @'(D.CPU, 0) (f . indexList t)

baseT :: M '[Batch 2]
baseT = fromNamed (tabulateList @'[Batch 2] @'D.Float @'(D.CPU, 0) (\[i] -> fromIntegral i - 0.5))

-- reference: run k per element through the graded bind (the oracle)
oracleG ::
  (forall a. (Floating a, Cond a) => a -> HList (ToFinites '[RGB]) -> a) ->
  M '[Batch 2, RGB]
oracleG k = gbind baseT (\x -> fromNamed (tabulate (k x)))

-- staged: run the same k once per inner index on the whole outer tensor
stagedG ::
  (forall a. (Floating a, Cond a) => a -> HList (ToFinites '[RGB]) -> a) ->
  M '[Batch 2, RGB]
stagedG k = gbindV @'[RGB] baseT k

spec :: Spec
spec = do
  describe "Staged element-wise ops" $ do
    it "emap agrees exactly with the element-wise reference (polynomial)" $ do
      elemsOf (emap fPoly timg) `shouldBe` elemsOf (emapRef fPoly timg)
    it "emap agrees exactly with the element-wise reference (whereE/relu)" $ do
      elemsOf (emap fRelu timg) `shouldBe` elemsOf (emapRef fRelu timg)
      elemsOf (emap fRelu timg) `shouldBe` [0, 0, 0, 0, 1, 2]
    it "emap agrees with the reference within float tolerance (transcendental)" $ do
      let d = zipWith (\a c -> abs (a - c)) (elemsOf (emap fTrans timg)) (elemsOf (emapRef fTrans timg))
      maximum d `shouldSatisfy` (< 1e-4)
    it "ezipWith computes an elementwise max via whereE" $ do
      let u = emap fRelu timg
      elemsOf (ezipWith fMax timg u) `shouldBe` zipWith max (elemsOf timg) (elemsOf u)
    it "emap preserves shape and dtype" $ do
      shape (emap fPoly timg) `shouldBe` [2, 3]
      dtype (emap fPoly timg) `shouldBe` D.Float
  describe "gbindV vs gbind (oracle)" $ do
    it "satisfies gbindV m k == gbind m (fromNamed . tabulate . k) (affine)" $ do
      elemsOf (toNamed (stagedG kAffine)) `shouldBe` elemsOf (toNamed (oracleG kAffine))
    it "satisfies the equation with value-dependent control flow (whereE)" $ do
      elemsOf (toNamed (stagedG kClip)) `shouldBe` elemsOf (toNamed (oracleG kClip))
    it "produces the concatenated shape" $ do
      D.shape (toDynamic (toNamed (stagedG kAffine))) `shouldBe` [2, 3]
