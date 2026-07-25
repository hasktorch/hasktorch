{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE QualifiedDo #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.GradedSpec (spec) where

import Data.Default.Class
import Data.Proxy
import Data.Vector.Sized (Vector)
import GHC.Generics
import GHC.TypeLits
import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import qualified Torch.Tensor as D
import Torch.Typed.Graded (GradedMonad (..), TensorMonad, fromNamed, toNamed)
import qualified Torch.Typed.Graded as G
import Torch.Typed.Representable
import Torch.Typed.Tensor

newtype Batch (n :: Nat) a = Batch (Vector n a) deriving (Show, Eq, Generic)

data RGB a = RGB
  { r :: a,
    g :: a,
    b :: a
  }
  deriving (Show, Eq, Generic, Default)

type M = TensorMonad '(D.CPU, 0) 'D.Float

-- The shape monoid: '[] is the unit and ++ is associative, definitionally.
testUnitL :: Proxy ('[] ++ '[Batch 2, RGB]) -> Proxy '[Batch 2, RGB]
testUnitL = id

testUnitR :: Proxy ('[Batch 2, RGB] ++ '[]) -> Proxy '[Batch 2, RGB]
testUnitR = id

testAssoc ::
  Proxy (('[Batch 2] ++ '[RGB]) ++ '[Batch 2]) ->
  Proxy ('[Batch 2] ++ ('[RGB] ++ '[Batch 2]))
testAssoc = id

-- a rank-1 tensor holding [0, 1]
base :: M '[Batch 2]
base = fromNamed (tabulate (\(i :. HNil) -> fromIntegral (fromEnum i)))

-- for each element x, an RGB triple [x*10, x*10+1, x*10+2]
toRGB :: Float -> M '[RGB]
toRGB x = fromNamed (tabulate (\(j :. HNil) -> x * 10 + fromIntegral (fromEnum j)))

toBatch :: Float -> M '[Batch 2]
toBatch y = fromNamed (tabulate (\(k :. HNil) -> y * 100 + fromIntegral (fromEnum k)))

elems :: M s -> [Float]
elems = D.asValue . D.reshape [-1] . toDynamic . toNamed

dimsM :: M s -> [Int]
dimsM = D.shape . toDynamic . toNamed

-- do-notation over the graded monad, via QualifiedDo
viaDo :: M '[Batch 2, RGB]
viaDo = G.do
  x <- base
  toRGB x

spec :: Spec
spec = do
  describe "GradedMonad TensorMonad" $ do
    it "gbind concatenates shapes" $ do
      let t = gbind base toRGB
      dimsM t `shouldBe` [2, 3]
      elems t `shouldBe` [0, 1, 2, 10, 11, 12]
    it "greturn produces a scalar" $ do
      let t = greturn 7 :: M '[]
      dimsM t `shouldBe` []
      elems t `shouldBe` [7]
    it "satisfies the left identity law" $ do
      elems (gbind (greturn 5 :: M '[]) toRGB) `shouldBe` elems (toRGB 5)
      dimsM (gbind (greturn 5 :: M '[]) toRGB) `shouldBe` dimsM (toRGB 5)
    it "satisfies the right identity law" $ do
      elems (gbind base greturn) `shouldBe` elems base
      dimsM (gbind base greturn) `shouldBe` dimsM base
    it "satisfies the associativity law" $ do
      let lhs = gbind (gbind base toRGB) toBatch :: M '[Batch 2, RGB, Batch 2]
          rhs = gbind base (\x -> gbind (toRGB x) toBatch) :: M '[Batch 2, RGB, Batch 2]
      dimsM lhs `shouldBe` [2, 3, 2]
      elems lhs `shouldBe` elems rhs
    it "works with QualifiedDo notation" $ do
      dimsM viaDo `shouldBe` [2, 3]
      elems viaDo `shouldBe` [0, 1, 2, 10, 11, 12]
