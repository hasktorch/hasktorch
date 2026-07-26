{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.RepresentableSpec (spec) where

import Data.Default.Class
import Data.Finite (Finite)
import Data.Proxy
import Data.Vector.Sized (Vector)
import GHC.Generics
import GHC.TypeLits
import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import qualified Torch.Tensor as D
import qualified Data.Vector.Sized as V
import Torch.Typed.Factories ()
import Torch.Typed.Representable
import Torch.Typed.Tensor

newtype Batch (n :: Nat) a = Batch (Vector n a) deriving (Show, Eq, Generic)

data RGB a = RGB
  { r :: a,
    g :: a,
    b :: a
  }
  deriving (Show, Eq, Generic, Default)

-- 'ToNat' is derived from 'Generic', so a user-defined structure needs no
-- registration to be usable as a dimension.
testToNatRGB :: Proxy (ToNat RGB) -> Proxy 3
testToNatRGB = id

-- The index type of a named shape is one 'Finite' per dimension.
testLog ::
  Proxy (Log (NamedTensor '(D.CPU, 0) 'D.Float '[Batch 2, RGB])) ->
  Proxy (HList '[Finite 2, Finite 3])
testLog = id

testElem ::
  Proxy (Elem (NamedTensor '(D.CPU, 0) 'D.Float '[Batch 2, RGB])) ->
  Proxy Float
testElem = id

type Image = NamedTensor '(D.CPU, 0) 'D.Float '[Batch 2, RGB]

spec :: Spec
spec = do
  describe "Representable NamedTensor" $ do
    it "reifies the dimensions of a named shape" $ do
      dimsOf (Proxy @(NamedTensor '(D.CPU, 0) 'D.Float '[Batch 2, RGB, Vector 4]))
        `shouldBe` [2, 3, 4]
    it "tabulate builds a tensor of the right shape and dtype" $ do
      let t = tabulate (\_ -> 1.0) :: Image
      shape t `shouldBe` [2, 3]
      dtype t `shouldBe` D.Float
    it "index reads back what tabulate wrote" $ do
      -- the element value encodes its own index, so a wrong lookup is visible
      let t =
            tabulate
              (\(i :. j :. HNil) -> fromIntegral (fromEnum i) * 10 + fromIntegral (fromEnum j))
              :: Image
      index t (0 :. 0 :. HNil) `shouldBe` 0
      index t (0 :. 2 :. HNil) `shouldBe` 2
      index t (1 :. 0 :. HNil) `shouldBe` 10
      index t (1 :. 2 :. HNil) `shouldBe` 12
    it "satisfies index (tabulate f) i == f i" $ do
      let f (i :. j :. HNil) = fromIntegral (fromEnum i) * 3 + fromIntegral (fromEnum j) :: Float
          t = tabulate f :: Image
          ix = [i :. j :. HNil | i <- [0 .. 1], j <- [0 .. 2]]
      map (index t) ix `shouldBe` map f ix
    it "satisfies tabulate (index t) == t" $ do
      let t = tabulate (\(i :. _ :. HNil) -> fromIntegral (fromEnum i)) :: Image
          t' = tabulate (index t) :: Image
      D.asValue (toDynamic t') `shouldBe` (D.asValue (toDynamic t) :: [[Float]])
    it "round-trips a named index through plain Ints" $ do
      toInts @'[Batch 2, RGB] (fromInts @'[Batch 2, RGB] [1, 2]) `shouldBe` [1, 2]
    it "indexes a field-structured dimension positionally" $ do
      -- RGB's fields are laid out in declaration order: r=0, g=1, b=2
      let t = tabulate (\(_ :. j :. HNil) -> fromIntegral (fromEnum j)) :: Image
      index t (0 :. 0 :. HNil) `shouldBe` 0
      index t (0 :. 1 :. HNil) `shouldBe` 1
      index t (0 :. 2 :. HNil) `shouldBe` 2
  describe "dimUp and dimDown" $ do
    it "dimUp splits the outermost dimension into a record, by name" $ do
      let t = tabulate (\(i :. j :. HNil) -> fromIntegral (fromEnum i) * 10 + fromIntegral (fromEnum j)) :: NamedTensor '(D.CPU, 0) 'D.Float '[RGB, Vector 2]
          chans = dimUp t :: RGB (NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2])
      (D.asValue (toDynamic (r chans)) :: [Float]) `shouldBe` [0, 1]
      (D.asValue (toDynamic (g chans)) :: [Float]) `shouldBe` [10, 11]
      (D.asValue (toDynamic (b chans)) :: [Float]) `shouldBe` [20, 21]
    it "dimDown is the inverse of dimUp" $ do
      let t = tabulate (\(i :. j :. HNil) -> fromIntegral (fromEnum i) * 10 + fromIntegral (fromEnum j)) :: NamedTensor '(D.CPU, 0) 'D.Float '[RGB, Vector 2]
      (D.asValue (D.reshape [-1] (toDynamic (dimDown (dimUp t)))) :: [Float])
        `shouldBe` (D.asValue (D.reshape [-1] (toDynamic t)) :: [Float])
    it "with Vector it is typed unbind and stack" $ do
      let t = tabulate (\(i :. j :. HNil) -> fromIntegral (fromEnum i) * 10 + fromIntegral (fromEnum j)) :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 3, RGB]
          rows = dimUp t :: Vector 3 (NamedTensor '(D.CPU, 0) 'D.Float '[RGB])
      (D.asValue (toDynamic (V.index rows 1)) :: [Float]) `shouldBe` [10, 11, 12]
