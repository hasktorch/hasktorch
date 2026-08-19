{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveFunctor #-}
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
import Data.Functor.Compose (Compose (..))
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
import Torch.Typed.Autograd (grad)
import Torch.Typed.Factories (ones)
import Torch.Typed.Functional (sumNamedDim)
import Torch.Typed.Parameter (makeIndependent, toDependent)
import Torch.Typed.Representable
import Torch.Typed.Tensor

newtype Batch (n :: Nat) a = Batch (Vector n a) deriving (Show, Eq, Generic)

data RGB a = RGB
  { r :: a,
    g :: a,
    b :: a
  }
  deriving (Show, Eq, Generic, Default, Functor)

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

-- a compound dimension named by an ordinary nullary synonym
type Triangle = Compose (Vector 3) RGB

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
  describe "vmap" $ do
    -- rows r=[0,1], g=[10,11], b=[20,21]
    let t = tabulate (\(i :. j :. HNil) -> fromIntegral (fromEnum i) * 10 + fromIntegral (fromEnum j)) :: NamedTensor '(D.CPU, 0) 'D.Float '[RGB, Vector 2]
    it "matches its specification, dimDown . fmap g . dimUp" $ do
      let g :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2] -> NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2]
          g x = fromUnnamed (toUnnamed x * toUnnamed x)
      (D.asValue (toDynamic (vmap g t)) :: [[Float]])
        `shouldBe` (D.asValue (toDynamic (dimDown (fmap g (dimUp t)))) :: [[Float]])
    it "maps a reduction, changing the shape under the mapped dimension" $ do
      let sums = vmap (sumNamedDim @(Vector 2)) t :: NamedTensor '(D.CPU, 0) 'D.Float '[RGB]
      (D.asValue (toDynamic sums) :: [Float]) `shouldBe` [1, 21, 41]
    it "vmap2 zips along the shared dimension" $ do
      let s = vmap2 (\a b -> a + b) t t :: NamedTensor '(D.CPU, 0) 'D.Float '[RGB, Vector 2]
      (D.asValue (toDynamic s) :: [[Float]]) `shouldBe` [[0, 2], [20, 22], [40, 42]]
    it "vscan carries state along the dimension" $ do
      let s = vscan (+) (def :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2]) t
      (D.asValue (toDynamic s) :: [[Float]]) `shouldBe` [[0, 1], [10, 12], [30, 33]]
    it "gradients flow through vmap" $ do
      w <- makeIndependent (ones :: Tensor '(D.CPU, 0) 'D.Float '[2])
      let wn = fromUnnamed (toDependent w) :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2]
          total = sumNamedDim @(Vector 2) (sumNamedDim @RGB (vmap (\c -> c * wn) t))
          gw :. HNil = grad (toUnnamed total) (w :. HNil)
      -- d/dw_j of sum_{c,j} t_{c,j} * w_j is the column sums of t
      (D.asValue (toDynamic gw) :: [Float]) `shouldBe` [30, 33]
  describe "dimGroup and dimUngroup" $ do
    let tri = tabulate (\(k :. i :. c :. HNil) -> fromIntegral (fromEnum k) * 100 + fromIntegral (fromEnum i) * 10 + fromIntegral (fromEnum c)) :: NamedTensor '(D.CPU, 0) 'D.Float '[Batch 2, Vector 3, RGB]
    it "a Compose dimension has the product size" $ do
      dimsOf (Proxy @(NamedTensor '(D.CPU, 0) 'D.Float '[Batch 2, Triangle]))
        `shouldBe` [2, 9]
    it "dimGroup merges two dimensions without moving data" $ do
      let grouped = dimGroup tri :: NamedTensor '(D.CPU, 0) 'D.Float '[Batch 2, Triangle]
      shape grouped `shouldBe` [2, 9]
      (D.asValue (D.reshape [-1] (toDynamic grouped)) :: [Float])
        `shouldBe` (D.asValue (D.reshape [-1] (toDynamic tri)) :: [Float])
    it "dimUngroup is the inverse of dimGroup" $ do
      let roundtrip = dimUngroup (dimGroup tri) :: NamedTensor '(D.CPU, 0) 'D.Float '[Batch 2, Vector 3, RGB]
      shape roundtrip `shouldBe` [2, 3, 3]
      (D.asValue (D.reshape [-1] (toDynamic roundtrip)) :: [Float])
        `shouldBe` (D.asValue (D.reshape [-1] (toDynamic tri)) :: [Float])
    it "indexes the grouped dimension flat, row-major" $ do
      -- flat position 5 is (vertex 1, channel 2)
      let grouped = dimGroup tri :: NamedTensor '(D.CPU, 0) 'D.Float '[Batch 2, Triangle]
      index grouped (1 :. 5 :. HNil) `shouldBe` 112
