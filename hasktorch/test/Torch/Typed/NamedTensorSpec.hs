{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.NamedTensorSpec (spec) where

import Data.Default.Class
import Data.Kind
import Data.Maybe (fromJust)
import Data.Proxy
import Data.Vector.Sized (Vector)
import qualified Data.Vector.Sized as V
import GHC.Exts
import GHC.Generics
import GHC.TypeLits
import Lens.Family
import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as F
import Torch.Lens
import qualified Torch.Tensor as D
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.Lens
import Torch.Typed.NamedTensor
import Torch.Typed.Tensor

newtype Batch (n :: Nat) a = Batch (Vector n a) deriving (Show, Eq, Generic)

newtype Height (n :: Nat) a = Height (Vector n a) deriving (Show, Eq, Generic)

newtype Width (n :: Nat) a = Width (Vector n a) deriving (Show, Eq, Generic)

data RGB a = RGB
  { r :: a,
    g :: a,
    b :: a
  }
  deriving (Show, Eq, Generic, Default)

data YCoCg a = YCoCg
  { y :: a,
    co :: a,
    cg :: a
  }
  deriving (Show, Eq, Generic, Default)

data RGBA a = RGBA a a a a deriving (Show, Eq, Generic, Default)

testFieldLens :: HasField "r" shape => Lens' (NamedTensor '(D.CPU, 0) 'D.Float shape) (NamedTensor '(D.CPU, 0) 'D.Float (DropField "r" shape))
testFieldLens = field @"r"

testFieldLens2 :: Lens' (NamedTensor '(D.CPU, 0) 'D.Float '[Vector n, RGB]) (NamedTensor '(D.CPU, 0) 'D.Float '[Vector n])
testFieldLens2 = field @"r"

testNamedLens :: Traversal' (NamedTensor '(D.CPU, 0) 'D.Float '[Vector n, RGB]) (NamedTensor '(D.CPU, 0) 'D.Float '[Vector n])
testNamedLens = name @RGB

testNamedLens2 :: Traversal' (NamedTensor '(D.CPU, 0) 'D.Float '[Vector 3, RGB, Vector 4]) (NamedTensor '(D.CPU, 0) 'D.Float '[Vector 3, Vector 4])
testNamedLens2 = name @RGB

testDropField :: Proxy (DropField "r" '[Vector 2, RGB]) -> Proxy '[Vector 2]
testDropField = id

testDropField2 :: Proxy (DropField "y" '[Vector 2, YCoCg]) -> Proxy '[Vector 2]
testDropField2 = id

testCountField :: Proxy (ToNat YCoCg) -> Proxy 3
testCountField = id

testCountField2 :: Proxy (ToNat (Vector n)) -> Proxy n
testCountField2 = id

testCountField3 :: Proxy (ToNat RGBA) -> Proxy 4
testCountField3 = id

testCountField4 :: Proxy (ToNat (Batch n)) -> Proxy n
testCountField4 = id

toYCoCG :: (KnownNat n, KnownDType dtype, KnownDevice device) => NamedTensor device dtype [Vector n, RGB] -> NamedTensor device dtype [Vector n, YCoCg]
toYCoCG rgb =
  set (field @"y") ((r + g * 2 + b) / 4) $
    set (field @"co") ((r - b) / 2) $
      set (field @"cg") ((- r + g * 2 - b) / 4) $
        def
  where
    r = rgb ^. field @"r"
    g = rgb ^. field @"g"
    b = rgb ^. field @"b"

checkDynamicTensorAttributes' ::
  forall device dtype shape t.
  ( IsUnnamed t device dtype shape,
    TensorOptions shape dtype device
  ) =>
  t ->
  IO ()
checkDynamicTensorAttributes' t = do
  D.device untyped `shouldBe` optionsRuntimeDevice @shape @dtype @device
  D.dtype untyped `shouldBe` optionsRuntimeDType @shape @dtype @device
  D.shape untyped `shouldBe` optionsRuntimeShape @shape @dtype @device
  where
    untyped = toDynamic t

spec :: Spec
spec = do
  describe "NamedTensor" $ do
    it "create by Typed Tensor" $ do
      let t :: Tensor '(D.CPU, 0) 'D.Float '[2, 3]
          t = ones
          t2 :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2, RGB]
          t2 = fromUnnamed t
          t3 :: NamedTensor '(D.CPU, 0) 'D.Float '[Batch 2, RGB]
          t3 = fromUnnamed t
      print t2
      checkDynamicTensorAttributes' t2
      checkDynamicTensorAttributes' t3
    it "create by default class" $ do
      let t :: NamedTensor '(D.CPU, 0) 'D.Float '[Batch 2, Height 3, Width 4, RGB]
          t = def
      checkDynamicTensorAttributes' t
    it "create by NamedTensorLike" $ do
      let t :: NamedTensor '(D.CPU, 0) 'D.Float '[Batch 1, Height 1, Width 1, RGB]
          t = asNamedTensor $ Batch (V.singleton (Height (V.singleton (Width (V.singleton (RGB {r = 0 :: Float, g = 1, b = 2}))))))
      checkDynamicTensorAttributes' t
    it "check a shape of lens" $ do
      let t :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2, Vector 3, Vector 4, RGB]
          t = def
          t2 :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2, Vector 3, Vector 4]
          t2 = t ^. field @"r"
      print $ shape t2
      checkDynamicTensorAttributes' t
      checkDynamicTensorAttributes' t2
    it "get fieldids" $ do
      let v = RGB () () ()
      fieldId @"r" (Proxy :: Proxy (RGB ())) `shouldBe` Just 0
      fieldId @"g" (Proxy :: Proxy (RGB ())) `shouldBe` Just 1
      fieldId @"b" (Proxy :: Proxy (RGB ())) `shouldBe` Just 2
      fieldId @"y" (Proxy :: Proxy (RGB ())) `shouldBe` Nothing
    it "sort" $ do
      let t :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2, Vector 3, Vector 4, RGB]
          t = def
          (v, idx) = sortNamedDim @RGB True t
      checkDynamicTensorAttributes' v
      checkDynamicTensorAttributes' idx
    it "mean" $ do
      let t :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2, Vector 3, Vector 4, RGB]
          t = def
          v0 = meanNamedDim @RGB t :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2, Vector 3, Vector 4]
          v1 = meanNamedDim @(Vector 3) t :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2, Vector 4, RGB]
      checkDynamicTensorAttributes' v0
      checkDynamicTensorAttributes' v1
    it "shape and dtype" $ do
      let t :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2, Vector 3, Vector 4, RGB]
          t = def
      shape t `shouldBe` [2, 3, 4, 3]
      dtype t `shouldBe` D.Float
    it "named index" $ do
      let t :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 2, Vector 3, Vector 4, RGB]
          t = def
          t2 = flattenValues (name @(Vector 3)) t
      map shape t2 `shouldBe` [[2, 4, 3], [2, 4, 3], [2, 4, 3]]
      map dtype t2 `shouldBe` [D.Float, D.Float, D.Float]
