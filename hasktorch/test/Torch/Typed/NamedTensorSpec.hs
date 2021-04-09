{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
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
{-# LANGUAGE DeriveGeneric #-}

module Torch.Typed.NamedTensorSpec (spec) where

import Data.Kind
import GHC.TypeLits
import Test.Hspec
import GHC.Exts
import qualified Torch.Device as D
import qualified Torch.Tensor as D
import qualified Torch.DType as D
import qualified Torch.Functional as F
import Torch.Typed.Tensor
import Torch.Typed.Factories
import Data.Vector.Sized (Vector)
import qualified Data.Vector.Sized as V
import Torch.Typed.Lens
import Lens.Family
import GHC.Generics
import Data.Proxy
import Data.Default.Class

data RGB a = RGB {
  r :: a,
  g :: a,
  b :: a
} deriving (Show, Eq, Generic)

data YCoCg a = YCoCg {
  y :: a,
  co :: a,
  cg :: a
} deriving (Show, Eq, Generic)

data RGBA a = RGBA a a a a deriving (Show, Eq, Generic)

testFieldLens :: HasField "r" shape => Lens' (NamedTensor '(D.CPU,0) 'D.Float shape) (NamedTensor '(D.CPU,0) 'D.Float (DropField "r" shape))
testFieldLens = field @"r"

testFieldLens2 :: Lens' (NamedTensor '(D.CPU,0) 'D.Float '[Vector n,RGB]) (NamedTensor '(D.CPU,0) 'D.Float '[Vector n])
testFieldLens2 = field @"r"

testDropField :: Proxy (DropField "r" '[Vector 2,RGB]) -> Proxy '[Vector 2]
testDropField = id

testDropField2 :: Proxy (DropField "y" '[Vector 2,YCoCg]) -> Proxy '[Vector 2]
testDropField2 = id

testCountField :: Proxy (ToNat YCoCg) -> Proxy 3
testCountField = id

testCountField2 :: Proxy (ToNat (Vector n)) -> Proxy n
testCountField2 = id

testCountField3 :: Proxy (ToNat RGBA) -> Proxy 4
testCountField3 = id

toYCoCG :: (KnownNat n, KnownDType dtype, KnownDevice device) => NamedTensor device dtype [Vector n, RGB] -> NamedTensor device dtype [Vector n, YCoCg]
toYCoCG rgb =
  set (field @"y")  ((r + g * 2+ b)/4) $
  set (field @"co")  ((r - b)/2)  $
  set (field @"cg")  ((-r + g * 2 - b)/4)  $
  def
  where
    r = rgb ^. field @"r"
    g = rgb ^. field @"g"
    b = rgb ^. field @"b" 

checkDynamicTensorAttributes' ::
  forall device dtype shape t.
  ( Unnamed t, device ~ (UTDevice t), dtype ~ (UTDType t), shape ~ (UTShape t)
  , TensorOptions shape dtype device) =>
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
    it "create" $ do
      let t :: Tensor '(D.CPU,0) 'D.Float '[2,3]
          t = ones
          t2 :: NamedTensor '(D.CPU,0) 'D.Float '[Vector 2,RGB]
          t2 = fromUnnamed t
      print t2
      checkDynamicTensorAttributes' t2
    it "check a shape of lens" $ do
      let t :: NamedTensor '(D.CPU,0) 'D.Float '[Vector 2,Vector 3,Vector 4,RGB]
          t = mempty
          t2 :: NamedTensor '(D.CPU,0) 'D.Float '[Vector 2,Vector 3,Vector 4]
          t2 = t ^. field @"r"
      print $ shape t2
      checkDynamicTensorAttributes' t
      checkDynamicTensorAttributes' t2
    it "check fieldids" $ do
      let v = RGB () () ()
      fieldId @"r" (Proxy :: Proxy (RGB ())) `shouldBe` Just 0
      fieldId @"g" (Proxy :: Proxy (RGB ())) `shouldBe` Just 1
      fieldId @"b" (Proxy :: Proxy (RGB ())) `shouldBe` Just 2
      fieldId @"y" (Proxy :: Proxy (RGB ())) `shouldBe` Nothing
