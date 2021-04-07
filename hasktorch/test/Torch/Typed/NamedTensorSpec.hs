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

module Torch.Typed.NamedTensorSpec (spec) where

import Data.Kind
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

data RGB a = RGB {
  r :: a,
  g :: a,
  b :: a
} deriving (Show, Eq)

type instance ToNat RGB = 3

testFieldLens :: HasField "r" shape => Lens' (NamedTensor '(D.CPU,0) 'D.Float shape) (NamedTensor '(D.CPU,0) 'D.Float (DropField "r" shape))
testFieldLens = field @"r"

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
