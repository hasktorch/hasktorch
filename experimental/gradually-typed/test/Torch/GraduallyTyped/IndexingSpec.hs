{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}

module Torch.GraduallyTyped.IndexingSpec where

import Control.Arrow ((<<<))
import Control.Monad ((<=<))
import Control.Monad.Trans (lift)
import Data.Foldable (asum)
import Data.List
import Data.Singletons.Prelude (Demote, SList (..), SingKind, SomeSing (..), fromSing, toSing)
import qualified Data.Vector.Sized as SV
import GHC.TypeLits (Nat)
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Language.Haskell.TH as TH
import qualified Language.Haskell.TH.Syntax as TH
import Test.Hspec (Spec, context, describe, it, shouldBe)
import Test.Hspec.Hedgehog (Gen, Range, forAll, hedgehog, (===))
import Torch.GraduallyTyped.DType (DType (..), DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Index (Index (..), SIndex (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.Prelude (IsChecked (..), forgetIsChecked, pattern (:|:))
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor (IndexType (..), Indices (..), SIndexType (..), SIndices (..), Tensor, arangeNaturals, fromTensor, getDims, parseSlice, reshape, (!), slice)
import Control.Applicative ((<|>))

-- | 2x2x3 tensor for testing.
-- >>> tensor
-- Tensor Int64 [2,2,3] [[[ 0,  1,  2],
--                        [ 3,  4,  5]],
--                       [[ 6,  7,  8],
--                        [ 9,  10,  11]]]
tensor ::
  Tensor
    ('Gradient 'WithoutGradient)
    ('Layout 'Dense)
    ('Device 'CPU)
    ('DataType 'Int64)
    ( 'Shape
        '[ 'Dim ('Name "*") ('Size 2),
           'Dim ('Name "*") ('Size 2),
           'Dim ('Name "*") ('Size 3)
         ]
    )
tensor = head $ reshape =<< arangeNaturals @('Size 12)

spec :: Spec
spec = describe "Indexing" $ do
  it "extracts a scalar value" $ do
    x <- tensor ! SIndices (SSliceAt (SIndex @1) :|: SSliceAt (SIndex @0) :|: SSliceAt (SIndex @2) :|: SNil)

    dimSize <$> getDims x `shouldBe` []
    fromTensor @Int x `shouldBe` 8

  it "slices a tensor" $ do
    x <- tensor ! SIndices (SSliceAt (SIndex @1) :|: SSliceAt (SIndex @0) :|: SNil)

    dimSize <$> getDims x `shouldBe` [3]
    fromTensor @(Int, Int, Int) x `shouldBe` (6, 7, 8)

  it "slices only the last dimension" $ do
    x <- tensor ! SIndices (SSliceAll :|: SSliceAll :|: SSliceAt (SIndex @1) :|: SNil)

    dimSize <$> getDims x `shouldBe` [2, 2]
    fromTensor @((Int, Int), (Int, Int)) x `shouldBe` ((1, 4), (7, 10))

  it "slices with ellipsis" $ do
    x <- tensor ! SIndices (SEllipsis :|: SSliceAt (SIndex @1) :|: SNil)

    dimSize <$> getDims x `shouldBe` [2, 2]
    fromTensor @((Int, Int), (Int, Int)) x `shouldBe` ((1, 4), (7, 10))

  it "slices with SSliceFrom" $ do
    x <- tensor ! SIndices (SSliceAll :|: SSliceFrom (SIndex @1) :|: SNil)

    dimSize <$> getDims x `shouldBe` [2, 1, 3]
    fromTensor @(SV.Vector _ (SV.Vector _ (SV.Vector _ Int))) x
      `shouldBe` SV.fromTuple
        ( SV.singleton $ SV.fromTuple (3, 4, 5),
          SV.singleton $ SV.fromTuple (9, 10, 11)
        )

  it "slices with SliceFromUpTo" $ do
    x <- tensor ! SIndices (SSliceAll :|: SSliceFrom (SIndex @1) :|: SSliceFromUpTo (SIndex @0) (SIndex @1) :|: SNil)

    dimSize <$> getDims x `shouldBe` [2, 1, 1]
    fromTensor @(SV.Vector _ (SV.Vector _ (SV.Vector _ Int))) x
      `shouldBe` SV.fromTuple
        ( SV.singleton $ SV.singleton 3,
          SV.singleton $ SV.singleton 9
        )

  it "slices with SliceFromUpToWithStep" $ do
    x <- tensor ! SIndices (SSliceAll :|: SSliceFrom (SIndex @1) :|: SSliceFromUpToWithStep (SIndex @0) (SIndex @1) (SIndex @2) :|: SNil)

    dimSize <$> getDims x `shouldBe` [2, 1, 1]
    fromTensor @(SV.Vector _ (SV.Vector _ (SV.Vector _ Int))) x
      `shouldBe` SV.fromTuple
        ( SV.singleton $ SV.singleton 3,
          SV.singleton $ SV.singleton 9
        )
