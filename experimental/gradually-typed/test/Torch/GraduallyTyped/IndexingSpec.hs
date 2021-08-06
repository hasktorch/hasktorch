{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}

module Torch.GraduallyTyped.IndexingSpec where

import Control.Arrow ((<<<))
import Control.Monad (join)
import Control.Monad.Trans (lift)
import Data.Bool (bool)
import Data.List
import Data.Maybe (fromJust)
import Data.Singletons.Prelude (SList (..), fromSing)
import qualified Data.Vector.Sized as SV
import GHC.TypeLits (Nat)
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Language.Haskell.TH as TH
import Test.Hspec (Spec, context, describe, it, shouldBe)
import Test.Hspec.Hedgehog (Gen, GenT, forAll, hedgehog, tripping, (===), MonadGen)
import Torch.GraduallyTyped.DType (DType (..), DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Index (Index (..), SIndex (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked, pattern (:|:))
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor (IndexType (..), Indices (..), SIndexType (..), SIndices (..), Tensor, arangeNaturals, fromTensor, getDims, parseSlice, reshape, (!), slice)
import Hedgehog.Internal.Property (forAllT)
import Hedgehog.Internal.Gen (mapGenT)
import Control.Monad.Morph (hoist)
import Data.Foldable (asum)

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

toIndexTypes :: SIndices (indices :: Indices [IndexType (Index Nat)]) -> [IndexType Integer]
toIndexTypes = fmap (forgetIsChecked <$>) . forgetIsChecked . fromSing

genIndexTH :: GenT TH.Q TH.Exp
genIndexTH =
    Gen.choice
    [ lift [|NewAxis|]
      -- pure $ TH.ConE 'Ellipsis,
      -- pure $ TH.ConE 'SliceAll
      -- TH.AppE (TH.ConE 'SliceBool) [TH.ConE (bool 'False 'True b) <$> Gen.bool]
    ]
      -- do
      --   at <- genInt
      --   TH.ConE 'SliceAt $at,
      -- do
      --   from <- genInt
      --   TH.ConE 'SliceFrom $from,
      -- do
      --   upTo <- genInt
      --   TH.ConE 'SliceUpTo $upTo,
      -- do
      --   step <- genInt
      --   TH.ConE 'SliceWithStep $step,
      -- do
      --   from <- genInt
      --   upTo <- genInt
      --   TH.ConE 'SliceFromUpTo $from $upTo,
      -- do
      --   from <- genInt
      --   step <- genInt
      --   TH.ConE 'SliceFromWithStep $from $step,
      -- do
      --   upTo <- genInt
      --   step <- genInt
      --   TH.ConE 'SliceUpToWithStep $upTo $step,
      -- do
      --   from <- genInt
      --   upTo <- genInt
      --   step <- genInt
      --   TH.ConE 'SliceFromUpToWithStep $from $upTo $step
  --   ]
  -- where
  --   genInt :: MonadGen m => m (TH.Q TH.Exp)
  --   genInt = TH.litE . TH.integerL <$> Gen.integral (Range.linear (-100) 100)

-- genIndex :: Gen (IndexType Integer)
-- genIndex =
--   Gen.choice
--     [ pure NewAxis,
--       pure Ellipsis,
--       pure SliceAll,
--       SliceBool <$> Gen.bool,
--       SliceAt <$> genInt,
--       SliceFrom <$> genInt,
--       SliceUpTo <$> genInt,
--       SliceWithStep <$> genInt,
--       SliceFromUpTo <$> genInt <*> genInt,
--       SliceFromWithStep <$> genInt <*> genInt,
--       SliceUpToWithStep <$> genInt <*> genInt,
--       SliceFromUpToWithStep <$> genInt <*> genInt <*> genInt
--     ]
--   where
--     genInt = Gen.integral (Range.linear (-100) 100)

-- genWs :: Gen String
-- genWs = _

-- pprintIndex :: IndexType Integer -> Gen String
-- pprintIndex = \case
--     NewAxis -> do
--         ws <- genWs
--         newAxis <- asum [ pure "+", pure "NewAxis"]
--         ws' <- genWs
--         pure $ ws <> newAxis <> ws'
--     Ellipsis -> [pure "...", pure "Ellipsis"]
--     SliceAll -> [pure ":", [":", ":"]]
--     SliceAt at -> [i at]
--     SliceBool b -> [pure $ show b]
--     -- [["-", "11", ":"], ["-", "11", ":", ":"]]


--     SliceFrom from -> [i from <> [":"], i from <> [":", ":"]]
--     SliceUpTo to -> [[":"] <> i to, [":"] <> i to <> [":"]]
--     SliceWithStep step -> [[":", ":"] <> i step]
--     SliceFromUpTo from upTo -> [i from <> [":"] <> i upTo, i from <> [":"] <> i upTo <> [":"]]
--     SliceFromWithStep from step -> [i from <> [":", ":"] <> i step]
--     SliceUpToWithStep upTo step -> [[":"] <> i upTo <> [":"] <> i step]
--     SliceFromUpToWithStep from upTo step -> [i from <> [":"] <> i upTo <> [":"] <> i step]
--   where
--     i :: Integer -> Gen String
--     i x
--       | x < 0 = foldr <$> genWs <*> ["-", show $ abs x]
--       | otherwise = pure $ show x

spec :: Spec
spec = describe "Indexing" $ do
  context "without slice quasiquoter" $ do
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

    context "with slice quasiquoter" $ do
      -- it "round-trips" $
      --   hedgehog $ do
      --     index <- forAllT $ hoist TH.runQ genIndexTH
      --     let
      --         pretty = join . head . pprintIndex
      --         parse = TH.runQ . parseSlice
      --     tripping index pretty parse

      it "+" $ do
        toIndexTypes [slice|+|] `shouldBe` [NewAxis]
      it "NewAxis" $ do
        toIndexTypes [slice|NewAxis|] `shouldBe` [NewAxis]
      it "Ellipsis" $ do
        toIndexTypes [slice|Ellipsis|] `shouldBe` [Ellipsis]
      it "..." $ do
        toIndexTypes [slice|...|] `shouldBe` [Ellipsis]
      it "123" $ do
        toIndexTypes [slice|123|] `shouldBe` [SliceAt 123]
      it "-123" $ do
        toIndexTypes [slice|-123|] `shouldBe` [SliceAt (-123)]
      it "True" $ do
        toIndexTypes [slice|True|] `shouldBe` [SliceBool True]
      it "False" $ do
        toIndexTypes [slice|False|] `shouldBe` [SliceBool False]
      it ":" $ do
        toIndexTypes [slice|:|] `shouldBe` [SliceAll]
      it "::" $ do
        toIndexTypes [slice|::|] `shouldBe` [SliceAll]
      it "1:" $ do
        toIndexTypes [slice|1:|] `shouldBe` [SliceFrom 1]
      it "1::" $ do
        toIndexTypes [slice|1::|] `shouldBe` [SliceFrom 1]
      it ":3" $ do
        toIndexTypes [slice|:3|] `shouldBe` [SliceUpTo 3]
      it ":3:" $ do
        toIndexTypes [slice|:3:|] `shouldBe` [SliceUpTo 3]
      it "::2" $ do
        toIndexTypes [slice|::2|] `shouldBe` [SliceWithStep 2]
      it "1:3" $ do
        toIndexTypes [slice|1:3|] `shouldBe` [SliceFromUpTo 1 3]
      it "1::2" $ do
        toIndexTypes [slice|1::2|] `shouldBe` [SliceFromWithStep 1 2]
      it ":3:2" $ do
        toIndexTypes [slice|:3:2|] `shouldBe` [SliceUpToWithStep 3 2]
      it "1:3:2" $ do
        toIndexTypes [slice|1:3:2|] `shouldBe` [SliceFromUpToWithStep 1 3 2]
      it "1,2,3" $ do
        toIndexTypes [slice|1,2,3|] `shouldBe` [SliceAt 1, SliceAt 2, SliceAt 3]
      it "1 , 2, 3" $ do
        toIndexTypes [slice|1 , 2, 3|] `shouldBe` [SliceAt 1, SliceAt 2, SliceAt 3]
      it "{SIndex @1}" $ do
        let i = SIndex @1
        toIndexTypes [slice|{i}|] `shouldBe` [SliceAt 1]
      it "{SNegativeIndex @1}" $ do
        let i = SNegativeIndex @1
        toIndexTypes [slice|{i}|] `shouldBe` [SliceAt (-1)]
      it "{SUncheckedIndex 1}" $ do
        let i = SUncheckedIndex 1
        toIndexTypes [slice|{i}|] `shouldBe` [SliceAt 1]
