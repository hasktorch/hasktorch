{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module TensorSpec (spec) where

import Control.Monad.Catch (MonadThrow)
import Data.Singletons.Prelude.List
import qualified Data.Vector as V
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Gen.QuickCheck as Gen
import qualified Hedgehog.Range as Range
import Test.Hspec
import Test.Hspec.Hedgehog (MonadGen, assert, forAll, hedgehog, (===))
import qualified Test.QuickCheck as QC
import Torch.GraduallyTyped

toCPUTensor ::
  (TensorLike a dType dims, MonadThrow m) =>
  a ->
  m (Tensor 'WithoutGradient ('Layout 'Dense) ('Device 'CPU) ('DataType dType) ('Shape dims))
toCPUTensor = toTensor

genReal :: MonadGen m => (RealFloat a, QC.Arbitrary a, Read a) => m a
genReal =
  Gen.frequency
    [ (1, denormalized),
      (10, Gen.arbitrary)
    ]
  where
    denormalized =
      Gen.element
        [ read "NaN",
          read "Infinity",
          read "-Infinity",
          read "-0"
        ]

spec :: Spec
spec = describe "TensorLike" $ do
  context "fromTensor ° toTensor ≡ id" $ do
    it "Bool" $
      hedgehog $ do
        x <- forAll Gen.bool
        t <- toCPUTensor x
        fromTensor t === x

    it "Int" $
      hedgehog $ do
        x <- forAll $ Gen.int Range.linearBounded
        t <- toCPUTensor x
        fromTensor t === x

    it "Float" $
      hedgehog $ do
        x :: Float <- forAll genReal
        t <- toCPUTensor x
        let x' = fromTensor t
        assert $ x' == x || isNaN x' && isNaN x

    it "Double" $
      hedgehog $ do
        x :: Double <- forAll genReal
        t <- toCPUTensor x
        let x' = fromTensor t
        assert $ x' == x || isNaN x' && isNaN x

    it "(Bool, Bool)" $
      hedgehog $ do
        x <- forAll $ (,) <$> Gen.bool <*> Gen.bool
        t <- toCPUTensor x
        fromTensor t === x

    it "(Int, Int)" $
      hedgehog $ do
        x <- forAll $ (,) <$> Gen.int Range.linearBounded <*> Gen.int Range.linearBounded
        t <- toCPUTensor x
        fromTensor t === x

    it "(Float, Float)" $
      hedgehog $ do
        (x, y) :: (Float, Float) <- forAll $ (,) <$> genReal <*> genReal
        t <- toCPUTensor (x, y)
        let (x', y') = fromTensor t
        assert $
          (x', y') == (x, y)
            || isNaN x' && isNaN x && y' == y
            || isNaN y' && isNaN y && x' == x

    it "(Double, Double)" $
      hedgehog $ do
        (x, y) :: (Double, Double) <- forAll $ (,) <$> genReal <*> genReal
        t <- toCPUTensor (x, y)
        let (x', y') = fromTensor t
        assert $
          (x', y') == (x, y)
            || isNaN x' && isNaN x && y' == y
            || isNaN y' && isNaN y && x' == x

    it "[Bool]" $
      hedgehog $ do
        x <- forAll $ Gen.list (Range.linear 0 1000) Gen.bool
        t <- toCPUTensor x
        fromTensor t === x

    it "[Int]" $
      hedgehog $ do
        x <- forAll $ Gen.list (Range.linear 0 1000) $ Gen.int Range.linearBounded
        t <- toCPUTensor x
        fromTensor t === x

    it "[Float]" $
      hedgehog $ do
        xs :: [Float] <- forAll $ Gen.list (Range.linear 0 1000) genReal
        t <- toCPUTensor xs
        let xs' = fromTensor t
        assert $ all (\(x, y) -> x == y || isNaN x && isNaN y) $ zip xs' xs

    it "[Double]" $
      hedgehog $ do
        xs :: [Double] <- forAll $ Gen.list (Range.linear 0 1000) genReal
        t <- toCPUTensor xs
        let xs' = fromTensor t
        assert $ all (\(x, y) -> x == y || isNaN x && isNaN y) $ zip xs' xs

    it "(Vector Float)" $
      hedgehog $ do
        xs :: V.Vector Double <- (V.fromList <$>) <$> forAll $ Gen.list (Range.linear 0 1000) genReal
        t <- toCPUTensor xs
        let xs' = fromTensor t
        assert $ all (\(x, y) -> x == y || isNaN x && isNaN y) $ V.zip xs' xs

    it "Tensor" $ do
      let t =
            ones
              @'WithoutGradient
              @('Layout 'Dense)
              @('Device 'CPU)
              @('DataType 'Int64)
              @('Shape '[ 'Dim ('Name "*") ('Size 4), 'Dim ('Name "*") ('Size 8)])
      t' <- toTensor @'WithoutGradient @('Layout 'Dense) @('Device 'CPU) t
      fromTensor (allT $ t' ==. t) `shouldBe` True
      let t'' =
            fromTensor
              @( Tensor
                  'WithoutGradient
                  ('Layout 'Dense)
                  ('Device 'CPU)
                  ('DataType 'Int64)
                  ('Shape '[ 'Dim ('Name "*") ('Size 4), 'Dim ('Name "*") ('Size 8)])
              )
              t'
      fromTensor (allT $ t'' ==. t) `shouldBe` True

  it "dims ([[]] :: [[[Int]]]) = 1x0x0" $ do
    x <- toCPUTensor @[[[Int]]] [[]]
    dims' <- dims x
    dims' `shouldBe` [Dim "*" 1, Dim "*" 0, Dim "*" 0]

  it "dims ([] :: [(Int, Int)]) = 0x2" $ do
    x <- toCPUTensor @[(Int, Int)] []
    dims' <- dims x
    dims' `shouldBe` [Dim "*" 0, Dim "*" 2]

  it "dims ([] :: [([Int], [Int])] = 0x2x0" $ do
    x <- toCPUTensor @[([Int], [Int])] []
    dims' <- dims x
    dims' `shouldBe` [Dim "*" 0, Dim "*" 2, Dim "*" 0]

  it "lists having different length" $ do
    let mkT = toCPUTensor @[[Double]] [[1], [1, 2]]
    mkT `shouldThrow` (== DimMismatchError [1] [2])
