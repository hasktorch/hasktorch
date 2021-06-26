{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.Tensor.Creation
  ( sOnes,
    sZeros,
    sFull,
    sRandn,
    sArangeNaturals,
    sEye,
    sEyeSquare,
  )
where

import Data.Monoid (All (..))
import Data.Singletons (fromSing)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.DType (SDataType)
import Torch.GraduallyTyped.Device (SDevice)
import Torch.GraduallyTyped.Internal.TensorOptions (tensorOptions)
import Torch.GraduallyTyped.Layout (SLayout (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (Generator, withGenerator)
import Torch.GraduallyTyped.RequiresGradient (SRequiresGradient)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SShape, SSize, Shape (..), dimName, dimSize)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import Torch.Internal.Cast (cast2, cast3, cast4)
import qualified Torch.Internal.Managed.TensorFactories as ATen

-- $setup
-- >>> import Data.Int (Int16)
-- >>> import Data.Singletons.Prelude.List (SList (..))
-- >>> import Torch.GraduallyTyped

-- | Create a tensor of ones.
--
-- >>> shape = SShape $ SName @"batch" :&: SSize @32 :|: SUncheckedName "feature" :&: SUncheckedSize 8 :|: SNil
-- >>> :type sOnes SWithoutGradient (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape
-- sOnes SWithoutGradient (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape
--   :: Tensor
--        'WithoutGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Int64)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim 'UncheckedName 'UncheckedSize])
sOnes ::
  forall requiresGradient layout device dataType shape.
  SRequiresGradient requiresGradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SShape shape ->
  Tensor requiresGradient layout device dataType shape
sOnes reqGradient layout device dataType shape =
  let opts = tensorOptions requiresGradient layoutType deviceType dType
      tensor = unsafePerformIO $ case (map dimName dims, map dimSize dims) of
        (names, sizes)
          | getAll . foldMap (All . (== "*")) $ names -> cast2 ATen.ones_lo sizes opts
          | otherwise -> cast3 ATen.ones_lNo sizes names opts
   in UnsafeTensor tensor
  where
    requiresGradient = fromSing reqGradient
    layoutType = forgetIsChecked . fromSing $ layout
    deviceType = forgetIsChecked . fromSing $ device
    dType = forgetIsChecked . fromSing $ dataType
    dims =
      fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
        . forgetIsChecked
        . fromSing
        $ shape

-- | Create a tensor of zeros.
--
-- >>> shape = SShape $ SName @"batch" :&: SSize @32 :|: SUncheckedName "feature" :&: SUncheckedSize 8 :|: SNil
-- >>> :type sZeros SWithoutGradient (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape
-- sZeros SWithoutGradient (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape
--   :: Tensor
--        'WithoutGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Int64)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim 'UncheckedName 'UncheckedSize])
sZeros ::
  forall requiresGradient layout device dataType shape.
  SRequiresGradient requiresGradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SShape shape ->
  Tensor requiresGradient layout device dataType shape
sZeros reqGradient layout device dataType shape =
  let opts = tensorOptions requiresGradient layoutType deviceType dType
      tensor = unsafePerformIO $ case (map dimName dims, map dimSize dims) of
        (names, sizes)
          | getAll . foldMap (All . (== "*")) $ names -> cast2 ATen.zeros_lo sizes opts
          | otherwise -> cast3 ATen.zeros_lNo sizes names opts
   in UnsafeTensor tensor
  where
    requiresGradient = fromSing reqGradient
    layoutType = forgetIsChecked . fromSing $ layout
    deviceType = forgetIsChecked . fromSing $ device
    dType = forgetIsChecked . fromSing $ dataType
    dims =
      fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
        . forgetIsChecked
        . fromSing
        $ shape

-- | Create a tensor filled with a given scalar value.
--
-- >>> shape = SShape $ SName @"batch" :&: SSize @32 :|: SUncheckedName "feature" :&: SUncheckedSize 8 :|: SNil
-- >>> input = 2
-- >>> :type sFull SWithoutGradient (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape input
-- sFull SWithoutGradient (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape input
--   :: Tensor
--        'WithoutGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Int64)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim 'UncheckedName 'UncheckedSize])
sFull ::
  forall requiresGradient layout device dataType shape input.
  Scalar input =>
  SRequiresGradient requiresGradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SShape shape ->
  input ->
  Tensor requiresGradient layout device dataType shape
sFull sRequiresGradient sLayout sDevice sDataType sShape input = UnsafeTensor tensor
  where
    tensor = unsafePerformIO $ case (dimName <$> dims, dimSize <$> dims) of
      (names, sizes)
        | getAll . foldMap (\name -> All $ name == "*") $ names -> cast3 ATen.full_lso sizes input opts
        | otherwise -> cast4 ATen.full_lsNo sizes input names opts
    opts = tensorOptions requiresGradient layoutType deviceType dType
    requiresGradient = fromSing sRequiresGradient
    layoutType = forgetIsChecked . fromSing $ sLayout
    deviceType = forgetIsChecked . fromSing $ sDevice
    dType = forgetIsChecked . fromSing $ sDataType
    dims =
      fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
        . forgetIsChecked
        . fromSing
        $ sShape

-- | Create a random tensor.
sRandn ::
  forall requiresGradient layout device dataType shape device'.
  SRequiresGradient requiresGradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SShape shape ->
  Generator device' ->
  (Tensor requiresGradient layout device dataType shape, Generator device')
sRandn reqGradient layout device dataType shape =
  let opts = tensorOptions requiresGradient layoutType deviceType dType
   in withGenerator
        ( \genPtr -> do
            tensor <- case (map dimName dims, map dimSize dims) of
              (names, sizes)
                | getAll . foldMap (\name -> All $ name == "*") $ names -> cast3 ATen.randn_lGo sizes genPtr opts
                | otherwise -> cast4 ATen.randn_lGNo sizes genPtr names opts
            pure $ UnsafeTensor tensor
        )
        ( unsafePerformIO $ do
            tensor <- case (map dimName dims, map dimSize dims) of
              (names, sizes)
                | getAll . foldMap (\name -> All $ name == "*") $ names -> cast2 ATen.zeros_lo sizes opts
                | otherwise -> cast3 ATen.zeros_lNo sizes names opts
            pure $ UnsafeTensor tensor
        )
  where
    requiresGradient = fromSing reqGradient
    layoutType = forgetIsChecked . fromSing $ layout
    deviceType = forgetIsChecked . fromSing $ device
    dType = forgetIsChecked . fromSing $ dataType
    dims =
      fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
        . forgetIsChecked
        . fromSing
        $ shape

-- | Create a one-dimensional tensor of the numbers @0@ to @size -1@.
sArangeNaturals ::
  forall requiresGradient layout device dataType shape size.
  shape ~ 'Shape '[ 'Dim ('Name "*") size] =>
  SRequiresGradient requiresGradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SSize size ->
  Tensor requiresGradient layout device dataType shape
sArangeNaturals sRequiresGradient sLayout sDevice sDataType sSizeDim = UnsafeTensor tensor
  where
    tensor = unsafePerformIO $ cast2 ATen.arange_so size opts
    opts = tensorOptions requiresGradient layoutType deviceType dType

    requiresGradient = fromSing sRequiresGradient
    layoutType = forgetIsChecked . fromSing $ sLayout
    deviceType = forgetIsChecked . fromSing $ sDevice
    dType = forgetIsChecked . fromSing $ sDataType
    size = forgetIsChecked . fromSing $ sSizeDim

sEye ::
  forall requiresGradient layout device dataType shape rows cols.
  (shape ~ 'Shape '[ 'Dim ('Name "*") rows, 'Dim ('Name "*") cols]) =>
  SRequiresGradient requiresGradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SSize rows ->
  SSize cols ->
  Tensor requiresGradient layout device dataType shape
sEye sRequiresGradient sLayout sDevice sDataType sRows sCols = UnsafeTensor tensor
  where
    tensor = unsafePerformIO $ cast3 ATen.eye_llo (fromInteger rows :: Int) (fromInteger cols :: Int) opts
    opts = tensorOptions requiresGradient layoutType deviceType dType

    requiresGradient = fromSing sRequiresGradient
    layoutType = forgetIsChecked . fromSing $ sLayout
    deviceType = forgetIsChecked . fromSing $ sDevice
    dType = forgetIsChecked . fromSing $ sDataType
    rows = forgetIsChecked . fromSing $ sRows
    cols = forgetIsChecked . fromSing $ sCols

sEyeSquare ::
  forall requiresGradient layout device dataType shape size.
  shape ~ 'Shape '[ 'Dim ('Name "*") size, 'Dim ('Name "*") size] =>
  SRequiresGradient requiresGradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SSize size ->
  Tensor requiresGradient layout device dataType shape
sEyeSquare sRequiresGradient sLayout sDevice sDataType sSize = UnsafeTensor tensor
  where
    tensor = unsafePerformIO $ cast2 ATen.eye_lo (fromInteger size :: Int) opts
    opts = tensorOptions requiresGradient layoutType deviceType dType

    requiresGradient = fromSing sRequiresGradient
    layoutType = forgetIsChecked . fromSing $ sLayout
    deviceType = forgetIsChecked . fromSing $ sDevice
    dType = forgetIsChecked . fromSing $ sDataType
    size = forgetIsChecked . fromSing $ sSize
