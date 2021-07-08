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
    ones,
    sZeros,
    zeros,
    sFull,
    full,
    sRandn,
    randn,
    sArangeNaturals,
    arangeNaturals,
    sEye,
    eye,
    sEyeSquare,
    eyeSquare,
  )
where

import Data.Monoid (All (..))
import Data.Singletons (SingI (..), SingKind (fromSing))
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.DType (SDataType)
import Torch.GraduallyTyped.Device (SDevice)
import Torch.GraduallyTyped.Internal.TensorOptions (tensorOptions)
import Torch.GraduallyTyped.Layout (SLayout (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (Generator, withGenerator)
import Torch.GraduallyTyped.RequiresGradient (SGradient)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SShape, SSize, Shape (..), dimName, dimSize)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.Internal.Cast (cast2, cast3, cast4)
import qualified Torch.Internal.Managed.TensorFactories as ATen

-- $setup
-- >>> import Data.Int (Int16)
-- >>> import Data.Singletons.Prelude.List (SList (..))
-- >>> import Torch.GraduallyTyped

-- | Create a gradually typed tensor of ones.
--
-- >>> shape = SShape $ SName @"batch" :&: SSize @32 :|: SUncheckedName "feature" :&: SUncheckedSize 8 :|: SNil
-- >>> :type sOnes (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape
-- sOnes (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape
--   :: Tensor
--        ('Gradient 'WithoutGradient)
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Int64)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim 'UncheckedName 'UncheckedSize])
sOnes ::
  forall gradient layout device dataType shape.
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SShape shape ->
  Tensor gradient layout device dataType shape
sOnes gradient layout device dataType shape =
  let opts = tensorOptions requiresGradient layoutType deviceType dType
      tensor = unsafePerformIO $ case (map dimName dims, map dimSize dims) of
        (names, sizes)
          | getAll . foldMap (All . (== "*")) $ names -> cast2 ATen.ones_lo sizes opts
          | otherwise -> cast3 ATen.ones_lNo sizes names opts
   in UnsafeTensor tensor
  where
    requiresGradient = forgetIsChecked . fromSing $ gradient
    layoutType = forgetIsChecked . fromSing $ layout
    deviceType = forgetIsChecked . fromSing $ device
    dType = forgetIsChecked . fromSing $ dataType
    dims =
      fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
        . forgetIsChecked
        . fromSing
        $ shape

-- | Create a typed tensor of ones.
--
-- >>> ones :: CPUParameter ('DataType 'Float) ('Shape '[])
-- Tensor Float []  1.0000
-- >>> ones :: CPUTensor ('DataType 'Int64) ('Shape '[ 'Dim ('Name "*") ('Size 1)])
-- Tensor Int64 [1] [ 1]
ones ::
  forall gradient layout device dataType shape.
  (SingI gradient, SingI layout, SingI device, SingI dataType, SingI shape) =>
  Tensor gradient layout device dataType shape
ones = sOnes (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @shape)

-- | Create a gradually typed tensor of zeros.
--
-- >>> shape = SShape $ SName @"batch" :&: SSize @32 :|: SUncheckedName "feature" :&: SUncheckedSize 8 :|: SNil
-- >>> :type sZeros (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape
-- sZeros (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape
--   :: Tensor
--        ('Gradient 'WithoutGradient)
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Int64)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim 'UncheckedName 'UncheckedSize])
sZeros ::
  forall gradient layout device dataType shape.
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SShape shape ->
  Tensor gradient layout device dataType shape
sZeros gradient layout device dataType shape =
  let opts = tensorOptions requiresGradient layoutType deviceType dType
      tensor = unsafePerformIO $ case (map dimName dims, map dimSize dims) of
        (names, sizes)
          | getAll . foldMap (All . (== "*")) $ names -> cast2 ATen.zeros_lo sizes opts
          | otherwise -> cast3 ATen.zeros_lNo sizes names opts
   in UnsafeTensor tensor
  where
    requiresGradient = forgetIsChecked . fromSing $ gradient
    layoutType = forgetIsChecked . fromSing $ layout
    deviceType = forgetIsChecked . fromSing $ device
    dType = forgetIsChecked . fromSing $ dataType
    dims =
      fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
        . forgetIsChecked
        . fromSing
        $ shape

-- | Create a typed tensor of zeros.
--
-- >>> zeros :: CPUParameter ('DataType 'Float) ('Shape '[])
-- Tensor Float []  0.0000
-- >>> zeros :: CPUTensor ('DataType 'Int64) ('Shape '[ 'Dim ('Name "*") ('Size 1)])
-- Tensor Int64 [1] [ 0]
zeros ::
  forall gradient layout device dataType shape.
  (SingI gradient, SingI layout, SingI device, SingI dataType, SingI shape) =>
  Tensor gradient layout device dataType shape
zeros = sZeros (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @shape)

-- | Create a gradually typed tensor filled with a given scalar value.
--
-- >>> shape = SShape $ SName @"batch" :&: SSize @32 :|: SUncheckedName "feature" :&: SUncheckedSize 8 :|: SNil
-- >>> input = -1
-- >>> :type sFull (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape input
-- sFull (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape input
--   :: Tensor
--        ('Gradient 'WithoutGradient)
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Int64)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim 'UncheckedName 'UncheckedSize])
sFull ::
  forall gradient layout device dataType shape input.
  Scalar input =>
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SShape shape ->
  input ->
  Tensor gradient layout device dataType shape
sFull gradient layout device dataType shape input = UnsafeTensor tensor
  where
    tensor = unsafePerformIO $ case (dimName <$> dims, dimSize <$> dims) of
      (names, sizes)
        | getAll . foldMap (\name -> All $ name == "*") $ names -> cast3 ATen.full_lso sizes input opts
        | otherwise -> cast4 ATen.full_lsNo sizes input names opts
    opts = tensorOptions requiresGradient layoutType deviceType dType

    requiresGradient = forgetIsChecked . fromSing $ gradient
    layoutType = forgetIsChecked . fromSing $ layout
    deviceType = forgetIsChecked . fromSing $ device
    dType = forgetIsChecked . fromSing $ dataType
    dims =
      fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
        . forgetIsChecked
        . fromSing
        $ shape

-- | Create a typed tensor filled with a given scalar value.
--
-- >>> full (-1) :: CPUParameter ('DataType 'Float) ('Shape '[])
-- Tensor Float [] -1.0000
-- >>> full (-1) :: CPUTensor ('DataType 'Int64) ('Shape '[ 'Dim ('Name "*") ('Size 1)])
-- Tensor Int64 [1] [-1]
full ::
  forall gradient layout device dataType shape input.
  (SingI gradient, SingI layout, SingI device, SingI dataType, SingI shape, Scalar input) =>
  input ->
  Tensor gradient layout device dataType shape
full = sFull (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @shape)

-- | Create a gradually typed random tensor.
sRandn ::
  forall gradient layout device dataType shape device'.
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SShape shape ->
  Generator device' ->
  (Tensor gradient layout (device <+> device') dataType shape, Generator (device <+> device'))
sRandn gradient layout device dataType shape =
  let opts = tensorOptions requiresGradient layoutType deviceType dType
   in withGenerator @device
        ( \genPtr -> do
            case (map dimName dims, map dimSize dims) of
              (names, sizes)
                | getAll . foldMap (\name -> All $ name == "*") $ names -> cast3 ATen.randn_lGo sizes genPtr opts
                | otherwise -> cast4 ATen.randn_lGNo sizes genPtr names opts
        )
  where
    requiresGradient = forgetIsChecked . fromSing $ gradient
    layoutType = forgetIsChecked . fromSing $ layout
    deviceType = forgetIsChecked . fromSing $ device
    dType = forgetIsChecked . fromSing $ dataType
    dims =
      fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
        . forgetIsChecked
        . fromSing
        $ shape

-- | Create typed random tensor.
randn ::
  forall gradient layout device dataType shape device'.
  (SingI gradient, SingI layout, SingI device, SingI dataType, SingI shape) =>
  Generator device' ->
  (Tensor gradient layout (device <+> device') dataType shape, Generator (device <+> device'))
randn = sRandn (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @shape)

-- | Create a gradually typed one-dimensional tensor of the numbers @0@ to @size -1@.
sArangeNaturals ::
  forall gradient layout device dataType size shape.
  shape ~ 'Shape '[ 'Dim ('Name "*") size] =>
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SSize size ->
  Tensor gradient layout device dataType shape
sArangeNaturals sGradient sLayout sDevice sDataType sSizeDim = UnsafeTensor tensor
  where
    tensor = unsafePerformIO $ cast2 ATen.arange_so size opts
    opts = tensorOptions requiresGradient layoutType deviceType dType

    requiresGradient = forgetIsChecked . fromSing $ sGradient
    layoutType = forgetIsChecked . fromSing $ sLayout
    deviceType = forgetIsChecked . fromSing $ sDevice
    dType = forgetIsChecked . fromSing $ sDataType
    size = forgetIsChecked . fromSing $ sSizeDim

-- | Create a typed one-dimensional tensor of the numbers @0@ to @size -1@.
arangeNaturals ::
  forall gradient layout device dataType size shape.
  ( shape ~ 'Shape '[ 'Dim ('Name "*") size],
    SingI gradient,
    SingI layout,
    SingI device,
    SingI dataType,
    SingI size
  ) =>
  Tensor gradient layout device dataType shape
arangeNaturals = sArangeNaturals (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @size)

-- | Create a gradually typed rectangular tensor with ones on the diagonal and zeros elsewhere.
sEye ::
  forall gradient layout device dataType rows cols shape.
  (shape ~ 'Shape '[ 'Dim ('Name "*") rows, 'Dim ('Name "*") cols]) =>
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SSize rows ->
  SSize cols ->
  Tensor gradient layout device dataType shape
sEye sGradient sLayout sDevice sDataType sRows sCols = UnsafeTensor tensor
  where
    tensor = unsafePerformIO $ cast3 ATen.eye_llo (fromInteger rows :: Int) (fromInteger cols :: Int) opts
    opts = tensorOptions requiresGradient layoutType deviceType dType

    requiresGradient = forgetIsChecked . fromSing $ sGradient
    layoutType = forgetIsChecked . fromSing $ sLayout
    deviceType = forgetIsChecked . fromSing $ sDevice
    dType = forgetIsChecked . fromSing $ sDataType
    rows = forgetIsChecked . fromSing $ sRows
    cols = forgetIsChecked . fromSing $ sCols

-- | Create a typed rectangular tensor with ones on the diagonal and zeros elsewhere.
eye ::
  forall gradient layout device dataType rows cols shape.
  ( shape ~ 'Shape '[ 'Dim ('Name "*") rows, 'Dim ('Name "*") cols],
    SingI gradient,
    SingI layout,
    SingI device,
    SingI dataType,
    SingI rows,
    SingI cols
  ) =>
  Tensor gradient layout device dataType shape
eye = sEye (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @rows) (sing @cols)

-- | Create a gradually typed square tensor with ones on the diagonal and zeros elsewhere.
sEyeSquare ::
  forall gradient layout device dataType size shape.
  shape ~ 'Shape '[ 'Dim ('Name "*") size, 'Dim ('Name "*") size] =>
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SSize size ->
  Tensor gradient layout device dataType shape
sEyeSquare sGradient sLayout sDevice sDataType sSize = UnsafeTensor tensor
  where
    tensor = unsafePerformIO $ cast2 ATen.eye_lo (fromInteger size :: Int) opts
    opts = tensorOptions requiresGradient layoutType deviceType dType

    requiresGradient = forgetIsChecked . fromSing $ sGradient
    layoutType = forgetIsChecked . fromSing $ sLayout
    deviceType = forgetIsChecked . fromSing $ sDevice
    dType = forgetIsChecked . fromSing $ sDataType
    size = forgetIsChecked . fromSing $ sSize

-- | Create a typed square tensor with ones on the diagonal and zeros elsewhere.
eyeSquare ::
  forall gradient layout device dataType size shape.
  ( shape ~ 'Shape '[ 'Dim ('Name "*") size, 'Dim ('Name "*") size],
    SingI gradient,
    SingI layout,
    SingI device,
    SingI dataType,
    SingI size
  ) =>
  Tensor gradient layout device dataType shape
eyeSquare = sEyeSquare (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @size)
