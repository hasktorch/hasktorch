{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

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

import Control.Monad.Catch (MonadThrow)
import Data.Monoid (All (..))
import Data.Singletons (SingI (..), SingKind (fromSing))
import Torch.GraduallyTyped.DType (SDataType)
import Torch.GraduallyTyped.Device (SDevice)
import Torch.GraduallyTyped.Internal.TensorOptions (tensorDims, tensorOptions)
import Torch.GraduallyTyped.Layout (SLayout (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (Generator (..), SGetGeneratorDevice, sCreateWithGenerator)
import Torch.GraduallyTyped.RequiresGradient (SGradient)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SSize, Shape (..), dimName, dimSize)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), TensorSpec (..))
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.Internal.Cast (cast2, cast3, cast4)
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.TensorFactories as ATen

-- $setup
-- >>> import Data.Int (Int16)
-- >>> import Torch.GraduallyTyped.Prelude.List (SList (..))
-- >>> import Torch.GraduallyTyped

-- | Create a gradually typed tensor of ones.
--
-- >>> shape = SShape $ SName @"batch" :&: SSize @32 :|: SUncheckedName "feature" :&: SUncheckedSize 8 :|: SNil
-- >>> :type sOnes $ TensorSpec (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape
-- sOnes $ TensorSpec (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape
--   :: MonadThrow m =>
--      m (Tensor
--           ('Gradient WithoutGradient)
--           ('Layout Dense)
--           ('Device CPU)
--           ('DataType Int64)
--           ('Shape
--              ['Dim ('Name "batch") ('Size 32),
--               'Dim UncheckedName UncheckedSize]))
sOnes ::
  forall gradient layout device dataType shape m.
  MonadThrow m =>
  TensorSpec gradient layout device dataType shape ->
  m (Tensor gradient layout device dataType shape)
sOnes TensorSpec {..} = unsafeThrowableIO $ do
  let opts = tensorOptions tsGradient tsLayout tsDevice tsDataType
      dims = tensorDims tsShape
  tPtr <- case (map dimName dims, map dimSize dims) of
    (names, sizes)
      | getAll . foldMap (All . (== "*")) $ names -> cast2 ATen.ones_lo sizes opts
      | otherwise -> cast3 ATen.ones_lNo sizes names opts
  pure . UnsafeTensor $ tPtr

-- | Create a typed tensor of ones.
--
-- >>> ones :: IO (CPUParameter ('DataType 'Float) ('Shape '[]))
-- Tensor Float []  1.0000
-- >>> ones :: IO (CPUTensor ('DataType 'Int64) ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
-- Tensor Int64 [1] [ 1]
ones ::
  forall gradient layout device dataType shape m.
  MonadThrow m =>
  (SingI gradient, SingI layout, SingI device, SingI dataType, SingI shape) =>
  m (Tensor gradient layout device dataType shape)
ones = sOnes $ TensorSpec (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @shape)

-- | Create a gradually typed tensor of zeros.
--
-- >>> shape = SShape $ SName @"batch" :&: SSize @32 :|: SUncheckedName "feature" :&: SUncheckedSize 8 :|: SNil
-- >>> :type sZeros $ TensorSpec (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape
-- sZeros $ TensorSpec (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape
--   :: MonadThrow m =>
--      m (Tensor
--           ('Gradient WithoutGradient)
--           ('Layout Dense)
--           ('Device CPU)
--           ('DataType Int64)
--           ('Shape
--              ['Dim ('Name "batch") ('Size 32),
--               'Dim UncheckedName UncheckedSize]))
sZeros ::
  forall gradient layout device dataType shape m.
  MonadThrow m =>
  TensorSpec gradient layout device dataType shape ->
  m (Tensor gradient layout device dataType shape)
sZeros TensorSpec {..} = unsafeThrowableIO $ do
  let opts = tensorOptions tsGradient tsLayout tsDevice tsDataType
      dims = tensorDims tsShape
  tPtr <- case (map dimName dims, map dimSize dims) of
    (names, sizes)
      | getAll . foldMap (All . (== "*")) $ names -> cast2 ATen.zeros_lo sizes opts
      | otherwise -> cast3 ATen.zeros_lNo sizes names opts
  pure . UnsafeTensor $ tPtr

-- | Create a typed tensor of zeros.
--
-- >>> zeros :: IO (CPUParameter ('DataType 'Float) ('Shape '[]))
-- Tensor Float []  0.0000
-- >>> zeros :: IO (CPUTensor ('DataType 'Int64) ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
-- Tensor Int64 [1] [ 0]
zeros ::
  forall gradient layout device dataType shape m.
  MonadThrow m =>
  (SingI gradient, SingI layout, SingI device, SingI dataType, SingI shape) =>
  m (Tensor gradient layout device dataType shape)
zeros = sZeros $ TensorSpec (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @shape)

-- | Create a gradually typed tensor filled with a given scalar value.
--
-- >>> shape = SShape $ SName @"batch" :&: SSize @32 :|: SUncheckedName "feature" :&: SUncheckedSize 8 :|: SNil
-- >>> input = -1
-- >>> :type sFull (TensorSpec (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape) input
-- sFull (TensorSpec (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt64) shape) input
--   :: MonadThrow m =>
--      m (Tensor
--           ('Gradient WithoutGradient)
--           ('Layout Dense)
--           ('Device CPU)
--           ('DataType Int64)
--           ('Shape
--              ['Dim ('Name "batch") ('Size 32),
--               'Dim UncheckedName UncheckedSize]))
sFull ::
  forall gradient layout device dataType shape input m.
  (MonadThrow m, Scalar input) =>
  TensorSpec gradient layout device dataType shape ->
  input ->
  m (Tensor gradient layout device dataType shape)
sFull TensorSpec {..} input = unsafeThrowableIO $ do
  let opts = tensorOptions tsGradient tsLayout tsDevice tsDataType
      dims = tensorDims tsShape
  tPtr <- case (dimName <$> dims, dimSize <$> dims) of
    (names, sizes)
      | getAll . foldMap (\name -> All $ name == "*") $ names -> cast3 ATen.full_lso sizes input opts
      | otherwise -> cast4 ATen.full_lsNo sizes input names opts
  pure . UnsafeTensor $ tPtr

-- | Create a typed tensor filled with a given scalar value.
--
-- >>> full (-1) :: IO (CPUParameter ('DataType 'Float) ('Shape '[]))
-- Tensor Float [] -1.0000
-- >>> full (-1) :: IO (CPUTensor ('DataType 'Int64) ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
-- Tensor Int64 [1] [-1]
full ::
  forall gradient layout device dataType shape input m.
  (MonadThrow m, SingI gradient, SingI layout, SingI device, SingI dataType, SingI shape, Scalar input) =>
  input ->
  m (Tensor gradient layout device dataType shape)
full = sFull $ TensorSpec (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @shape)

-- | Create a gradually typed random tensor.
sRandn ::
  forall gradient layout device dataType shape generatorDevice m.
  (SGetGeneratorDevice generatorDevice, MonadThrow m) =>
  TensorSpec gradient layout device dataType shape ->
  Generator generatorDevice ->
  m (Tensor gradient layout (device <+> generatorDevice) dataType shape, Generator (device <+> generatorDevice))
sRandn tSpec generator =
  sCreateWithGenerator tSpec generator $
    \opts dims genPtr ->
      case (map dimName dims, map dimSize dims) of
        (names, sizes)
          | getAll . foldMap (\name -> All $ name == "*") $ names -> cast3 ATen.randn_lGo sizes genPtr opts
          | otherwise -> cast4 ATen.randn_lGNo sizes genPtr names opts

-- | Create typed random tensor.
randn ::
  forall gradient layout device dataType shape generatorDevice m.
  (SGetGeneratorDevice generatorDevice, MonadThrow m) =>
  (SingI gradient, SingI layout, SingI device, SingI dataType, SingI shape) =>
  Generator generatorDevice ->
  m (Tensor gradient layout (device <+> generatorDevice) dataType shape, Generator (device <+> generatorDevice))
randn = sRandn $ TensorSpec (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @shape)

-- | Create a gradually typed one-dimensional tensor of the numbers @0@ to @size -1@.
sArangeNaturals ::
  forall m gradient layout device dataType size shape.
  (MonadThrow m, shape ~ 'Shape '[ 'Dim ('Name "*") size]) =>
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SSize size ->
  m (Tensor gradient layout device dataType shape)
sArangeNaturals gradient layout device dataType size = unsafeThrowableIO $ do
  let opts = tensorOptions gradient layout device dataType
      size' = forgetIsChecked . fromSing $ size
  tPtr <- cast2 ATen.arange_so size' opts
  pure . UnsafeTensor $ tPtr

-- | Create a typed one-dimensional tensor of the numbers @0@ to @size -1@.
arangeNaturals ::
  forall size gradient layout device dataType shape m.
  ( MonadThrow m,
    shape ~ 'Shape '[ 'Dim ('Name "*") size],
    SingI gradient,
    SingI layout,
    SingI device,
    SingI dataType,
    SingI size
  ) =>
  m (Tensor gradient layout device dataType shape)
arangeNaturals = sArangeNaturals (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @size)

-- | Create a gradually typed rectangular tensor with ones on the diagonal and zeros elsewhere.
sEye ::
  forall gradient layout device dataType rows cols shape m.
  (MonadThrow m, shape ~ 'Shape '[ 'Dim ('Name "*") rows, 'Dim ('Name "*") cols]) =>
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SSize rows ->
  SSize cols ->
  m (Tensor gradient layout device dataType shape)
sEye gradient layout device dataType rows cols = unsafeThrowableIO $ do
  let opts = tensorOptions gradient layout device dataType
      rows' :: Int = fromInteger . forgetIsChecked . fromSing $ rows
      cols' :: Int = fromInteger . forgetIsChecked . fromSing $ cols
  tPtr <- cast3 ATen.eye_llo rows' cols' opts
  pure . UnsafeTensor $ tPtr

-- | Create a typed rectangular tensor with ones on the diagonal and zeros elsewhere.
eye ::
  forall rows cols gradient layout device dataType shape m.
  ( MonadThrow m,
    shape ~ 'Shape '[ 'Dim ('Name "*") rows, 'Dim ('Name "*") cols],
    SingI gradient,
    SingI layout,
    SingI device,
    SingI dataType,
    SingI rows,
    SingI cols
  ) =>
  m (Tensor gradient layout device dataType shape)
eye = sEye (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @rows) (sing @cols)

-- | Create a gradually typed square tensor with ones on the diagonal and zeros elsewhere.
sEyeSquare ::
  forall gradient layout device dataType size shape m.
  (MonadThrow m, shape ~ 'Shape '[ 'Dim ('Name "*") size, 'Dim ('Name "*") size]) =>
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SSize size ->
  m (Tensor gradient layout device dataType shape)
sEyeSquare gradient layout device dataType size = unsafeThrowableIO $ do
  let opts = tensorOptions gradient layout device dataType
      size' :: Int = fromInteger . forgetIsChecked . fromSing $ size
  tPtr <- cast2 ATen.eye_lo size' opts
  pure . UnsafeTensor $ tPtr

-- | Create a typed square tensor with ones on the diagonal and zeros elsewhere.
eyeSquare ::
  forall size gradient layout device dataType shape m.
  ( MonadThrow m,
    shape ~ 'Shape '[ 'Dim ('Name "*") size, 'Dim ('Name "*") size],
    SingI gradient,
    SingI layout,
    SingI device,
    SingI dataType,
    SingI size
  ) =>
  m (Tensor gradient layout device dataType shape)
eyeSquare = sEyeSquare (sing @gradient) (sing @layout) (sing @device) (sing @dataType) (sing @size)
