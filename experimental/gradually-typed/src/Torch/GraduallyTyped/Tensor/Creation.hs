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

module Torch.GraduallyTyped.Tensor.Creation
  ( WithCreateC (..),
    ones,
    sOnes,
    -- checkedOnes,
    uncheckedOnes,
    zeros,
    full,
    randn,
    -- checkedRandn,
    uncheckedRandn,
    arangeNaturals,
    eye,
    eyeSquare,
  )
where

import Data.Int (Int16)
import Data.Kind (Type)
import Data.Monoid (All (..))
import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDType, SDataType, WithDataTypeC (..), sDType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDeviceType, SDevice, WithDeviceC (..), sDeviceType)
import Torch.GraduallyTyped.Internal.TensorOptions (tensorOptions)
import Torch.GraduallyTyped.Internal.Void (Void)
import Torch.GraduallyTyped.Layout (KnownLayoutType (layoutTypeVal), Layout (..), LayoutType (..), SLayout (..), WithLayoutC (..), sLayoutType)
import Torch.GraduallyTyped.Prelude (Catch)
import Torch.GraduallyTyped.Random (Generator, withGenerator)
import Torch.GraduallyTyped.RequiresGradient (KnownRequiresGradient, RequiresGradient (..), requiresGradientVal, SRequiresGradient, sRequiresGradient)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SShape, Shape (..), Size (..), WithDimC (..), WithShapeC (..), dimName, dimSize, sShape)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import Torch.Internal.Cast (cast2, cast3, cast4)
import qualified Torch.Internal.Managed.TensorFactories as ATen

-- $setup
-- >>> import Data.Int (Int16)
-- >>> import Torch.DType (DType (..))
-- >>> import Torch.GraduallyTyped.DType (SDataType (..))
-- >>> import Torch.GraduallyTyped.Device (DeviceType (..), SDevice (..))
-- >>> import Torch.GraduallyTyped.Layout (LayoutType (..), SLayout (..))
-- >>> import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), SRequiresGradient (..))
-- >>> import Torch.GraduallyTyped.Shape (Dim (..), SDims (..), SName (..), SShape (..), SSize (..), pattern (:&:), pattern (:|:))

class
  WithCreateC
    (createOut :: Type)
    (requiresGradient :: RequiresGradient)
    (layout :: Layout LayoutType)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (shape :: Shape [Dim (Name Symbol) (Size Nat)])
  where
  type
    WithCreateF createOut requiresGradient layout device dataType shape ::
      Type
  withCreate ::
    ( RequiresGradient ->
      LayoutType ->
      DeviceType Int16 ->
      DType ->
      [Dim String Integer] ->
      createOut
    ) ->
    WithCreateF createOut requiresGradient layout device dataType shape
  withoutCreate ::
    WithCreateF createOut requiresGradient layout device dataType shape ->
    ( RequiresGradient ->
      LayoutType ->
      DeviceType Int16 ->
      DType ->
      [Dim String Integer] ->
      createOut
    )

-- | Auxiliary instance to make 'WithCreateC' opaque.
instance {-# OVERLAPPING #-} WithCreateC Void requiresGradient layout device dataType shape where
  type
    WithCreateF Void requiresGradient layout device dataType shape =
      WithLayoutF
        layout
        ( WithDeviceF
            device
            ( WithDataTypeF
                dataType
                ( WithShapeF
                    shape
                    Void
                )
            )
        )
  withCreate = undefined
  withoutCreate = undefined

-- | Catch-all instance.
instance
  ( KnownRequiresGradient requiresGradient,
    WithLayoutC layout (WithDeviceF device (WithDataTypeF dataType (WithShapeF shape createOut))),
    WithDeviceC device (WithDataTypeF dataType (WithShapeF shape createOut)),
    WithDataTypeC dataType (WithShapeF shape createOut),
    WithShapeC shape createOut
  ) =>
  WithCreateC createOut requiresGradient layout device dataType shape
  where
  type
    WithCreateF createOut requiresGradient layout device dataType shape =
      WithLayoutF
        layout
        ( WithDeviceF
            device
            ( WithDataTypeF
                dataType
                ( WithShapeF
                    shape
                    createOut
                )
            )
        )
  withCreate go =
    withLayout @layout $
      \layoutType ->
        withDevice @device $
          \deviceType ->
            withDataType @dataType $
              \dType ->
                withShape @shape $
                  \shape ->
                    go (requiresGradientVal @requiresGradient) layoutType deviceType dType shape
  withoutCreate go =
    \_requiresGradient layoutType deviceType dType shape ->
      withoutShape @shape
        ( withoutDataType @dataType
            ( withoutDevice @device
                ( withoutLayout @layout
                    go
                    layoutType
                )
                deviceType
            )
            dType
        )
        shape

-- | Create a tensor of ones.
--
-- >>> :type ones @'WithoutGradient @'UncheckedLayout @'UncheckedDevice @'UncheckedDataType @'UncheckedShape
-- ones @'WithoutGradient @'UncheckedLayout @'UncheckedDevice @'UncheckedDataType @'UncheckedShape
--   :: LayoutType
--      -> DeviceType Int16
--      -> DType
--      -> [Dim String Integer]
--      -> Tensor
--           'WithoutGradient
--           'UncheckedLayout
--           'UncheckedDevice
--           'UncheckedDataType
--           'UncheckedShape
-- >>> :type ones @'WithoutGradient @('Layout 'Dense) @'UncheckedDevice @'UncheckedDataType @'UncheckedShape
-- ones @'WithoutGradient @('Layout 'Dense) @'UncheckedDevice @'UncheckedDataType @'UncheckedShape
--   :: DeviceType Int16
--      -> DType
--      -> [Dim String Integer]
--      -> Tensor
--           'WithoutGradient
--           ('Layout 'Dense)
--           'UncheckedDevice
--           'UncheckedDataType
--           'UncheckedShape
-- >>> :type ones @'WithoutGradient @('Layout 'Dense) @('Device ('CUDA 0)) @'UncheckedDataType @'UncheckedShape
-- ones @'WithoutGradient @('Layout 'Dense) @('Device ('CUDA 0)) @'UncheckedDataType @'UncheckedShape
--   :: DType
--      -> [Dim String Integer]
--      -> Tensor
--           'WithoutGradient
--           ('Layout 'Dense)
--           ('Device ('CUDA 0))
--           'UncheckedDataType
--           'UncheckedShape
-- >>> :type ones @'WithoutGradient @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @'UncheckedShape
-- ones @'WithoutGradient @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @'UncheckedShape
--   :: [Dim String Integer]
--      -> Tensor
--           'WithoutGradient
--           ('Layout 'Dense)
--           ('Device ('CUDA 0))
--           ('DataType 'Half)
--           'UncheckedShape
-- >>> :type ones @'WithoutGradient @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- ones @'WithoutGradient @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
--   :: Tensor
--        'WithoutGradient
--        ('Layout 'Dense)
--        ('Device ('CUDA 0))
--        ('DataType 'Half)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
ones ::
  forall requiresGradient layout device dataType shape.
  WithCreateC (Tensor requiresGradient layout device dataType shape) requiresGradient layout device dataType shape =>
  WithCreateF (Tensor requiresGradient layout device dataType shape) requiresGradient layout device dataType shape
ones =
  withCreate
    @(Tensor requiresGradient layout device dataType shape)
    @requiresGradient
    @layout
    @device
    @dataType
    @shape
    go
  where
    go requiresGradient layoutType deviceType dType dims =
      let opts = tensorOptions requiresGradient layoutType deviceType dType
          tensor = unsafePerformIO $ case (map dimName dims, map dimSize dims) of
            (names, sizes)
              | getAll . foldMap (\name -> All $ name == "*") $ names -> cast2 ATen.ones_lo sizes opts
              | otherwise -> cast3 ATen.ones_lNo sizes names opts
       in UnsafeTensor tensor

-- | Create tensor of ones.
--
-- >>> shape = SShape $ SName @"batch" :&: SSize @32 :|: SUncheckedName "feature" :&: SUncheckedSize 8 :|: SDimsNil
-- >>> :type sOnes SWithoutGradient (SLayout @Dense) (SDevice @CPU) (SDataType @Int64) shape
-- sOnes SWithoutGradient (SLayout @Dense) (SDevice @CPU) (SDataType @Int64) shape
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
          | getAll . foldMap (\name -> All $ name == "*") $ names -> cast2 ATen.ones_lo sizes opts
          | otherwise -> cast3 ATen.ones_lNo sizes names opts
   in UnsafeTensor tensor
  where
    requiresGradient = sRequiresGradient reqGradient
    layoutType = sLayoutType layout
    deviceType = sDeviceType device
    dType = sDType dataType
    dims = sShape shape

-- checkedOnes ::
--   forall requiresGradient layoutType deviceType dType dims.
--   ( KnownRequiresGradient requiresGradient,
--     KnownLayoutType layoutType,
--     KnownDeviceType deviceType,
--     KnownDType dType,
--     WithShapeC ( 'Shape dims) (Tensor requiresGradient ( 'Layout layoutType) ( 'Device deviceType) ( 'DataType dType) ( 'Shape dims))
--   ) =>
--   WithShapeF
--     ( 'Shape dims)
--     ( Tensor
--         requiresGradient
--         ( 'Layout layoutType)
--         ( 'Device deviceType)
--         ( 'DataType dType)
--         ( 'Shape dims)
--     )
-- checkedOnes = ones @requiresGradient @( 'Layout layoutType) @( 'Device deviceType) @( 'DataType dType) @( 'Shape dims)

-- | Like 'ones', but specialized to the case in which all arguments are unchecked at compile time.
uncheckedOnes ::
  -- | Memory layout of the tensor.
  LayoutType ->
  -- | Compute device of the tensor.
  DeviceType Int16 ->
  -- | Data type of the tensor.
  DType ->
  -- | Shape of the tensor.
  [Dim String Integer] ->
  -- | Returned tensor.
  Tensor
    'WithoutGradient
    'UncheckedLayout
    'UncheckedDevice
    'UncheckedDataType
    'UncheckedShape
uncheckedOnes = ones @'WithoutGradient @'UncheckedLayout @'UncheckedDevice @'UncheckedDataType @'UncheckedShape

zeros ::
  forall requiresGradient layout device dataType shape.
  WithCreateC (Tensor requiresGradient layout device dataType shape) requiresGradient layout device dataType shape =>
  WithCreateF (Tensor requiresGradient layout device dataType shape) requiresGradient layout device dataType shape
zeros =
  withCreate
    @(Tensor requiresGradient layout device dataType shape)
    @requiresGradient
    @layout
    @device
    @dataType
    @shape
    go
  where
    go requiresGradient layoutType deviceType dType dims =
      let opts = tensorOptions requiresGradient layoutType deviceType dType
          tensor = unsafePerformIO $ case (map dimName dims, map dimSize dims) of
            (names, sizes)
              | getAll . foldMap (\name -> All $ name == "*") $ names -> cast2 ATen.zeros_lo sizes opts
              | otherwise -> cast3 ATen.zeros_lNo sizes names opts
       in UnsafeTensor tensor

full ::
  forall requiresGradient layout device dataType shape input.
  ( Scalar input,
    WithCreateC (input -> Tensor requiresGradient layout device dataType shape) requiresGradient layout device dataType shape
  ) =>
  WithCreateF (input -> Tensor requiresGradient layout device dataType shape) requiresGradient layout device dataType shape
full =
  withCreate
    @(input -> Tensor requiresGradient layout device dataType shape)
    @requiresGradient
    @layout
    @device
    @dataType
    @shape
    go
  where
    go requiresGradient layoutType deviceType dType dims input =
      let opts = tensorOptions requiresGradient layoutType deviceType dType
          tensor = unsafePerformIO $ case (map dimName dims, map dimSize dims) of
            (names, sizes)
              | getAll . foldMap (\name -> All $ name == "*") $ names -> cast3 ATen.full_lso sizes input opts
              | otherwise -> cast4 ATen.full_lsNo sizes input names opts
       in UnsafeTensor tensor

randn ::
  forall requiresGradient layout device dataType shape device'.
  ( WithCreateC (Generator device' -> (Tensor requiresGradient layout device dataType shape, Generator device')) requiresGradient layout device dataType shape
  ) =>
  WithCreateF (Generator device' -> (Tensor requiresGradient layout device dataType shape, Generator device')) requiresGradient layout device dataType shape
randn = withCreate @(Generator device' -> (Tensor requiresGradient layout device dataType shape, Generator device')) @requiresGradient @layout @device @dataType @shape go
  where
    go requiresGradient layoutType deviceType dType shape =
      let opts = tensorOptions requiresGradient layoutType deviceType dType
       in withGenerator
            ( \genPtr -> do
                tensor <- case (map dimName shape, map dimSize shape) of
                  (names, sizes)
                    | getAll . foldMap (\name -> All $ name == "*") $ names -> cast3 ATen.randn_lGo sizes genPtr opts
                    | otherwise -> cast4 ATen.randn_lGNo sizes genPtr names opts
                pure $ UnsafeTensor tensor
            )
            ( unsafePerformIO $ do
                tensor <- case (map dimName shape, map dimSize shape) of
                  (names, sizes)
                    | getAll . foldMap (\name -> All $ name == "*") $ names -> cast2 ATen.zeros_lo sizes opts
                    | otherwise -> cast3 ATen.zeros_lNo sizes names opts
                pure $ UnsafeTensor tensor
            )

-- checkedRandn ::
--   forall requiresGradient layoutType deviceType dType dims.
--   ( KnownRequiresGradient requiresGradient,
--     KnownLayoutType layoutType,
--     KnownDeviceType deviceType,
--     KnownDType dType,
--     Catch deviceType,
--     WithShapeC ( 'Shape dims) (Generator ( 'Device deviceType) -> (Tensor requiresGradient ( 'Layout layoutType) ( 'Device deviceType) ( 'DataType dType) ( 'Shape dims), Generator ( 'Device deviceType)))
--   ) =>
--   ( WithShapeF
--       ( 'Shape dims)
--       ( Generator ( 'Device deviceType) ->
--         ( Tensor
--             requiresGradient
--             ( 'Layout layoutType)
--             ( 'Device deviceType)
--             ( 'DataType dType)
--             ( 'Shape dims),
--           Generator ( 'Device deviceType)
--         )
--       )
--   )
-- checkedRandn = randn @requiresGradient @( 'Layout layoutType) @( 'Device deviceType) @( 'DataType dType) @( 'Shape dims) @( 'Device deviceType)

uncheckedRandn ::
  -- | Memory layout of the tensor.
  LayoutType ->
  -- | Compute device of the tensor.
  DeviceType Int16 ->
  -- | Data type of the tensor.
  DType ->
  -- | Shape of the tensor.
  [Dim String Integer] ->
  -- | Random number generator.
  Generator 'UncheckedDevice ->
  -- | Returned tensor and generator.
  ( Tensor
      'WithoutGradient
      'UncheckedLayout
      'UncheckedDevice
      'UncheckedDataType
      'UncheckedShape,
    Generator 'UncheckedDevice
  )
uncheckedRandn = randn @'WithoutGradient @'UncheckedLayout @'UncheckedDevice @'UncheckedDataType @'UncheckedShape

arangeNaturals ::
  forall requiresGradient layout device dataType sizeDim shape createOut.
  ( shape ~ 'Shape '[sizeDim],
    createOut ~ Tensor requiresGradient layout device dataType shape,
    KnownRequiresGradient requiresGradient,
    WithLayoutC layout (WithDeviceF device (WithDataTypeF dataType (WithDimF sizeDim createOut))),
    WithDeviceC device (WithDataTypeF dataType (WithDimF sizeDim createOut)),
    WithDataTypeC dataType (WithDimF sizeDim createOut),
    WithDimC sizeDim createOut
  ) =>
  WithLayoutF layout (WithDeviceF device (WithDataTypeF dataType (WithDimF sizeDim createOut)))
arangeNaturals =
  withLayout @layout $
    \layoutType ->
      withDevice @device $
        \deviceType ->
          withDataType @dataType $
            \dType ->
              withDim @sizeDim @createOut $
                \sizeDim ->
                  go (requiresGradientVal @requiresGradient) layoutType deviceType dType sizeDim
  where
    go requiresGradient layoutType deviceType dType sizeDim =
      let opts = tensorOptions requiresGradient layoutType deviceType dType
          Dim _ size = sizeDim
          tensor = unsafePerformIO $ do
            t <- cast2 ATen.arange_so size opts
            -- renaming of dimension goes here
            pure t
       in UnsafeTensor tensor

eye ::
  forall requiresGradient layout device dataType rowsDim colsDim shape createOut.
  ( shape ~ 'Shape '[rowsDim, colsDim],
    createOut ~ Tensor requiresGradient layout device dataType shape,
    KnownRequiresGradient requiresGradient,
    WithLayoutC layout (WithDeviceF device (WithDataTypeF dataType (WithDimF rowsDim (WithDimF colsDim createOut)))),
    WithDeviceC device (WithDataTypeF dataType (WithDimF rowsDim (WithDimF colsDim createOut))),
    WithDataTypeC dataType (WithDimF rowsDim (WithDimF colsDim createOut)),
    WithDimC rowsDim (WithDimF colsDim createOut),
    WithDimC colsDim createOut
  ) =>
  WithLayoutF layout (WithDeviceF device (WithDataTypeF dataType (WithDimF rowsDim (WithDimF colsDim createOut))))
eye =
  withLayout @layout $
    \layoutType ->
      withDevice @device $
        \deviceType ->
          withDataType @dataType $
            \dType ->
              withDim @rowsDim $
                \rowsDim ->
                  withDim @colsDim @createOut $
                    \colsDim ->
                      go (requiresGradientVal @requiresGradient) layoutType deviceType dType rowsDim colsDim
  where
    go requiresGradient layoutType deviceType dType rowsDim colsDim =
      let opts = tensorOptions requiresGradient layoutType deviceType dType
          Dim _ rows = rowsDim
          Dim _ cols = colsDim
          tensor = unsafePerformIO $ cast3 ATen.eye_llo (fromInteger rows :: Int) (fromInteger cols :: Int) opts
       in UnsafeTensor tensor

eyeSquare ::
  forall requiresGradient layout device dataType sizeDim shape createOut.
  ( shape ~ 'Shape '[sizeDim],
    createOut ~ Tensor requiresGradient layout device dataType shape,
    KnownRequiresGradient requiresGradient,
    WithLayoutC layout (WithDeviceF device (WithDataTypeF dataType (WithDimF sizeDim createOut))),
    WithDeviceC device (WithDataTypeF dataType (WithDimF sizeDim createOut)),
    WithDataTypeC dataType (WithDimF sizeDim createOut),
    WithDimC sizeDim createOut
  ) =>
  WithLayoutF layout (WithDeviceF device (WithDataTypeF dataType (WithDimF sizeDim createOut)))
eyeSquare =
  withLayout @layout $
    \layoutType ->
      withDevice @device $
        \deviceType ->
          withDataType @dataType $
            \dType ->
              withDim @sizeDim @createOut $
                \sizeDim ->
                  go (requiresGradientVal @requiresGradient) layoutType deviceType dType sizeDim
  where
    go requiresGradient layoutType deviceType dType sizeDim =
      let opts = tensorOptions requiresGradient layoutType deviceType dType
          Dim _ size = sizeDim
          tensor = unsafePerformIO $ cast2 ATen.eye_lo (fromInteger size :: Int) opts
       in UnsafeTensor tensor
