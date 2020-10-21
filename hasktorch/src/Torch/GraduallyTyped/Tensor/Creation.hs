{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.Tensor.Creation
  ( create,
    CreateF,
    CreateC,
    unCreate,
    ones,
    checkedOnes,
    uncheckedOnes,
    randn,
    uncheckedRandn,
  )
where

import Control.Monad.State.Strict (runState)
import Data.Int (Int16)
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType)
import Torch.GraduallyTyped.DType (DataType (..), KnownDType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType, KnownDeviceType, WithDeviceC (..))
import Torch.GraduallyTyped.Internal.TensorOptions (tensorOptions)
import Torch.GraduallyTyped.Layout (KnownLayoutType, Layout (..), LayoutType, WithLayoutC (..))
import Torch.GraduallyTyped.Prelude (KnownList)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (KnownRequiresGradient, RequiresGradient (..), requiresGradientVal)
import Torch.GraduallyTyped.Shape (Dim, DimType, Shape (..), WithShapeC (..), namedDims, sizedDims)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import Torch.Internal.Cast (cast2, cast3)
import qualified Torch.Internal.Managed.TensorFactories as ATen

-- $setup
-- >>> import Data.Int (Int16)
-- >>> import Torch.DType (DType (..))
-- >>> import Torch.GraduallyTyped.Device (DeviceType (..))
-- >>> import Torch.GraduallyTyped.Layout (LayoutType (..))
-- >>> import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
-- >>> import Torch.GraduallyTyped.Shape (Dim (..), DimType (..))

type CreateC
  (createOut :: Type)
  (requiresGradient :: RequiresGradient)
  (layout :: Layout LayoutType)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (shape :: Shape [Dim (DimType Symbol Nat)]) =
  ( KnownRequiresGradient requiresGradient,
    WithLayoutC layout (WithDeviceF device (WithDataTypeF dataType (WithShapeF shape createOut))),
    WithDeviceC device (WithDataTypeF dataType (WithShapeF shape createOut)),
    WithDataTypeC dataType (WithShapeF shape createOut),
    WithShapeC shape createOut
  )

type CreateF
  (createOut :: Type)
  (requiresGradient :: RequiresGradient)
  (layout :: Layout LayoutType)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (shape :: Shape [Dim (DimType Symbol Nat)]) =
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

create ::
  forall createOut requiresGradient layout device dataType shape.
  CreateC createOut requiresGradient layout device dataType shape =>
  ( RequiresGradient ->
    LayoutType ->
    DeviceType Int16 ->
    DType ->
    [DimType String Integer] ->
    createOut
  ) ->
  CreateF createOut requiresGradient layout device dataType shape
create go =
  withLayout @layout $
    \layoutType ->
      withDevice @device $
        \deviceType ->
          withDataType @dataType $
            \dType ->
              withShape @shape $
                \shape ->
                  go (requiresGradientVal @requiresGradient) layoutType deviceType dType shape

unCreate ::
  forall createOut requiresGradient layout device dataType shape.
  CreateC createOut requiresGradient layout device dataType shape =>
  CreateF createOut requiresGradient layout device dataType shape ->
  ( RequiresGradient ->
    LayoutType ->
    DeviceType Int16 ->
    DType ->
    [DimType String Integer] ->
    createOut
  )
unCreate go =
  \_requiresGradient layoutType deviceType dType shape ->
    ( withoutShape @shape
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
    )

-- | Create a tensor of ones.
--
-- >>> :type ones @'Dependent @'UncheckedLayout @'UncheckedDevice @'UncheckedDataType @'UncheckedShape
-- ones @'Dependent @'UncheckedLayout @'UncheckedDevice @'UncheckedDataType @'UncheckedShape
--   :: MonadFail m =>
--      LayoutType
--      -> DeviceType Int16
--      -> DType
--      -> [DimType String Integer]
--      -> m (Tensor
--              'Dependent
--              'UncheckedLayout
--              'UncheckedDevice
--              'UncheckedDataType
--              'UncheckedShape)
--
-- >>> :type ones @'Dependent @('Layout 'Dense) @'UncheckedDevice @'UncheckedDataType @'UncheckedShape
-- ones @'Dependent @('Layout 'Dense) @'UncheckedDevice @'UncheckedDataType @'UncheckedShape
--   :: MonadFail m =>
--      DeviceType Int16
--      -> DType
--      -> [DimType String Integer]
--      -> m (Tensor
--              'Dependent
--              ('Layout 'Dense)
--              'UncheckedDevice
--              'UncheckedDataType
--              'UncheckedShape)
--
-- >>> :type ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @'UncheckedDataType @'UncheckedShape
-- ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @'UncheckedDataType @'UncheckedShape
--   :: MonadFail m =>
--      DType
--      -> [DimType String Integer]
--      -> m (Tensor
--              'Dependent
--              ('Layout 'Dense)
--              ('Device ('CUDA 0))
--              'UncheckedDataType
--              'UncheckedShape)
--
-- >>> :type ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @'UncheckedShape
-- ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @'UncheckedShape
--   :: MonadFail m =>
--      [DimType String Integer]
--      -> m (Tensor
--              'Dependent
--              ('Layout 'Dense)
--              ('Device ('CUDA 0))
--              ('DataType 'Half)
--              'UncheckedShape)
--
-- >>> :type ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @('Shape '[ 'NamedSized "batch" 32, 'NamedSized "feature" 8])
-- ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @('Shape '[ 'NamedSized "batch" 32, 'NamedSized "feature" 8])
--   :: MonadFail m =>
--      m (Tensor
--           'Dependent
--           ('Layout 'Dense)
--           ('Device ('CUDA 0))
--           ('DataType 'Half)
--           ('Shape
--              '[ 'Dim ('NamedSized "batch" 32), 'Dim ('NamedSized "feature" 8)]))
ones ::
  forall requiresGradient layout device dataType shape.
  CreateC (Tensor requiresGradient layout device dataType shape) requiresGradient layout device dataType shape =>
  CreateF (Tensor requiresGradient layout device dataType shape) requiresGradient layout device dataType shape
ones =
  create
    @(Tensor requiresGradient layout device dataType shape)
    @requiresGradient
    @layout
    @device
    @dataType
    @shape
    go
  where
    go requiresGradient layoutType deviceType dType shape =
      let opts = tensorOptions requiresGradient layoutType deviceType dType
          tensor = unsafePerformIO $ case (namedDims shape, sizedDims shape) of
            (Just names, Just sizes) -> cast3 ATen.ones_lNo sizes names opts
            (Nothing, Just sizes) -> cast2 ATen.ones_lo sizes opts
            _ -> fail $ "Invalid tensor shape specification " <> show shape <> "."
       in UnsafeTensor tensor

checkedOnes ::
  forall layoutType deviceType dType dims.
  ( KnownLayoutType layoutType,
    KnownDeviceType deviceType,
    KnownDType dType,
    WithShapeC ( 'Shape dims) (Tensor 'Dependent ( 'Layout layoutType) ( 'Device deviceType) ( 'DataType dType) ( 'Shape dims))
  ) =>
  WithShapeF
    ( 'Shape dims)
    ( Tensor
        'Dependent
        ( 'Layout layoutType)
        ( 'Device deviceType)
        ( 'DataType dType)
        ( 'Shape dims)
    )
checkedOnes = ones @ 'Dependent @( 'Layout layoutType) @( 'Device deviceType) @( 'DataType dType) @( 'Shape dims)

-- | Like 'ones', but specialized to the case in which all arguments are unchecked at compile time.
uncheckedOnes ::
  -- | Memory layout of the tensor.
  LayoutType ->
  -- | Compute device of the tensor.
  DeviceType Int16 ->
  -- | Data type of the tensor.
  DType ->
  -- | Shape of the tensor.
  [DimType String Integer] ->
  -- | Returned tensor.
  Tensor
    'Dependent
    'UncheckedLayout
    'UncheckedDevice
    'UncheckedDataType
    'UncheckedShape
uncheckedOnes = ones @ 'Dependent @ 'UncheckedLayout @ 'UncheckedDevice @ 'UncheckedDataType @ 'UncheckedShape

randn ::
  forall requiresGradient layout device dataType shape.
  CreateC (Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape =>
  CreateF (Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
randn = create @(Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) @requiresGradient @layout @device @dataType @shape go
  where
    go requiresGradient layoutType deviceType dType shape = runState $ do
      opts <- pure $ tensorOptions requiresGradient layoutType deviceType dType
      tensor <- pure . unsafePerformIO $ case (namedDims shape, sizedDims shape) of
        (Just names, Just sizes) -> pure . unsafePerformIO $ cast3 ATen.randn_lNo sizes names opts
        (Nothing, Just sizes) -> pure . unsafePerformIO $ cast2 ATen.randn_lo sizes opts
        _ -> fail $ "Invalid tensor shape specification " <> show shape <> "."
      return $ UnsafeTensor tensor

uncheckedRandn ::
  LayoutType ->
  DeviceType Int16 ->
  DType ->
  [DimType String Integer] ->
  Generator 'UncheckedDevice ->
  (Tensor 'Dependent 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape, Generator 'UncheckedDevice)
uncheckedRandn = randn @ 'Dependent @ 'UncheckedLayout @ 'UncheckedDevice @ 'UncheckedDataType @ 'UncheckedShape
