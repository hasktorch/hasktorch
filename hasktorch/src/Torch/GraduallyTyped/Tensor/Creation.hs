{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.Tensor.Creation where

import Data.Type.Equality (type (==))
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.DType (DataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), WithDeviceC (..))
import Torch.GraduallyTyped.Internal.TensorOptions (tensorOptions)
import Torch.GraduallyTyped.Layout (Layout (..), WithLayoutC (..))
import Torch.GraduallyTyped.RequiresGradient (KnownRequiresGradient, requiresGradientVal)
import Torch.GraduallyTyped.Shape (Shape (..), WidenShapeF, WithShapeC (..), namedDims, sizedDims)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import Torch.Internal.Cast (cast2, cast3)
import qualified Torch.Internal.Managed.TensorFactories as ATen

-- $setup
-- >>> import Torch.DType (DType (..))
-- >>> import Torch.GraduallyTyped.Device (DeviceType (..))
-- >>> import Torch.GraduallyTyped.Layout (LayoutType (..))
-- >>> import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
-- >>> import Torch.GraduallyTyped.Shape (Dim (..))

-- | Create a tensor of ones.
--
-- >>> :type ones @'Dependent @'UncheckedLayout @'UncheckedDevice @'UncheckedDataType @'UncheckedShape
-- ones @'Dependent @'UncheckedLayout @'UncheckedDevice @'UncheckedDataType @'UncheckedShape
--   :: MonadFail m =>
--      LayoutType
--      -> DeviceType GHC.Int.Int16
--      -> DType
--      -> [Dim String Integer]
--      -> m (Tensor
--              'Dependent 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape)
--
-- >>> :type ones @'Dependent @('Layout 'Dense) @'UncheckedDevice @'UncheckedDataType @'UncheckedShape
-- ones @'Dependent @('Layout 'Dense) @'UncheckedDevice @'UncheckedDataType @'UncheckedShape
--   :: MonadFail m =>
--      DeviceType GHC.Int.Int16
--      -> DType
--      -> [Dim String Integer]
--      -> m (Tensor
--              'Dependent ('Layout 'Dense) 'UncheckedDevice 'UncheckedDataType 'UncheckedShape)
--
-- >>> :type ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @'UncheckedDataType @'UncheckedShape
-- ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @'UncheckedDataType @'UncheckedShape
--   :: MonadFail m =>
--      DType
--      -> [Dim String Integer]
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
--      [Dim String Integer]
--      -> m (Tensor
--              'Dependent
--              ('Layout 'Dense)
--              ('Device ('CUDA 0))
--              ('DataType 'Half)
--              'UncheckedShape)
--
-- >>> :type ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @('Shape '[ 'Dim ( 'NamedSized "batch" 32), 'Dim ( 'NamedSized "feature" 8)])
-- ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @('Shape '[ 'Dim ( 'NamedSized "batch" 32), 'Dim ( 'NamedSized "feature" 8)])
--   :: MonadFail m =>
--      m (Tensor
--           'Dependent
--           ('Layout 'Dense)
--           ('Device ('CUDA 0))
--           ('DataType 'Half)
--           ('Shape '[ 'Dim ( 'NamedSized "batch" 32), 'Dim ( 'NamedSized "feature" 8)]))
ones ::
  forall requiresGradient layout device dataType shape m.
  ( KnownRequiresGradient requiresGradient,
    WithLayoutC (layout == 'UncheckedLayout) layout (WithDeviceF (device == 'UncheckedDevice) (WithDataTypeF (dataType == 'UncheckedDataType) (WithShapeF (shape == 'UncheckedShape) (m (Tensor requiresGradient layout device dataType (WidenShapeF shape)))))),
    WithDeviceC (device == 'UncheckedDevice) device (WithDataTypeF (dataType == 'UncheckedDataType) (WithShapeF (shape == 'UncheckedShape) (m (Tensor requiresGradient layout device dataType (WidenShapeF shape))))),
    WithDataTypeC (dataType == 'UncheckedDataType) dataType (WithShapeF (shape == 'UncheckedShape) (m (Tensor requiresGradient layout device dataType (WidenShapeF shape)))),
    WithShapeC (shape == 'UncheckedShape) shape (m (Tensor requiresGradient layout device dataType (WidenShapeF shape))),
    MonadFail m
  ) =>
  ( WithLayoutF
      (layout == 'UncheckedLayout)
      ( WithDeviceF
          (device == 'UncheckedDevice)
          ( WithDataTypeF
              (dataType == 'UncheckedDataType)
              (WithShapeF (shape == 'UncheckedShape) (m (Tensor requiresGradient layout device dataType (WidenShapeF shape))))
          )
      )
  )
ones =
  withLayout @(layout == 'UncheckedLayout) @layout @(WithDeviceF (device == 'UncheckedDevice) (WithDataTypeF (dataType == 'UncheckedDataType) (WithShapeF (shape == 'UncheckedShape) (m (Tensor requiresGradient layout device dataType (WidenShapeF shape)))))) $
    \layoutType ->
      withDevice @(device == 'UncheckedDevice) @device @(WithDataTypeF (dataType == 'UncheckedDataType) (WithShapeF (shape == 'UncheckedShape) (m (Tensor requiresGradient layout device dataType (WidenShapeF shape))))) $
        \deviceType ->
          withDataType @(dataType == 'UncheckedDataType) @dataType @(WithShapeF (shape == 'UncheckedShape) (m (Tensor requiresGradient layout device dataType (WidenShapeF shape)))) $
            \dType ->
              withShape @(shape == 'UncheckedShape) @shape @(m (Tensor requiresGradient layout device dataType (WidenShapeF shape))) $
                \shape ->
                  go (requiresGradientVal @requiresGradient) layoutType deviceType dType shape
  where
    go requiresGradient layoutType deviceType dType shape = do
      opts <- pure $ tensorOptions requiresGradient layoutType deviceType dType
      tensor <- case (namedDims shape, sizedDims shape) of
        (Just names, Just sizes) -> pure . unsafePerformIO $ cast3 ATen.ones_lNo sizes names opts
        (Nothing, Just sizes) -> pure . unsafePerformIO $ cast2 ATen.ones_lo sizes opts
        _ -> fail $ "Invalid tensor shape specification " <> show shape <> "."
      return $ UnsafeTensor tensor
