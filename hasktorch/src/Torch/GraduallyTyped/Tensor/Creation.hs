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
import Torch.GraduallyTyped.Shape (Shape (..), WithShapeC (..), namedDims, sizedDims)
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
-- >>> :type ones @'Dependent @'AnyLayout @'AnyDevice @'AnyDataType @'AnyShape
-- ones @'Dependent @'AnyLayout @'AnyDevice @'AnyDataType @'AnyShape
--   :: MonadFail m =>
--      LayoutType
--      -> DeviceType GHC.Int.Int16
--      -> DType
--      -> [Dim String Integer]
--      -> m (Tensor
--              'Dependent 'AnyLayout 'AnyDevice 'AnyDataType 'AnyShape)
--
-- >>> :type ones @'Dependent @('Layout 'Dense) @'AnyDevice @'AnyDataType @'AnyShape
-- ones @'Dependent @('Layout 'Dense) @'AnyDevice @'AnyDataType @'AnyShape ::
--   DeviceType Int16 ->
--   DType ->
--   [Dim String Integer] ->
--   IO (Tensor 'Dependent ('Layout 'Dense) 'AnyDevice 'AnyDataType 'AnyShape)
--
-- >>> :type ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @'AnyDataType @'AnyShape
-- ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @'AnyDataType @'AnyShape ::
--   DType ->
--   [Dim String Integer] ->
--   IO (Tensor 'Dependent ('Layout 'Dense) ('Device ('CUDA 0)) 'AnyDataType 'AnyShape)
--
-- >>> :type ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @'AnyShape
-- ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @'AnyShape ::
--   [Dim String Integer] ->
--   IO (Tensor 'Dependent ('Layout 'Dense) ('Device ('CUDA 0)) ('DataType 'Half) 'AnyShape)
--
-- >>> :type ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- ones @'Dependent @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) ::
--   IO (Tensor 'Dependent ('Layout 'Dense) ('Device ('CUDA 0)) ('DataType 'Half) ('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]))
ones ::
  forall requiresGradient layout device dataType shape m.
  ( KnownRequiresGradient requiresGradient,
    WithLayoutC (layout == 'AnyLayout) layout (WithDeviceF (device == 'AnyDevice) (WithDataTypeF (dataType == 'AnyDataType) (WithShapeF (shape == 'AnyShape) (m (Tensor requiresGradient layout device dataType shape))))),
    WithDeviceC (device == 'AnyDevice) device (WithDataTypeF (dataType == 'AnyDataType) (WithShapeF (shape == 'AnyShape) (m (Tensor requiresGradient layout device dataType shape)))),
    WithDataTypeC (dataType == 'AnyDataType) dataType (WithShapeF (shape == 'AnyShape) (m (Tensor requiresGradient layout device dataType shape))),
    WithShapeC (shape == 'AnyShape) shape (m (Tensor requiresGradient layout device dataType shape)),
    MonadFail m
  ) =>
  ( WithLayoutF
      (layout == 'AnyLayout)
      ( WithDeviceF
          (device == 'AnyDevice)
          ( WithDataTypeF
              (dataType == 'AnyDataType)
              (WithShapeF (shape == 'AnyShape) (m (Tensor requiresGradient layout device dataType shape)))
          )
      )
  )
ones =
  withLayout @(layout == 'AnyLayout) @layout @(WithDeviceF (device == 'AnyDevice) (WithDataTypeF (dataType == 'AnyDataType) (WithShapeF (shape == 'AnyShape) (m (Tensor requiresGradient layout device dataType shape))))) $
    \layoutType ->
      withDevice @(device == 'AnyDevice) @device @(WithDataTypeF (dataType == 'AnyDataType) (WithShapeF (shape == 'AnyShape) (m (Tensor requiresGradient layout device dataType shape)))) $
        \deviceType ->
          withDataType @(dataType == 'AnyDataType) @dataType @(WithShapeF (shape == 'AnyShape) (m (Tensor requiresGradient layout device dataType shape))) $
            \dType ->
              withShape @(shape == 'AnyShape) @shape @(m (Tensor requiresGradient layout device dataType shape)) $
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
