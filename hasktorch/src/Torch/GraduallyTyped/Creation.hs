{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.Creation where

import Control.Monad (MonadPlus, foldM, mzero)
import Data.Int (Int16)
import Data.Kind (Type)
import Data.Type.Equality (type (==))
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType(..))
import Torch.GraduallyTyped.DType (DataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (DeviceType (..), Device (..), WithDeviceC (..))
import Torch.GraduallyTyped.Internal.TensorOptions (tensorOptions)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), WithLayoutC (..))
import Torch.GraduallyTyped.Shape (Dim (..), Shape (..), WithShapeC (..), namedDims, sizedDims)
import Torch.GraduallyTyped.Tensor (RequiresGradient (..), Tensor (..))
import Torch.Internal.Cast (cast2, cast3)
import qualified Torch.Internal.Managed.TensorFactories as ATen
import Torch.Internal.Type (DimnameList, IntArray)

-- | Create a tensor of ones.
-- >>> :type ones @'AnyLayout @'AnyDevice @'AnyDataType @'AnyShape
-- ones @'AnyLayout @'AnyDevice @'AnyDataType @'AnyShape ::
--   LayoutType ->
--   DeviceType Int16 ->
--   DType ->
--   [Dim String Integer] ->
--   IO (Tensor 'Dependent 'AnyLayout 'AnyDevice 'AnyDataType 'AnyShape)
-- >>> :type ones @('Layout 'Dense) @'AnyDevice @'AnyDataType @'AnyShape
-- ones @('Layout 'Dense) @'AnyDevice @'AnyDataType @'AnyShape ::
--   DeviceType Int16 ->
--   DType ->
--   [Dim String Integer] ->
--   IO (Tensor 'Dependent ('Layout 'Dense) 'AnyDevice 'AnyDataType 'AnyShape)
-- >>> :type ones @('Layout 'Dense) @('Device ('CUDA 0)) @'AnyDataType @'AnyShape
-- ones @('Layout 'Dense) @('Device ('CUDA 0)) @'AnyDataType @'AnyShape ::
--   DType ->
--   [Dim String Integer] ->
--   IO (Tensor 'Dependent ('Layout 'Dense) ('Device ('CUDA 0)) 'AnyDataType 'AnyShape)
-- >>> :type ones @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @'AnyShape
-- ones @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @'AnyShape ::
--   [Dim String Integer] ->
--   IO (Tensor 'Dependent ('Layout 'Dense) ('Device ('CUDA 0)) ('DataType 'Half) 'AnyShape)
-- >>> :type ones @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- ones @('Layout 'Dense) @('Device ('CUDA 0)) @('DataType 'Half) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) ::
--   IO (Tensor 'Dependent ('Layout 'Dense) ('Device ('CUDA 0)) ('DataType 'Half) ('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]))
ones ::
  forall layout device dataType shape.
  ( WithLayoutC (layout == 'AnyLayout) layout (WithDeviceF (device == 'AnyDevice) (WithDataTypeF (dataType == 'AnyDataType) (WithShapeF (shape == 'AnyShape) (IO (Tensor 'Dependent layout device dataType shape))))),
    WithDeviceC (device == 'AnyDevice) device (WithDataTypeF (dataType == 'AnyDataType) (WithShapeF (shape == 'AnyShape) (IO (Tensor 'Dependent layout device dataType shape)))),
    WithDataTypeC (dataType == 'AnyDataType) dataType (WithShapeF (shape == 'AnyShape) (IO (Tensor 'Dependent layout device dataType shape))),
    WithShapeC (shape == 'AnyShape) shape (IO (Tensor 'Dependent layout device dataType shape))
  ) =>
  ( WithLayoutF
      (layout == 'AnyLayout)
      ( WithDeviceF
          (device == 'AnyDevice)
          ( WithDataTypeF
              (dataType == 'AnyDataType)
              (WithShapeF (shape == 'AnyShape) (IO (Tensor 'Dependent layout device dataType shape)))
          )
      )
  )
ones =
  withLayout @(layout == 'AnyLayout) @layout @(WithDeviceF (device == 'AnyDevice) (WithDataTypeF (dataType == 'AnyDataType) (WithShapeF (shape == 'AnyShape) (IO (Tensor 'Dependent layout device dataType shape))))) $
    \layoutType ->
      withDevice @(device == 'AnyDevice) @device @(WithDataTypeF (dataType == 'AnyDataType) (WithShapeF (shape == 'AnyShape) (IO (Tensor 'Dependent layout device dataType shape)))) $
        \deviceType ->
          withDataType @(dataType == 'AnyDataType) @dataType @(WithShapeF (shape == 'AnyShape) (IO (Tensor 'Dependent layout device dataType shape))) $
            \dType ->
              withShape @(shape == 'AnyShape) @shape @(IO (Tensor 'Dependent layout device dataType shape)) $
                \shape ->
                  go layoutType deviceType dType shape
  where
    go layoutType deviceType dType shape = do
      opts <- pure $ tensorOptions layoutType deviceType dType
      tensor <- case (namedDims shape, sizedDims shape) of
        (Just names, Just sizes) -> cast3 ATen.ones_lNo sizes names opts
        (Nothing, Just sizes) -> cast2 ATen.ones_lo sizes opts
        _ -> fail $ "invalid tensor shape specification " <> show shape
      return $ UnsafeTensor tensor
