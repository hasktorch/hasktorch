{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.GraduallyTyped.RewriteRules where

import Torch.GraduallyTyped.DType (GetDataTypes, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (GetDevices, WithDeviceC (..))
import Torch.GraduallyTyped.Layout (GetLayouts, WithLayoutC (..))
import Torch.GraduallyTyped.Shape.Type (GetShapes, WithShapeC (..))

type LayoutDeviceRule device f = GetLayouts (WithDeviceF device f) ~ GetLayouts f

type LayoutDataTypeRule dataType f = GetLayouts (WithDataTypeF dataType f) ~ GetLayouts f

type LayoutShapeRule shape f = GetLayouts (WithShapeF shape f) ~ GetLayouts f

type DeviceLayoutRule layout f = GetDevices (WithLayoutF layout f) ~ GetDevices f

type DeviceDataTypeRule dataType f = GetDevices (WithDataTypeF dataType f) ~ GetDevices f

type DeviceShapeRule shape f = GetDevices (WithShapeF shape f) ~ GetDevices f

type DataTypeLayoutRule layout f = GetDataTypes (WithLayoutF layout f) ~ GetDataTypes f

type DataTypeDeviceRule device f = GetDataTypes (WithDeviceF device f) ~ GetDataTypes f

type DataTypeShapeRule shape f = GetDataTypes (WithShapeF shape f) ~ GetDataTypes f

type ShapeLayoutRule layout f = GetShapes (WithLayoutF layout f) ~ GetShapes f

type ShapeDeviceRule device f = GetShapes (WithDeviceF device f) ~ GetShapes f

type ShapeDataTypeRule dataType f = GetShapes (WithDataTypeF dataType f) ~ GetShapes f
