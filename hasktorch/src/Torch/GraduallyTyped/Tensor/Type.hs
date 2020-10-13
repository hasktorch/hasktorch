{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.GraduallyTyped.Tensor.Type where

import Control.Monad (foldM)
import Data.Coerce (coerce)
import Data.Int (Int16)
import Data.Kind (Type)
import Data.Monoid (Sum (..))
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice (..))
import Torch.GraduallyTyped.Layout (KnownLayout (..), Layout (..), LayoutType (..))
import Torch.GraduallyTyped.Prelude (ifM, (&&^))
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), KnownShape (..), Shape (..))
import Torch.HList (HList (..))
import Torch.Internal.Cast (cast0, cast1, cast2)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Extra as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen (Tensor, TensorList)

-- $setup
-- import Torch.GraduallyTyped.Creation (ones)

-- | A gradually typed tensor.
newtype
  Tensor
    (requiresGradient :: RequiresGradient)
    (layout :: Layout LayoutType)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (shape :: Shape [Dim Symbol Nat])
  where
  -- | Do not call this constructor directly, use the smart constructors instead.
  UnsafeTensor ::
    forall requiresGradient layout device dataType shape.
    ForeignPtr ATen.Tensor ->
    Tensor requiresGradient layout device dataType shape

type family
  TensorF
    ( parameters ::
        ( RequiresGradient,
          Layout LayoutType,
          Device (DeviceType Nat),
          DataType DType,
          Shape [Dim Symbol Nat]
        )
    ) ::
    Type
  where
  TensorF '(requiresGradient, layout, device, dataType, shape) = Tensor requiresGradient layout device dataType shape

instance
  Castable
    (Tensor requiresGradient layout device dataType shape)
    (ForeignPtr ATen.Tensor)
  where
  cast (UnsafeTensor atenTensor) f = f atenTensor
  uncast atenTensor f = f $ UnsafeTensor atenTensor

instance
  Castable
    [Tensor requiresGradient layout device dataType shape]
    (ForeignPtr ATen.TensorList)
  where
  cast xs f = do
    ptrList <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Tensor))) xs
    cast ptrList f
  uncast xs f = uncast xs $ \ptrList -> do
    tensorList <- mapM (\(x :: ForeignPtr ATen.Tensor) -> uncast x return) ptrList
    f tensorList

instance Castable (HList tensors) [ForeignPtr ATen.Tensor] where
  cast xs f = undefined
  uncast xs f = undefined

instance
  Castable (HList l) [ForeignPtr ATen.Tensor] =>
  Castable (HList l) (ForeignPtr ATen.TensorList)
  where
  cast xs f = do
    ts <- cast xs return :: IO [ForeignPtr ATen.Tensor]
    cast ts f
  uncast xs f = uncast xs $ \(ptrList :: [ForeignPtr ATen.Tensor]) -> do
    ts <- uncast ptrList return :: IO (HList l)
    f ts

-- | Alias for an untyped tensor without gradients.
type UntypedTensor = Tensor 'Dependent 'AnyLayout 'AnyDevice 'AnyDataType 'AnyShape

-- | Alias for an untyped tensor with gradients.
type UntypedParameter = Tensor 'Independent 'AnyLayout 'AnyDevice 'AnyDataType 'AnyShape

-- | Alias for a tensor on CPU memory without gradients.
type CPUTensor = Tensor 'Dependent ( 'Layout 'Dense) ( 'Device 'CPU)

-- | Alias for a tensor on CPU memory with gradients.
type CPUParameter = Tensor 'Independent ( 'Layout 'Dense) ( 'Device 'CPU)

-- | Alias for a sparse tensor on CPU memory without gradients.
type SparseCPUTensor = Tensor 'Dependent ( 'Layout 'Sparse) ( 'Device 'CPU)

-- | Alias for a sparse tensor on CPU memory with gradients.
type SparseCPUParameter = Tensor 'Independent ( 'Layout 'Sparse) ( 'Device 'CPU)

-- | Alias for a tensor on CUDA memory without gradients.
type CUDATensor deviceId = Tensor 'Dependent ( 'Layout 'Dense) ( 'Device ( 'CUDA deviceId))

-- | Alias for a tensor on CUDA memory with gradients.
type CUDAParameter deviceId = Tensor 'Independent ( 'Layout 'Dense) ( 'Device ( 'CUDA deviceId))

-- | Alias for a sparse tensor on CUDA memory without gradients.
type SparseCUDATensor deviceId = Tensor 'Dependent ( 'Layout 'Sparse) ( 'Device ( 'CUDA deviceId))

-- | Alias for a sparse tensor on CUDA memory with gradients.
type SparseCUDAParameter deviceId = Tensor 'Independent ( 'Layout 'Sparse) ( 'Device ( 'CUDA deviceId))

-- | Returns an independent copy of the tensor that requires gradients.
makeIndependent ::
  forall layout device dataType shape.
  -- | input
  Tensor 'Dependent layout device dataType shape ->
  -- | copy with gradients
  IO (Tensor 'Independent layout device dataType shape)
makeIndependent tensor = cast2 ATen.tensor_set_requires_grad_b tensor True

-- | Returns a dependent copy of the tensor that does not require gradients.
makeDependent ::
  forall layout device dataType shape.
  -- | input
  Tensor 'Independent layout device dataType shape ->
  -- | copy without gradients
  IO (Tensor 'Dependent layout device dataType shape)
makeDependent tensor = cast2 ATen.tensor_set_requires_grad_b tensor False

-- | Returns a dense copy of the tensor.
toDense ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | dense copy
  Tensor requiresGradient ( 'Layout 'Dense) device dataType shape
toDense = unsafePerformIO . cast1 ATen.tensor_to_dense

-- | Returns a sparse copy of the tensor.
toSparse ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | sparse copy
  Tensor requiresGradient ( 'Layout 'Sparse) device dataType shape
toSparse = unsafePerformIO . cast1 ATen.tensor_to_sparse

-- Returns the memory layout of the input tensor.
--
-- >>> t <- ones @('Layout 'Sparse) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- >>> layout t
-- Sparse
-- >>> t <- ones @'AnyLayout @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) (Layout Sparse)
-- >>> layout t
-- Sparse
layout ::
  forall requiresGradient layout device dataType shape.
  KnownLayout layout =>
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | memory layout
  LayoutType
layout tensor =
  case layoutVal @layout of
    AnyLayout ->
      if unsafePerformIO . cast1 ATen.tensor_is_sparse $ tensor
        then Sparse
        else Dense
    Layout layoutType -> layoutType

-- | Returns the input tensor but with 'AnyLayout' as memory layout type annotation.
-- Any static information about the tensor's memory layout is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t <- ones @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- >>> :type uncheckedLayout t
-- uncheckedLayout t
--   :: Tensor
--        'Dependent
--        'AnyLayout
--        '('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
uncheckedLayout ::
  forall requiresGradient layout device dataType shape.
  -- | input tensor
  Tensor requiresGradient layout device dataType shape ->
  -- | tensor without checked layout
  Tensor requiresGradient 'AnyLayout device dataType shape
uncheckedLayout = coerce

-- | Returns 'True' if the tensor has the memory layout 'layout' and 'False' otherwise.
-- If 'layout' is 'AnyLayout', 'True' is returned for consistency.
--
-- >>> t <- ones @'AnyLayout @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) (Layout Sparse)
-- >>> checkLayout @('Layout 'Sparse) t
-- True
-- >>> checkLayout @('Layout 'Dense) t
-- False
-- >>> checkLayout @'AnyLayout t
-- True
checkLayout ::
  forall (layout :: Layout LayoutType) requiresGradient device dataType shape.
  (KnownLayout layout) =>
  -- | tensor under consideration
  Tensor requiresGradient 'AnyLayout device dataType shape ->
  -- | whether or not the input tensor has the memory layout 'layout'
  Bool
checkLayout tensor =
  case layoutVal @layout of
    AnyLayout -> True
    Layout Sparse -> undefined
    Layout Dense -> undefined

-- | Checks whether or not the input tensor has the memory layout 'layout'
-- and returns a statically annotated copy of it wrapped in a 'MonadFail' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor has indeed the memory layout 'layout'.
-- If it does not have it, then the result will be 'Nothing'.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t <- ones @'AnyLayout @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) (Layout Dense)
-- >>> t' <- checkedLayout @('Layout 'Dense) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- >>> t' <- checkedLayout @('Layout 'Sparse) t
-- *** Exception: user error (The tensor does not have the memory layout "Layout Sparse".)
checkedLayout ::
  forall (layout :: Layout LayoutType) m requiresGradient device dataType shape.
  (KnownLayout layout, MonadFail m) =>
  -- | input tensor
  Tensor requiresGradient 'AnyLayout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor requiresGradient layout device dataType shape)
checkedLayout tensor
  | checkLayout @layout tensor = pure . coerce $ tensor
  | otherwise = fail $ "The tensor does not have the memory layout \"" <> show (layoutVal @layout) <> "\"."

-- | Unsafe version of 'checkedLayout'.
-- If the tensor does not have the memory layout 'layout', then the execution is stopped and an error message is displayed.
--
-- >>> t <- ones @'AnyLayout @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) CPU
-- >>> t' = unsafeCheckedLayout @('Layout 'Dense) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- >>> t' = unsafeCheckedLayout @('Layout 'Sparse) t
-- *** Exception: The tensor does not have the memory layout "Layout Sparse".
unsafeCheckedLayout ::
  forall (layout :: Layout LayoutType) requiresGradient device dataType shape.
  KnownLayout layout =>
  -- | input tensor
  Tensor requiresGradient 'AnyLayout device dataType shape ->
  -- | annotated output tensor
  Tensor requiresGradient layout device dataType shape
unsafeCheckedLayout tensor = case checkedLayout @layout tensor of
  Right tensor' -> tensor'
  Left err -> error err

-- | Returns a copy of the tensor in CPU memory.
cpu ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | copy in CPU memory
  Tensor requiresGradient layout ( 'Device 'CPU) dataType shape
cpu = unsafePerformIO . cast1 ATen.tensor_cpu

-- | Returns a copy of the tensor in CUDA memory.
cuda ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | copy in CUDA memory
  Tensor requiresGradient layout ( 'Device ( 'CUDA 0)) dataType shape
cuda = unsafePerformIO . cast1 ATen.tensor_cuda

-- | Returns the compute device of the input tensor.
--
-- >>> t <- ones @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- >>> device t
-- CPU
-- >>> t <- ones @('Layout 'Dense) @'AnyDevice @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) (Device CPU)
-- >>> device t
-- CPU
device ::
  forall requiresGradient layout device dataType shape.
  KnownDevice device =>
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | compute device of the input tensor
  DeviceType Int16
device tensor =
  case deviceVal @device of
    AnyDevice ->
      unsafePerformIO $ do
        hasCUDA <- cast0 ATen.hasCUDA
        if hasCUDA
          then do
            isCUDA <- cast1 ATen.tensor_is_cuda tensor
            if isCUDA
              then do
                deviceIndex :: Int <- cast1 ATen.tensor_get_device tensor
                pure . CUDA . fromIntegral $ deviceIndex
              else pure $ CPU
          else pure $ CPU
    Device deviceType -> deviceType

-- | Returns the input tensor but with 'AnyDevice' as device type annotation.
-- Any static information about the tensor's device is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t <- ones @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- >>> :type uncheckedDevice t
-- uncheckedDevice t
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        'AnyDevice
--        ('DataType 'Float)
--        ('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
uncheckedDevice ::
  forall requiresGradient layout device dataType shape.
  -- | input tensor
  Tensor requiresGradient layout device dataType shape ->
  -- | tensor without checked device
  Tensor requiresGradient layout 'AnyDevice dataType shape
uncheckedDevice = coerce

-- | Returns 'True' if the tensor is in the memory of 'device' and 'False' otherwise.
-- If 'device' is 'AnyDevice', 'True' is returned for consistency.
--
-- >>> t <- ones @('Layout 'Dense) @'AnyDevice @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) CPU
-- >>> checkDevice @('Device 'CPU) t
-- True
-- >>> checkDevice @('Device ('CUDA 0)) t
-- False
-- >>> checkDevice @'AnyDevice t
-- True
checkDevice ::
  forall (device :: Device (DeviceType Nat)) requiresGradient layout dataType shape.
  (KnownDevice device) =>
  -- | tensor under consideration
  Tensor requiresGradient layout 'AnyDevice dataType shape ->
  -- | whether or not the input tensor is on the 'device'
  Bool
checkDevice tensor =
  case deviceVal @device of
    AnyDevice -> True
    Device CPU -> not . unsafePerformIO $ cast0 ATen.hasCUDA &&^ cast1 ATen.tensor_is_cuda tensor
    Device (CUDA deviceIndex) ->
      unsafePerformIO $
        cast0 ATen.hasCUDA
          &&^ cast1 ATen.tensor_is_cuda tensor
          &&^ ((deviceIndex ==) . fromIntegral) <$> (cast1 ATen.tensor_get_device tensor :: IO Int)

-- | Checks whether or not the input tensor is in the memory of 'device'
-- and returns a statically annotated copy of it wrapped in a 'MonadFail' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor is indeed on 'device'.
-- If it is not, then the result will be 'Nothing'.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t <- ones @('Layout 'Dense) @'AnyDevice @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) CPU
-- >>> t' <- checkedDevice @('Device 'CPU) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- >>> t' <- checkedDevice @('Device ('CUDA 0)) t
-- *** Exception: user error (The tensor is not in the memory of the device "Device (CUDA 0)".)
checkedDevice ::
  forall (device :: Device (DeviceType Nat)) m requiresGradient layout dataType shape.
  (KnownDevice device, MonadFail m) =>
  -- | input tensor
  Tensor requiresGradient layout 'AnyDevice dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor requiresGradient layout device dataType shape)
checkedDevice tensor
  | checkDevice @device tensor = pure . coerce $ tensor
  | otherwise = fail $ "The tensor is not in the memory of the device \"" <> show (deviceVal @device) <> "\"."

-- | Unsafe version of 'checkedDevice'.
-- If the tensor is not on 'device', then the execution is stopped and an error message is displayed.
--
-- >>> t <- ones @('Layout 'Dense) @'AnyDevice @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) CPU
-- >>> t' = unsafeCheckedDevice @('Device 'CPU) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- >>> t' = unsafeCheckedDevice @('Device ('CUDA 0)) t
-- *** Exception: The tensor is not in the memory of the device "Device (CUDA 0)".
unsafeCheckedDevice ::
  forall (device :: Device (DeviceType Nat)) requiresGradient layout dataType shape.
  KnownDevice device =>
  -- | input tensor
  Tensor requiresGradient layout 'AnyDevice dataType shape ->
  -- | annotated output tensor
  Tensor requiresGradient layout device dataType shape
unsafeCheckedDevice tensor = case checkedDevice @device tensor of
  Right tensor' -> tensor'
  Left err -> error err

-- | Returns a copy of the tensor converted to 'Bool'.
bool ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Bool' copy
  Tensor 'Dependent layout device ( 'DataType 'Bool) shape
bool tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Bool

-- | Returns a copy of the tensor converted to 'UInt8'.
byte ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'UInt8' copy
  Tensor 'Dependent layout device ( 'DataType 'UInt8) shape
byte tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor UInt8

-- | Returns a copy of the tensor converted to 'Int8'.
char ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Int8' copy
  Tensor 'Dependent layout device ( 'DataType 'Int8) shape
char tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Int8

-- | Returns a copy of the tensor converted to 'Int16'.
short ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Int16' copy
  Tensor 'Dependent layout device ( 'DataType 'Int16) shape
short tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Int16

-- | Returns a copy of the tensor converted to 'Int32'.
int ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Int32' copy
  Tensor 'Dependent layout device ( 'DataType 'Int32) shape
int tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Int32

-- | Returns a copy of the tensor converted to 'Int64'.
long ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Int64' copy
  Tensor 'Dependent layout device ( 'DataType 'Int64) shape
long tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Int64

-- | Returns a copy of the tensor converted to the 16-bit floating point format 'Half'.
half ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Half' copy
  Tensor requiresGradient layout device ( 'DataType 'Half) shape
half tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Half

-- | Returns a copy of the tensor converted to the 32-bit floating point format 'Float'.
float ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Float' copy
  Tensor requiresGradient layout device ( 'DataType 'Float) shape
float tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Float

-- | Returns a copy of the tensor converted to the 32-bit floating point format 'Double'.
double ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Double' copy
  Tensor requiresGradient layout device ( 'DataType 'Double) shape
double tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Double

-- | Returns the data type of the input tensor.
--
-- >>> t <- ones @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- >>> dtype t
-- Float
-- >>> t <- ones @('Layout 'Dense) @('Device 'CPU) @'AnyDataType @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) (DataType Float)
-- >>> dtype t
-- Float
dataType ::
  forall dataType requiresGradient layout device shape.
  KnownDataType dataType =>
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | data type of the input tensor
  DType
dataType tensor =
  case dataTypeVal @dataType of
    AnyDataType -> unsafePerformIO $ cast1 ATen.tensor_scalar_type tensor
    DataType dtype -> dtype

-- | Alias for 'dataType'.
dtype ::
  forall dataType requiresGradient layout device shape.
  KnownDataType dataType =>
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | data type of the input tensor
  DType
dtype = dataType @dataType

-- | Returns the input tensor but with 'AnyDataType' as data-type type annotation.
-- Any static information about the tensor's data type is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t <- ones @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- >>> :type uncheckedDataType t
-- uncheckedDataType t
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        'AnyDataType
--        ('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
uncheckedDataType ::
  forall requiresGradient layout device dataType shape.
  -- | input tensor
  Tensor requiresGradient layout device dataType shape ->
  -- | tensor without checked data type
  Tensor requiresGradient layout device 'AnyDataType shape
uncheckedDataType = coerce

-- | Returns 'True' if the tensor has the data type 'dataType' and 'False' otherwise.
-- If 'dataType' is 'AnyDataType', 'True' is returned for consistency.
--
-- >>> t <- ones @('Layout 'Dense) @'AnyDevice @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) CPU
-- >>> checkDataType @('DataType 'Float) t
-- True
-- >>> checkDataType @('DataType 'Double) t
-- False
-- >>> checkDataType @'AnyDataType t
-- True
checkDataType ::
  forall (dataType :: DataType DType) requiresGradient layout device shape.
  (KnownDataType dataType) =>
  -- | tensor under consideration
  Tensor requiresGradient layout device 'AnyDataType shape ->
  -- | whether or not the input tensor has the data type 'dataType'
  Bool
checkDataType tensor =
  case dataTypeVal @dataType of
    AnyDataType -> True
    DataType dtype -> unsafePerformIO $ (dtype ==) <$> cast1 ATen.tensor_scalar_type tensor

-- | Checks whether or not the input tensor has the data type 'dataType'
-- and returns a statically annotated copy of it wrapped in a 'MonadFail' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor has indeed the data type 'dataType'.
-- If it does not have it, then the result will be 'Nothing'.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t <- ones @('Layout 'Dense) @('Device 'CPU) @'AnyDataType @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) (DataType Float)
-- >>> t' <- checkedDataType @('DataType 'Float) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- >>> t' <- checkedDataType @('DataType 'Double) t
-- *** Exception: user error (The tensor does not have the data type "DataType Double".)
checkedDataType ::
  forall (dataType :: DataType DType) m requiresGradient layout device shape.
  (KnownDataType dataType, MonadFail m) =>
  -- | input tensor
  Tensor requiresGradient layout device 'AnyDataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor requiresGradient layout device dataType shape)
checkedDataType tensor
  | checkDataType @dataType tensor = pure . coerce $ tensor
  | otherwise = fail $ "The tensor does not have the data type \"" <> show (dataTypeVal @dataType) <> "\"."

-- | Unsafe version of 'checkedDataType'.
-- If the tensor does not have the data type 'dataType', then the execution is stopped and an error message is displayed.
--
-- >>> t <- ones @('Layout 'Dense) @('Device 'CPU) @'AnyDataType @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) (DataType Float)
-- >>> t' = checkedDataType @('DataType 'Float) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
-- >>> t' = unsafeCheckedDataType @('DataType 'Double) t
-- *** Exception: user error (The tensor does not have the data type "DataType Double".)
unsafeCheckedDataType ::
  forall (dataType :: DataType DType) requiresGradient layout device shape.
  KnownDataType dataType =>
  -- | input tensor
  Tensor requiresGradient layout device 'AnyDataType shape ->
  -- | annotated output tensor
  Tensor requiresGradient layout device dataType shape
unsafeCheckedDataType tensor = case checkedDataType @dataType tensor of
  Right tensor' -> tensor'
  Left err -> error err

shape ::
  forall requiresGradient layout device dataType shape.
  KnownShape shape =>
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | shape of the input tensor
  [Dim String Integer]
shape tensor =
  case shapeVal @shape of
    AnyShape -> unsafePerformIO $ do
      sizes <- cast1 ATen.tensor_sizes tensor
      ifM
        (cast1 ATen.tensor_has_names tensor)
        ( do
            names <- cast1 ATen.tensor_names tensor
            return $ zipWith NamedSizedDim names sizes
        )
        (return $ SizedDim <$> sizes)
    Shape shape ->
      unsafePerformIO $
        snd <$> foldM step' mempty shape
      where
        inc = (<>) (Sum 1)
        step' (s, as) a = (\a' -> (inc s, a' : as)) <$> step s a
        step :: Sum Int -> Dim String Integer -> IO (Dim String Integer)
        step dim AnyDim = do
          size :: Int <- cast2 ATen.tensor_size_l tensor (getSum dim)
          ifM
            (cast1 ATen.tensor_has_names tensor)
            ( do
                name :: String <- undefined
                return $ NamedSizedDim name (fromIntegral size)
            )
            (return $ SizedDim (fromIntegral size))
        step dim (NamedDim name) = do
          size :: Int <- cast2 ATen.tensor_size_l tensor (getSum dim)
          (return $ NamedSizedDim name (fromIntegral size))
        step dim (SizedDim size) = do
          ifM
            (cast1 ATen.tensor_has_names tensor)
            ( do
                name :: String <- undefined
                return $ NamedSizedDim name (fromIntegral size)
            )
            (return $ SizedDim size)
        step _ namedSizedDim@(NamedSizedDim _ _) =
          return namedSizedDim