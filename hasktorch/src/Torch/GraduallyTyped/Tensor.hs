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

module Torch.GraduallyTyped.Tensor where

import Data.Coerce (coerce)
import Data.Int (Int16)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice (..))
import Torch.GraduallyTyped.Layout (KnownLayout (..), Layout (..), LayoutType (..))
import Torch.GraduallyTyped.Prelude ((&&^))
import Torch.GraduallyTyped.Shape (Dim, Shape (..))
import Torch.Internal.Cast (cast0, cast1)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Autograd as ATen
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen (Tensor, TensorList)

data RequiresGradient
  = -- | The tensor requires gradients.
    Independent
  | -- | The tensor does not require gradients.
    Dependent

class KnownRequiresGradient (requiresGradient :: RequiresGradient) where
  requiresGradientVal :: RequiresGradient

instance KnownRequiresGradient 'Independent where
  requiresGradientVal = Independent

instance KnownRequiresGradient 'Dependent where
  requiresGradientVal = Dependent

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

-- | Alias for an untyped tensor without gradients.
type UntypedTensor = Tensor 'Dependent 'AnyLayout 'AnyDevice 'AnyDataType 'AnyShape

-- | Alias for an untyped tensor with gradients.
type UntypedParameter = Tensor 'Independent 'AnyLayout 'AnyDevice 'AnyDataType 'AnyShape

-- | Alias for a tensor on CPU memory without gradients.
type CPUTensor = Tensor 'Dependent ('Layout 'Dense) ('Device 'CPU)

-- | Alias for a tensor on CPU memory with gradients.
type CPUParameter = Tensor 'Independent ('Layout 'Dense) ('Device 'CPU)

-- | Alias for a sparse tensor on CPU memory without gradients.
type SparseCPUTensor = Tensor 'Dependent ('Layout 'Sparse) ('Device 'CPU)

-- | Alias for a sparse tensor on CPU memory with gradients.
type SparseCPUParameter = Tensor 'Independent ('Layout 'Sparse) ('Device 'CPU)

-- | Alias for a tensor on CUDA memory without gradients.
type CUDATensor deviceId = Tensor 'Dependent ('Layout 'Dense) ('Device ('CUDA deviceId))

-- | Alias for a tensor on CUDA memory with gradients.
type CUDAParameter deviceId = Tensor 'Independent ('Layout 'Dense) ('Device ('CUDA deviceId))

-- | Alias for a sparse tensor on CUDA memory without gradients.
type SparseCUDATensor deviceId = Tensor 'Dependent ('Layout 'Sparse) ('Device ('CUDA deviceId))

-- | Alias for a sparse tensor on CUDA memory with gradients.
type SparseCUDAParameter deviceId = Tensor 'Independent ('Layout 'Sparse) ('Device ('CUDA deviceId))

-- | Returns an independent copy of the tensor that requires gradients.
makeIndependent ::
  forall layout device dataType shape.
  -- | input
  Tensor 'Dependent layout device dataType shape ->
  -- | copy with gradients
  IO (Tensor 'Independent layout device dataType shape)
makeIndependent = cast1 ATen.makeIndependent

-- | Returns a dependent copy of the tensor that does not require gradients.
makeDependent ::
  forall layout device dataType shape.
  -- | input
  Tensor 'Independent layout device dataType shape ->
  -- | copy without gradients
  IO (Tensor 'Dependent layout device dataType shape)
makeDependent = undefined -- TODO: implement set_requires_grad(false)

-- | Returns a dense copy of the tensor.
toDense ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | dense copy
  Tensor requiresGradient ('Layout 'Dense) device dataType shape
toDense = unsafePerformIO . cast1 ATen.tensor_to_dense

-- | Returns a sparse copy of the tensor.
toSparse ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | sparse copy
  Tensor requiresGradient ('Layout 'Sparse) device dataType shape
toSparse = unsafePerformIO . cast1 ATen.tensor_to_sparse

-- Returns the memory layout of the input tensor.
layout ::
  forall requiresGradient layout device dataType shape.
  KnownLayout layout =>
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | memory layout
  Layout LayoutType
layout tensor =
  case layoutVal @layout of
    AnyLayout ->
      if unsafePerformIO . cast1 ATen.tensor_is_sparse $ tensor
        then Layout Sparse
        else Layout Dense
    Layout layoutType -> Layout layoutType

-- | Returns a copy of the tensor in CPU memory.
cpu ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | copy in CPU memory
  Tensor requiresGradient layout ('Device 'CPU) dataType shape
cpu = unsafePerformIO . cast1 ATen.tensor_cpu

-- | Returns a copy of the tensor in CUDA memory.
cuda ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | copy in CUDA memory
  Tensor requiresGradient layout ('Device ('CUDA 0)) dataType shape
cuda = unsafePerformIO . cast1 ATen.tensor_cuda

-- | Returns the compute device of the input tensor.
-- >>> t <- ones @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8])
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
-- Any static information about the device is thus erased.
-- The underlying tensor data structure is not changed.
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
-- >>> t <- ones @('Layout 'Dense) @'AnyDevice @('DataType 'Float) @('Shape '[ 'NamedSizedDim "Batch" 32, 'NamedSizedDim "Feature" 8]) CPU
-- >>> checkDevice @('Device 'CPU) t
-- True
-- >>> checkDevice @('Device ('CUDA 0)) t
-- False
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
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor is indeed on 'device'.
-- If it is not, then the result will be 'Nothing'.
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
-- If the tensor is not on 'device', the execution is stopped and an error message is displayed.
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
-- >>> t' = checkedDevice @('Device ('CUDA 0)) t
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

-- | Returns the data type of the input tensor.
dtype ::
  forall requiresGradient layout device dataType shape.
  KnownDataType dataType =>
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | data type of the input tensor
  DType
dtype tensor =
  case dataTypeVal @dataType of
    AnyDataType -> unsafePerformIO $ cast1 ATen.tensor_scalar_type tensor
    DataType dtype -> dtype
