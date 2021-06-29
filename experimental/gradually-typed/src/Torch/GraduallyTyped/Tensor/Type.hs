{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.Tensor.Type where

import Data.Bifunctor (bimap)
import Data.Coerce (coerce)
import Data.Foldable (Foldable (fold))
import Data.Int (Int16)
import Data.Monoid (All (..))
import Data.Proxy (Proxy (..))
import Data.Singletons (SingI (sing), SingKind (fromSing))
import Data.Singletons.Prelude.List (SList (..))
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (KnownNat, KnownSymbol, Nat, Symbol, natVal, symbolVal)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (KnownLayout (..), Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked, ifM, (&&^))
import Torch.GraduallyTyped.RequiresGradient (KnownRequiresGradient, RequiresGradient (..))
import Torch.GraduallyTyped.Scalar ()
import Torch.GraduallyTyped.Shape.Class (ReplaceDimF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownShape (..), Name (..), SDim (..), SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:|:))
import Torch.HList (HList (..), pattern (:.))
import Torch.Internal.Cast (cast0, cast1, cast2)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Extra as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen (Tensor, TensorList)
import qualified Torch.Tensor (Tensor (Unsafe))

-- $setup
-- >>> import Data.Singletons.Prelude.List (SList (..))
-- >>> import Torch.GraduallyTyped

-- | A gradually typed tensor.
--
-- @
--                                  ┌─► Compute device, e.g. 'Device 'CPU
--                                  │
--                                  │               ┌─► List of dimensions, e.g. 'Shape '[ 'Dim 'UncheckedName ('Size 8), 'Dim 'UncheckedName ('Size 1) ]
--                                  │               │
-- Tensor requiresGradient layout device dataType shape
--               │           │              │
--               │           │              └─► Data type, e.g. 'DataType 'Float
--               │           │
--               │           └─► Memory layout, e.g. 'Layout 'Dense
--               │
--               └─► Whether or not the tensor requires a gradient, e.g. 'WithGradient for one that does
-- @
newtype
  Tensor
    (requiresGradient :: RequiresGradient)
    (layout :: Layout LayoutType)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (shape :: Shape [Dim (Name Symbol) (Size Nat)])
  where
  -- | Unsafe constructor for tensors.
  -- Do not call this constructor directly,
  -- use smart constructors like 'ones' or 'randn' instead.
  UnsafeTensor ::
    forall requiresGradient layout device dataType shape.
    ForeignPtr ATen.Tensor ->
    Tensor requiresGradient layout device dataType shape

type role Tensor nominal nominal nominal nominal nominal

instance Show (Tensor requiresGradient layout device dataType shape) where
  show (UnsafeTensor t) = show (Torch.Tensor.Unsafe t)

-- | Alias for an untyped tensor without gradients.
type UntypedTensor = Tensor 'WithoutGradient 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape

-- | Alias for an untyped tensor with gradients.
type UntypedParameter = Tensor 'WithGradient 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape

-- | Alias for a tensor on CPU memory without gradients.
type CPUTensor = Tensor 'WithoutGradient ('Layout 'Dense) ('Device 'CPU)

-- | Alias for a tensor on CPU memory with gradients.
type CPUParameter = Tensor 'WithGradient ('Layout 'Dense) ('Device 'CPU)

-- | Alias for a sparse tensor on CPU memory without gradients.
type SparseCPUTensor = Tensor 'WithoutGradient ('Layout 'Sparse) ('Device 'CPU)

-- | Alias for a sparse tensor on CPU memory with gradients.
type SparseCPUParameter = Tensor 'WithGradient ('Layout 'Sparse) ('Device 'CPU)

-- | Alias for a tensor on CUDA memory without gradients.
type CUDATensor deviceId = Tensor 'WithoutGradient ('Layout 'Dense) ('Device ('CUDA deviceId))

-- | Alias for a tensor on CUDA memory with gradients.
type CUDAParameter deviceId = Tensor 'WithGradient ('Layout 'Dense) ('Device ('CUDA deviceId))

-- | Alias for a sparse tensor on CUDA memory without gradients.
type SparseCUDATensor deviceId = Tensor 'WithoutGradient ('Layout 'Sparse) ('Device ('CUDA deviceId))

-- | Alias for a sparse tensor on CUDA memory with gradients.
type SparseCUDAParameter deviceId = Tensor 'WithGradient ('Layout 'Sparse) ('Device ('CUDA deviceId))

instance
  Num (Tensor requiresGradient layout device dataType shape)
  where
  (+) = (unsafePerformIO .) . cast2 ATen.add_tt
  (-) = (unsafePerformIO .) . cast2 ATen.sub_tt
  (*) = (unsafePerformIO .) . cast2 ATen.mul_tt
  negate = unsafePerformIO . cast1 ATen.neg_t
  abs = unsafePerformIO . cast1 ATen.abs_t
  signum = unsafePerformIO . cast1 ATen.sign_t
  fromInteger _a = undefined

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

instance Castable (HList '[]) [ForeignPtr ATen.Tensor] where
  cast HNil f = f []
  uncast [] f = f HNil
  uncast (_ : _) _ = fail "The list of tensors has more elements than expected. This means that the runtime length of the list exceeded its compile-time length."

instance
  ( Castable (HList tensors) [ForeignPtr ATen.Tensor]
  ) =>
  Castable (HList (Tensor requiresGradient layout device dataType shape ': tensors)) [ForeignPtr ATen.Tensor]
  where
  cast (tensor :. tensors) f = do
    ptr <- cast tensor pure
    ptrList <- cast tensors pure
    f (ptr : ptrList)
  uncast [] _ = fail "The list of tensors ended prematurely. This means that the runtime length of the list was smaller than its compile-time length."
  uncast (ptr : ptrList) f = do
    tensor <- uncast ptr pure
    tensors <- uncast ptrList pure
    f (tensor :. tensors)

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

-- | Takes a tensor that may or may not require gradients and returns a copy that does not require them.
withoutGradient ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | copy without gradients
  IO (Tensor 'WithoutGradient layout device dataType shape)
withoutGradient tensor = cast2 ATen.tensor_set_requires_grad_b tensor False

-- | Takes a tensor that does not requires gradients and returns a copy that requires them.
withGradient ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | copy with gradients
  IO (Tensor 'WithGradient layout device dataType shape)
withGradient tensor = cast2 ATen.tensor_set_requires_grad_b tensor True

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

class SGetLayout (layout :: Layout LayoutType) where
  -- | Returns the gradually typed memory layout of the input tensor.
  --
  -- >>> sOnes' layout = sOnes SWithGradient layout (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t = sOnes' $ SLayout SDense
  -- >>> sLayout t
  -- SLayout SDense
  -- >>> t = sOnes' $ SUncheckedLayout Dense
  -- >>> sLayout t
  -- SUncheckedLayout Dense
  sLayout ::
    forall m requiresGradient device dataType shape.
    MonadFail m =>
    -- | input
    Tensor requiresGradient layout device dataType shape ->
    -- | memory layout
    m (SLayout layout)

  -- | Returns the untyped memory layout of the input tensor.
  --
  -- >>> sOnes' layout = sOnes SWithGradient layout (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t = sOnes' $ SLayout SDense
  -- >>> layoutType t
  -- Dense
  -- >>> t = sOnes' $ SUncheckedLayout Dense
  -- >>> layoutType t
  -- Dense
  layoutType ::
    forall m requiresGradient device dataType shape.
    MonadFail m =>
    -- | input
    Tensor requiresGradient layout device dataType shape ->
    -- | memory layout
    m LayoutType
  layoutType tensor = forgetIsChecked . fromSing <$> sLayout tensor

instance SGetLayout 'UncheckedLayout where
  sLayout tensor
    | unsafePerformIO (cast1 ATen.tensor_is_sparse tensor) = pure $ SUncheckedLayout Sparse
    | otherwise = pure $ SUncheckedLayout Dense

instance SGetLayout ('Layout 'Sparse) where
  sLayout tensor
    | unsafePerformIO (cast1 ATen.tensor_is_sparse tensor) = pure $ SLayout SSparse
    | otherwise = fail "The tensor should be sparse but isn't. Please open a ticket on GitHub."

instance SGetLayout ('Layout 'Dense) where
  sLayout tensor
    | unsafePerformIO (cast1 ATen.tensor_is_sparse tensor) = fail "The tensor should be dense but isn't. Please open a ticket on GitHub."
    | otherwise = pure $ SLayout SDense

-- | Returns the input tensor but with 'UncheckedLayout' as memory layout type annotation.
-- Any static information about the tensor's memory layout is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t = ones @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedLayout t
-- uncheckedLayout t
--   :: Tensor
--        'WithGradient
--        'UncheckedLayout
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
uncheckedLayout ::
  forall requiresGradient layout device dataType shape.
  -- | input tensor
  Tensor requiresGradient layout device dataType shape ->
  -- | tensor without checked layout
  Tensor requiresGradient 'UncheckedLayout device dataType shape
uncheckedLayout = coerce

-- | Returns 'True' if the tensor has the memory layout 'layout' and 'False' otherwise.
-- If 'layout' is 'UncheckedLayout', 'True' is returned for consistency.
--
-- >>> t = sOnes SWithGradient (SUncheckedLayout Dense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> checkLayout @('Layout 'Sparse) t
-- False
-- >>> checkLayout @('Layout 'Dense) t
-- True
-- >>> checkLayout @'UncheckedLayout t
-- True
checkLayout ::
  forall (layout :: Layout LayoutType) requiresGradient device dataType shape.
  (KnownLayout layout) =>
  -- | tensor under consideration
  Tensor requiresGradient 'UncheckedLayout device dataType shape ->
  -- | whether or not the input tensor has the memory layout 'layout'
  Bool
checkLayout tensor =
  case layoutVal @layout of
    UncheckedLayout -> True
    Layout layoutType ->
      unsafePerformIO $
        (==) layoutType <$> cast1 ATen.tensor_layout tensor

-- | Checks whether or not the input tensor has the memory layout 'layout'
-- and returns a statically annotated copy of it wrapped in a 'MonadFail' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor has indeed the memory layout 'layout'.
-- If it does not have it, then the result will be 'Nothing'.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t = sOnes SWithGradient (SUncheckedLayout Dense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' <- checkedLayout @('Layout 'Dense) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
-- >>> t' <- checkedLayout @('Layout 'Sparse) t
-- *** Exception: user error (The tensor does not have the memory layout "Layout Sparse".)
checkedLayout ::
  forall (layout :: Layout LayoutType) m requiresGradient device dataType shape.
  (KnownLayout layout, MonadFail m) =>
  -- | input tensor
  Tensor requiresGradient 'UncheckedLayout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor requiresGradient layout device dataType shape)
checkedLayout tensor
  | checkLayout @layout tensor = pure . coerce $ tensor
  | otherwise = fail $ "The tensor does not have the memory layout \"" <> show (layoutVal @layout) <> "\"."

-- | Unsafe version of 'checkedLayout'.
-- If the tensor does not have the memory layout 'layout', then the execution is stopped and an error message is displayed.
--
-- >>> t = sOnes SWithGradient (SUncheckedLayout Dense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' = unsafeCheckedLayout @('Layout 'Dense) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
-- >>> unsafeCheckedLayout @('Layout 'Sparse) t
-- *** Exception: The tensor does not have the memory layout "Layout Sparse".
-- CallStack (from HasCallStack):
--   error, called at ...
unsafeCheckedLayout ::
  forall (layout :: Layout LayoutType) requiresGradient device dataType shape.
  KnownLayout layout =>
  -- | input tensor
  Tensor requiresGradient 'UncheckedLayout device dataType shape ->
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

class SGetDevice (device :: Device (DeviceType Nat)) where
  -- | Returns the gradually typed compute device of the input tensor.
  --
  -- >>> ones' device = sOnes SWithGradient (SLayout SDense) device (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t = ones' $ SDevice SCPU
  -- >>> fromSing <$> sDevice t
  -- Checked CPU
  -- >>> t = ones' $ SUncheckedDevice CPU
  -- >>> fromSing <$> sDevice t
  -- Unchecked CPU
  sDevice ::
    forall m requiresGradient layout dataType shape.
    MonadFail m =>
    -- | input
    Tensor requiresGradient layout device dataType shape ->
    -- | compute device of the input tensor
    m (SDevice device)

  -- | Returns the untyped compute device of the input tensor.
  --
  -- >>> ones' device = sOnes SWithGradient (SLayout SDense) device (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t = ones' $ SDevice SCPU
  -- >>> deviceType t
  -- CPU
  -- >>> t = ones' $ SUncheckedDevice CPU
  -- >>> deviceType t
  -- CPU
  deviceType ::
    forall m requiresGradient layout dataType shape.
    MonadFail m =>
    -- | input
    Tensor requiresGradient layout device dataType shape ->
    -- | compute device of the input tensor
    m (DeviceType Int16)
  deviceType tensor = forgetIsChecked . fromSing <$> sDevice tensor

instance SGetDevice 'UncheckedDevice where
  sDevice tensor
    | unsafePerformIO (cast0 ATen.hasCUDA) && unsafePerformIO (cast1 ATen.tensor_is_cuda tensor) =
      case unsafePerformIO (cast1 ATen.tensor_get_device tensor) :: Int of
        deviceIndex -> pure . SUncheckedDevice . CUDA . fromIntegral $ deviceIndex
    | otherwise = pure . SUncheckedDevice $ CPU

instance SGetDevice ('Device 'CPU) where
  sDevice tensor
    | unsafePerformIO (cast0 ATen.hasCUDA) && unsafePerformIO (cast1 ATen.tensor_is_cuda tensor) =
      fail "The tensor should be on CPU but is on CUDA. Please open a ticket on GitHub."
    | otherwise = pure . SDevice $ SCPU

instance KnownNat deviceIndex => SGetDevice ('Device ('CUDA deviceIndex)) where
  sDevice tensor
    | unsafePerformIO (cast0 ATen.hasCUDA) && unsafePerformIO (cast1 ATen.tensor_is_cuda tensor) =
      case unsafePerformIO (cast1 ATen.tensor_get_device tensor) :: Int of
        deviceIndex
          | deviceIndex == fromIntegral (natVal (Proxy @deviceIndex)) -> pure . SDevice $ SCUDA
          | otherwise ->
            fail $
              "The tensor should be on CUDA device "
                <> show (natVal (Proxy @deviceIndex))
                <> " but is on device "
                <> show deviceIndex
                <> ". Please open a ticket on GitHub."
    | otherwise =
      fail "The tensor should be on CUDA but is on CPU. Please open a ticket on GitHub."

-- | Returns the input tensor but with 'UncheckedDevice' as device type annotation.
-- Any static information about the tensor's device is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t = ones @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedDevice t
-- uncheckedDevice t
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        'UncheckedDevice
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
uncheckedDevice ::
  forall requiresGradient layout device dataType shape.
  -- | input tensor
  Tensor requiresGradient layout device dataType shape ->
  -- | tensor without checked device
  Tensor requiresGradient layout 'UncheckedDevice dataType shape
uncheckedDevice = coerce

-- | Returns 'True' if the tensor is in the memory of 'device' and 'False' otherwise.
-- If 'device' is 'UncheckedDevice', 'True' is returned for consistency.
--
-- >>> t = sOnes SWithGradient (SLayout SDense) (SUncheckedDevice CPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> checkDevice @('Device 'CPU) t
-- True
-- >>> checkDevice @('Device ('CUDA 0)) t
-- False
-- >>> checkDevice @'UncheckedDevice t
-- True
checkDevice ::
  forall (device :: Device (DeviceType Nat)) requiresGradient layout dataType shape.
  (KnownDevice device) =>
  -- | tensor under consideration
  Tensor requiresGradient layout 'UncheckedDevice dataType shape ->
  -- | whether or not the input tensor is on the 'device'
  Bool
checkDevice tensor =
  case deviceVal @device of
    UncheckedDevice -> True
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
-- >>> t = sOnes SWithGradient (SLayout SDense) (SUncheckedDevice CPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' <- checkedDevice @('Device 'CPU) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
-- >>> t' <- checkedDevice @('Device ('CUDA 0)) t
-- *** Exception: user error (The tensor is not in the memory of the device "Device (CUDA 0)".)
checkedDevice ::
  forall (device :: Device (DeviceType Nat)) m requiresGradient layout dataType shape.
  (KnownDevice device, MonadFail m) =>
  -- | input tensor
  Tensor requiresGradient layout 'UncheckedDevice dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor requiresGradient layout device dataType shape)
checkedDevice tensor
  | checkDevice @device tensor = pure . coerce $ tensor
  | otherwise = fail $ "The tensor is not in the memory of the device \"" <> show (deviceVal @device) <> "\"."

-- | Unsafe version of 'checkedDevice'.
-- If the tensor is not on 'device', then the execution is stopped and an error message is displayed.
--
-- >>> t = sOnes SWithGradient (SLayout SDense) (SUncheckedDevice CPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' = unsafeCheckedDevice @('Device 'CPU) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
-- >>> unsafeCheckedDevice @('Device ('CUDA 0)) t
-- *** Exception: The tensor is not in the memory of the device "Device (CUDA 0)".
-- CallStack (from HasCallStack):
--   error, called at ...
unsafeCheckedDevice ::
  forall (device :: Device (DeviceType Nat)) requiresGradient layout dataType shape.
  KnownDevice device =>
  -- | input tensor
  Tensor requiresGradient layout 'UncheckedDevice dataType shape ->
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
  Tensor 'WithoutGradient layout device ('DataType 'Bool) shape
bool tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Bool

-- | Returns a copy of the tensor converted to 'UInt8'.
byte ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'UInt8' copy
  Tensor 'WithoutGradient layout device ('DataType 'UInt8) shape
byte tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor UInt8

-- | Returns a copy of the tensor converted to 'Int8'.
char ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Int8' copy
  Tensor 'WithoutGradient layout device ('DataType 'Int8) shape
char tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Int8

-- | Returns a copy of the tensor converted to 'Int16'.
short ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Int16' copy
  Tensor 'WithoutGradient layout device ('DataType 'Int16) shape
short tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Int16

-- | Returns a copy of the tensor converted to 'Int32'.
int ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Int32' copy
  Tensor 'WithoutGradient layout device ('DataType 'Int32) shape
int tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Int32

-- | Returns a copy of the tensor converted to 'Int64'.
long ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Int64' copy
  Tensor 'WithoutGradient layout device ('DataType 'Int64) shape
long tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Int64

-- | Returns a copy of the tensor converted to the 16-bit floating point format 'Half'.
half ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Half' copy
  Tensor requiresGradient layout device ('DataType 'Half) shape
half tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Half

-- | Returns a copy of the tensor converted to the 32-bit floating point format 'Float'.
float ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Float' copy
  Tensor requiresGradient layout device ('DataType 'Float) shape
float tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Float

-- | Returns a copy of the tensor converted to the 32-bit floating point format 'Double'.
double ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | 'Double' copy
  Tensor requiresGradient layout device ('DataType 'Double) shape
double tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Double

class SGetDataType (dataType :: DataType DType) where
  -- | Returns the gradually typed compute data type of the input tensor.
  --
  -- >>> sOnes' dataType = sOnes SWithGradient (SLayout SDense) (SDevice SCPU) dataType (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t = sOnes' $ SDataType SFloat
  -- >>> sDataType t
  -- SDataType SFloat
  -- >>> t = sOnes' $ SUncheckedDataType Float
  -- >>> sDataType t
  -- SUncheckedDataType Float
  sDataType ::
    forall m requiresGradient layout device shape.
    MonadFail m =>
    -- | input
    Tensor requiresGradient layout device dataType shape ->
    -- | data type of the input tensor
    m (SDataType dataType)

  -- | Returns the untyped compute data type of the input tensor.
  --
  -- >>> sOnes' dataType = sOnes SWithGradient (SLayout SDense) (SDevice SCPU) dataType (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t = sOnes' $ SDataType SFloat
  -- >>> dType t
  -- Float
  -- >>> t = sOnes' $ SUncheckedDataType Float
  -- >>> dType t
  -- Float
  dType ::
    forall m requiresGradient layout device shape.
    MonadFail m =>
    -- | input
    Tensor requiresGradient layout device dataType shape ->
    -- | data type of the input tensor
    m DType
  dType tensor = forgetIsChecked . fromSing <$> sDataType tensor

instance SGetDataType 'UncheckedDataType where
  sDataType tensor = pure . SUncheckedDataType . unsafePerformIO $ cast1 ATen.tensor_scalar_type tensor

instance SingI dType => SGetDataType ('DataType dType) where
  sDataType tensor
    | unsafePerformIO (cast1 ATen.tensor_scalar_type tensor) == fromSing (sing @dType) = pure . SDataType $ sing @dType
    | otherwise = fail $ "The tensor should have data type " <> show (fromSing $ sing @dType) <> " but hasn't. Please open a ticket on GitHub."

-- | Returns the input tensor but with 'UncheckedDataType' as data-type type annotation.
-- Any static information about the tensor's data type is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t = ones @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedDataType t
-- uncheckedDataType t
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        'UncheckedDataType
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
uncheckedDataType ::
  forall requiresGradient layout device dataType shape.
  -- | input tensor
  Tensor requiresGradient layout device dataType shape ->
  -- | tensor without checked data type
  Tensor requiresGradient layout device 'UncheckedDataType shape
uncheckedDataType = coerce

-- | Returns 'True' if the tensor has the data type 'dataType' and 'False' otherwise.
-- If 'dataType' is 'UncheckedDataType', 'True' is returned for consistency.
--
-- >>> t = sOnes SWithGradient (SLayout SDense) (SDevice SCPU) (SUncheckedDataType Float) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> checkDataType @('DataType 'Float) t
-- True
-- >>> checkDataType @('DataType 'Double) t
-- False
-- >>> checkDataType @'UncheckedDataType t
-- True
checkDataType ::
  forall (dataType :: DataType DType) requiresGradient layout device shape.
  (KnownDataType dataType) =>
  -- | tensor under consideration
  Tensor requiresGradient layout device 'UncheckedDataType shape ->
  -- | whether or not the input tensor has the data type 'dataType'
  Bool
checkDataType tensor =
  case dataTypeVal @dataType of
    UncheckedDataType -> True
    DataType dtype -> unsafePerformIO $ (dtype ==) <$> cast1 ATen.tensor_scalar_type tensor

-- | Checks whether or not the input tensor has the data type 'dataType'
-- and returns a statically annotated copy of it wrapped in a 'MonadFail' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor has indeed the data type 'dataType'.
-- If it does not have it, then the result will be 'Nothing'.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t = sOnes SWithGradient (SLayout SDense) (SDevice SCPU) (SUncheckedDataType Float) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' <- checkedDataType @('DataType 'Float) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
-- >>> t' <- checkedDataType @('DataType 'Double) t
-- *** Exception: user error (The tensor does not have the data type "DataType Double".)
checkedDataType ::
  forall (dataType :: DataType DType) m requiresGradient layout device shape.
  (KnownDataType dataType, MonadFail m) =>
  -- | input tensor
  Tensor requiresGradient layout device 'UncheckedDataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor requiresGradient layout device dataType shape)
checkedDataType tensor
  | checkDataType @dataType tensor = pure . coerce $ tensor
  | otherwise = fail $ "The tensor does not have the data type \"" <> show (dataTypeVal @dataType) <> "\"."

-- | Unsafe version of 'checkedDataType'.
-- If the tensor does not have the data type 'dataType', then the execution is stopped and an error message is displayed.
--
-- >>> t = sOnes SWithGradient (SLayout SDense) (SDevice SCPU) (SUncheckedDataType Float) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' = unsafeCheckedDataType @('DataType 'Float) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
-- >>> unsafeCheckedDataType @('DataType 'Double) t
-- *** Exception: The tensor does not have the data type "DataType Double".
-- CallStack (from HasCallStack):
--   error, called at ...
unsafeCheckedDataType ::
  forall (dataType :: DataType DType) requiresGradient layout device shape.
  KnownDataType dataType =>
  -- | input tensor
  Tensor requiresGradient layout device 'UncheckedDataType shape ->
  -- | annotated output tensor
  Tensor requiresGradient layout device dataType shape
unsafeCheckedDataType tensor = case checkedDataType @dataType tensor of
  Right tensor' -> tensor'
  Left err -> error err

class SGetShape (shape :: Shape [Dim (Name Symbol) (Size Nat)]) where
  -- | Returns the gradually typed shape of the input tensor.
  --
  -- >>> sOnes' = sOnes SWithGradient (SLayout SDense) (SDevice SCPU) (SDataType SFloat)
  -- >>> t = sOnes' . SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil
  -- >>> sShape t
  -- SShape (SCons (SDim {sDimName = SName, sDimSize = SSize}) (SCons (SDim {sDimName = SName, sDimSize = SSize}) SNil))
  -- >>> t = sOnes' . SUncheckedShape $ [Dim "batch" 32, Dim "feature" 8]
  -- >>> sShape t
  -- SUncheckedShape [Dim {dimName = "batch", dimSize = 32},Dim {dimName = "feature", dimSize = 8}]
  -- >>> t = sOnes' . SShape $ SUncheckedName "batch" :&: SUncheckedSize 32 :|: SUncheckedName "feature" :&: SSize @32 :|: SNil
  -- >>> sShape t
  -- SShape (SCons (SDim {sDimName = SUncheckedName "batch", sDimSize = SUncheckedSize 32}) (SCons (SDim {sDimName = SUncheckedName "feature", sDimSize = SSize}) SNil))
  -- >>> t = sOnes' . SShape $ SName @"batch" :&: SUncheckedSize 32 :|: SName @"feature" :&: SUncheckedSize 8 :|: SNil
  -- >>> sShape t
  -- SShape (SCons (SDim {sDimName = SName, sDimSize = SUncheckedSize 32}) (SCons (SDim {sDimName = SName, sDimSize = SUncheckedSize 8}) SNil))
  sShape ::
    forall requiresGradient layout device dataType m.
    MonadFail m =>
    Tensor requiresGradient layout device dataType shape ->
    m (SShape shape)

  -- | Returns the untyped shape of the input tensor.
  --
  -- >>> sOnes' = sOnes SWithGradient (SLayout SDense) (SDevice SCPU) (SDataType SFloat)
  -- >>> t = sOnes' . SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil
  -- >>> dims t
  -- [Dim {dimName = "batch", dimSize = 32},Dim {dimName = "feature", dimSize = 8}]
  -- >>> t = sOnes' . SUncheckedShape $ [Dim "batch" 32, Dim "feature" 8]
  -- >>> dims t
  -- [Dim {dimName = "batch", dimSize = 32},Dim {dimName = "feature", dimSize = 8}]
  -- >>> t = sOnes' . SShape $ SUncheckedName "batch" :&: SUncheckedSize 32 :|: SUncheckedName "feature" :&: SSize @32 :|: SNil
  -- >>> dims t
  -- [Dim {dimName = "batch", dimSize = 32},Dim {dimName = "feature", dimSize = 32}]
  -- >>> t = sOnes' . SShape $ SName @"batch" :&: SUncheckedSize 32 :|: SName @"feature" :&: SUncheckedSize 8 :|: SNil
  -- >>> dims t
  -- [Dim {dimName = "batch", dimSize = 32},Dim {dimName = "feature", dimSize = 8}]
  dims ::
    forall requiresGradient layout device dataType m.
    MonadFail m =>
    Tensor requiresGradient layout device dataType shape ->
    m [Dim String Integer]
  dims tensor = fmap (bimap forgetIsChecked forgetIsChecked) . forgetIsChecked . fromSing <$> sShape tensor

instance SGetShape 'UncheckedShape where
  sShape tensor = pure . SUncheckedShape . unsafePerformIO $ do
    sizes <- cast1 ATen.tensor_sizes tensor
    ifM
      (cast1 ATen.tensor_has_names tensor)
      ( do
          names <- cast1 ATen.tensor_names tensor
          return $ zipWith Dim names sizes
      )
      (return $ Dim "*" <$> sizes)

instance SGetDims dims => SGetShape ('Shape dims) where
  sShape tensor =
    let sizes =
          unsafePerformIO $
            ifM
              ((> (0 :: Int)) <$> cast1 ATen.tensor_dim tensor)
              (cast1 ATen.tensor_sizes tensor)
              (pure [])
        names =
          unsafePerformIO $
            ifM
              (cast1 ATen.tensor_has_names tensor)
              (cast1 ATen.tensor_names tensor)
              (pure $ map (const "*") sizes)
     in SShape <$> sDims names sizes

class SGetDims (dims :: [Dim (Name Symbol) (Size Nat)]) where
  sDims :: forall m. MonadFail m => [String] -> [Integer] -> m (SList dims)

dimsError :: forall m a. MonadFail m => m a
dimsError = fail "The numbers of compile- and runtime dimensions are not the same. Please open a ticket on GitHub."

dimNameError :: forall m a. MonadFail m => String -> String -> m a
dimNameError name name' =
  fail $
    "The compile- and runtime dimension names are not the same, '"
      <> name
      <> "' != '"
      <> name'
      <> "'. Please open a ticket on GitHub."

dimSizeError :: forall m a b. (MonadFail m, Show a) => a -> a -> m b
dimSizeError size size' =
  fail $
    "The compile- and runtime dimension sizes are not the same, '"
      <> show size
      <> "' != '"
      <> show size'
      <> "'. Please open a ticket on GitHub."

dimNameSizeError :: forall m a b. (MonadFail m, Show a) => String -> String -> a -> a -> m b
dimNameSizeError name name' size size' =
  fail $
    "The compile- and runtime dimension names and sizes are not the same, '"
      <> name
      <> "' != '"
      <> name'
      <> "' and '"
      <> show size
      <> "' != '"
      <> show size'
      <> "'. Please open a ticket on GitHub."

instance SGetDims '[] where
  sDims [] [] = pure SNil
  sDims _ _ = dimsError

instance (SGetDim dim, SGetDims dims) => SGetDims (dim : dims) where
  sDims (name : names) (size : sizes) = (:|:) <$> sDim name size <*> sDims names sizes
  sDims _ _ = dimsError

class SGetDim (dim :: Dim (Name Symbol) (Size Nat)) where
  sDim :: forall m. MonadFail m => String -> Integer -> m (SDim dim)

instance SGetDim ('Dim 'UncheckedName 'UncheckedSize) where
  sDim name size = pure $ SDim (SUncheckedName name) (SUncheckedSize size)

instance KnownSymbol name => SGetDim ('Dim ('Name name) 'UncheckedSize) where
  sDim name size = case symbolVal $ Proxy @name of
    name'
      | name == name' -> pure $ SDim (SName @name) (SUncheckedSize size)
      | otherwise -> dimNameError name name'

instance KnownNat size => SGetDim ('Dim 'UncheckedName ('Size size)) where
  sDim name size = case natVal $ Proxy @size of
    size'
      | size == size' -> pure $ SDim (SUncheckedName name) (SSize @size)
      | otherwise -> dimSizeError size size'

instance (KnownSymbol name, KnownNat size) => SGetDim ('Dim ('Name name) ('Size size)) where
  sDim name size = case (symbolVal $ Proxy @name, natVal $ Proxy @size) of
    (name', size')
      | name == name' && size == size' -> pure $ SDim (SName @name) (SSize @size)
      | name /= name' && size == size' -> dimNameError name name'
      | name == name' && size /= size' -> dimSizeError size size'
      | otherwise -> dimNameSizeError name name' size size'

-- | Returns the input tensor but with the selected dimension replaces with 'UncheckedDim' as dimension type annotation.
-- The static information about the selected tensor dimension is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t = ones @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedDim @('SelectDim ('ByName "batch")) t
-- uncheckedDim @('SelectDim ('ByName "batch")) t
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim 'UncheckedName 'UncheckedSize,
--              'Dim ('Name "feature") ('Size 8)])
-- >>> t = ones @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedDim @('SelectDim ('ByIndex 1)) t
-- uncheckedDim @('SelectDim ('ByIndex 1)) t
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim 'UncheckedName 'UncheckedSize])
uncheckedDim ::
  forall selectDim requiresGradient layout device dataType shape.
  -- | input tensor
  Tensor requiresGradient layout device dataType shape ->
  -- | tensor with the selected dimensions unchecked
  Tensor requiresGradient layout device dataType (ReplaceDimF selectDim shape ('Dim 'UncheckedName 'UncheckedSize))
uncheckedDim = coerce

-- | Returns the input tensor but with 'UncheckedShape' as shape type annotation.
-- Any static information about the tensor's shape is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t = ones @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedShape t
-- uncheckedShape t
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        'UncheckedShape
uncheckedShape ::
  forall requiresGradient layout device dataType shape.
  -- | input tensor
  Tensor requiresGradient layout device dataType shape ->
  -- | tensor without checked shape
  Tensor requiresGradient layout device dataType 'UncheckedShape
uncheckedShape = coerce

-- | Returns 'True' if the tensor has the shape 'shape' and 'False' otherwise.
-- If 'shape' is 'UncheckedShape', 'True' is returned for consistency.
--
-- >>> t = sOnes SWithGradient (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SUncheckedShape [Dim "batch" 32, Dim "feature" 8])
-- >>> checkShape @('Shape [ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) t
-- True
-- >>> checkShape @('Shape [ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 32)]) t
-- False
-- >>> checkShape @'UncheckedShape t
-- True
-- >>> checkShape @('Shape [ 'Dim 'UncheckedName ('Size 32), 'Dim ('Name "feature") 'UncheckedSize]) t
-- True
checkShape ::
  forall (shape :: Shape [Dim (Name Symbol) (Size Nat)]) requiresGradient layout device dataType.
  (KnownShape shape) =>
  -- | tensor under consideration
  Tensor requiresGradient layout device dataType 'UncheckedShape ->
  -- | whether or not the input tensor has the shape 'shape'
  Bool
checkShape tensor =
  case shapeVal @shape of
    UncheckedShape -> True
    Shape dims ->
      let sizes =
            unsafePerformIO $
              ifM
                ((> (0 :: Int)) <$> cast1 ATen.tensor_dim tensor)
                (cast1 ATen.tensor_sizes tensor)
                (pure [])
          names =
            unsafePerformIO $
              ifM
                (cast1 ATen.tensor_has_names tensor)
                (cast1 ATen.tensor_names tensor)
                (pure $ map (const "*") sizes)
          f (Dim UncheckedName UncheckedSize) _ _ = mempty
          f (Dim (Name name) UncheckedSize) name' _ = All $ name == name'
          f (Dim UncheckedName (Size size)) _ size' = All $ size == size'
          f (Dim (Name name) (Size size)) name' size' = All $ name == name' && size == size'
       in length dims == length names
            && length names == length sizes
            && (getAll . fold) (zipWith3 f dims names sizes)

-- | Checks whether or not the input tensor has the shape 'shape'
-- and returns a statically annotated copy of it wrapped in a 'MonadFail' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor has indeed the shape 'shape'.
-- If it is not, then the result will be 'Nothing'.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t = sOnes SWithGradient (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SUncheckedShape [Dim "batch" 32, Dim "feature" 8])
-- >>> t' <- checkedShape @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
-- >>> t' <- checkedShape @('Shape '[ 'Dim 'UncheckedName ('Size 32), 'Dim ('Name "feature") 'UncheckedSize]) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim 'UncheckedName ('Size 32),
--              'Dim ('Name "feature") 'UncheckedSize])
-- >>> t' <- checkedShape @('Shape [ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 32)]) t
-- *** Exception: user error (The tensor does not have the shape "Shape [Dim {dimName = Name "batch", dimSize = Size 32},Dim {dimName = Name "feature", dimSize = Size 32}]".)
checkedShape ::
  forall (shape :: Shape [Dim (Name Symbol) (Size Nat)]) m requiresGradient layout device dataType.
  (KnownShape shape, MonadFail m) =>
  -- | input tensor
  Tensor requiresGradient layout device dataType 'UncheckedShape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor requiresGradient layout device dataType shape)
checkedShape tensor
  | checkShape @shape tensor = pure . coerce $ tensor
  | otherwise = fail $ "The tensor does not have the shape \"" <> show (shapeVal @shape) <> "\"."

-- | Unsafe version of 'checkedShape'.
-- If the tensor does not have the shape 'shape', then the execution is stopped and an error message is displayed.
--
-- >>> t = sOnes SWithGradient (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SUncheckedShape [Dim "batch" 32, Dim "feature" 8])
-- >>> t' = unsafeCheckedShape @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
-- >>> unsafeCheckedShape @('Shape [ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 32)]) t
-- *** Exception: The tensor does not have the shape "Shape [Dim {dimName = Name "batch", dimSize = Size 32},Dim {dimName = Name "feature", dimSize = Size 32}]".
-- CallStack (from HasCallStack):
--   error, called at ...
unsafeCheckedShape ::
  forall (shape :: Shape [Dim (Name Symbol) (Size Nat)]) requiresGradient layout device dataType.
  KnownShape shape =>
  -- | input tensor
  Tensor requiresGradient layout device dataType 'UncheckedShape ->
  -- | annotated output tensor
  Tensor requiresGradient layout device dataType shape
unsafeCheckedShape tensor = case checkedShape @shape tensor of
  Right tensor' -> tensor'
  Left err -> error err
