{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
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

import Control.Exception (Exception (..))
import Control.Monad.Catch (MonadThrow (..))
import Data.Bifunctor (bimap)
import Data.Coerce (coerce)
import Data.Int (Int16)
import Data.Proxy (Proxy (..))
import Data.Singletons (SingI (sing), SingKind (fromSing))
import Data.Singletons.Prelude.List (SList (..))
import Data.Typeable (Typeable)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (KnownNat, KnownSymbol, Nat, Symbol, natVal, symbolVal)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.Prelude (Seq, forgetIsChecked, ifM)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Scalar ()
import Torch.GraduallyTyped.Shape.Class (ReplaceDimF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim (..), SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:|:))
import Torch.GraduallyTyped.Unify (type (<+>))
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
--                          ┌─► Compute device, e.g. `'Device 'CPU`
--                          │
--                          │               ┌─► List of dimensions, e.g. `'Shape '[ 'Dim 'UncheckedName ('Size 8), 'Dim 'UncheckedName ('Size 1) ]`
--                          │               │
-- Tensor gradient layout device dataType shape
--           │       │              │
--           │       │              └─► Data type, e.g. `'DataType 'Float`
--           │       │
--           │       └─► Memory layout, e.g. `'Layout 'Dense`
--           │
--           └─► Whether or not the tensor requires a gradient, e.g. `'Gradient 'WithGradient` for one that does
-- @
newtype
  Tensor
    (gradient :: Gradient RequiresGradient)
    (layout :: Layout LayoutType)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (shape :: Shape [Dim (Name Symbol) (Size Nat)])
  where
  -- | Unsafe constructor for tensors.
  -- Do not call this constructor directly,
  -- use smart constructors like 'ones' or 'randn' instead.
  UnsafeTensor ::
    forall gradient layout device dataType shape.
    ForeignPtr ATen.Tensor ->
    Tensor gradient layout device dataType shape

type role Tensor nominal nominal nominal nominal nominal

instance Show (Tensor gradient layout device dataType shape) where
  show (UnsafeTensor t) = show (Torch.Tensor.Unsafe t)

-- | Alias for an untyped tensor without gradients.
type UncheckedTensor = Tensor 'UncheckedGradient 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape

-- | Alias for an untyped tensor with gradients.
type UncheckedParameter = Tensor ('Gradient 'WithGradient) 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape

-- | Alias for a tensor on CPU memory without gradients.
type CPUTensor = Tensor ('Gradient 'WithoutGradient) ('Layout 'Dense) ('Device 'CPU)

-- | Alias for a tensor on CPU memory with gradients.
type CPUParameter = Tensor ('Gradient 'WithGradient) ('Layout 'Dense) ('Device 'CPU)

-- | Alias for a sparse tensor on CPU memory without gradients.
type SparseCPUTensor = Tensor ('Gradient 'WithoutGradient) ('Layout 'Sparse) ('Device 'CPU)

-- | Alias for a sparse tensor on CPU memory with gradients.
type SparseCPUParameter = Tensor ('Gradient 'WithGradient) ('Layout 'Sparse) ('Device 'CPU)

-- | Alias for a tensor on CUDA memory without gradients.
type CUDATensor deviceId = Tensor ('Gradient 'WithoutGradient) ('Layout 'Dense) ('Device ('CUDA deviceId))

-- | Alias for a tensor on CUDA memory with gradients.
type CUDAParameter deviceId = Tensor ('Gradient 'WithGradient) ('Layout 'Dense) ('Device ('CUDA deviceId))

-- | Alias for a sparse tensor on CUDA memory without gradients.
type SparseCUDATensor deviceId = Tensor ('Gradient 'WithoutGradient) ('Layout 'Sparse) ('Device ('CUDA deviceId))

-- | Alias for a sparse tensor on CUDA memory with gradients.
type SparseCUDAParameter deviceId = Tensor ('Gradient 'WithGradient) ('Layout 'Sparse) ('Device ('CUDA deviceId))

instance Num (Tensor gradient layout device dataType shape) where
  (+) = (unsafePerformIO .) . cast2 ATen.add_tt
  (-) = (unsafePerformIO .) . cast2 ATen.sub_tt
  (*) = (unsafePerformIO .) . cast2 ATen.mul_tt
  negate = unsafePerformIO . cast1 ATen.neg_t
  abs = unsafePerformIO . cast1 ATen.abs_t
  signum = unsafePerformIO . cast1 ATen.sign_t
  fromInteger _a = undefined

instance
  Castable
    (Tensor gradient layout device dataType shape)
    (ForeignPtr ATen.Tensor)
  where
  cast (UnsafeTensor atenTensor) f = f atenTensor
  uncast atenTensor f = f $ UnsafeTensor atenTensor

instance
  Castable
    [Tensor gradient layout device dataType shape]
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
  Castable (HList (Tensor gradient layout device dataType shape ': tensors)) [ForeignPtr ATen.Tensor]
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

-- | Takes a tensor that may or may not require gradient computations
-- and returns a copy that does not require them.
withoutGradient ::
  forall gradient layout device dataType shape.
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | copy of the input tensor without gradient computations turned off.
  IO (Tensor ('Gradient 'WithoutGradient) layout device dataType shape)
withoutGradient tensor = cast2 ATen.tensor_set_requires_grad_b tensor False

-- | Takes a tensor that does not requires gradient computations
-- and returns a copy that requires them.
withGradient ::
  forall gradient layout device dataType shape.
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | copy of the input tensor with gradient computations turned on.
  IO (Tensor ('Gradient 'WithGradient) layout device dataType shape)
withGradient tensor = cast2 ATen.tensor_set_requires_grad_b tensor True

class SGetGradient (gradient :: Gradient RequiresGradient) where
  -- | Returns the gradually typed information for whether or not gradient computations for the tensor are turned on.
  --
  -- >>> sOnes' gradient = sOnes gradient (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t = sOnes' $ SGradient SWithGradient
  -- >>> sGradient t
  -- SGradient SWithGradient
  -- >>> t = sOnes' $ SUncheckedGradient WithoutGradient
  -- >>> sGradient t
  -- SUncheckedGradient WithoutGradient
  sGradient ::
    forall layout device dataType shape.
    -- | input tensor
    Tensor gradient layout device dataType shape ->
    -- | information about whether or not gradient computations are required
    SGradient gradient

  -- | Returns the untyped memory layout of the input tensor.
  --
  -- >>> sOnes' gradient = sOnes gradient (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t = sOnes' $ SGradient SWithGradient
  -- >>> requiresGradient t
  -- WithGradient
  -- >>> t = sOnes' $ SUncheckedGradient WithoutGradient
  -- >>> requiresGradient t
  -- WithoutGradient
  requiresGradient ::
    forall layout device dataType shape.
    -- | input tensor
    Tensor gradient layout device dataType shape ->
    -- | information about whether or not gradient computations are required
    RequiresGradient
  requiresGradient tensor = forgetIsChecked . fromSing $ sGradient tensor

instance SGetGradient 'UncheckedGradient where
  sGradient tensor
    | unsafePerformIO (cast1 ATen.tensor_requires_grad tensor) = SUncheckedGradient WithGradient
    | otherwise = SUncheckedGradient WithoutGradient

instance SGetGradient ('Gradient 'WithGradient) where
  sGradient tensor
    | unsafePerformIO (cast1 ATen.tensor_requires_grad tensor) = SGradient SWithGradient
    | otherwise =
      error $
        "The tensor should require gradient computations but doesn't. "
          <> gitHubErrorMsg

instance SGetGradient ('Gradient 'WithoutGradient) where
  sGradient tensor
    | unsafePerformIO (cast1 ATen.tensor_requires_grad tensor) =
      error $
        "The tensor should not require gradient computations but does. "
          <> gitHubErrorMsg
    | otherwise = SGradient SWithoutGradient

data GradientError = GradientError {geExpected :: RequiresGradient, geActual :: RequiresGradient}
  deriving stock (Show, Typeable)

instance Exception GradientError where
  displayException GradientError {..} =
    "The tensor's information about whether or not gradient computations are required reads `"
      <> show geActual
      <> "` instead of `"
      <> show geExpected
      <> "`."

-- | Checks whether or not the input tensor has the memory layout 'layout'
-- and returns a statically annotated copy of it wrapped in a 'MonadThrow' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor has indeed the memory layout 'layout'.
-- If it does not have it, then the result will be 'Nothing'.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t = sOnes SWithGradient (SUncheckedLayout Dense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' <- sCheckedLayout (SLayout SDense) t
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
-- >>> t' <- sCheckedLayout (SLayout SSparse) t
-- *** Exception: user error (The tensor does not have the memory layout "Layout Sparse".)
sCheckedGradient ::
  forall gradient' m gradient layout device dataType shape.
  (SGetGradient gradient, MonadThrow m) =>
  -- | layout
  SGradient gradient' ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor (Seq (gradient <+> gradient') gradient') layout device dataType shape)
sCheckedGradient gradient' tensor =
  let actualGradient = forgetIsChecked . fromSing $ sGradient tensor
      expectedGradient = forgetIsChecked . fromSing $ gradient'
   in if actualGradient == expectedGradient
        then pure . coerce $ tensor
        else throwM $ GradientError expectedGradient actualGradient

checkedGradient ::
  forall gradient' m gradient layout device dataType shape.
  (SingI gradient', SGetGradient gradient, MonadThrow m) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor (Seq (gradient <+> gradient') gradient') layout device dataType shape)
checkedGradient = sCheckedGradient (sing @gradient')

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
uncheckedGradient ::
  forall gradient layout device dataType shape.
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | tensor without checked layout
  Tensor 'UncheckedGradient layout device dataType shape
uncheckedGradient = coerce

-- | Returns a dense copy of the tensor.
toDense ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | dense copy
  Tensor gradient ('Layout 'Dense) device dataType shape
toDense = unsafePerformIO . cast1 ATen.tensor_to_dense

-- | Returns a sparse copy of the tensor.
toSparse ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | sparse copy
  Tensor gradient ('Layout 'Sparse) device dataType shape
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
    forall gradient device dataType shape.
    -- | input
    Tensor gradient layout device dataType shape ->
    -- | memory layout
    SLayout layout

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
    forall gradient device dataType shape.
    -- | input
    Tensor gradient layout device dataType shape ->
    -- | memory layout
    LayoutType
  layoutType tensor = forgetIsChecked . fromSing $ sLayout tensor

instance SGetLayout 'UncheckedLayout where
  sLayout tensor
    | unsafePerformIO (cast1 ATen.tensor_is_sparse tensor) = SUncheckedLayout Sparse
    | otherwise = SUncheckedLayout Dense

instance SGetLayout ('Layout 'Sparse) where
  sLayout tensor
    | unsafePerformIO (cast1 ATen.tensor_is_sparse tensor) = SLayout SSparse
    | otherwise =
      error $
        "The tensor should be sparse but isn't. "
          <> gitHubErrorMsg

instance SGetLayout ('Layout 'Dense) where
  sLayout tensor
    | unsafePerformIO (cast1 ATen.tensor_is_sparse tensor) =
      error $
        "The tensor should be dense but isn't. "
          <> gitHubErrorMsg
    | otherwise = SLayout SDense

data LayoutError = LayoutError {leExpected :: LayoutType, leActual :: LayoutType}
  deriving stock (Show, Typeable)

instance Exception LayoutError where
  displayException LayoutError {..} =
    "The tensor does not have the memory layout `"
      <> show leExpected
      <> "` but `"
      <> show leActual
      <> "`."

-- | Checks whether or not the input tensor has the memory layout 'layout'
-- and returns a statically annotated copy of it wrapped in a 'MonadThrow' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor has indeed the memory layout 'layout'.
-- If it does not have it, then the result will be 'Nothing'.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t = sOnes SWithGradient (SUncheckedLayout Dense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' <- sCheckedLayout (SLayout SDense) t
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
-- >>> t' <- sCheckedLayout (SLayout SSparse) t
-- *** Exception: user error (The tensor does not have the memory layout "Layout Sparse".)
sCheckedLayout ::
  forall layout' m gradient layout device dataType shape.
  (SGetLayout layout, MonadThrow m) =>
  -- | layout
  SLayout layout' ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient (Seq (layout <+> layout') layout') device dataType shape)
sCheckedLayout layout' tensor =
  let actualLayout = forgetIsChecked . fromSing $ sLayout tensor
      expectedLayout = forgetIsChecked . fromSing $ layout'
   in if actualLayout == expectedLayout
        then pure . coerce $ tensor
        else throwM $ LayoutError expectedLayout actualLayout

checkedLayout ::
  forall layout' m gradient layout device dataType shape.
  (SingI layout', SGetLayout layout, MonadThrow m) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient (Seq (layout <+> layout') layout') device dataType shape)
checkedLayout = sCheckedLayout (sing @layout')

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
  forall gradient layout device dataType shape.
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | tensor without checked layout
  Tensor gradient 'UncheckedLayout device dataType shape
uncheckedLayout = coerce

-- | Returns a copy of the tensor in CPU memory.
cpu ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | copy in CPU memory
  Tensor gradient layout ('Device 'CPU) dataType shape
cpu = unsafePerformIO . cast1 ATen.tensor_cpu

-- | Returns a copy of the tensor in CUDA memory.
cuda ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | copy in CUDA memory
  Tensor gradient layout ('Device ('CUDA 0)) dataType shape
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
    forall gradient layout dataType shape.
    -- | input
    Tensor gradient layout device dataType shape ->
    -- | compute device of the input tensor
    SDevice device

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
    forall gradient layout dataType shape.
    -- | input
    Tensor gradient layout device dataType shape ->
    -- | compute device of the input tensor
    DeviceType Int16
  deviceType tensor = forgetIsChecked . fromSing $ sDevice tensor

instance SGetDevice 'UncheckedDevice where
  sDevice tensor
    | unsafePerformIO (cast0 ATen.hasCUDA) && unsafePerformIO (cast1 ATen.tensor_is_cuda tensor) =
      case unsafePerformIO (cast1 ATen.tensor_get_device tensor) :: Int of
        deviceIndex -> SUncheckedDevice . CUDA . fromIntegral $ deviceIndex
    | otherwise = SUncheckedDevice CPU

instance SGetDevice ('Device 'CPU) where
  sDevice tensor
    | unsafePerformIO (cast0 ATen.hasCUDA) && unsafePerformIO (cast1 ATen.tensor_is_cuda tensor) =
      error $
        "The tensor should be on CPU but is on CUDA. "
          <> gitHubErrorMsg
    | otherwise = SDevice SCPU

instance KnownNat deviceIndex => SGetDevice ('Device ('CUDA deviceIndex)) where
  sDevice tensor
    | unsafePerformIO (cast0 ATen.hasCUDA) && unsafePerformIO (cast1 ATen.tensor_is_cuda tensor) =
      case unsafePerformIO (cast1 ATen.tensor_get_device tensor) :: Int of
        deviceIndex
          | deviceIndex == fromIntegral (natVal (Proxy @deviceIndex)) -> SDevice SCUDA
          | otherwise ->
            error $
              "The tensor should be on CUDA device "
                <> show (natVal (Proxy @deviceIndex))
                <> " but is on device "
                <> show deviceIndex
                <> ". "
                <> gitHubErrorMsg
    | otherwise =
      error $
        "The tensor should be on CUDA but is on CPU. "
          <> gitHubErrorMsg

data DeviceError = DeviceError {deExpected :: DeviceType Int16, deActual :: DeviceType Int16}
  deriving stock (Show, Typeable)

instance Exception DeviceError where
  displayException DeviceError {..} =
    "The tensor is not in the memory of the device `"
      <> show deExpected
      <> "` but `"
      <> show deActual
      <> "`."

-- | Checks whether or not the input tensor is in the memory of 'device'
-- and returns a statically annotated copy of it wrapped in a 'MonadThrow' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor is indeed on 'device'.
-- If it is not, then the result will be 'Nothing'.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t = sOnes SWithGradient (SLayout SDense) (SUncheckedDevice CPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' <- sCheckedDevice (SDevice SCPU) t
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
-- >>> t' <- sCheckedDevice (SDevice SCUDA @0) t
-- *** Exception: user error (The tensor is not in the memory of the device "Device (CUDA 0)".)
sCheckedDevice ::
  forall device' m gradient layout device dataType shape.
  (SGetDevice device, MonadThrow m) =>
  -- | device
  SDevice device' ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout (Seq (device <+> device') device') dataType shape)
sCheckedDevice device' tensor =
  let actualDevice = forgetIsChecked . fromSing $ sDevice tensor
      expectedDevice = forgetIsChecked . fromSing $ device'
   in if actualDevice == expectedDevice
        then pure . coerce $ tensor
        else throwM $ DeviceError expectedDevice actualDevice

checkedDevice ::
  forall device' m gradient layout device dataType shape.
  (SingI device', SGetDevice device, MonadThrow m) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout (Seq (device <+> device') device') dataType shape)
checkedDevice = sCheckedDevice (sing @device')

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
  forall gradient layout device dataType shape.
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | tensor without checked device
  Tensor gradient layout 'UncheckedDevice dataType shape
uncheckedDevice = coerce

-- | Returns a copy of the tensor converted to 'Bool'.
bool ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Bool' copy
  Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'Bool) shape
bool tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Bool

-- | Returns a copy of the tensor converted to 'UInt8'.
byte ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'UInt8' copy
  Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'UInt8) shape
byte tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor UInt8

-- | Returns a copy of the tensor converted to 'Int8'.
char ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Int8' copy
  Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'Int8) shape
char tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Int8

-- | Returns a copy of the tensor converted to 'Int16'.
short ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Int16' copy
  Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'Int16) shape
short tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Int16

-- | Returns a copy of the tensor converted to 'Int32'.
int ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Int32' copy
  Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'Int32) shape
int tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Int32

-- | Returns a copy of the tensor converted to 'Int64'.
long ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Int64' copy
  Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'Int64) shape
long tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Int64

-- | Returns a copy of the tensor converted to the 16-bit floating point format 'Half'.
half ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Half' copy
  Tensor gradient layout device ('DataType 'Half) shape
half tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Half

-- | Returns a copy of the tensor converted to the 32-bit floating point format 'Float'.
float ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Float' copy
  Tensor gradient layout device ('DataType 'Float) shape
float tensor = unsafePerformIO $ cast2 ATen.tensor_toType_s tensor Float

-- | Returns a copy of the tensor converted to the 32-bit floating point format 'Double'.
double ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Double' copy
  Tensor gradient layout device ('DataType 'Double) shape
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
    forall gradient layout device shape.
    -- | input
    Tensor gradient layout device dataType shape ->
    -- | data type of the input tensor
    SDataType dataType

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
    forall gradient layout device shape.
    -- | input
    Tensor gradient layout device dataType shape ->
    -- | data type of the input tensor
    DType
  dType tensor = forgetIsChecked . fromSing $ sDataType tensor

instance SGetDataType 'UncheckedDataType where
  sDataType tensor = SUncheckedDataType . unsafePerformIO $ cast1 ATen.tensor_scalar_type tensor

instance SingI dType => SGetDataType ('DataType dType) where
  sDataType tensor
    | unsafePerformIO (cast1 ATen.tensor_scalar_type tensor) == fromSing (sing @dType) = SDataType $ sing @dType
    | otherwise =
      error $
        "The tensor should have data type "
          <> show (fromSing $ sing @dType)
          <> " but hasn't. "
          <> gitHubErrorMsg

data DataTypeError = DataTypeError {dtExpected :: DType, dtActual :: DType}
  deriving stock (Show, Typeable)

instance Exception DataTypeError where
  displayException DataTypeError {..} =
    "The tensor does not have the data type `"
      <> show dtExpected
      <> "` but `"
      <> show dtActual
      <> "`."

-- | Checks whether or not the input tensor has the data type 'dataType'
-- and returns a statically annotated copy of it wrapped in a 'MonadThrow' 'm'.
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
sCheckedDataType ::
  forall dataType' m gradient layout device dataType shape.
  (SGetDataType dataType, MonadThrow m) =>
  -- | data type
  SDataType dataType' ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout device (Seq (dataType <+> dataType') dataType') shape)
sCheckedDataType dataType' tensor =
  let actualDataType = forgetIsChecked . fromSing $ sDataType tensor
      expectedDataType = forgetIsChecked . fromSing $ dataType'
   in if actualDataType == expectedDataType
        then pure . coerce $ tensor
        else throwM $ DataTypeError expectedDataType actualDataType

checkedDataType ::
  forall dataType' m gradient layout device dataType shape.
  (SingI dataType', SGetDataType dataType, MonadThrow m) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout device (Seq (dataType <+> dataType') dataType') shape)
checkedDataType = sCheckedDataType (sing @dataType')

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
  forall gradient layout device dataType shape.
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | tensor without checked data type
  Tensor gradient layout device 'UncheckedDataType shape
uncheckedDataType = coerce

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
    forall gradient layout device dataType.
    Tensor gradient layout device dataType shape ->
    SShape shape

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
    forall gradient layout device dataType.
    Tensor gradient layout device dataType shape ->
    [Dim String Integer]
  dims tensor = fmap (bimap forgetIsChecked forgetIsChecked) . forgetIsChecked . fromSing $ sShape tensor

instance SGetShape 'UncheckedShape where
  sShape tensor = SUncheckedShape . unsafePerformIO $ do
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
     in SShape $ sDims names sizes

class SGetDims (dims :: [Dim (Name Symbol) (Size Nat)]) where
  sDims :: [String] -> [Integer] -> SList dims

dimsError :: forall a. a
dimsError = error $ "The numbers of compile- and runtime dimensions are not the same. " <> gitHubErrorMsg

dimNameError :: forall a. String -> String -> a
dimNameError name name' =
  error $
    "The compile- and runtime dimension names are not the same, '"
      <> name
      <> "' != '"
      <> name'
      <> "'. "
      <> gitHubErrorMsg

dimSizeError :: forall a b. Show a => a -> a -> b
dimSizeError size size' =
  error $
    "The compile- and runtime dimension sizes are not the same, '"
      <> show size
      <> "' != '"
      <> show size'
      <> "'. "
      <> gitHubErrorMsg

dimNameSizeError :: forall a b. Show a => String -> String -> a -> a -> b
dimNameSizeError name name' size size' =
  error $
    "The compile- and runtime dimension names and sizes are not the same, '"
      <> name
      <> "' != '"
      <> name'
      <> "' and '"
      <> show size
      <> "' != '"
      <> show size'
      <> "'. "
      <> gitHubErrorMsg

instance SGetDims '[] where
  sDims [] [] = SNil
  sDims _ _ = dimsError

instance (SGetDim dim, SGetDims dims) => SGetDims (dim : dims) where
  sDims (name : names) (size : sizes) = sDim name size :|: sDims names sizes
  sDims _ _ = dimsError

class SGetDim (dim :: Dim (Name Symbol) (Size Nat)) where
  sDim :: String -> Integer -> SDim dim

instance SGetDim ('Dim 'UncheckedName 'UncheckedSize) where
  sDim name size = SDim (SUncheckedName name) (SUncheckedSize size)

instance KnownSymbol name => SGetDim ('Dim ('Name name) 'UncheckedSize) where
  sDim name size = case symbolVal $ Proxy @name of
    name'
      | name == name' -> SDim (SName @name) (SUncheckedSize size)
      | otherwise -> dimNameError name name'

instance KnownNat size => SGetDim ('Dim 'UncheckedName ('Size size)) where
  sDim name size = case natVal $ Proxy @size of
    size'
      | size == size' -> SDim (SUncheckedName name) (SSize @size)
      | otherwise -> dimSizeError size size'

instance (KnownSymbol name, KnownNat size) => SGetDim ('Dim ('Name name) ('Size size)) where
  sDim name size = case (symbolVal $ Proxy @name, natVal $ Proxy @size) of
    (name', size')
      | name == name' && size == size' -> SDim (SName @name) (SSize @size)
      | name /= name' && size == size' -> dimNameError name name'
      | name == name' && size /= size' -> dimSizeError size size'
      | otherwise -> dimNameSizeError name name' size size'

data ShapeError = ShapeError {seExpected :: [Dim String Integer], seActual :: [Dim String Integer]}
  deriving stock (Show)

instance Exception ShapeError where
  displayException ShapeError {..} =
    "The tensor does not have the shape `"
      <> show seExpected
      <> "` but `"
      <> show seActual
      <> "`."

-- | Checks whether or not the input tensor has the shape 'shape'
-- and returns a statically annotated copy of it wrapped in a 'MonadThrow' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor has indeed the shape 'shape'.
-- If it is not, then the result will be 'Nothing'.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t = sOnes SWithGradient (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SUncheckedShape [Dim "batch" 32, Dim "feature" 8])
-- >>> t' <- sCheckedShape (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil) t
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
-- >>> t' <- sCheckedShape (SShape $ SUncheckedName "batch" :&: SSize @32 :|: SName @"feature" :&: SUncheckedSize 8 :|: SNil) t
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
-- >>> t' <- sCheckedShape (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SUncheckedSize 32 :|: SNil) t
-- *** Exception: user error (The tensor does not have the shape "Shape [Dim {dimName = Name "batch", dimSize = Size 32},Dim {dimName = Name "feature", dimSize = Size 32}]".)
sCheckedShape ::
  forall shape' m gradient layout device dataType shape.
  (SGetShape shape, MonadThrow m) =>
  -- | shape
  SShape shape' ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout device dataType (Seq (shape <+> shape') shape'))
sCheckedShape shape' tensor =
  let f = fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size)) . forgetIsChecked . fromSing
      actualShape = f $ sShape tensor
      expectedShape = f shape'
   in if actualShape == expectedShape
        then pure . coerce $ tensor
        else throwM $ ShapeError expectedShape actualShape

checkedShape ::
  forall shape' m gradient layout device dataType shape.
  (SingI shape', SGetShape shape, MonadThrow m) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout device dataType (Seq (shape <+> shape') shape'))
checkedShape = sCheckedShape (sing @shape')

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
  forall selectDim gradient layout device dataType shape.
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | tensor with the selected dimensions unchecked
  Tensor gradient layout device dataType (ReplaceDimF selectDim shape ('Dim 'UncheckedName 'UncheckedSize))
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
  forall gradient layout device dataType shape.
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | tensor without checked shape
  Tensor gradient layout device dataType 'UncheckedShape
uncheckedShape = coerce

gitHubErrorMsg :: String
gitHubErrorMsg = "Please open a ticket on GitHub."