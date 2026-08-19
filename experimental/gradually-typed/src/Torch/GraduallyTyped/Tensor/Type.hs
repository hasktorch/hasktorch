{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE ViewPatterns #-}

module Torch.GraduallyTyped.Tensor.Type where

import Control.Applicative (empty)
import Control.Category ((>>>))
import Control.Exception (Exception (..))
import Control.Monad (forM, forM_, when, (<=<), (>=>))
import Control.Monad.Catch (MonadThrow (..))
import Data.Bifunctor (bimap)
import Data.Coerce (coerce)
import Data.Foldable (traverse_)
import Data.Functor ((<&>))
import Data.Int (Int16,Int64)
import Data.List.NonEmpty (NonEmpty ((:|)), nonEmpty, unzip)
import Data.Maybe (maybeToList)
import Data.Proxy (Proxy (..))
import Data.Singletons (SingI (sing), fromSing)
import Data.Typeable (Typeable)
import qualified Data.Vector as V hiding (uncons)
import qualified Data.Vector.Generic.Sized.Internal as SVI
import qualified Data.Vector.Sized as SV
import Foreign (Ptr, Word8, castPtr, fromBool, peekElemOff, pokeElemOff, withForeignPtr)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, KnownSymbol, Nat, Symbol, natVal, symbolVal)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Internal.TensorOptions (tensorOptions)
import qualified Torch.GraduallyTyped.Internal.Vector as V
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.Prelude (Catch, forgetIsChecked, ifM, pattern (:|:))
import Torch.GraduallyTyped.Prelude.List (SList (..))
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (InsertDimF, ReplaceDimF)
import Torch.GraduallyTyped.Shape.Type (By (ByIndex), Dim (..), Name (..), SDim (..), SName (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.HList (HList (..), pattern (:.))
import Torch.Internal.Cast (cast0, cast1, cast2, cast4)
import Torch.Internal.Class (Castable (..))
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Extra as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Type.TensorOptions as ATen
import qualified Torch.Internal.Type as ATen (Tensor, TensorList, TensorOptions)
import qualified Torch.Internal.Unmanaged.Type.Tensor as Unmanaged (tensor_data_ptr)
import qualified Torch.Tensor (Tensor (Unsafe))
import Prelude hiding (unzip, unzip3)

-- $setup
-- >>> import Torch.GraduallyTyped.Prelude.List (SList (..))
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

data
  TensorSpec
    (gradient :: Gradient RequiresGradient)
    (layout :: Layout LayoutType)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (shape :: Shape [Dim (Name Symbol) (Size Nat)])
  where
  TensorSpec ::
    forall gradient layout device dataType shape.
    { tsGradient :: SGradient gradient,
      tsLayout :: SLayout layout,
      tsDevice :: SDevice device,
      tsDataType :: SDataType dataType,
      tsShape :: SShape shape
    } ->
    TensorSpec gradient layout device dataType shape
  deriving stock (Show, Generic)

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
  cast (HCons (tensor, tensors)) f = do
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

-- | Turn gradient computations off or on for a tensor.
sSetGradient ::
  forall gradient gradient' layout device dataType shape.
  SGradient gradient ->
  Tensor gradient' layout device dataType shape ->
  IO (Tensor gradient layout device dataType shape)
sSetGradient gradient tensor =
  case forgetIsChecked (fromSing gradient) of
    WithoutGradient -> cast2 ATen.tensor_set_requires_grad_b tensor False
    WithGradient -> cast2 ATen.tensor_set_requires_grad_b tensor True

class SGetGradient (gradient :: Gradient RequiresGradient) where
  -- | Returns the gradually typed information for whether or not gradient computations for the tensor are turned on.
  --
  -- >>> sOnes' gradient = sOnes $ TensorSpec gradient (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t <- sOnes' $ SGradient SWithGradient
  -- >>> sGetGradient t
  -- SGradient SWithGradient
  -- >>> t <- sOnes' $ SUncheckedGradient WithoutGradient
  -- >>> sGetGradient t
  -- SUncheckedGradient WithoutGradient
  sGetGradient ::
    forall layout device dataType shape.
    -- | input tensor
    Tensor gradient layout device dataType shape ->
    -- | information about whether or not gradient computations are required
    SGradient gradient

  -- | Returns the untyped memory layout of the input tensor.
  --
  -- >>> sOnes' gradient = sOnes $ TensorSpec gradient (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t <- sOnes' $ SGradient SWithGradient
  -- >>> getRequiresGradient t
  -- WithGradient
  -- >>> t <- sOnes' $ SUncheckedGradient WithoutGradient
  -- >>> getRequiresGradient t
  -- WithoutGradient
  getRequiresGradient ::
    forall layout device dataType shape.
    -- | input tensor
    Tensor gradient layout device dataType shape ->
    -- | information about whether or not gradient computations are required
    RequiresGradient
  getRequiresGradient tensor = forgetIsChecked . fromSing $ sGetGradient tensor

instance SGetGradient 'UncheckedGradient where
  sGetGradient tensor
    | unsafePerformIO (cast1 ATen.tensor_requires_grad tensor) = SUncheckedGradient WithGradient
    | otherwise = SUncheckedGradient WithoutGradient

instance SGetGradient ('Gradient 'WithGradient) where
  sGetGradient tensor
    | unsafePerformIO (cast1 ATen.tensor_requires_grad tensor) = SGradient SWithGradient
    | otherwise =
      error $
        "The tensor should require gradient computations but doesn't. "
          <> gitHubErrorMsg

instance SGetGradient ('Gradient 'WithoutGradient) where
  sGetGradient tensor
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

-- | Checks whether or not gradient computations are required for a tensor
-- and returns a statically annotated copy of it wrapped in a 'MonadThrow' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just'
-- if and only if gradients are computed for the tensor according to the argument @gradient@.
-- If gradients are expected but none are computed, then the result will be 'Nothing'.
-- If gradients are not expected but are computed, then the result will be 'Nothing' as well.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t <- sOnes $ TensorSpec (SUncheckedGradient WithGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' <- sCheckedGradient (SGradient SWithGradient) t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "batch") ('Size 32),
--            'Dim ('Name "feature") ('Size 8)])
-- >>> t' <- sCheckedGradient (SGradient SWithoutGradient) t
-- *** Exception: GradientError {geExpected = WithoutGradient, geActual = WithGradient}
sCheckedGradient ::
  forall gradient' m gradient layout device dataType shape.
  (SGetGradient gradient, MonadThrow m, Catch (gradient <+> gradient')) =>
  -- | layout
  SGradient gradient' ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient' layout device dataType shape)
sCheckedGradient gradient' tensor =
  let actualGradient = forgetIsChecked . fromSing $ sGetGradient tensor
      expectedGradient = forgetIsChecked . fromSing $ gradient'
   in if actualGradient == expectedGradient
        then pure . coerce $ tensor
        else throwM $ GradientError expectedGradient actualGradient

checkedGradient ::
  forall gradient' m gradient layout device dataType shape.
  (SingI gradient', SGetGradient gradient, MonadThrow m, Catch (gradient <+> gradient')) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient' layout device dataType shape)
checkedGradient = sCheckedGradient (sing @gradient')

-- | Returns the input tensor but with 'UncheckedGradeint' as gradient type annotation.
-- Any static information about whether or not the gradient computation is required for the tensor is lost.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t <- ones @('Gradient 'WithGradient) @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedGradient t
-- uncheckedGradient t
--   :: Tensor
--        UncheckedGradient
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "batch") ('Size 32),
--            'Dim ('Name "feature") ('Size 8)])
uncheckedGradient ::
  forall gradient layout device dataType shape.
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | tensor without checked layout
  Tensor 'UncheckedGradient layout device dataType shape
uncheckedGradient = coerce

-- | Returns a dense copy of the tensor.
toDense ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | dense copy
  m (Tensor gradient ('Layout 'Dense) device dataType shape)
toDense = unsafeThrowableIO . cast1 ATen.tensor_to_dense

-- | Returns a sparse copy of the tensor.
toSparse ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | sparse copy
  m (Tensor gradient ('Layout 'Sparse) device dataType shape)
toSparse input =
  unsafeThrowableIO $ do
   dim <- cast1 ATen.tensor_dim_c_unsafe input :: IO Int64
   cast2 ATen.tensor_to_sparse_l input dim

-- | Set the memory layout of a tensor to a given layout.
sSetLayout ::
  forall m gradient layout layout' device dataType shape.
  MonadThrow m =>
  SLayout layout ->
  Tensor gradient layout' device dataType shape ->
  m (Tensor gradient layout device dataType shape)
sSetLayout layout input =
  case forgetIsChecked (fromSing layout) of
    Dense -> unsafeThrowableIO . cast1 ATen.tensor_to_dense $ input
    Sparse -> unsafeThrowableIO $ do
      dim <- cast1 ATen.tensor_dim_c_unsafe input :: IO Int64
      cast2 ATen.tensor_to_sparse_l input dim

class SGetLayout (layout :: Layout LayoutType) where
  -- | Returns the gradually typed memory layout of the input tensor.
  --
  -- >>> sOnes' layout = sOnes $ TensorSpec (SGradient SWithGradient) layout (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t <- sOnes' $ SLayout SDense
  -- >>> sGetLayout t
  -- SLayout SDense
  -- >>> t <- sOnes' $ SUncheckedLayout Dense
  -- >>> sGetLayout t
  -- SUncheckedLayout Dense
  sGetLayout ::
    forall gradient device dataType shape.
    -- | input
    Tensor gradient layout device dataType shape ->
    -- | memory layout
    SLayout layout

  -- | Returns the untyped memory layout of the input tensor.
  --
  -- >>> sOnes' layout = sOnes $ TensorSpec (SGradient SWithGradient) layout (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t <- sOnes' $ SLayout SDense
  -- >>> getLayoutType t
  -- Dense
  -- >>> t <- sOnes' $ SUncheckedLayout Dense
  -- >>> getLayoutType t
  -- Dense
  getLayoutType ::
    forall gradient device dataType shape.
    -- | input
    Tensor gradient layout device dataType shape ->
    -- | memory layout
    LayoutType
  getLayoutType tensor = forgetIsChecked . fromSing $ sGetLayout tensor

instance SGetLayout 'UncheckedLayout where
  sGetLayout tensor
    | unsafePerformIO (cast1 ATen.tensor_is_sparse tensor) = SUncheckedLayout Sparse
    | otherwise = SUncheckedLayout Dense

instance SGetLayout ('Layout 'Sparse) where
  sGetLayout tensor
    | unsafePerformIO (cast1 ATen.tensor_is_sparse tensor) = SLayout SSparse
    | otherwise =
      error $
        "The tensor should be sparse but isn't. "
          <> gitHubErrorMsg

instance SGetLayout ('Layout 'Dense) where
  sGetLayout tensor
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
-- >>> t <- sOnes $ TensorSpec (SGradient SWithGradient) (SUncheckedLayout Dense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' <- sCheckedLayout (SLayout SDense) t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "batch") ('Size 32),
--            'Dim ('Name "feature") ('Size 8)])
-- >>> t' <- sCheckedLayout (SLayout SSparse) t
-- *** Exception: LayoutError {leExpected = Sparse, leActual = Dense}
sCheckedLayout ::
  forall layout' m gradient layout device dataType shape.
  (SGetLayout layout, MonadThrow m, Catch (layout <+> layout')) =>
  -- | layout
  SLayout layout' ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout' device dataType shape)
sCheckedLayout layout' tensor =
  let actualLayout = forgetIsChecked . fromSing $ sGetLayout tensor
      expectedLayout = forgetIsChecked . fromSing $ layout'
   in if actualLayout == expectedLayout
        then pure . coerce $ tensor
        else throwM $ LayoutError expectedLayout actualLayout

checkedLayout ::
  forall layout' m gradient layout device dataType shape.
  (SingI layout', SGetLayout layout, MonadThrow m, Catch (layout <+> layout')) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout' device dataType shape)
checkedLayout = sCheckedLayout (sing @layout')

-- | Returns the input tensor but with 'UncheckedLayout' as memory layout type annotation.
-- Any static information about the tensor's memory layout is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t <- ones @('Gradient 'WithGradient) @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedLayout t
-- uncheckedLayout t
--   :: Tensor
--        ('Gradient WithGradient)
--        UncheckedLayout
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "batch") ('Size 32),
--            'Dim ('Name "feature") ('Size 8)])
uncheckedLayout ::
  forall gradient layout device dataType shape.
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | tensor without checked layout
  Tensor gradient 'UncheckedLayout device dataType shape
uncheckedLayout = coerce

-- | Returns a copy of the tensor in CPU memory.
cpu ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | copy in CPU memory
  m (Tensor gradient layout ('Device 'CPU) dataType shape)
cpu = unsafeThrowableIO . cast1 ATen.tensor_cpu

-- | Returns a copy of the tensor in CUDA memory.
cuda ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | copy in CUDA memory
  m (Tensor gradient layout ('Device ('CUDA 0)) dataType shape)
cuda = unsafeThrowableIO . cast1 ATen.tensor_cuda

-- | Reallocates a tensor on the specified device.
sSetDevice ::
  forall m gradient layout device device' dataType shape.
  MonadThrow m =>
  SDevice device ->
  Tensor gradient layout device' dataType shape ->
  m (Tensor gradient layout device dataType shape)
sSetDevice device input =
  case forgetIsChecked (fromSing device) of
    CPU ->
      unsafeThrowableIO . cast1 ATen.tensor_cpu $ input
    CUDA 0 ->
      unsafeThrowableIO . cast1 ATen.tensor_cuda $ input
    CUDA idx ->
      unsafeThrowableIO $ do
        opts :: ForeignPtr ATen.TensorOptions <- cast1 ATen.tensor_options input
        opts' :: ForeignPtr ATen.TensorOptions <- cast2 ATen.tensorOptions_device_index_s opts idx
        cast4 ATen.tensor_to_obb input opts' nonBlocking copy
      where
        nonBlocking = False
        copy = False

class SGetDevice (device :: Device (DeviceType Nat)) where
  -- | Returns the gradually typed compute device of the input tensor.
  --
  -- >>> ones' device = sOnes $ TensorSpec (SGradient SWithGradient) (SLayout SDense) device (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t <- ones' $ SDevice SCPU
  -- >>> sGetDevice t
  -- SDevice SCPU
  -- >>> t <- ones' $ SUncheckedDevice CPU
  -- >>> sGetDevice t
  -- SUncheckedDevice CPU
  sGetDevice ::
    forall gradient layout dataType shape.
    -- | input
    Tensor gradient layout device dataType shape ->
    -- | compute device of the input tensor
    SDevice device

  -- | Returns the untyped compute device of the input tensor.
  --
  -- >>> ones' device = sOnes $ TensorSpec (SGradient SWithGradient) (SLayout SDense) device (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t <- ones' $ SDevice SCPU
  -- >>> getDeviceType t
  -- CPU
  -- >>> t <- ones' $ SUncheckedDevice CPU
  -- >>> getDeviceType t
  -- CPU
  getDeviceType ::
    forall gradient layout dataType shape.
    -- | input
    Tensor gradient layout device dataType shape ->
    -- | compute device of the input tensor
    DeviceType Int16
  getDeviceType tensor = forgetIsChecked . fromSing $ sGetDevice tensor

instance SGetDevice 'UncheckedDevice where
  sGetDevice tensor
    | unsafePerformIO (cast0 ATen.hasCUDA) && unsafePerformIO (cast1 ATen.tensor_is_cuda tensor) =
      case unsafePerformIO (cast1 ATen.tensor_get_device tensor) :: Int of
        deviceIndex -> SUncheckedDevice . CUDA . fromIntegral $ deviceIndex
    | otherwise = SUncheckedDevice CPU

instance SGetDevice ('Device 'CPU) where
  sGetDevice tensor
    | unsafePerformIO (cast0 ATen.hasCUDA) && unsafePerformIO (cast1 ATen.tensor_is_cuda tensor) =
      error $
        "The tensor should be on CPU but is on CUDA. "
          <> gitHubErrorMsg
    | otherwise = SDevice SCPU

instance KnownNat deviceIndex => SGetDevice ('Device ('CUDA deviceIndex)) where
  sGetDevice tensor
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
-- >>> t <- sOnes $ TensorSpec (SGradient SWithGradient) (SLayout SDense) (SUncheckedDevice CPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' <- sCheckedDevice (SDevice SCPU) t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "batch") ('Size 32),
--            'Dim ('Name "feature") ('Size 8)])
-- >>> t' <- sCheckedDevice (SDevice (SCUDA @0)) t
-- *** Exception: DeviceError {deExpected = CUDA 0, deActual = CPU}
sCheckedDevice ::
  forall device' m gradient layout device dataType shape.
  (SGetDevice device, MonadThrow m, Catch (device <+> device')) =>
  -- | device
  SDevice device' ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout device' dataType shape)
sCheckedDevice device' tensor =
  let actualDevice = forgetIsChecked . fromSing $ sGetDevice tensor
      expectedDevice = forgetIsChecked . fromSing $ device'
   in if actualDevice == expectedDevice
        then pure . coerce $ tensor
        else throwM $ DeviceError expectedDevice actualDevice

checkedDevice ::
  forall device' m gradient layout device dataType shape.
  (SingI device', SGetDevice device, MonadThrow m, Catch (device <+> device')) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout device' dataType shape)
checkedDevice = sCheckedDevice (sing @device')

-- | Returns the input tensor but with 'UncheckedDevice' as device type annotation.
-- Any static information about the tensor's device is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t <- ones @('Gradient 'WithGradient) @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedDevice t
-- uncheckedDevice t
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        UncheckedDevice
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "batch") ('Size 32),
--            'Dim ('Name "feature") ('Size 8)])
uncheckedDevice ::
  forall gradient layout device dataType shape.
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | tensor without checked device
  Tensor gradient layout 'UncheckedDevice dataType shape
uncheckedDevice = coerce

-- | Returns a copy of the tensor converted to 'Bool'.
bool ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Bool' copy
  m (Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'Bool) shape)
bool tensor = unsafeThrowableIO $ cast2 ATen.tensor_toType_s tensor Bool

-- | Returns a copy of the tensor converted to 'UInt8'.
byte ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'UInt8' copy
  m (Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'UInt8) shape)
byte tensor = unsafeThrowableIO $ cast2 ATen.tensor_toType_s tensor UInt8

-- | Returns a copy of the tensor converted to 'Int8'.
char ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Int8' copy
  m (Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'Int8) shape)
char tensor = unsafeThrowableIO $ cast2 ATen.tensor_toType_s tensor Int8

-- | Returns a copy of the tensor converted to 'Int16'.
short ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Int16' copy
  m (Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'Int16) shape)
short tensor = unsafeThrowableIO $ cast2 ATen.tensor_toType_s tensor Int16

-- | Returns a copy of the tensor converted to 'Int32'.
int ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Int32' copy
  m (Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'Int32) shape)
int tensor = unsafeThrowableIO $ cast2 ATen.tensor_toType_s tensor Int32

-- | Returns a copy of the tensor converted to 'Int64'.
long ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Int64' copy
  m (Tensor ('Gradient 'WithoutGradient) layout device ('DataType 'Int64) shape)
long tensor = unsafeThrowableIO $ cast2 ATen.tensor_toType_s tensor Int64

-- | Returns a copy of the tensor converted to the 16-bit floating point format 'Half'.
half ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Half' copy
  m (Tensor gradient layout device ('DataType 'Half) shape)
half tensor = unsafeThrowableIO $ cast2 ATen.tensor_toType_s tensor Half

-- | Returns a copy of the tensor converted to the 32-bit floating point format 'Float'.
float ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Float' copy
  m (Tensor gradient layout device ('DataType 'Float) shape)
float tensor = unsafeThrowableIO $ cast2 ATen.tensor_toType_s tensor Float

-- | Returns a copy of the tensor converted to the 32-bit floating point format 'Double'.
double ::
  forall m gradient layout device dataType shape.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | 'Double' copy
  m (Tensor gradient layout device ('DataType 'Double) shape)
double tensor = unsafeThrowableIO $ cast2 ATen.tensor_toType_s tensor Double

-- | Set the data type of a tensor to the specified data type.
sSetDataType ::
  forall m gradient layout device dataType dataType' shape.
  MonadThrow m =>
  SDataType dataType ->
  Tensor gradient layout device dataType' shape ->
  m (Tensor gradient layout device dataType shape)
sSetDataType dataType input =
  case forgetIsChecked (fromSing dataType) of
    dType -> unsafeThrowableIO $ do
      opts :: ForeignPtr ATen.TensorOptions <- cast1 ATen.tensor_options input
      opts' :: ForeignPtr ATen.TensorOptions <- cast2 ATen.tensorOptions_dtype_s opts dType
      cast4 ATen.tensor_to_obb input opts' nonBlocking copy
      where
        nonBlocking = False
        copy = False

class SGetDataType (dataType :: DataType DType) where
  -- | Returns the gradually typed compute data type of the input tensor.
  --
  -- >>> sOnes' dataType = sOnes $ TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) dataType (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t <- sOnes' $ SDataType SFloat
  -- >>> sGetDataType t
  -- SDataType SFloat
  -- >>> t <- sOnes' $ SUncheckedDataType Float
  -- >>> sGetDataType t
  -- SUncheckedDataType Float
  sGetDataType ::
    forall gradient layout device shape.
    -- | input
    Tensor gradient layout device dataType shape ->
    -- | data type of the input tensor
    SDataType dataType

  -- | Returns the untyped compute data type of the input tensor.
  --
  -- >>> sOnes' dataType = sOnes $ TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) dataType (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
  -- >>> t <- sOnes' $ SDataType SFloat
  -- >>> getDType t
  -- Float
  -- >>> t <- sOnes' $ SUncheckedDataType Float
  -- >>> getDType t
  -- Float
  getDType ::
    forall gradient layout device shape.
    -- | input
    Tensor gradient layout device dataType shape ->
    -- | data type of the input tensor
    DType
  getDType tensor = forgetIsChecked . fromSing $ sGetDataType tensor

instance SGetDataType 'UncheckedDataType where
  sGetDataType tensor = SUncheckedDataType . unsafePerformIO $ cast1 ATen.tensor_scalar_type tensor

instance SingI dType => SGetDataType ('DataType dType) where
  sGetDataType tensor
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
-- >>> t <- sOnes $ TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) (SUncheckedDataType Float) (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil)
-- >>> t' <- checkedDataType @('DataType 'Float) t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "batch") ('Size 32),
--            'Dim ('Name "feature") ('Size 8)])
-- >>> t' <- checkedDataType @('DataType 'Double) t
-- *** Exception: DataTypeError {dtExpected = Double, dtActual = Float}
sCheckedDataType ::
  forall dataType' m gradient layout device dataType shape.
  (SGetDataType dataType, MonadThrow m, Catch (dataType <+> dataType')) =>
  -- | data type
  SDataType dataType' ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout device dataType' shape)
sCheckedDataType dataType' tensor =
  let actualDataType = forgetIsChecked . fromSing $ sGetDataType tensor
      expectedDataType = forgetIsChecked . fromSing $ dataType'
   in if actualDataType == expectedDataType
        then pure . coerce $ tensor
        else throwM $ DataTypeError expectedDataType actualDataType

checkedDataType ::
  forall dataType' m gradient layout device dataType shape.
  (SingI dataType', SGetDataType dataType, MonadThrow m, Catch (dataType <+> dataType')) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout device dataType' shape)
checkedDataType = sCheckedDataType (sing @dataType')

-- | Returns the input tensor but with 'UncheckedDataType' as data-type type annotation.
-- Any static information about the tensor's data type is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t <- ones @('Gradient 'WithGradient) @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedDataType t
-- uncheckedDataType t
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        UncheckedDataType
--        ('Shape
--           ['Dim ('Name "batch") ('Size 32),
--            'Dim ('Name "feature") ('Size 8)])
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
  -- >>> sOnes' = sOnes . TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat)
  -- >>> t <- sOnes' . SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil
  -- >>> sGetShape t
  -- SShape (SCons (SDim {sDimName = SName, sDimSize = SSize}) (SCons (SDim {sDimName = SName, sDimSize = SSize}) SNil))
  -- >>> t <- sOnes' . SUncheckedShape $ [Dim "batch" 32, Dim "feature" 8]
  -- >>> sGetShape t
  -- SUncheckedShape [Dim {dimName = "batch", dimSize = 32},Dim {dimName = "feature", dimSize = 8}]
  -- >>> t <- sOnes' . SShape $ SUncheckedName "batch" :&: SUncheckedSize 32 :|: SUncheckedName "feature" :&: SSize @32 :|: SNil
  -- >>> sGetShape t
  -- SShape (SCons (SDim {sDimName = SUncheckedName "batch", sDimSize = SUncheckedSize 32}) (SCons (SDim {sDimName = SUncheckedName "feature", sDimSize = SSize}) SNil))
  -- >>> t <- sOnes' . SShape $ SName @"batch" :&: SUncheckedSize 32 :|: SName @"feature" :&: SUncheckedSize 8 :|: SNil
  -- >>> sGetShape t
  -- SShape (SCons (SDim {sDimName = SName, sDimSize = SUncheckedSize 32}) (SCons (SDim {sDimName = SName, sDimSize = SUncheckedSize 8}) SNil))
  sGetShape ::
    forall gradient layout device dataType.
    Tensor gradient layout device dataType shape ->
    SShape shape

  -- | Returns the untyped shape of the input tensor.
  --
  -- >>> sOnes' = sOnes . TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat)
  -- >>> t <- sOnes' . SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil
  -- >>> getDims t
  -- [Dim {dimName = "batch", dimSize = 32},Dim {dimName = "feature", dimSize = 8}]
  -- >>> t <- sOnes' . SUncheckedShape $ [Dim "batch" 32, Dim "feature" 8]
  -- >>> getDims t
  -- [Dim {dimName = "batch", dimSize = 32},Dim {dimName = "feature", dimSize = 8}]
  -- >>> t <- sOnes' . SShape $ SUncheckedName "batch" :&: SUncheckedSize 32 :|: SUncheckedName "feature" :&: SSize @32 :|: SNil
  -- >>> getDims t
  -- [Dim {dimName = "batch", dimSize = 32},Dim {dimName = "feature", dimSize = 32}]
  -- >>> t <- sOnes' . SShape $ SName @"batch" :&: SUncheckedSize 32 :|: SName @"feature" :&: SUncheckedSize 8 :|: SNil
  -- >>> getDims t
  -- [Dim {dimName = "batch", dimSize = 32},Dim {dimName = "feature", dimSize = 8}]
  getDims ::
    forall gradient layout device dataType.
    Tensor gradient layout device dataType shape ->
    [Dim String Integer]
  getDims tensor = fmap (bimap forgetIsChecked forgetIsChecked) . forgetIsChecked . fromSing $ sGetShape tensor

instance SGetShape 'UncheckedShape where
  sGetShape tensor = SUncheckedShape . unsafePerformIO $ do
    sizes <- cast1 ATen.tensor_sizes tensor
    ifM
      (cast1 ATen.tensor_has_names tensor)
      ( do
          names <- cast1 ATen.tensor_names tensor
          return $ zipWith Dim names sizes
      )
      (return $ Dim "*" <$> sizes)

instance SGetDims dims => SGetShape ('Shape dims) where
  sGetShape tensor =
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
     in SShape $ sGetDims names sizes

class SGetDims (dims :: [Dim (Name Symbol) (Size Nat)]) where
  sGetDims :: [String] -> [Integer] -> SList dims

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
  sGetDims [] [] = SNil
  sGetDims _ _ = dimsError

instance (SGetDim dim, SGetDims dims) => SGetDims (dim : dims) where
  sGetDims (name : names) (size : sizes) = sGetDim name size :|: sGetDims names sizes
  sGetDims _ _ = dimsError

class SGetDim (dim :: Dim (Name Symbol) (Size Nat)) where
  sGetDim :: String -> Integer -> SDim dim

instance SGetDim ('Dim 'UncheckedName 'UncheckedSize) where
  sGetDim name size = SDim (SUncheckedName name) (SUncheckedSize size)

instance KnownSymbol name => SGetDim ('Dim ('Name name) 'UncheckedSize) where
  sGetDim name size = case symbolVal $ Proxy @name of
    name'
      | name == name' -> SDim (SName @name) (SUncheckedSize size)
      | otherwise -> dimNameError name name'

instance KnownNat size => SGetDim ('Dim 'UncheckedName ('Size size)) where
  sGetDim name size = case natVal $ Proxy @size of
    size'
      | size == size' -> SDim (SUncheckedName name) (SSize @size)
      | otherwise -> dimSizeError size size'

instance (KnownSymbol name, KnownNat size) => SGetDim ('Dim ('Name name) ('Size size)) where
  sGetDim name size = case (symbolVal $ Proxy @name, natVal $ Proxy @size) of
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
-- >>> t <- sOnes $ TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SUncheckedShape [Dim "batch" 32, Dim "feature" 8])
-- >>> t' <- sCheckedShape (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SSize @8 :|: SNil) t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "batch") ('Size 32),
--            'Dim ('Name "feature") ('Size 8)])
-- >>> t' <- sCheckedShape (SShape $ SUncheckedName "batch" :&: SSize @32 :|: SName @"feature" :&: SUncheckedSize 8 :|: SNil) t
-- >>> :type t'
-- t'
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim UncheckedName ('Size 32),
--            'Dim ('Name "feature") UncheckedSize])
-- >>> t' <- sCheckedShape (SShape $ SName @"batch" :&: SSize @32 :|: SName @"feature" :&: SUncheckedSize 32 :|: SNil) t
-- *** Exception: ShapeError {seExpected = [Dim {dimName = "batch", dimSize = 32},Dim {dimName = "feature", dimSize = 32}], seActual = [Dim {dimName = "batch", dimSize = 32},Dim {dimName = "feature", dimSize = 8}]}
sCheckedShape ::
  forall shape' m gradient layout device dataType shape.
  (SGetShape shape, MonadThrow m, Catch (shape <+> shape')) =>
  -- | shape
  SShape shape' ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout device dataType shape')
sCheckedShape shape' tensor =
  let f = fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size)) . forgetIsChecked . fromSing
      actualShape = f $ sGetShape tensor
      expectedShape = f shape'
   in if actualShape == expectedShape
        then pure . coerce $ tensor
        else throwM $ ShapeError expectedShape actualShape

checkedShape ::
  forall shape' m gradient layout device dataType shape.
  (SingI shape', SGetShape shape, MonadThrow m, Catch (shape <+> shape')) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | annotated output tensor wrapped in 'm'
  m (Tensor gradient layout device dataType shape')
checkedShape = sCheckedShape (sing @shape')

-- | Returns the input tensor but with the selected dimension replaces with 'UncheckedDim' as dimension type annotation.
-- The static information about the selected tensor dimension is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t <- ones @('Gradient 'WithGradient) @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedDim @('SelectDim ('ByName "batch")) t
-- uncheckedDim @('SelectDim ('ByName "batch")) t
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim UncheckedName UncheckedSize,
--            'Dim ('Name "feature") ('Size 8)])
-- >>> t <- ones @('Gradient 'WithGradient) @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedDim @('SelectDim ('ByIndex 1)) t
-- uncheckedDim @('SelectDim ('ByIndex 1)) t
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "batch") ('Size 32),
--            'Dim UncheckedName UncheckedSize])
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
-- >>> t <- ones @('Gradient 'WithGradient) @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedShape t
-- uncheckedShape t
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        UncheckedShape
uncheckedShape ::
  forall gradient layout device dataType shape.
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | tensor without checked shape
  Tensor gradient layout device dataType 'UncheckedShape
uncheckedShape = coerce

gitHubErrorMsg :: String
gitHubErrorMsg = "Please open a ticket on GitHub."

isContiguous ::
  Tensor gradient layout device dataType shape ->
  Bool
isContiguous t = unsafePerformIO $ cast1 ATen.tensor_is_contiguous t

contiguous ::
  Tensor gradient layout device dataType shape ->
  Tensor gradient layout device dataType shape
contiguous t = unsafePerformIO $ cast1 ATen.tensor_contiguous t

withTensor :: Tensor gradient layout device dataType shape -> (Ptr () -> IO a) -> IO a
withTensor t fn =
  let contiguousTensor = if isContiguous t then t else contiguous t
   in cast contiguousTensor $ \ct -> withForeignPtr ct $ Unmanaged.tensor_data_ptr >=> fn

class TensorLikeRaw a where
  -- | Guesses outer dim.
  --
  -- >>> guessDim @[[Int]] $ pure [[1, 2], [3, 4], [5, 6]]
  -- Just 3
  guessDim ::
    -- | value
    -- 'Nothing' if the data type wrapping 'a' is empty.
    Maybe a ->
    -- | dimension
    -- 'Nothing' if 'a' is a scalar.
    Maybe Int

  -- | Guesses inner dims.
  --
  -- >>> guessInnerDims @[[Int]] $ pure [[1, 2], [3, 4], [5, 6]]
  -- [2]
  guessInnerDims ::
    MonadThrow m =>
    -- | value
    -- 'Nothing' if the data type wrapping 'a' is empty.
    Maybe a ->
    -- | inner dimensions
    m [Int]

  -- | Reads a value from a tensor.
  tensorPeekElemOff ::
    -- | pointer to tensor
    Ptr () ->
    -- | offset
    Int ->
    -- | tensor dimensions
    [Int] ->
    -- | value
    IO a

  -- | Writes a value to a tensor.
  tensorPokeElemOff ::
    -- | pointer to tensor
    Ptr () ->
    -- | offset
    Int ->
    -- | tensor dimensions
    [Int] ->
    -- | value
    a ->
    IO ()

-- | Guesses dims: concatenates 'guessDim' with 'guessInnerDims'.
--
-- >>> guessDims @[[Int]] $ pure [[1, 2], [3, 4], [5, 6]]
-- [3,2]
guessDims :: forall a m. (TensorLikeRaw a, MonadThrow m) => Maybe a -> m [Int]
guessDims x = (outerDim <>) <$> guessInnerDims x
  where
    outerDim = maybeToList $ guessDim x

unexpectedDimsError :: forall a m b. (TensorLikeRaw a, MonadThrow m) => [Int] -> Maybe a -> m b
unexpectedDimsError dims' x = do
  expected <- guessDims x
  error $ "Expected shape to be " <> show expected <> " got: " <> show dims'

class TensorLike a (dType :: DType) (dims :: [Dim (Name Symbol) (Size Nat)]) | a -> dims, a -> dType where
  -- | Creates a tensor from a 'TensorLike' value.
  --
  -- >>> t <- sToTensor (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) ([(1, 2), (3, 4), (5, 6)] :: [(Int, Int)])
  -- >>> t
  -- Tensor Int64 [3,2] [[ 1,  2],
  --                     [ 3,  4],
  --                     [ 5,  6]]
  -- >>> :type t
  -- t :: Tensor
  --        ('Gradient WithoutGradient)
  --        ('Layout Dense)
  --        ('Device CPU)
  --        ('DataType 'Int64)
  --        ('Shape
  --           ['Dim ('Name "*") UncheckedSize, 'Dim ('Name "*") ('Size 2)])
  sToTensor ::
    forall gradient layout device m.
    MonadThrow m =>
    SGradient gradient ->
    SLayout layout ->
    SDevice device ->
    a ->
    m (Tensor gradient layout device ('DataType dType) ('Shape dims))

  -- | Creates a 'TensorLike' from a tensor.
  fromTensor ::
    forall gradient layout device.
    Tensor gradient layout device ('DataType dType) ('Shape dims) ->
    a

-- | Non-singleton version of 'sToTensor'.
toTensor ::
  forall gradient layout device a dType dims m.
  ( TensorLike a dType dims,
    SingI gradient,
    SingI layout,
    SingI device,
    MonadThrow m
  ) =>
  a ->
  m (Tensor gradient layout device ('DataType dType) ('Shape dims))
toTensor = sToTensor (sing @gradient) (sing @layout) (sing @device)

sToTensorRaw ::
  forall gradient layout device a dType dims m.
  (TensorLike a dType dims, TensorLikeRaw a, SingI dType, MonadThrow m) =>
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  a ->
  m (Tensor gradient layout device ('DataType dType) ('Shape dims))
sToTensorRaw gradient' layout device x = do
  dims' <- guessDims $ pure x

  pure $
    unsafePerformIO $ do
      t <- UnsafeTensor <$> cast2 ATen.empty_lo dims' opts
      withTensor t $ \ptr ->
        tensorPokeElemOff ptr 0 dims' x
      sSetDevice device t
  where
    opts = tensorOptions gradient' layout (SDevice SCPU) (sing @('DataType dType))

fromTensorRaw ::
  forall gradient layout device a dType dims.
  (TensorLike a dType dims, TensorLikeRaw a, SGetDims dims) =>
  Tensor gradient layout device ('DataType dType) ('Shape dims) ->
  a
fromTensorRaw t =
  unsafePerformIO $
    sSetDevice (SDevice SCPU) t
      >>= flip withTensor (\ptr -> tensorPeekElemOff ptr 0 $ fromInteger . dimSize <$> getDims t)

instance TensorLike Bool 'Bool '[] where
  sToTensor = sToTensorRaw
  fromTensor = fromTensorRaw

instance TensorLikeRaw Bool where
  guessDim = const empty

  guessInnerDims = const $ pure mempty

  tensorPeekElemOff ptr offset [] = peekElemOff @Word8 (castPtr ptr) offset <&> (== 1)
  tensorPeekElemOff _ _ dims' = unexpectedDimsError @Bool dims' empty

  tensorPokeElemOff ptr offset [] x = pokeElemOff @Word8 (castPtr ptr) offset (fromBool x)
  tensorPokeElemOff _ _ dims' x = unexpectedDimsError dims' $ pure x

instance TensorLike Int 'Int64 '[] where
  sToTensor = sToTensorRaw
  fromTensor = fromTensorRaw

instance TensorLikeRaw Int where
  guessDim = const empty

  guessInnerDims = const $ pure empty

  tensorPeekElemOff ptr offset [] = peekElemOff (castPtr ptr) offset
  tensorPeekElemOff _ _ dims' = unexpectedDimsError @Int dims' empty

  tensorPokeElemOff ptr offset [] x = pokeElemOff (castPtr ptr) offset x
  tensorPokeElemOff _ _ dims' x = unexpectedDimsError dims' $ pure x

instance TensorLike Float 'Float '[] where
  sToTensor = sToTensorRaw
  fromTensor = fromTensorRaw

instance TensorLikeRaw Float where
  guessDim = const empty

  guessInnerDims = const $ pure empty

  tensorPeekElemOff ptr offset [] = peekElemOff (castPtr ptr) offset
  tensorPeekElemOff _ _ dims' = unexpectedDimsError @Float dims' empty

  tensorPokeElemOff ptr offset [] x = pokeElemOff (castPtr ptr) offset x
  tensorPokeElemOff _ _ dims' x = unexpectedDimsError dims' $ pure x

instance TensorLike Double 'Double '[] where
  sToTensor = sToTensorRaw
  fromTensor = fromTensorRaw

instance TensorLikeRaw Double where
  guessDim = const empty

  guessInnerDims = const $ pure empty

  tensorPeekElemOff ptr offset [] = peekElemOff (castPtr ptr) offset
  tensorPeekElemOff _ _ dims' = unexpectedDimsError @Double dims' empty

  tensorPokeElemOff ptr offset [] x = pokeElemOff (castPtr ptr) offset x
  tensorPokeElemOff _ _ dims' x = unexpectedDimsError dims' $ pure x

data DimMismatchError = DimMismatchError {dmeFirst :: [Int], dmeOther :: [Int]}
  deriving (Show, Eq)

instance Exception DimMismatchError where
  displayException DimMismatchError {..} =
    "When converting to a tensor, all elements on the same dimension must have the same shape, "
      <> "but the first element has shape "
      <> show dmeFirst
      <> " while another element has shape "
      <> show dmeOther
      <> "."

checkDims :: MonadThrow m => [Int] -> [Int] -> m ()
checkDims firstDims otherDims = when (firstDims /= otherDims) $ throwM $ DimMismatchError firstDims otherDims

instance
  ( TensorLike a dType dims,
    TensorLike b dType dims',
    TensorLikeRaw a,
    TensorLikeRaw b,
    SingI dType,
    SGetDims dimsOut,
    'Shape dimsOut ~ InsertDimF ('SelectDim ('ByIndex 0)) ('Shape (dims <+> dims')) ('Dim ('Name "*") ('Size 2))
  ) =>
  TensorLike (a, b) dType dimsOut
  where
  sToTensor = sToTensorRaw
  fromTensor = fromTensorRaw

instance (TensorLikeRaw a, TensorLikeRaw b) => TensorLikeRaw (a, b) where
  guessDim = const $ pure 2

  guessInnerDims (unzip -> (x, y)) = do
    xDims <- guessDims x
    yDims <- guessDims y
    checkDims xDims yDims
    pure xDims

  tensorPeekElemOff ptr offset (2 : innerDims) =
    (,)
      <$> tensorPeekElemOff ptr offset innerDims
      <*> tensorPeekElemOff ptr (offset + width) innerDims
    where
      width = product innerDims
  tensorPeekElemOff _ _ dims' = unexpectedDimsError @(a, b) dims' empty

  tensorPokeElemOff ptr offset (2 : innerDims) (x, y) = do
    tensorPokeElemOff ptr offset innerDims x
    tensorPokeElemOff ptr (offset + width) innerDims y
    where
      width = product innerDims
  tensorPokeElemOff _ _ dims' x = unexpectedDimsError dims' $ pure x

instance
  ( TensorLike a dType dims,
    TensorLike b dType dims',
    TensorLike c dType dims',
    TensorLikeRaw a,
    TensorLikeRaw b,
    TensorLikeRaw c,
    SingI dType,
    SGetDims dimsOut,
    'Shape dimsOut ~ InsertDimF ('SelectDim ('ByIndex 0)) ('Shape (dims <+> dims')) ('Dim ('Name "*") ('Size 3))
  ) =>
  TensorLike (a, b, c) dType dimsOut
  where
  sToTensor = sToTensorRaw
  fromTensor = fromTensorRaw

unzip3 :: Functor f => f (a, b, c) -> (f a, f b, f c)
unzip3 xyz =
  ( (\(x, _, _) -> x) <$> xyz,
    (\(_, y, _) -> y) <$> xyz,
    (\(_, _, z) -> z) <$> xyz
  )

instance (TensorLikeRaw a, TensorLikeRaw b, TensorLikeRaw c) => TensorLikeRaw (a, b, c) where
  guessDim = const $ pure 2

  guessInnerDims (unzip3 -> (x, y, z)) = do
    xDims <- guessDims x
    yDims <- guessDims y
    zDims <- guessDims z
    traverse_ (checkDims xDims) [yDims, zDims]
    pure xDims

  tensorPeekElemOff ptr offset (3 : innerDims) =
    (,,)
      <$> tensorPeekElemOff ptr offset innerDims
      <*> tensorPeekElemOff ptr (offset + width) innerDims
      <*> tensorPeekElemOff ptr (offset + 2 * width) innerDims
    where
      width = product innerDims
  tensorPeekElemOff _ _ dims' = unexpectedDimsError @(a, b) dims' empty

  tensorPokeElemOff ptr offset (3 : innerDims) (x, y, z) = do
    tensorPokeElemOff ptr offset innerDims x
    tensorPokeElemOff ptr (offset + width) innerDims y
    tensorPokeElemOff ptr (offset + 2 * width) innerDims z
    where
      width = product innerDims
  tensorPokeElemOff _ _ dims' x = unexpectedDimsError dims' $ pure x

instance
  ( TensorLike a dType dims,
    TensorLikeRaw a,
    SingI dType,
    SGetDims dimsOut,
    'Shape dimsOut ~ InsertDimF ('SelectDim ('ByIndex 0)) ('Shape dims) ('Dim ('Name "*") 'UncheckedSize)
  ) =>
  TensorLike [a] dType dimsOut
  where
  sToTensor = sToTensorRaw
  fromTensor = fromTensorRaw

instance TensorLikeRaw a => TensorLikeRaw [a] where
  guessDim = pure . maybe 0 length

  guessInnerDims =
    (>>= nonEmpty) >>> \case
      Nothing -> guessDims @a empty
      Just (x :| xs) -> do
        xDims <- guessDims $ pure x
        traverse_ (checkDims xDims <=< guessDims . pure) xs
        pure xDims

  tensorPeekElemOff ptr offset (d : innerDims) =
    forM [0 .. d - 1] $ \i -> do
      tensorPeekElemOff ptr (offset + i * width) innerDims
    where
      width = product innerDims
  tensorPeekElemOff _ _ dims' = unexpectedDimsError @[a] dims' empty

  tensorPokeElemOff ptr offset (d : innerDims) xs =
    forM_ (zip [0 .. d - 1] xs) $ \(i, x) ->
      tensorPokeElemOff ptr (offset + i * width) innerDims x
    where
      width = product innerDims
  tensorPokeElemOff _ _ dims' x = unexpectedDimsError dims' $ pure x

instance
  ( TensorLike a dType dims,
    TensorLikeRaw a,
    SingI dType,
    SGetDims dimsOut,
    'Shape dimsOut ~ InsertDimF ('SelectDim ('ByIndex 0)) ('Shape dims) ('Dim ('Name "*") 'UncheckedSize)
  ) =>
  TensorLike (V.Vector a) dType dimsOut
  where
  sToTensor = sToTensorRaw
  fromTensor = fromTensorRaw

instance
  TensorLikeRaw a =>
  TensorLikeRaw (V.Vector a)
  where
  guessDim = pure . maybe 0 length

  guessInnerDims =
    (>>= V.uncons) >>> \case
      Nothing -> guessDims @a empty
      Just (x, xs) -> do
        xDims <- guessDims $ pure x
        traverse_ (checkDims xDims <=< guessDims . pure) xs
        pure xDims

  tensorPeekElemOff ptr offset (d : innerDims) =
    forM (V.enumFromTo 0 (d - 1)) $ \i -> do
      tensorPeekElemOff ptr (offset + i * width) innerDims
    where
      width = product innerDims
  tensorPeekElemOff _ _ dims' = unexpectedDimsError @(V.Vector a) dims' empty

  tensorPokeElemOff ptr offset (d : innerDims) xs = do
    forM_ (V.zip (V.enumFromTo 0 (d - 1)) xs) $ \(i, x) ->
      tensorPokeElemOff ptr (offset + i * width) innerDims x
    where
      width = product innerDims
  tensorPokeElemOff _ _ dims' x = unexpectedDimsError dims' $ pure x

instance
  ( KnownNat n,
    TensorLike a dType dims,
    TensorLikeRaw a,
    SingI dType,
    SGetDims dimsOut,
    'Shape dimsOut ~ InsertDimF ('SelectDim ('ByIndex 0)) ('Shape dims) ('Dim ('Name "*") ('Size n))
  ) =>
  TensorLike (SV.Vector n a) dType dimsOut
  where
  sToTensor = sToTensorRaw
  fromTensor = fromTensorRaw

instance
  ( KnownNat n,
    TensorLikeRaw a
  ) =>
  TensorLikeRaw (SV.Vector n a)
  where
  guessDim = pure . maybe 0 length

  guessInnerDims = guessInnerDims . fmap SV.SomeSized

  tensorPeekElemOff ptr offset dims' = SVI.Vector <$> tensorPeekElemOff ptr offset dims'

  tensorPokeElemOff ptr offset dims' = tensorPokeElemOff ptr offset dims' . SV.SomeSized

sSetTensorOptions ::
  forall gradient layout device dataType gradientFrom layoutFrom deviceFrom dataTypeFrom shape.
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  Tensor gradientFrom layoutFrom deviceFrom dataTypeFrom shape ->
  IO (Tensor gradient layout device dataType shape)
sSetTensorOptions gradient' layout device dataType t =
  UnsafeTensor <$> cast4 ATen.tensor_to_obb t opts nonBlocking copy
  where
    opts = tensorOptions gradient' layout device dataType

    nonBlocking = False
    copy = False

setTensorOptions ::
  forall gradient layout device dataType gradientFrom layoutFrom deviceFrom dataTypeFrom shape.
  ( SingI gradient,
    SingI layout,
    SingI device,
    SingI dataType
  ) =>
  Tensor gradientFrom layoutFrom deviceFrom dataTypeFrom shape ->
  IO (Tensor gradient layout device dataType shape)
setTensorOptions = sSetTensorOptions (sing @gradient) (sing @layout) (sing @device) (sing @dataType)

-- instance
--   ( SingI gradient,
--     SingI layout,
--     SingI device,
--     SingI dType
--   ) =>
--   TensorLike (Tensor gradient layout device ('DataType dType) ('Shape dims)) dType dims
--   where
--   sToTensor gradient' layout device t = pure $ sSetTensorOptions gradient' layout device dataType t
--     where
--       dataType = SDataType $ sing @dType

--   fromTensor = setTensorOptions @gradient @layout @device @('DataType dType)
