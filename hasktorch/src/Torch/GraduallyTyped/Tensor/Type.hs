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
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

module Torch.GraduallyTyped.Tensor.Type where

import Data.Coerce (coerce)
import Data.Foldable (Foldable (fold))
import Data.Int (Int16)
import Data.Kind (Type)
import Data.Monoid (All (..))
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice (..))
import Torch.GraduallyTyped.Internal.TensorOptions (tensorOptions)
import Torch.GraduallyTyped.Layout (KnownLayout (..), Layout (..), LayoutType (..))
import Torch.GraduallyTyped.Prelude (ifM, (&&^))
import Torch.GraduallyTyped.RequiresGradient (KnownRequiresGradient, RequiresGradient (..), requiresGradientVal)
import Torch.GraduallyTyped.Scalar ()
import Torch.GraduallyTyped.Shape (Dim (..), KnownShape (..), Name (..), ReplaceDimF, Shape (..), Size (..))
import Torch.HList (HList (..), pattern (:.))
import Torch.Internal.Cast (cast0, cast1, cast2, cast3, cast4)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Extra as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen (Tensor, TensorList)

-- $setup
-- >>> import Torch.GraduallyTyped.Tensor.Creation (ones)
-- >>> import Torch.GraduallyTyped.Shape (By (..), SelectDim(..))

-- | A gradually typed tensor.
--
-- @
--                                +-> Compute device, e.g. 'Device 'CPU
--                                |
--                                |               +-> List of dimensions, e.g. 'Shape '[ 'Dim ('Sized 8), 'Dim ('Sized 1) ]
--                                +               +
-- Tensor requiresGradient layout device dataType shape
--        +                +             +
--        |                |             +-> Data type, e.g. 'DataType 'Float
--        |                |
--        |                +-> Memory layout, e.g. 'Layout 'Dense
--        |
--        +-> Whether or not the tensor requires a gradient, e.g. 'Independent for one that does
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
  -- use smart constructors like 'ones' instead.
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
          Shape [Dim (Name Symbol) (Size Nat)]
        )
    ) ::
    Type
  where
  TensorF '(requiresGradient, layout, device, dataType, shape) = Tensor requiresGradient layout device dataType shape

-- | Alias for an untyped tensor without gradients.
type UntypedTensor = Tensor 'Dependent 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape

-- | Alias for an untyped tensor with gradients.
type UntypedParameter = Tensor 'Independent 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape

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

instance
  ( KnownRequiresGradient requiresGradient,
    KnownLayout layout,
    KnownDevice device,
    KnownDataType dataType,
    KnownShape shape
  ) =>
  Num (Tensor requiresGradient layout device dataType shape)
  where
  (+) = (unsafePerformIO .) . cast2 ATen.add_tt
  (-) = (unsafePerformIO .) . cast2 ATen.sub_tt
  (*) = (unsafePerformIO .) . cast2 ATen.mul_tt
  negate = unsafePerformIO . cast1 ATen.neg_t
  abs = unsafePerformIO . cast1 ATen.abs_t
  signum = unsafePerformIO . cast1 ATen.sign_t
  fromInteger a = undefined

-- go
--   (requiresGradientVal @requiresGradient)
--   (layoutVal @layout)
--   (deviceVal @device)
--   (dataTypeVal @dataType)
--   (shapeVal @(Dim (Name Symbol) (Size Nat)) @shape)
-- where
--   prefix = "Unable to convert the integer " <> show a <> " to a tensor. "
--   go _ UncheckedLayout _ _ _ = error $ prefix <> "The memory layout is unknown."
--   go _ _ UncheckedDevice _ _ = error $ prefix <> "The tensor device is unknown."
--   go _ _ _ UncheckedDataType _ = error $ prefix <> "The tensor data type is unknown."
--   go _ _ _ _ UncheckedShape = error $ prefix <> "The tensor shape is unknown."
--   go requiresGradient (Layout layoutType) (Device deviceType) (DataType dType) (Shape shape) =
--     let opts = tensorOptions requiresGradient layoutType deviceType dType
--         shape' =
--           foldr
--             ( \case
--                 UncheckedDim -> \_ -> error $ prefix <> "Not all dimensions are known."
--                 Dim dimType -> \dimTypes -> dimType : dimTypes
--             )
--             []
--             shape
--      in case (namedDims shape', sizedDims shape') of
--           (Just names, Just sizes) -> unsafePerformIO $ cast4 ATen.full_lsNo sizes a names opts
--           (_, Just sizes) -> unsafePerformIO $ cast3 ATen.full_lso sizes a opts
--           _ -> error $ prefix <> "Invalid tensor shape specification " <> show shape <> "."

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
-- >>> t <- ones @'Dependent @('Layout 'Sparse) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('NamedSized "batch" 32), 'Dim ('NamedSized "feature" 8)])
-- >>> layout t
-- Sparse
-- >>> t <- ones @'Dependent @'UncheckedLayout @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('NamedSized "batch" 32), 'Dim ('NamedSized "feature" 8)]) (Layout Sparse)
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
    UncheckedLayout ->
      if unsafePerformIO . cast1 ATen.tensor_is_sparse $ tensor
        then Sparse
        else Dense
    Layout layoutType -> layoutType

-- | Returns the input tensor but with 'UncheckedLayout' as memory layout type annotation.
-- Any static information about the tensor's memory layout is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedLayout t
-- uncheckedLayout t
--   :: Tensor
--        'Dependent
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
-- >>> t = ones @'Dependent @'UncheckedLayout @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) Dense
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
        ((==) layoutType) <$> cast1 ATen.tensor_layout tensor

-- | Checks whether or not the input tensor has the memory layout 'layout'
-- and returns a statically annotated copy of it wrapped in a 'MonadFail' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor has indeed the memory layout 'layout'.
-- If it does not have it, then the result will be 'Nothing'.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t = ones @'Dependent @'UncheckedLayout @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) Dense
-- >>> t' <- checkedLayout @('Layout 'Dense) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
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
-- >>> t = ones @'Dependent @'UncheckedLayout @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) Dense
-- >>> t' = unsafeCheckedLayout @('Layout 'Dense) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
-- >>> t' = unsafeCheckedLayout @('Layout 'Sparse) t
-- *** Exception: The tensor does not have the memory layout "Layout Sparse".
-- CallStack (from HasCallStack):
--   error, called at /root/hasktorch/hasktorch/src/Torch/GraduallyTyped/Tensor/Type.hs:339:15 in main:Torch.GraduallyTyped.Tensor.Type
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
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> device t
-- CPU
-- >>> t = ones @'Dependent @('Layout 'Dense) @'UncheckedDevice @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) CPU
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
    UncheckedDevice ->
      unsafePerformIO $ do
        hasCUDA <- cast0 ATen.hasCUDA
        if hasCUDA
          then do
            isCUDA <- cast1 ATen.tensor_is_cuda tensor
            if isCUDA
              then do
                deviceIndex :: Int <- cast1 ATen.tensor_get_device tensor
                pure . CUDA . fromIntegral $ deviceIndex
              else pure CPU
          else pure CPU
    Device deviceType -> deviceType

-- | Returns the input tensor but with 'UncheckedDevice' as device type annotation.
-- Any static information about the tensor's device is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedDevice t
-- uncheckedDevice t
--   :: Tensor
--        'Dependent
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
-- >>> t = ones @'Dependent @('Layout 'Dense) @'UncheckedDevice @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) CPU
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
-- >>> t = ones @'Dependent @('Layout 'Dense) @'UncheckedDevice @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) CPU
-- >>> t' <- checkedDevice @('Device 'CPU) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
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
-- >>> t = ones @'Dependent @('Layout 'Dense) @'UncheckedDevice @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) CPU
-- >>> t' = unsafeCheckedDevice @('Device 'CPU) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
-- >>> t' = unsafeCheckedDevice @('Device ('CUDA 0)) t
-- *** Exception: The tensor is not in the memory of the device "Device (CUDA 0)".
-- CallStack (from HasCallStack):
--   error, called at /root/hasktorch/hasktorch/src/Torch/GraduallyTyped/Tensor/Type.hs:497:15 in main:Torch.GraduallyTyped.Tensor.Type
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
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> dtype t
-- Float
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @'UncheckedDataType @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) Float
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
    UncheckedDataType -> unsafePerformIO $ cast1 ATen.tensor_scalar_type tensor
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

-- | Returns the input tensor but with 'UncheckedDataType' as data-type type annotation.
-- Any static information about the tensor's data type is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedDataType t
-- uncheckedDataType t
--   :: Tensor
--        'Dependent
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
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @'UncheckedDataType  @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) Float
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
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @'UncheckedDataType @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) Float
-- >>> t' <- checkedDataType @('DataType 'Float) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
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
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @'UncheckedDataType @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) Float
-- >>> t' = unsafeCheckedDataType @('DataType 'Float) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('NamedSized "batch" 32), 'Dim ('NamedSized "feature" 8)])
-- >>> t' = unsafeCheckedDataType @('DataType 'Double) t
-- *** Exception: The tensor does not have the data type "DataType Double".
-- CallStack (from HasCallStack):
--   error, called at /root/hasktorch/hasktorch/src/Torch/GraduallyTyped/Tensor/Type.hs:712:15 in main:Torch.GraduallyTyped.Tensor.Type
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

-- | Returns the shape of the input tensor.
--
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> shape t
-- [Dim "batch" 32,Dim "feature" 8]
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @'UncheckedShape [Dim "batch" 32, Dim "feature" 8]
-- >>> shape t
-- [Dim "batch" 32,Dim "feature" 8]
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim 'UncheckedName 'UncheckedSize, 'Dim 'UncheckedName ('Size 8)]) (Dim "batch" 32) "feature"
-- >>> shape t
-- [Dim "batch" 32,Dim "feature" 8]
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") 'UncheckedSize, 'Dim ('Name "feature") 'UncheckedSize]) 32 8
-- >>> shape t
-- [Dim "batch" 32,Dim "feature" 8]
shape ::
  forall requiresGradient layout device dataType shape.
  KnownShape shape =>
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | shape of the input tensor
  [Dim String Integer]
shape tensor =
  case shapeVal @shape of
    UncheckedShape -> unsafePerformIO $ do
      sizes <- cast1 ATen.tensor_sizes tensor
      ifM
        (cast1 ATen.tensor_has_names tensor)
        ( do
            names <- cast1 ATen.tensor_names tensor
            return $ zipWith Dim names sizes
        )
        (return $ Dim "" <$> sizes)
    Shape dims ->
      let sizes = unsafePerformIO $ cast1 ATen.tensor_sizes tensor
          names =
            unsafePerformIO $
              ifM
                (cast1 ATen.tensor_has_names tensor)
                (cast1 ATen.tensor_names tensor)
                (pure $ map (const mempty) sizes)
          f (Dim UncheckedName UncheckedSize) name size = Dim name size
          f (Dim (Name name) UncheckedSize) name' size
            | name == name' = Dim name size
            | otherwise =
              error $
                "The compile- and runtime dimension names are not the same, '"
                  <> name
                  <> "' != '"
                  <> name'
                  <> "'. Please open a ticket on GitHub."
          f (Dim UncheckedName (Size size)) name size'
            | size == size' = Dim name size
            | otherwise =
              error $
                "The compile- and runtime dimension sizes are not the same, '"
                  <> show size
                  <> "' != '"
                  <> show size'
                  <> "'. Please open a ticket on GitHub."
          f (Dim (Name name) (Size size)) name' size'
            | name == name' && size == size' = Dim name size
            | otherwise =
              error $
                "The compile- and runtime dimension names and sizes are not the same, '"
                  <> name
                  <> "' != '"
                  <> name'
                  <> "' and '"
                  <> show size
                  <> "' != '"
                  <> show size'
                  <> "'. Please open a ticket on GitHub."
       in zipWith3 f dims names sizes

-- | Returns the input tensor but with the selected dimension replaces with 'UncheckedDim' as dimension type annotation.
-- The static information about the selected tensor dimension is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedDim @('SelectDim ('ByName "batch")) t
-- uncheckedDim @('SelectDim ('ByName "batch")) t
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim 'UncheckedName 'UncheckedSize,
--              'Dim ('Name "feature") ('Size 8)])
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedDim @('SelectDim ('ByIndex 1)) t
-- uncheckedDim @('SelectDim ('ByIndex 1)) t
--   :: Tensor
--        'Dependent
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
  Tensor requiresGradient layout device dataType (ReplaceDimF selectDim shape ( 'Dim 'UncheckedName 'UncheckedSize))
uncheckedDim = coerce

-- | Returns the input tensor but with 'UncheckedShape' as shape type annotation.
-- Any static information about the tensor's shape is thus erased.
-- However, the tensor's underlying data structure is not changed.
--
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
-- >>> :type uncheckedShape t
-- uncheckedShape t
--   :: Tensor
--        'Dependent
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
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @'UncheckedShape [Dim "batch" 32, Dim "feature" 8]
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
      let sizes = unsafePerformIO $ cast1 ATen.tensor_sizes tensor
          names =
            unsafePerformIO $
              ifM
                (cast1 ATen.tensor_has_names tensor)
                (cast1 ATen.tensor_names tensor)
                (pure $ map (const mempty) sizes)
          f (Dim UncheckedName UncheckedSize) _ _ = mempty
          f (Dim (Name name) UncheckedSize) name' _ = All $ name == name'
          f (Dim UncheckedName (Size size)) _ size' = All $ size == size'
          f (Dim (Name name) (Size size)) name' size' = All $ name == name' && size == size'
       in getAll . fold $ zipWith3 f dims names sizes

-- | Checks whether or not the input tensor has the shape 'shape'
-- and returns a statically annotated copy of it wrapped in a 'MonadFail' 'm'.
--
-- For instance, if 'm' is 'Maybe', then the result will be wrapped in 'Just' if and only if the tensor has indeed the shape 'shape'.
-- If it is not, then the result will be 'Nothing'.
--
-- In the REPL, 'm' will default to 'IO':
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @'UncheckedShape [Dim "batch" 32, Dim "feature" 8]
-- >>> t' <- checkedShape @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
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
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim 'UncheckedName ('Size 32),
--              'Dim ('Name "feature") 'UncheckedSize])
-- >>> t' <- checkedShape @('Shape [ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 32)]) t
-- *** Exception: user error (The tensor does not have the shape "Shape [Dim (Name "batch") (Size 32),Dim (Name "feature") (Size 32)]".)
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
-- >>> t = ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @'UncheckedShape [Dim "batch" 32, Dim "feature" 8]
-- >>> t' = unsafeCheckedShape @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)]) t
-- >>> :type t'
-- t'
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "batch") ('Size 32),
--              'Dim ('Name "feature") ('Size 8)])
-- >>> t' = unsafeCheckedShape @('Shape [ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 32)]) t
-- *** Exception: The tensor does not have the shape "Shape [Dim (Name "batch") (Size 32),Dim (Name "feature") (Size 32)]".
-- CallStack (from HasCallStack):
--   error, called at /root/hasktorch/hasktorch/src/Torch/GraduallyTyped/Tensor/Type.hs:910:15 in main:Torch.GraduallyTyped.Tensor.Type
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
