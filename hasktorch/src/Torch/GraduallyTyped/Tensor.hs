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

import Data.Int (Int16)
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits
  ( KnownNat,
    KnownSymbol,
    Nat,
    Symbol,
    natVal,
    symbolVal,
  )
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.Internal.Cast (cast0, cast1)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Autograd as ATen
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Cast as ATen ()
import qualified Torch.Internal.Type as ATen (TensorList, Tensor)

data RequiresGradient
  = -- | The tensor requires gradients
    Independent
  | -- | The tensor does not require gradients
    Dependent

class KnownRequiresGradient (requiresGradient :: RequiresGradient) where
  requiresGradientVal :: RequiresGradient

instance KnownRequiresGradient 'Independent where
  requiresGradientVal = Independent

instance KnownRequiresGradient 'Dependent where
  requiresGradientVal = Dependent

data LayoutType = Dense | Sparse

data Layout (layoutType :: Type) where
  AnyLayout :: forall layoutType. Layout layoutType
  Layout :: forall layoutType. layoutType -> Layout layoutType

class KnownLayout (layout :: Layout LayoutType) where
  layoutVal :: Layout LayoutType

instance KnownLayout 'AnyLayout where
  layoutVal = AnyLayout

instance KnownLayout ('Layout 'Dense) where
  layoutVal = Layout Dense

instance KnownLayout ('Layout 'Sparse) where
  layoutVal = Layout Sparse

data DeviceType (deviceId :: Type) where
  CPU :: forall deviceId. DeviceType deviceId
  CUDA :: forall deviceId. deviceId -> DeviceType deviceId

data Device (deviceType :: Type) where
  AnyDevice :: forall deviceType. Device deviceType
  Device :: forall deviceType. deviceType -> Device deviceType

class KnownDevice (device :: Device (DeviceType Nat)) where
  deviceVal :: Device (DeviceType Int16)

instance KnownDevice 'AnyDevice where
  deviceVal = AnyDevice

instance KnownDevice ('Device 'CPU) where
  deviceVal = Device CPU

instance (KnownNat deviceId) => KnownDevice ('Device ('CUDA deviceId)) where
  deviceVal = Device (CUDA (fromIntegral . natVal $ Proxy @deviceId))

data DataType (dType :: Type) where
  AnyDataType :: forall dType. DataType dType
  DataType :: forall dType. dType -> DataType dType

class KnownDataType (dataType :: DataType DType) where
  dataTypeVal :: DataType DType

instance KnownDataType 'AnyDataType where
  dataTypeVal = AnyDataType

instance KnownDataType ('DataType 'Bool) where
  dataTypeVal = DataType Bool

instance KnownDataType ('DataType 'UInt8) where
  dataTypeVal = DataType UInt8

instance KnownDataType ('DataType 'Int8) where
  dataTypeVal = DataType Int8

instance KnownDataType ('DataType 'Int16) where
  dataTypeVal = DataType Int16

instance KnownDataType ('DataType 'Int32) where
  dataTypeVal = DataType Int32

instance KnownDataType ('DataType 'Int64) where
  dataTypeVal = DataType Int64

instance KnownDataType ('DataType 'Half) where
  dataTypeVal = DataType Half

instance KnownDataType ('DataType 'Float) where
  dataTypeVal = DataType Float

instance KnownDataType ('DataType 'Double) where
  dataTypeVal = DataType Double

data Dim (name :: Type) (size :: Type) where
  AnyDim :: forall name size. Dim name size
  NamedDim :: forall name size. name -> Dim name size
  SizedDim :: forall name size. size -> Dim name size
  NamedSizedDim :: forall name size. name -> size -> Dim name size

data Shape (shapeList :: Type) where
  AnyShape :: forall shapeList. Shape shapeList
  Shape :: forall shapeList. shapeList -> Shape shapeList

class KnownShape (shape :: Shape [Dim Symbol Nat]) where
  shapeVal :: Shape [Dim String Integer]

instance KnownShape 'AnyShape where
  shapeVal = AnyShape

instance KnownShape ('Shape '[]) where
  shapeVal = Shape []

instance
  ( KnownShape ('Shape t),
    KnownSymbol name
  ) =>
  KnownShape ('Shape ('NamedDim name ': t))
  where
  shapeVal =
    let name = symbolVal $ Proxy @name
     in case shapeVal @('Shape t) of
          Shape t -> Shape $ NamedDim name : t

instance
  ( KnownShape ('Shape t),
    KnownNat size
  ) =>
  KnownShape ('Shape ('SizedDim size ': t))
  where
  shapeVal =
    let size = natVal $ Proxy @size
     in case shapeVal @('Shape t) of
          Shape t -> Shape $ SizedDim size : t

instance
  ( KnownShape ('Shape t),
    KnownSymbol name,
    KnownNat size
  ) =>
  KnownShape ('Shape ('NamedSizedDim name size ': t))
  where
  shapeVal =
    let name = symbolVal $ Proxy @name
        size = natVal $ Proxy @size
     in case shapeVal @('Shape t) of
          Shape t -> Shape $ NamedSizedDim name size : t

-- | A gradually typed tensor
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

instance Castable 
    [Tensor requiresGradient layout device dataType shape] 
    (ForeignPtr ATen.TensorList) where
  cast xs f = do
    ptrList <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Tensor))) xs
    cast ptrList f
  uncast xs f = uncast xs $ \ptrList -> do
    tensorList <- mapM (\(x :: ForeignPtr ATen.Tensor) -> uncast x return) ptrList
    f tensorList

type UntypedTensor = Tensor 'Dependent 'AnyLayout 'AnyDevice 'AnyDataType 'AnyShape

type UntypedParameter = Tensor 'Independent 'AnyLayout 'AnyDevice 'AnyDataType 'AnyShape

type CPUTensor = Tensor 'Dependent ('Layout 'Dense) ('Device 'CPU)

type CPUParameter = Tensor 'Independent ('Layout 'Dense) ('Device 'CPU)

type SparseCPUTensor = Tensor 'Dependent ('Layout 'Sparse) ('Device 'CPU)

type SparseCPUParameter = Tensor 'Independent ('Layout 'Sparse) ('Device 'CPU)

type CUDATensor deviceId = Tensor 'Dependent ('Layout 'Dense) ('Device ('CUDA deviceId))

type CUDAParameter deviceId = Tensor 'Independent ('Layout 'Dense) ('Device ('CUDA deviceId))

type SparseCUDATensor deviceId = Tensor 'Dependent ('Layout 'Sparse) ('Device ('CUDA deviceId))

type SparseCUDAParameter deviceId = Tensor 'Independent ('Layout 'Sparse) ('Device ('CUDA deviceId))

makeIndependent ::
  forall layout device dataType shape.
  Tensor 'Dependent layout device dataType shape ->
  IO (Tensor 'Independent layout device dataType shape)
makeIndependent = cast1 ATen.makeIndependent

makeDependent ::
  forall layout device dataType shape.
  Tensor 'Independent layout device dataType shape ->
  IO (Tensor 'Dependent layout device dataType shape)
makeDependent = undefined -- TODO: implement set_requires_grad(false)

toDense ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient ('Layout 'Dense) device dataType shape
toDense = unsafePerformIO . cast1 ATen.tensor_to_dense

toSparse ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient ('Layout 'Sparse) device dataType shape
toSparse = unsafePerformIO . cast1 ATen.tensor_to_sparse

layout ::
  forall requiresGradient layout device dataType shape.
  KnownLayout layout =>
  Tensor requiresGradient layout device dataType shape ->
  Layout LayoutType
layout tensor =
  case layoutVal @layout of
    AnyLayout ->
      if unsafePerformIO . cast1 ATen.tensor_is_sparse $ tensor
        then Layout Sparse
        else Layout Dense
    Layout layoutType -> Layout layoutType

cpu ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout ('Device 'CPU) dataType shape
cpu = unsafePerformIO . cast1 ATen.tensor_cpu

cuda ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout ('Device ('CUDA 0)) dataType shape
cuda = unsafePerformIO . cast1 ATen.tensor_cuda

device ::
  forall requiresGradient layout device dataType shape.
  KnownDevice device =>
  Tensor requiresGradient layout device dataType shape ->
  Device (DeviceType Int16)
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
                deviceId :: Int <- cast1 ATen.tensor_get_device $ tensor
                pure . Device . CUDA . fromIntegral $ deviceId
              else pure . Device $ CPU
          else pure . Device $ CPU
    Device deviceType -> Device deviceType
