{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.Factories where

import Control.Arrow ((&&&))
import Data.Default.Class
import Data.Finite
import Data.Kind (Constraint)
import Data.Proxy
import Data.Reflection
import GHC.TypeLits
import System.IO.Unsafe
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D
import Torch.Internal.Cast
import qualified Torch.Scalar as D
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import qualified Torch.TensorOptions as D
import Torch.Typed.Auxiliary
import Torch.Typed.Tensor
import Prelude hiding (sin)

instance (TensorOptions shape dtype device) => Default (Tensor device dtype shape) where
  def = zeros

instance (TensorOptions shape' dtype device, shape' ~ ToNats shape) => Default (NamedTensor device dtype shape) where
  def = fromUnnamed zeros

zeros ::
  forall shape dtype device.
  (TensorOptions shape dtype device) =>
  Tensor device dtype shape
zeros =
  UnsafeMkTensor $
    D.zeros
      (optionsRuntimeShape @shape @dtype @device)
      ( D.withDevice (optionsRuntimeDevice @shape @dtype @device)
          . D.withDType (optionsRuntimeDType @shape @dtype @device)
          $ D.defaultOpts
      )

full ::
  forall shape dtype device a.
  (TensorOptions shape dtype device, D.Scalar a) =>
  a ->
  Tensor device dtype shape
full value =
  UnsafeMkTensor $
    D.full
      (optionsRuntimeShape @shape @dtype @device)
      value
      ( D.withDevice (optionsRuntimeDevice @shape @dtype @device)
          . D.withDType (optionsRuntimeDType @shape @dtype @device)
          $ D.defaultOpts
      )

ones ::
  forall shape dtype device.
  (TensorOptions shape dtype device) =>
  Tensor device dtype shape
ones =
  UnsafeMkTensor $
    D.ones
      (optionsRuntimeShape @shape @dtype @device)
      ( D.withDevice (optionsRuntimeDevice @shape @dtype @device)
          . D.withDType (optionsRuntimeDType @shape @dtype @device)
          $ D.defaultOpts
      )

type family RandDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  RandDTypeIsValid '( 'D.CPU, 0) dtype =
    ( DTypeIsNotBool '( 'D.CPU, 0) dtype,
      DTypeIsNotHalf '( 'D.CPU, 0) dtype
    )
  RandDTypeIsValid '( 'D.CUDA, _) dtype = ()
  RandDTypeIsValid '(deviceType, _) dtype = UnsupportedDTypeForDevice deviceType dtype

rand ::
  forall shape dtype device.
  ( TensorOptions shape dtype device,
    RandDTypeIsValid device dtype
  ) =>
  IO (Tensor device dtype shape)
rand =
  UnsafeMkTensor
    <$> D.randIO
      (optionsRuntimeShape @shape @dtype @device)
      ( D.withDevice (optionsRuntimeDevice @shape @dtype @device)
          . D.withDType (optionsRuntimeDType @shape @dtype @device)
          $ D.defaultOpts
      )

randn ::
  forall shape dtype device.
  ( TensorOptions shape dtype device,
    RandDTypeIsValid device dtype
  ) =>
  IO (Tensor device dtype shape)
randn =
  UnsafeMkTensor
    <$> D.randnIO
      (optionsRuntimeShape @shape @dtype @device)
      ( D.withDevice (optionsRuntimeDevice @shape @dtype @device)
          . D.withDType (optionsRuntimeDType @shape @dtype @device)
          $ D.defaultOpts
      )

randint ::
  forall shape dtype device.
  ( TensorOptions shape dtype device,
    RandDTypeIsValid device dtype
  ) =>
  Int ->
  Int ->
  IO (Tensor device dtype shape)
randint low high =
  UnsafeMkTensor
    <$> (D.randintIO low high)
      (optionsRuntimeShape @shape @dtype @device)
      ( D.withDevice (optionsRuntimeDevice @shape @dtype @device)
          . D.withDType (optionsRuntimeDType @shape @dtype @device)
          $ D.defaultOpts
      )

-- | linspace
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Float]) $ linspace @7 @'( 'D.CPU, 0) 0 3
-- (Float,([7],[0.0,0.5,1.0,1.5,2.0,2.5,3.0]))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Float]) $ linspace @3 @'( 'D.CPU, 0) 0 2
-- (Float,([3],[0.0,1.0,2.0]))
linspace ::
  forall steps device start end.
  ( D.Scalar start,
    D.Scalar end,
    KnownNat steps,
    TensorOptions '[steps] 'D.Float device
  ) =>
  -- | start
  start ->
  -- | end
  end ->
  -- | output
  Tensor device 'D.Float '[steps]
linspace start end =
  UnsafeMkTensor $
    D.linspace
      start
      end
      (natValI @steps)
      ( D.withDevice (optionsRuntimeDevice @'[steps] @D.Float @device)
          . D.withDType (optionsRuntimeDType @'[steps] @D.Float @device)
          $ D.defaultOpts
      )

eyeSquare ::
  forall n dtype device.
  ( KnownNat n,
    TensorOptions '[n, n] dtype device
  ) =>
  -- | output
  Tensor device dtype '[n, n]
eyeSquare =
  UnsafeMkTensor $
    D.eyeSquare
      (natValI @n)
      ( D.withDevice (optionsRuntimeDevice @'[n, n] @dtype @device)
          . D.withDType (optionsRuntimeDType @'[n, n] @dtype @device)
          $ D.defaultOpts
      )
