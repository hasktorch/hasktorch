{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DeriveGeneric #-}

module Torch.Typed.Device where

import           Torch.HList
import           Data.Kind                    (Type)
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           GHC.Generics
import           System.IO.Unsafe

import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Managed.Autograd as LibTorch
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import qualified Torch.Tensor                  as D
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter

class HasToDevice (device' :: (D.DeviceType, Nat)) (device :: (D.DeviceType, Nat)) f g | device' device f -> g, device' device g -> f where
  -- >>> model <- A.sample (Torch.Typed.NN.LinearSpec @1 @1 @'D.Float @'( 'D.CPU, 0))
  -- >>> :type Torch.Typed.Device.toDevice @'( 'D.CUDA, 0) @'( 'D.CPU, 0) model
  -- Torch.Typed.Device.toDevice @'( 'D.CUDA, 0) @'( 'D.CPU, 0) model
  -- :: Torch.Typed.NN.Linear 1 1 'Float '( 'CUDA, 0)
  toDevice :: f -> g

-- >>> :kind! PutDevice (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) '( 'D.CUDA, 0) '( 'D.CPU, 0)
-- PutDevice (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) '( 'D.CUDA, 0) '( 'D.CPU, 0) :: *
-- = Torch.Typed.NN.Linear 1 1 'Float '( 'CUDA, 0)
type family PutDevice (f :: k) (device' :: (D.DeviceType, Nat)) (device :: (D.DeviceType, Nat)) :: k where
  PutDevice (t device) device' device = t device'
  PutDevice (t a)      device' device = (PutDevice t device' device) a
  PutDevice t          _       _      = t

instance
  ( g ~ PutDevice f device' device
  , f ~ PutDevice g device device'
  , Generic f
  , Generic g
  , GHasToDevice device' device (Rep f) (Rep g)
  ) => HasToDevice device' device f g where
  toDevice = to . gToDevice @device' @device . from

class GHasToDevice
  (device' :: (D.DeviceType, Nat))
  (device :: (D.DeviceType, Nat))
  (f :: Type -> Type)
  (g :: Type -> Type) where
  gToDevice :: forall a . f a -> g a

instance
  ( GHasToDevice device' device l l'
  , GHasToDevice device' device r r'
  ) => GHasToDevice device' device (l :*: r) (l' :*: r') where
  gToDevice (l :*: r) =
    let l' = gToDevice @device' @device l
        r' = gToDevice @device' @device r
    in  l' :*: r'

instance {-# OVERLAPS #-} (KnownDevice device') => HasToDevice device' device (Tensor device dtype shape) (Tensor device' dtype shape) where
  toDevice = Torch.Typed.Tensor.toDevice

instance {-# OVERLAPS #-} (KnownDevice device') => HasToDevice device' device (Parameter device dtype shape) (Parameter device' dtype shape) where
  toDevice = Torch.Typed.Parameter.toDevice

instance {-# OVERLAPPABLE #-} (HasToDevice device' device f g) => GHasToDevice device' device (K1 i f) (K1 i g) where
  gToDevice = K1 . Torch.Typed.Device.toDevice @device' @device . unK1

instance (GHasToDevice device' device f g) => GHasToDevice device' device (M1 i t f) (M1 i t g) where
  gToDevice = M1 . gToDevice @device' @device . unM1

instance GHasToDevice device' device U1 U1 where
  gToDevice = id
