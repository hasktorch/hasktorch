{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.Device where

import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import GHC.Generics
import GHC.TypeLits
import GHC.TypeLits.Extra
import System.IO.Unsafe
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Managed.Autograd as LibTorch
import qualified Torch.Tensor as D
import Torch.Typed.Auxiliary
import Torch.Typed.Functional
import Torch.Typed.Parameter
import Torch.Typed.Tensor

class
  HasToDevice
    (device' :: (D.DeviceType, Nat))
    (device :: (D.DeviceType, Nat))
    (f :: Type)
    (g :: Type)
    | device' device f -> g,
      device' device g -> f
  where
  -- >>> model <- A.sample (Torch.Typed.NN.LinearSpec @1 @1 @'D.Float @'( 'D.CPU, 0))
  -- >>> :type Torch.Typed.Device.toDevice @'( 'D.CUDA, 0) @'( 'D.CPU, 0) model
  -- Torch.Typed.Device.toDevice @'( 'D.CUDA, 0) @'( 'D.CPU, 0) model
  -- :: Torch.Typed.NN.Linear 1 1 'Float '( 'CUDA, 0)
  toDevice :: f -> g

-- In a data type `f` parameterized by zero or more device type variables, replace the given device type `device` with the device type `device'`.
--
-- >>> :kind! ReplaceDevice (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) '( 'D.CUDA, 0) '( 'D.CPU, 0)
-- ReplaceDevice (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) '( 'D.CUDA, 0) '( 'D.CPU, 0) :: *
-- = Torch.Typed.NN.Linear 1 1 'Float '( 'CUDA, 0)
type family ReplaceDevice (f :: k) (device' :: (D.DeviceType, Nat)) (device :: (D.DeviceType, Nat)) :: k where
  ReplaceDevice (t device) device' device = t device'
  ReplaceDevice (t a) device' device = (ReplaceDevice t device' device) (ReplaceDevice a device' device)
  ReplaceDevice t _ _ = t

-- In a data type `f` parameterized by zero or one device type variables, replace the only occurring device type with the device type `device'`.
--
-- >>> :kind! ReplaceDevice' (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) '( 'D.CUDA, 0)
-- ReplaceDevice' (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) '( 'D.CUDA, 0) :: *
-- = Torch.Typed.NN.Linear 1 1 'Float '( 'CUDA, 0)
type family ReplaceDevice' (f :: k) (device' :: (D.DeviceType, Nat)) :: k where
  ReplaceDevice' (t (device :: (D.DeviceType, Nat))) device' = t device'
  ReplaceDevice' (t a) device' = (ReplaceDevice' t device') (ReplaceDevice' a device')
  ReplaceDevice' t _ = t

instance
  ( g ~ ReplaceDevice f device' device,
    f ~ ReplaceDevice g device device',
    Generic f,
    Generic g,
    GHasToDevice device' device (Rep f) (Rep g)
  ) =>
  HasToDevice device' device f g
  where
  toDevice = to . gToDevice @device' @device . from

class
  GHasToDevice
    (device' :: (D.DeviceType, Nat))
    (device :: (D.DeviceType, Nat))
    (f :: Type -> Type)
    (g :: Type -> Type)
  where
  gToDevice :: forall a. f a -> g a

instance
  ( GHasToDevice device' device l l',
    GHasToDevice device' device r r'
  ) =>
  GHasToDevice device' device (l :*: r) (l' :*: r')
  where
  gToDevice (l :*: r) =
    let l' = gToDevice @device' @device l
        r' = gToDevice @device' @device r
     in l' :*: r'

instance {-# OVERLAPS #-} HasToDevice device' device Double Double where
  toDevice = id

instance {-# OVERLAPS #-} (KnownDevice device') => HasToDevice device' device (Tensor device dtype shape) (Tensor device' dtype shape) where
  toDevice = Torch.Typed.Tensor.toDevice

instance {-# OVERLAPS #-} (KnownDevice device') => HasToDevice device' device (Parameter device dtype shape) (Parameter device' dtype shape) where
  toDevice = Torch.Typed.Parameter.parameterToDevice

instance {-# OVERLAPS #-} HasToDevice device' device (HList ('[] :: [Type])) (HList ('[] :: [Type])) where
  toDevice = id

instance {-# OVERLAPS #-} (HasToDevice device' device x x', HasToDevice device' device (HList xs) (HList xs')) => HasToDevice device' device (HList (x ': xs)) (HList (x' ': xs')) where
  toDevice (x :. xs) = Torch.Typed.Device.toDevice @device' @device x :. Torch.Typed.Device.toDevice @device' @device xs

instance {-# OVERLAPPABLE #-} (HasToDevice device' device f g) => GHasToDevice device' device (K1 i f) (K1 i g) where
  gToDevice = K1 . Torch.Typed.Device.toDevice @device' @device . unK1

instance (GHasToDevice device' device f g) => GHasToDevice device' device (M1 i t f) (M1 i t g) where
  gToDevice = M1 . gToDevice @device' @device . unM1

instance GHasToDevice device' device U1 U1 where
  gToDevice = id

class HasReplicate (devices' :: [(D.DeviceType, Nat)]) (device :: (D.DeviceType, Nat)) (f :: Type) (gs :: [Type]) | devices' device f -> gs where
  replicate :: f -> HList gs

instance HasReplicate '[] device f '[] where
  replicate _ = HNil

instance
  ( HasReplicate devices' device f gs,
    HasToDevice device' device f g
  ) =>
  HasReplicate (device' ': devices') device f (g ': gs)
  where
  replicate f = Torch.Typed.Device.toDevice @device' @device f :. Torch.Typed.Device.replicate @devices' @device @f @gs f

class
  HasToDevices
    (devices' :: [(D.DeviceType, Nat)])
    (devices :: [(D.DeviceType, Nat)])
    (fs :: [Type])
    (gs :: [Type])
    | devices' devices fs -> gs,
      devices' devices gs -> fs
  where
  toDevices :: HList fs -> HList gs

instance HasToDevices '[] '[] '[] '[] where
  toDevices HNil = HNil

instance
  ( HasToDevices devices' devices fs gs,
    HasToDevice device' device f g
  ) =>
  HasToDevices (device' ': devices') (device ': devices) (f ': fs) (g ': gs)
  where
  toDevices (f :. fs) = Torch.Typed.Device.toDevice @device' @device f :. toDevices @devices' @devices @fs @gs fs

-- >>> :kind! GetDevice (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0))
-- GetDevice (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) :: Maybe (D.DeviceType, Nat)
-- = 'Just '( 'D.CPU, 0)
--
-- >>> :kind! GetDevice (Torch.Typed.Tensor.Tensor '( 'D.CUDA, 0) 'D.Float '[1])
-- GetDevice (Torch.Typed.Tensor.Tensor '( 'D.CUDA, 0) 'D.Float '[1]) :: Maybe (D.DeviceType, Nat)
-- = 'Just '( 'D.CUDA, 0)
type family GetDevice (f :: k) :: Maybe (D.DeviceType, Nat) where
  GetDevice (t (device :: (D.DeviceType, Nat))) = Just device
  GetDevice (t a) = GetDevice t
  GetDevice t = Nothing

-- >>> :kind! GetDevices '[Torch.Typed.Tensor.Tensor '( 'D.CUDA, 0) 'D.Float '[1], Torch.Typed.Tensor.Tensor '( 'D.CUDA, 1) 'D.Float '[1]]
-- GetDevices '[Torch.Typed.Tensor.Tensor '( 'D.CUDA, 0) 'D.Float '[1], Torch.Typed.Tensor.Tensor '( 'D.CUDA, 1) 'D.Float '[1]] :: [(D.DeviceType, Nat)]
-- = '[ '( 'D.CUDA, 0), '( 'D.CUDA, 1)]
type family GetDevices (fs :: [k]) :: [(D.DeviceType, Nat)] where
  GetDevices '[] = '[]
  GetDevices (f ': fs) = MaybePrepend (GetDevice f) (GetDevices fs)

-- class HasChunk chunks f gs | chunks f -> gs where
--   chunk :: f -> HList gs

-- class GHasChunk
--   (chunks :: Nat)
--   (f :: Type -> Type)
--   (gs :: [Type]) | chunks f -> gs where
--   gChunk :: forall a . f a -> HList gs

-- class GZipChunks (gs :: [k]) (gs' :: [k]) (gs'' :: [k]) | gs gs' -> gs'' where
--   gZipChunks   :: HList gs -> HList gs' -> HList gs''
--   -- gUnzipChunks :: HList gs'' -> HList gs :*: HList gs'

-- instance GZipChunks '[] '[] '[] where
--   gZipChunks _ _ = HNil
--   -- gUnzipChunks _ = HNil :*: HNil

-- instance
--   ( (g :*: g') ~ g''
--   , GZipChunks gs gs' gs''
--   ) => GZipChunks (g ': gs) (g' ': gs') (g'' ': gs'') where
--   gZipChunks (g :. gs) (g' :. gs') = (g :*: g') :. gZipChunks gs gs'
-- gZipChunks x y = _undefined
-- gUnzipChunks (~(g :*: g') :. gs'') =
--   let ~(gs :*: gs') = gUnzipChunks gs''
--   in  (g :. gs') :*: (y :. ys)

-- instance
--   (

--   ) =>

-- class HasCat fs g | fs -> g where
--   cat :: HList fs -> g

class HasScatter devices' device f gs | devices' device f -> gs where
  scatter :: f -> HList gs

instance
  ( chunks ~ ListLength devices',
    tensorChunks ~ Chunk chunks 0 shape dtype device,
    ATen.Castable (HList tensorChunks) [D.ATenTensor],
    devices ~ HReplicateR chunks device,
    HasToDevices devices' devices tensorChunks gs,
    KnownNat chunks
  ) =>
  HasScatter devices' device (Tensor device dtype shape) gs
  where
  scatter = toDevices @devices' @devices . Torch.Typed.Functional.chunk @chunks @0

class HasGather device' devices fs g | device' devices fs -> g where
  gather :: HList fs -> g

instance
  ( chunks ~ ListLength fs,
    devices ~ GetDevices fs,
    devices' ~ HReplicateR chunks device',
    HasToDevices devices' devices fs tensorChunks,
    '(shape, dtype, device') ~ Cat 0 tensorChunks,
    ATen.Castable (HList tensorChunks) [D.ATenTensor]
  ) =>
  HasGather device' devices fs (Tensor device' dtype shape)
  where
  gather = Torch.Typed.Functional.cat @0 . toDevices @devices' @devices
