{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.Aux where

import qualified Data.Int                      as I
import           Data.Kind                      ( Constraint )
import           Data.Proxy
import           GHC.TypeLits

import qualified Torch.Device                  as D
import qualified Torch.DType                   as D

type family Fst (t :: (a, b)) :: a where
  Fst '(x, _) = x

type family Snd (t :: (a, b)) :: b where
  Snd '(_, x) = x

type family Fst3 (t :: (a, b, c)) :: a where
  Fst3 '(x, _, _) = x

type family Snd3 (t :: (a, b, c)) :: b where
  Snd3 '(_, x, _) = x

type family Trd3 (t :: (a, b, c)) :: c where
  Trd3 '(_, _, x) = x

natValI :: forall n . KnownNat n => Int
natValI = fromIntegral $ natVal $ Proxy @n

natValInt16 :: forall n . KnownNat n => I.Int16
natValInt16 = fromIntegral $ natVal $ Proxy @n

type family DTypeIsFloatingPoint (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  DTypeIsFloatingPoint _                'D.Half   = ()
  DTypeIsFloatingPoint _                'D.Float  = ()
  DTypeIsFloatingPoint _                'D.Double = ()
  DTypeIsFloatingPoint '(deviceType, _) dtype     = UnsupportedDTypeForDevice deviceType dtype

type family DTypeIsIntegral (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  DTypeIsIntegral _                     'D.Bool  = ()
  DTypeIsIntegral _                     'D.UInt8 = ()
  DTypeIsIntegral _                     'D.Int8  = ()
  DTypeIsIntegral _                     'D.Int16 = ()
  DTypeIsIntegral _                     'D.Int32 = ()
  DTypeIsIntegral _                     'D.Int64 = ()
  DTypeIsFloatingPoint '(deviceType, _) dtype    = UnsupportedDTypeForDevice deviceType dtype

type family DTypeIsNotHalf (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  DTypeIsNotHalf '(deviceType, _) D.Half = UnsupportedDTypeForDevice deviceType D.Half
  DTypeIsNotHalf _                _      = ()

type family DTypeIsNotBool (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  DTypeIsNotHalf '(deviceType, _) D.Bool = UnsupportedDTypeForDevice deviceType D.Bool
  DTypeIsNotHalf _                _      = ()

type family UnsupportedDTypeForDevice (deviceType :: D.DeviceType) (dtype :: D.DType) :: Constraint where
  UnsupportedDTypeForDevice deviceType dtype = TypeError (    Text "This operation does not support "
                                                         :<>: ShowType dtype
                                                         :<>: Text " tensors on devices of type "
                                                         :<>: ShowType deviceType
                                                         :<>: Text "."
                                                         )
