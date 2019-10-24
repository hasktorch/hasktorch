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

type family DTypeIsNotHalf (dtype :: D.DType) :: Constraint where
  DTypeIsNotHalf D.Half = TypeError (Text "This operation does not support " :<>: ShowType D.Half :<>: Text " tensors.")
  DTypeIsNotHalf _      = ()

type family DTypeIsNotBool (dtype :: D.DType) :: Constraint where
  DTypeIsNotHalf D.Bool = TypeError (Text "This operation does not support " :<>: ShowType D.Bool :<>: Text " tensors.")
  DTypeIsNotHalf _      = ()
