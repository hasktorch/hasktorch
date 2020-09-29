{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.GraduallyTyped.Prelude
  ( module Data.Kind,
    module Data.Proxy,
    module Data.Type.Bool,
    module GHC.TypeLits,
    Fst,
    Snd,
    Elem,
    Contains,
  )
where

import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Data.Type.Bool (If, type (||))
import GHC.TypeLits (ErrorMessage (..), TypeError (..))

-- | Returns the first element of a type-level tuple with the kind @(k, k')@ marked by a prefix quote.
--
-- >>> :kind! Fst '(Int, String)
-- Fst '(Int, String) :: Type
-- = Int
-- >>> :kind! Fst '(Functor, String)
-- Fst '(Functor, String) :: (Type -> Type) -> Constraint
-- = Functor
type family Fst (t :: (k, k')) :: k where
  Fst '(x, _) = x

-- | Returns the second element of a type-level tuple with the kind @(k, k')@ marked by a prefix quote.
--
-- >>> :kind! Fst '(Int, String)
-- Fst '(Int, String) :: Type
-- = Int
-- >>> :kind! Fst '(Functor, String)
-- Fst '(Functor, String) :: (Type -> Type) -> Constraint
-- = Functor
type family Snd (t :: (k, k')) :: k' where
  Snd '(_, y) = y

-- | Check that a given type is an element of a type-level list:
--
-- >>> :kind! Elem String '[]
-- Elem String '[] :: Bool
-- = 'False
-- >>> :kind! Elem String '[Int, String]
-- Elem String '[Int, String] :: Bool
-- = 'True
-- >>> :kind! Elem String '[Int, Bool]
-- Elem String '[Int, Bool] :: Bool
-- = 'False
type family Elem (e :: t) (es :: [t]) :: Bool where
  Elem _ '[] = 'False
  Elem x (x ': xs) = 'True
  Elem x (_ ': xs) = Elem x xs

-- | Test whether or not a given type contains another:
--
-- >>> :kind! Contains (Either Int String) Int
-- Contains (Either Int String) Int :: Bool
-- = 'True
-- >>> :kind! Contains (Either Int String) Bool
-- Contains (Either Int String) Bool :: Bool
-- = 'False
-- >>> :kind! Contains (Either Int String) Either
-- Contains (Either Int String) Either :: Bool
-- = 'True
type family Contains (f :: k) (a :: k') :: Bool where
  Contains a a = 'True
  Contains (f g) a = Contains f a || Contains g a
  Contains _ _ = 'False
