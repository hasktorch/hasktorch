{-# LANGUAGE DataKinds #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Prelude
  ( module Data.Kind,
    module Data.Proxy,
    module Data.Type.Bool,
    module GHC.TypeLits,
    Assert,
    Fst,
    Snd,
    Elem,
    Contains,
    PrependMaybe,
    MapMaybe,
    BindMaybe,
    JoinMaybe,
    whenM,
    unlessM,
    ifM,
    guardM,
    (&&^),
    (||^),
    (<&&>),
    (<||>),
  )
where

import Control.Applicative (Applicative (liftA2))
import Control.Monad (MonadPlus, guard, unless, when)
import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))
import Data.String (IsString, fromString)
import Data.Type.Bool (If, type (||))
import GHC.Exts (Any)
import GHC.TypeLits (ErrorMessage (..), TypeError (..))

-- $setup
-- >>> :seti -XNoStarIsType

data T1

-- | Can be used to report stuck type families,
-- see https://kcsongor.github.io/report-stuck-families/
type family Assert (err :: Constraint)  (a :: k) :: k where
  Assert _ T1 = Any
  Assert _ k = k

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
  Elem x (x ': _) = 'True
  Elem x (_ ': xs) = Elem x xs

type family PrependMaybe (h :: Maybe a) (t :: Maybe [a]) :: Maybe [a] where
  PrependMaybe 'Nothing _ = 'Nothing
  PrependMaybe _ 'Nothing = 'Nothing
  PrependMaybe ( 'Just h) ( 'Just t) = 'Just (h : t)

type family MapMaybe (f :: k -> k') (a :: Maybe k) :: Maybe k' where
  MapMaybe _ 'Nothing = 'Nothing
  MapMaybe f ( 'Just k) = 'Just (f k)

type family BindMaybe (f :: k -> Maybe k') (a :: Maybe k) :: Maybe k' where
  BindMaybe _ 'Nothing = 'Nothing
  BindMaybe f ( 'Just k) = f k

type family JoinMaybe (a :: Maybe (Maybe k)) :: Maybe k where
  JoinMaybe 'Nothing = 'Nothing
  JoinMaybe ( 'Just 'Nothing) = 'Nothing
  JoinMaybe ( 'Just ( 'Just k)) = 'Just k

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

whenM :: Monad m => m Bool -> m () -> m ()
whenM p m =
  p >>= flip when m

unlessM :: Monad m => m Bool -> m () -> m ()
unlessM p m =
  p >>= flip unless m

ifM :: Monad m => m Bool -> m a -> m a -> m a
ifM p x y = p >>= \b -> if b then x else y

guardM :: MonadPlus m => m Bool -> m ()
guardM f = guard =<< f

-- | The '||' operator lifted to a monad. If the first
-- argument evaluates to 'True' the second argument will not
-- be evaluated.
infixr 2 ||^ -- same as (||)

(||^) :: Monad m => m Bool -> m Bool -> m Bool
(||^) a b = ifM a (return True) b

infixr 2 <||>

-- | '||' lifted to an Applicative.
-- Unlike '||^' the operator is __not__ short-circuiting.
(<||>) :: Applicative a => a Bool -> a Bool -> a Bool
(<||>) = liftA2 (||)
{-# INLINE (<||>) #-}

-- | The '&&' operator lifted to a monad. If the first
-- argument evaluates to 'False' the second argument will not
-- be evaluated.
infixr 3 &&^ -- same as (&&)

(&&^) :: Monad m => m Bool -> m Bool -> m Bool
(&&^) a b = ifM a b (return False)

infixr 3 <&&>

-- | '&&' lifted to an Applicative.
-- Unlike '&&^' the operator is __not__ short-circuiting.
(<&&>) :: Applicative a => a Bool -> a Bool -> a Bool
(<&&>) = liftA2 (&&)
{-# INLINE (<&&>) #-}

instance IsString str => MonadFail (Either str) where
  fail :: String -> Either str a
  fail = Left . fromString