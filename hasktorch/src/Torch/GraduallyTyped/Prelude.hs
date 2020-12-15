{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Prelude
  ( module Data.Kind,
    module Data.Proxy,
    module Data.Type.Bool,
    module GHC.TypeLits,
    All,
    KnownElem (..),
    KnownList (..),
    Assert,
    Catch,
    Seq,
    Fst,
    Snd,
    Elem,
    Head,
    Tail,
    Contains,
    FromMaybe,
    FstMaybe,
    SndMaybe,
    PrependMaybe,
    MapMaybe,
    BindMaybe,
    JoinMaybe,
    LiftM2Maybe,
    LiftTimesMaybe,
    LiftTypeEqMaybe,
    Concat,
    Reverse,
    whenM,
    unlessM,
    ifM,
    guardM,
    (&&^),
    (||^),
    (<&&>),
    (<||>)
  )
where

import Control.Applicative (Applicative (liftA2))
import Control.Monad (MonadPlus, guard, unless, when)
import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))
import Data.String (IsString, fromString)
import Data.Type.Bool (If, type (||))
import GHC.Exts (Any)
import GHC.TypeLits (Nat, ErrorMessage (..), TypeError (..), type (*))

type family All (c :: k -> Constraint) (xs :: [k]) :: Constraint where
  All _ '[] = ()
  All c (h ': t) = (c h, All c t)

class KnownElem k x where
  type ElemValF k :: Type
  elemVal :: ElemValF k

class KnownList k (xs :: [k]) where
  listVal :: [ElemValF k]

instance KnownList k '[] where
  listVal = []

instance (KnownElem k x, KnownList k xs) => KnownList k (x ': xs) where
  listVal = elemVal @k @x : listVal @k @xs

-- | Just a dummy type for 'Assert' type family, see below.
data T

-- | Can be used to report stuck type families,
-- see https://kcsongor.github.io/report-stuck-families/.
-- This family is able to check whether its argument 'a' is stuck and
-- report an error 'err' in that case.
type family Assert (err :: Constraint) (a :: k) :: k where
  Assert _ T = Any
  Assert _ a = a

-- | Approximates a normal form on the type level.
-- 'Catch' forces its argument 'a' and returns an empty 'Constraint'
-- if and only if the argument does not produce a 'TypeError'.
--
-- The first equation will recursively force the kind of the argument
-- until it reaches 'Type' or a 'TypeError'.
-- In the former case, it falls over to the second equation which will produce the
-- empty constraint.
-- In the latter case, it gets stuck with 'Catch (TypeError ...)',
-- and the compiler will report the error message.
--
-- Thanks to <https://kcsongor.github.io/kcsongor> for the suggestion.
type family Catch (a :: k) :: Constraint where
  Catch (f a) = (Catch f, Catch a)
  Catch _  = ()

type family Seq (a :: k) (b :: k') :: k' where
  Seq (f a) b = Seq (Seq f a) b
  Seq _ b = b

-- | Returns the first element of a type-level tuple with the kind @(k, k')@ marked by a prefix quote.
--
-- >>> :kind! Fst '(Int, String)
-- Fst '(Int, String) :: *
-- = Int
-- >>> :kind! Fst '(Functor, String)
-- Fst '(Functor, String) :: (* -> *) -> Constraint
-- = Functor
type family Fst (t :: (k, k')) :: k where
  Fst '(x, _) = x

-- | Returns the second element of a type-level tuple with the kind @(k, k')@ marked by a prefix quote.
--
-- >>> :kind! Snd '(Int, String)
-- Snd '(Int, String) :: *
-- = [Char]
-- >>> :kind! Snd '(Int, Monad)
-- Snd '(Int, Monad) :: (* -> *) -> Constraint
-- = Monad
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

type family Head (xs :: [a]) :: Maybe a where
  Head '[] = 'Nothing
  Head (x ': _) = 'Just x

type family Tail (xs :: [a]) :: Maybe [a] where
  Tail '[] = 'Nothing
  Tail (_ ': xs) = 'Just xs 

type family ReverseImplF (xs :: [a]) (acc :: [a]) :: [a] where
  ReverseImplF '[] acc = acc
  ReverseImplF (h ': t) acc = ReverseImplF t (h ': acc)

type Reverse xs = ReverseImplF xs '[]

type family FromMaybe (d :: k) (x :: Maybe k) :: k where
  FromMaybe d 'Nothing = d
  FromMaybe _ ('Just v) = v

type family FstMaybe (t :: Maybe (k, k')) :: Maybe k where
  FstMaybe 'Nothing = 'Nothing
  FstMaybe ('Just '(x, _)) = 'Just x

type family SndMaybe (t :: Maybe (k, k')) :: Maybe k' where
  SndMaybe 'Nothing = 'Nothing
  SndMaybe ('Just '(_, y)) = 'Just y

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

type family LiftM2Maybe (f :: k -> k' -> k'') (a :: Maybe k) (b :: Maybe k') :: Maybe k'' where
  LiftM2Maybe _ 'Nothing _ = 'Nothing
  LiftM2Maybe _ ('Just _) 'Nothing = 'Nothing
  LiftM2Maybe f ('Just a) ('Just b) = 'Just (f a b)

type family LiftTimesMaybe (a :: Maybe Nat) (b :: Maybe Nat) :: Maybe Nat where
  LiftTimesMaybe 'Nothing _ = 'Nothing
  LiftTimesMaybe ('Just _) 'Nothing = 'Nothing
  LiftTimesMaybe ('Just a) ('Just b) = 'Just (a * b)

type family LiftTypeEqMaybe (a :: Maybe k) (b :: Maybe k') :: Constraint where
  LiftTypeEqMaybe 'Nothing _ = ()
  LiftTypeEqMaybe ('Just _) 'Nothing = ()
  LiftTypeEqMaybe ('Just a) ('Just b) = a ~ b

type family Concat (xs :: [k]) (ys :: [k]) :: [k] where
  Concat '[] ys = ys
  Concat (x ': xs) ys = x ': Concat xs ys

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