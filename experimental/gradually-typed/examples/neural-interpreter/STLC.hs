{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}

module STLC where

import Bound (Scope, abstract1, fromScope, instantiate1, toScope, (>>>=))
import Control.Monad (MonadPlus (mzero), ap, guard)
import Control.Monad.Fresh (Fresh, MonadFresh (fresh), runFreshFrom)
import Control.Monad.Trans.Maybe (MaybeT (..))
import Data.Deriving (deriveEq1, deriveOrd1, deriveShow1)
import Data.Functor.Classes (compare1, eq1, showsPrec1)
import Data.Map (Map)
import qualified Data.Map as Map
import GHC.Generics (Generic, Generic1)
import qualified Language.Haskell.TH as TH

data Ty
  = -- | Arrow type (functions).
    Arr Ty Ty
  | -- | Natural number type.
    Nat
  deriving (Eq, Ord, Show, Generic)

data Exp a
  = -- | Variable.
    Var a
  | -- | Lambda abstraction.
    Lam {ty :: Ty, lamExp :: Scope () Exp a}
  | -- | Term application.
    (:@) {function :: Exp a, argument :: Exp a}
  | -- | Successor.
    Succ (Exp a)
  | -- | Zero.
    Zero
  deriving stock (Functor, Foldable, Traversable, Generic, Generic1)

deriveEq1 ''Exp
deriveShow1 ''Exp
deriveOrd1 ''Exp

instance Applicative Exp where
  pure = Var
  (<*>) = ap

instance Monad Exp where
  return = Var
  Var a >>= f = f a
  (x :@ y) >>= f = (x >>= f) :@ (y >>= f)
  Lam ty' e >>= f = Lam ty' (e >>>= f)
  Succ e >>= f = Succ (e >>= f)
  Zero >>= _ = Zero

instance Eq a => Eq (Exp a) where (==) = eq1

instance Ord a => Ord (Exp a) where compare = compare1

instance Show a => Show (Exp a) where showsPrec = showsPrec1

-- | Smart constructor for lambda terms
lam :: forall a. Eq a => Ty -> a -> Exp a -> Exp a
lam ty' uname bind = Lam ty' (abstract1 uname bind)

-- | Smart constructor that converts the given positive integer to a corresponding Nat.
nat :: forall a n. (Num n, Eq n) => n -> Exp a
nat 0 = Zero
nat n = Succ $ nat (n -1)

-- | Compute the normal form of an expression.
nf :: forall a. Exp a -> Exp a
nf e@Var {} = e
nf (Lam ty' b) = Lam ty' (toScope . nf . fromScope $ b)
nf (f :@ a) = case whnf f of
  Lam _ b -> nf (instantiate1 a b)
  f' -> nf f' :@ nf a
nf (Succ e) = Succ (nf e)
nf e@Zero = e

-- | Reduce a term to weak head normal form.
whnf :: forall a. Exp a -> Exp a
whnf e@Var {} = e
whnf e@Lam {} = e
whnf (f :@ a) = case whnf f of
  Lam _ b -> whnf (instantiate1 a b)
  f' -> f' :@ a
whnf e@Succ {} = e
whnf e@Zero = e

-- | Monad stack for type checking.
type TyM a = MaybeT (Fresh a)

-- | Guard against a type error.
assertTy :: Ord a => Map a Ty -> Exp a -> Ty -> TyM a ()
assertTy env e t = typeCheck env e >>= guard . (== t)

-- | Check the type of an expression.
typeCheck :: forall a. Ord a => Map a Ty -> Exp a -> TyM a Ty
typeCheck _ Zero = return Nat
typeCheck env (Succ e) = assertTy env e Nat >> return Nat
typeCheck env (Var a) = MaybeT . return $ Map.lookup a env
typeCheck env (f :@ a) =
  typeCheck env f >>= \case
    Arr fTy tTy -> assertTy env a fTy >> return tTy
    _ -> mzero
typeCheck env (Lam ty' bind) = do
  uname <- fresh
  Arr ty' <$> typeCheck (Map.insert uname ty' env) (instantiate1 (Var uname) bind)

type TyTH a = Fresh a TH.Exp

toTH :: forall a. Enum a => a -> Exp a -> TH.Exp
toTH a e = runFreshFrom a $ go e
  where
    zero = TH.mkName "Zero"
    succ' = TH.mkName "Succ"
    toName a' = TH.mkName $ "x" <> (show . fromEnum $ a')
    go :: Exp a -> TyTH a
    go Zero = pure $ TH.ConE zero
    go (Succ e') = TH.AppE (TH.ConE succ') <$> go e'
    go (Var a') =
      let name = toName a'
       in pure $ TH.VarE name
    go (f :@ a') = TH.AppE <$> go f <*> go a'
    go (Lam _ bind) = do
      a' <- fresh
      let name = toName a'
      TH.LamE [TH.VarP name] <$> go (instantiate1 (Var a') bind)

pprint :: Exp Int -> String
pprint = TH.pprint . toTH 0
