{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE GADTs #-}


module STLC where

import Bound (Scope, Var, abstract1, fromScope, instantiate1, toScope, (>>>=))
import Control.Monad (MonadPlus (mzero), ap, guard)
import Control.Monad.Fresh (Fresh, MonadFresh (fresh), runFreshFrom)
import Control.Monad.State (MonadState, modify)
import Control.Monad.Trans.Maybe (MaybeT (..))
import Data.Aeson (FromJSON, ToJSON)
import Data.Deriving (deriveEq1, deriveOrd1, deriveShow1)
import Data.Functor.Classes (compare1, eq1, showsPrec1, Eq1)
import Data.Hashable (Hashable)
import Data.Hashable.Lifted (Hashable1)
import Data.Map (Map)
import qualified Data.Map as Map
import GHC.Generics (Generic, Generic1)
import qualified Language.Haskell.TH as TH

data Ty
  = -- | Arrow type (functions).
    TArr Ty Ty
  | -- | Integer number type.
    TInt
  deriving stock (Eq, Ord, Show, Generic)
  deriving anyclass (Hashable)

data Exp a
  = -- | Variable.
    Var a
  | -- | Lambda abstraction.
    Lam {ty :: Ty, lamExp :: Scope () Exp a}
  | -- | Term application.
    (:@) {function :: Exp a, argument :: Exp a}
  | -- | Integer constant.
    Const {val :: Integer}
  | -- | Addition.
    Add {left :: Exp a, right :: Exp a}
  | -- | Subtraction.
    Sub {left :: Exp a, right :: Exp a}
  | -- | Multiplication.
    Mul {left :: Exp a, right :: Exp a}
  | -- | Negate.
    Neg {exp :: Exp a}
  | -- | Absolute value.
    Abs {exp :: Exp a}
  | -- | Sign of a number.
    Sign {exp :: Exp a}
  deriving stock (Functor, Foldable, Traversable, Generic, Generic1)

instance Applicative Exp where
  pure = Var
  (<*>) = ap

instance Monad Exp where
  return = Var
  Var a >>= f = f a
  (x :@ y) >>= f = (x >>= f) :@ (y >>= f)
  Lam ty' e >>= f = Lam ty' (e >>>= f)
  Const i >>= _ = Const i
  (Add l r) >>= f = Add (l >>= f) (r >>= f)
  (Sub l r) >>= f = Sub (l >>= f) (r >>= f)
  (Mul l r) >>= f = Mul (l >>= f) (r >>= f)
  (Neg e) >>= f = Neg (e >>= f)
  (Abs e) >>= f = Abs (e >>= f)
  (Sign e) >>= f = Sign (e >>= f)

instance FromJSON Ty

instance FromJSON a => FromJSON (Var () (Exp a))

instance FromJSON a => FromJSON (Scope () Exp a)

instance FromJSON a => FromJSON (Exp a)

instance ToJSON Ty

instance ToJSON a => ToJSON (Var () (Exp a))

instance ToJSON a => ToJSON (Scope () Exp a)

instance ToJSON a => ToJSON (Exp a)

fmap concat $ sequence
  [ deriveEq1   ''Exp
  , deriveOrd1  ''Exp
  , deriveShow1 ''Exp
  , [d| instance Eq a => Eq (Exp a) where (==) = eq1
        instance Ord a => Ord (Exp a) where compare = compare1
        instance Show a => Show (Exp a) where showsPrec = showsPrec1
      |]
  ]

instance (Eq a, Hashable a) => Hashable (Exp a)

instance Hashable1 Exp

-- | Smart constructor for lambda terms
lam :: forall a. Eq a => Ty -> a -> Exp a -> Exp a
lam ty' uname bind = Lam ty' (abstract1 uname bind)

-- | Smart constructor that converts the given integral number to a corresponding integer term.
int :: forall a n. Integral n => n -> Exp a
int = Const . fromIntegral

-- | Compute the normal form of an expression.
nf :: forall a n m. (Enum n, MonadState n m) => Exp a -> m (Exp a)
nf e = do
  modify succ
  go e
  where
    go e@Var {} = pure e
    go (Lam ty' b) = do
      b' <- toScope <$> nf (fromScope b)
      pure $ Lam ty' b'
    go (f :@ a) = do
      r <- whnf f
      case r of
        Lam _ b -> nf (instantiate1 a b)
        f' -> (:@) <$> nf f' <*> nf a
    go e@(Const _) = pure e
    go (Add l r) = do
      l' <- nf l
      r' <- nf r
      case (l', r') of
        (Const i, Const j) -> pure $ Const (i + j)
        (l', r') -> pure $ Add l' r'
    go (Sub l r) = do
      l' <- nf l
      r' <- nf r
      case (l', r') of
        (Const i, Const j) -> pure $ Const (i - j)
        (l', r') -> pure $ Sub l' r'
    go (Mul l r) = do
      l' <- nf l
      r' <- nf r
      case (l', r') of
        (Const i, Const j) -> pure $ Const (i * j)
        (l', r') -> pure $ Mul l' r'
    go (Neg e) = do
      e' <- nf e
      case e' of
        Const i -> pure $ Const (negate i)
        e' -> pure $ Neg e'
    go (Abs e) = do
      e' <- nf e
      case e' of
        Const i -> pure $ Const (abs i)
        e' -> pure $ Abs e'
    go (Sign e) = do
      e' <- nf e
      case e' of
        Const i -> pure $ Const (signum i)
        e' -> pure $ Sign e'

-- | Reduce a term to weak head normal form.
whnf :: forall a n m. (Enum n, MonadState n m) => Exp a -> m (Exp a)
whnf e = do
  modify succ
  go e
  where
    go e@Var {} = pure e
    go e@Lam {} = pure e
    go (f :@ a) = do
      r <- whnf f
      case r of
        Lam _ b -> whnf (instantiate1 a b)
        f' -> pure $ f' :@ a
    go e@(Const _) = pure e
    go e@Add {} = pure e
    go e@Sub {} = pure e
    go e@Mul {} = pure e
    go e@Neg {} = pure e
    go e@Abs {} = pure e
    go e@Sign {} = pure e

-- | Monad stack for type checking.
type TyM a = MaybeT (Fresh a)

-- | Guard against a type error.
assertTy :: Ord a => Map a Ty -> Exp a -> Ty -> TyM a ()
assertTy env e t = typeCheck env e >>= guard . (== t)

-- | Check the type of an expression.
typeCheck :: forall a. Ord a => Map a Ty -> Exp a -> TyM a Ty
typeCheck env (Var a) = MaybeT . return $ Map.lookup a env
typeCheck env (f :@ a) =
  typeCheck env f >>= \case
    TArr fTy tTy -> assertTy env a fTy >> return tTy
    _ -> mzero
typeCheck env (Lam ty' bind) = do
  uname <- fresh
  TArr ty' <$> typeCheck (Map.insert uname ty' env) (instantiate1 (Var uname) bind)
typeCheck _ (Const _) = return TInt
typeCheck env (Add l r) = assertTy env l TInt >> assertTy env r TInt >> return TInt
typeCheck env (Sub l r) = assertTy env l TInt >> assertTy env r TInt >> return TInt
typeCheck env (Mul l r) = assertTy env l TInt >> assertTy env r TInt >> return TInt
typeCheck env (Neg e) = assertTy env e TInt >> return TInt
typeCheck env (Abs e) = assertTy env e TInt >> return TInt
typeCheck env (Sign e) = assertTy env e TInt >> return TInt

type TyTH a = Fresh a TH.Exp

toTH :: forall a. Enum a => a -> Exp a -> TH.Exp
toTH a e = runFreshFrom a $ go e
  where
    -- prefix = "GHC.Num."
    prefix = mempty

    plus = TH.mkName $ prefix <> "+"
    minus = TH.mkName $ prefix <> "-"
    times = TH.mkName $ prefix <> "*"
    negate' = TH.mkName $ prefix <> "negate"
    abs' = TH.mkName $ prefix <> "abs"
    signum' = TH.mkName $ prefix <> "signum"

    toName a' = TH.mkName $ "x" <> (show . fromEnum $ a')

    go :: Exp a -> TyTH a
    go (Const i) = pure $ TH.LitE (TH.IntegerL i)
    go (Add l r) = TH.UInfixE <$> go l <*> pure (TH.VarE plus) <*> go r
    go (Sub l r) = TH.UInfixE <$> go l <*> pure (TH.VarE minus) <*> go r
    go (Mul l r) = TH.UInfixE <$> go l <*> pure (TH.VarE times) <*> go r
    go (Neg e') = TH.AppE (TH.VarE negate') <$> go e'
    go (Abs e') = TH.AppE (TH.VarE abs') <$> go e'
    go (Sign e') = TH.AppE (TH.VarE signum') <$> go e'
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
