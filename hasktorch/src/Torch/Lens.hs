{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Lens where

import Control.Monad.Identity
import Control.Monad.State.Strict
import GHC.Generics

-- | Type synonym for lens
type Lens s t a b = forall f. Functor f => (a -> f b) -> s -> f t

type Lens' s a = Lens s s a a

type Traversal s t a b = forall f. Applicative f => (a -> f b) -> s -> f t

type Traversal' s a = Traversal s s a a

class HasTypes s a where
  types_ :: Traversal' s a
  default types_ :: (Generic s, GHasTypes (Rep s) a) => Traversal' s a
  types_ func s = to <$> gtypes func (from s)
  {-# INLINE types_ #-}

instance {-# OVERLAPS #-} (Generic s, GHasTypes (Rep s) a) => HasTypes s a

over :: Traversal' s a -> (a -> a) -> s -> s
over l f = runIdentity . l (Identity . f)

flattenValues :: forall a s. Traversal' s a -> s -> [a]
flattenValues func orgData = reverse . snd $ runState (func push orgData) []
  where
    push :: a -> State [a] a
    push v = do
      d <- get
      put $ v : d
      return v

replaceValues :: forall a s. Traversal' s a -> s -> [a] -> s
replaceValues func orgData newValues = fst $ runState (func pop orgData) newValues
  where
    pop :: a -> State [a] a
    pop _ = do
      d <- get
      case d of
        [] -> error "Not enough values supplied to replaceValues"
        x : xs -> do
          put xs
          return x

types :: forall a s. HasTypes s a => Traversal' s a
types = types_ @s @a

class GHasTypes s a where
  gtypes :: forall b. Traversal' (s b) a

instance GHasTypes U1 a where
  gtypes _ = pure
  {-# INLINE gtypes #-}

instance (GHasTypes f a, GHasTypes g a) => GHasTypes (f :+: g) a where
  gtypes func (L1 x) = L1 <$> gtypes func x
  gtypes func (R1 x) = R1 <$> gtypes func x

instance (GHasTypes f a, GHasTypes g a) => GHasTypes (f :*: g) a where
  gtypes func (x :*: y) = (:*:) <$> gtypes func x <*> gtypes func y
  {-# INLINE gtypes #-}

instance (HasTypes s a) => GHasTypes (K1 i s) a where
  gtypes func (K1 x) = K1 <$> types func x
  {-# INLINE gtypes #-}

instance GHasTypes s a => GHasTypes (M1 i t s) a where
  gtypes func (M1 x) = M1 <$> gtypes func x
  {-# INLINE gtypes #-}

instance {-# OVERLAPS #-} (HasTypes s a) => HasTypes [s] a where
  types_ func [] = pure []
  types_ func (x : xs) = (:) <$> types_ func x <*> types_ func xs
  {-# INLINE types_ #-}

instance {-# OVERLAPS #-} (HasTypes s0 a, HasTypes s1 a) => HasTypes (s0, s1) a where
  types_ func (s0, s1) = (,) <$> types_ func s0 <*> types_ func s1
  {-# INLINE types_ #-}

instance {-# OVERLAPS #-} (HasTypes s0 a, HasTypes s1 a, HasTypes s2 a) => HasTypes (s0, s1, s2) a where
  types_ func (s0, s1, s2) = (,,) <$> types_ func s0 <*> types_ func s1 <*> types_ func s2
  {-# INLINE types_ #-}
