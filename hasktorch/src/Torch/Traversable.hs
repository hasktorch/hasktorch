{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Traversable where

import Control.Monad.State.Strict
import Data.Foldable (toList)
import GHC.Generics
import Torch.Scalar

class GTraversable a f where
  gflatten :: f -> [a]
  default gflatten :: (Generic f, GGTraversable a (Rep f)) => f -> [a]
  gflatten f = gflatten' (from f)

  gupdate :: f -> State [a] f
  default gupdate :: (Generic f, GGTraversable a (Rep f)) => f -> State [a] f
  gupdate f = to <$> gupdate' (from f)

class GGTraversable a f where
  gflatten' :: forall p. f p -> [a]
  gupdate' :: forall p. f p -> State [a] (f p)

instance GGTraversable a U1 where
  gflatten' U1 = []
  gupdate' U1 = return U1

instance (GGTraversable a f, GGTraversable a g) => GGTraversable a (f :+: g) where
  gflatten' (L1 x) = gflatten' x
  gflatten' (R1 x) = gflatten' x
  gupdate' (L1 x) = do
    x' <- gupdate' x
    return $ L1 x'
  gupdate' (R1 x) = do
    x' <- gupdate' x
    return $ R1 x'

instance (GGTraversable a f, GGTraversable a g) => GGTraversable a (f :*: g) where
  gflatten' (x :*: y) = gflatten' x ++ gflatten' y
  gupdate' (x :*: y) = do
    x' <- gupdate' x
    y' <- gupdate' y
    return $ x' :*: y'

instance (GTraversable a c) => GGTraversable a (K1 i c) where
  gflatten' (K1 x) = gflatten x
  gupdate' (K1 x) = do
    x' <- gupdate x
    return $ K1 x'

instance (GGTraversable a f) => GGTraversable a (M1 i t f) where
  gflatten' (M1 x) = gflatten' x
  gupdate' (M1 x) = do
    x' <- gupdate' x
    return $ M1 x'

instance GTraversable a a where
  gflatten = pure
  gupdate _ = gpop

instance {-# OVERLAPS #-} (Generic f, GGTraversable a (Rep f)) => GTraversable a f

instance (GTraversable a a0, GTraversable a b0) => GTraversable a (a0, b0) where
  gflatten (a, b) = gflatten a ++ gflatten b
  gupdate (a, b) = do
    a' <- gupdate a
    b' <- gupdate b
    return (a', b')

instance (GTraversable a a0, GTraversable a b0, GTraversable a c0) => GTraversable a (a0, b0, c0) where
  gflatten (a, b, c) = gflatten a ++ gflatten b ++ gflatten c
  gupdate (a, b, c) = do
    a' <- gupdate a
    b' <- gupdate b
    c' <- gupdate c
    return (a', b', c')

instance (GTraversable a a0, GTraversable a b0, GTraversable a c0, GTraversable a d0) => GTraversable a (a0, b0, c0, d0) where
  gflatten (a, b, c, d) = gflatten a ++ gflatten b ++ gflatten c ++ gflatten d
  gupdate (a, b, c, d) = do
    a' <- gupdate a
    b' <- gupdate b
    c' <- gupdate c
    d' <- gupdate d
    return (a', b', c', d')

instance (GTraversable a f) => GTraversable a [f] where
  gflatten = (=<<) gflatten . toList
  gupdate = mapM gupdate

gpop :: State [a] a
gpop = do
  v <- get
  case v of
    [] -> error "Not enough values supplied to gupdate"
    (p : t) -> do put t; return p

gmap :: (GTraversable a f) => (a -> a) -> f -> f
gmap func x =
  let (f', remaining) = runState (gupdate x) (map func (gflatten x))
   in if null remaining
        then f'
        else error "Some values in a call to gmap haven't been consumed!"

greplace :: (GTraversable a f) => f -> [a] -> f
greplace f x =
  let (f', remaining) = runState (gupdate f) x
   in if null remaining
        then f'
        else error "Some values in a call to greplace haven't been consumed!"
