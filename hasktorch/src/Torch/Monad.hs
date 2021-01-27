{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Monad where

import qualified Torch.Tensor as T
import Control.Monad
import Control.Applicative
import GHC.TypeLits
import Data.Proxy

data Batch a where
  ToBatch :: Tensor [a] -> Batch a
  Concat :: [a] -> Batch a

data Tensor a where
  Prim   :: (T.TensorLike a) => T.Tensor -> Tensor a
  Return :: a -> Tensor a
  Bind   :: Tensor a -> (a -> Tensor b) -> Tensor b

instance (T.TensorLike a) => T.TensorLike (Tensor a) where
  asTensor' = error "Not implemented for Tensor-a-type"
  asTensor = toTensor
  _asValue = Prim
  _dtype = error "Not implemented for Tensor-a-type"
  _dims v = error "Not implemented for Tensor-a-type"
  _deepDims v = error "Not implemented for Tensor-a-type"
  _peekElemOff = error "Not implemented for Tensor-a-type"
  _pokeElemOff = error "Not implemented for Tensor-a-type"

toTensor :: (T.TensorLike a) => Tensor a -> T.Tensor
toTensor (Prim s)                        = s
toTensor (Return a)                      = T.asTensor $ a
toTensor (Bind (Prim s) f)               = toTensor (f (T.asValue s))
toTensor (Bind (Return a) f)             = toTensor (f a)
toTensor (Bind (Bind ma f) g)            = toTensor (Bind ma (\a -> Bind (f a) g))

instance Functor Tensor where
  fmap = liftM

instance Applicative Tensor where
  pure  = return
  (<*>) = ap

instance Monad Tensor where
  return = Return
  (>>=)  = Bind

foo :: Tensor [Float]
foo = return [1,2,3]

foo' :: T.Tensor
foo' = toTensor foo
