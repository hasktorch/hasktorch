{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE IncoherentInstances #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Monad where

import qualified Torch.Tensor as T
import Control.Monad
import Control.Applicative
import GHC.TypeLits
import Data.Proxy
import Data.Coerce
import Data.Finite
import Data.Kind
import qualified Data.Vector.Sized as V
import Data.Functor.Compose
import Data.Singletons.Prelude (Reverse)

newtype Batch a = Batch [a]
newtype Channel (ch::Nat) a = Channel [a]
newtype DX a = DX [a]
newtype DY a = DY [a]
newtype CPU a = CPU a
newtype CUDA a = CUDA a

data Tensor a where
  Prim   :: (T.TensorLike a) => T.Tensor -> Tensor a
  Return :: a -> Tensor a
  Bind   :: Tensor a -> (a -> Tensor b) -> Tensor b

instance {-# OVERLAPPING #-} (T.TensorLike a) => T.TensorLike (Batch a) where
  asTensor' v opt = T.asTensor' @[a] (coerce v) opt
  asTensor v = T.asTensor @[a] (coerce v)
  _asValue v = coerce $ T._asValue @[a] v
  _dtype = T._dtype @[a]
  _dims v = T._dims @[a] (coerce v)
  _deepDims v = T._deepDims @[a] (coerce v)
  _peekElemOff ptr offset v = coerce $ T._peekElemOff @[a] ptr offset v
  _pokeElemOff ptr offset v = T._pokeElemOff @[a] ptr offset (coerce v)

instance {-# OVERLAPPING #-} (T.TensorLike a) => T.TensorLike (Channel ch a) where
  asTensor' v opt = T.asTensor' @[a] (coerce v) opt
  asTensor v = T.asTensor @[a] (coerce v)
  _asValue v = coerce $ T._asValue @[a] v
  _dtype = T._dtype @[a]
  _dims v = T._dims @[a] (coerce v)
  _deepDims v = T._deepDims @[a] (coerce v)
  _peekElemOff ptr offset v = coerce $ T._peekElemOff @[a] ptr offset v
  _pokeElemOff ptr offset v = T._pokeElemOff @[a] ptr offset (coerce v)

instance {-# OVERLAPPING #-} (T.TensorLike a) => T.TensorLike (DX a) where
  asTensor' v opt = T.asTensor' @[a] (coerce v) opt
  asTensor v = T.asTensor @[a] (coerce v)
  _asValue v = coerce $ T._asValue @[a] v
  _dtype = T._dtype @[a]
  _dims v = T._dims @[a] (coerce v)
  _deepDims v = T._deepDims @[a] (coerce v)
  _peekElemOff ptr offset v = coerce $ T._peekElemOff @[a] ptr offset v
  _pokeElemOff ptr offset v = T._pokeElemOff @[a] ptr offset (coerce v)

instance {-# OVERLAPPING #-} (T.TensorLike a) => T.TensorLike (DY a) where
  asTensor' v opt = T.asTensor' @[a] (coerce v) opt
  asTensor v = T.asTensor @[a] (coerce v)
  _asValue v = coerce $ T._asValue @[a] v
  _dtype = T._dtype @[a]
  _dims v = T._dims @[a] (coerce v)
  _deepDims v = T._deepDims @[a] (coerce v)
  _peekElemOff ptr offset v = coerce $ T._peekElemOff @[a] ptr offset v
  _pokeElemOff ptr offset v = T._pokeElemOff @[a] ptr offset (coerce v)

instance (T.TensorLike a) => T.TensorLike (Tensor a) where
  asTensor' = error "Not implemented for Tensor-a-type"
  asTensor = toTensor
  _asValue = Prim
  _dtype = error "Not implemented for Tensor-a-type"
  _dims v = error "Not implemented for Tensor-a-type"
  _deepDims v = error "Not implemented for Tensor-a-type"
  _peekElemOff = error "Not implemented for Tensor-a-type"
  _pokeElemOff = error "Not implemented for Tensor-a-type"

{-
instance (T.TensorLike a) => T.TensorLike (CPU a) where
  asTensor' = error "Not implemented for Tensor-a-type"
  asTensor = T.toCPU . toTensor
  _asValue = Prim . T.toCPU
  _dtype = error "Not implemented for Tensor-a-type"
  _dims v = error "Not implemented for Tensor-a-type"
  _deepDims v = error "Not implemented for Tensor-a-type"
  _peekElemOff = error "Not implemented for Tensor-a-type"
  _pokeElemOff = error "Not implemented for Tensor-a-type"

instance (T.TensorLike a) => T.TensorLike (CUDA a) where
  asTensor' = error "Not implemented for Tensor-a-type"
  asTensor = T.toCUDA . toTensor
  _asValue = Prim
  _dtype = error "Not implemented for Tensor-a-type"
  _dims v = error "Not implemented for Tensor-a-type"
  _deepDims v = error "Not implemented for Tensor-a-type"
  _peekElemOff = error "Not implemented for Tensor-a-type"
  _pokeElemOff = error "Not implemented for Tensor-a-type"
-}

toTensor :: (T.TensorLike a) => Tensor a -> T.Tensor
toTensor (Prim s)                        = s
toTensor (Return a)                      = T.asTensor $ a
toTensor (Bind (Prim s) f)               = toTensor (f (T.asValue s))
toTensor (Bind (Return a) f)             = toTensor (f a)
toTensor (Bind (Bind ma f) g)            = toTensor (Bind ma (\a -> Bind (f a) g))

-- (!!) :: Int -> Tensor [a] -> Tensor a
-- (!!) n tensor = return $ (toTensor tensor) T.! n

instance Functor Tensor where
  fmap = liftM

instance Applicative Tensor where
  pure  = return
  (<*>) = ap

instance Monad Tensor where
  return = Return
  (>>=)  = Bind

asValue :: (T.TensorLike a) => Tensor a -> a
asValue = T.asValue . toTensor

instance Functor Batch where
  fmap = liftM

instance Applicative Batch where
  pure  = return
  (<*>) = ap

instance Monad Batch where
  return v = Batch [v]
  (>>=) (Batch xs) f =
    Batch $ do
      x <- xs
      let Batch y = f x
      y

instance Functor (Channel ch) where
  fmap = liftM

instance Applicative (Channel ch) where
  pure  = return
  (<*>) = ap

instance Monad (Channel ch) where
  return v = Channel [v]
  (>>=) (Channel xs) f =
    Channel $ do
      x <- xs
      let Channel y = f x
      y

instance Functor DX where
  fmap = liftM

instance Applicative DX where
  pure  = return
  (<*>) = ap

instance Monad DX where
  return v = DX [v]
  (>>=) (DX xs) f =
    DX $ do
      x <- xs
      let DX y = f x
      y

instance Functor DY where
  fmap = liftM

instance Applicative DY where
  pure  = return
  (<*>) = ap

instance Monad DY where
  return v = DY [v]
  (>>=) (DY xs) f =
    DY $ do
      x <- xs
      let DY y = f x
      y

-- concat :: [Tensor a] -> Tensor [a]
--



foo :: Tensor [Float]
foo = return [1,2,3]

foo' :: T.Tensor
foo' = toTensor foo

bfoo :: Tensor (Batch [Float])
bfoo = return $ return [1,2,3]

bfoo' :: T.Tensor
bfoo' = toTensor bfoo

bcfoo :: Tensor (Batch (Channel 1 [Float]))
bcfoo = return $ return $ return [1,2,3]

bcfoo' :: T.Tensor
bcfoo' = toTensor bcfoo

bcxyfoo :: Tensor (Batch (Channel 1 (DY (DX [Float]))))
bcxyfoo = return $ return $ return $ return $ return [1,2,3]

bcxyfoo' :: T.Tensor
bcxyfoo' = toTensor bcxyfoo
