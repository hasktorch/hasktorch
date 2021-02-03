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
{-# LANGUAGE AllowAmbiguousTypes #-}

module Torch.Typed.Monad where

import qualified Torch.Tensor as T
import Control.Monad
import Control.Applicative
import GHC.TypeLits
import Data.Proxy
import Data.Coerce
import Data.Finite
import Data.Kind
import qualified Data.Vector.Sized as V
import Data.Functor.Identity
import Data.Functor.Compose
import Data.Singletons.Prelude (Reverse)
import qualified Torch.Monad as M
import Unsafe.Coerce (unsafeCoerce)

-- Refence :
--   https://github.com/jasigal/hasktorch-naperian/blob/master/src/Data/Naperian.hs
--   https://gist.github.com/lotz84/78474ac9ee307d50376e025093316d0f

type family Tensor_ (xs :: [Nat]) = r | r -> xs where
  Tensor_ '[] = Identity
  Tensor_ (n ': ns) = Compose (V.Vector n) (Tensor_ ns)

newtype Tensor ns a = MkTensor (M.Tensor (Tensor_ ns a))

class Functor f => Representable f where
  type Log f
  index    :: (T.TensorLike a) => f a -> (Log f -> a)
  tabulate :: (Log f -> a) -> f a

  positions :: f (Log f)
  tabulate h = fmap h positions
  positions  = tabulate id

instance Representable M.Tensor where
  type Log (M.Tensor) = ()
  index a _ = T.asValue (M.toTensor a)
  tabulate func = M.Return $ func ()

instance KnownNat n => Representable (V.Vector n) where
  type Log (V.Vector n) = Finite n
  index = V.index
  positions = V.generate id

instance Functor (Tensor '[]) where
  fmap func v =  toTensor $ fmap func $ fromTensor v

instance (Functor (Tensor ns)) => Functor (Tensor (n ': ns)) where
  fmap func v =  fromVector $ fmap (fmap func) $ toVector v

instance Monad (Tensor '[]) where
  return = toTensor. M.return . fromTensor
  (>>=)

instance Representable (Tensor '[]) where
  type Log (Tensor '[]) = ()
  index a _ = T.asValue (M.toTensor $ fromTensor a)
  tabulate = toTensor . tabulate

instance (KnownNat n, Representable (Tensor ns)) => Representable (Tensor (n ': ns)) where
  type Log (Tensor (n ': ns)) = (Finite n , (Log (Tensor ns)))
  index a (i, j) = index (index (toVector a) j) i
  tabulate = undefined

class Shapely (ns :: [Nat]) where
  replicateT :: a -> Tensor ns a

instance Shapely '[] where
  replicateT a = toTensor (return a)

instance (KnownNat n, Shapely ns, Representable (Tensor ns)) => Shapely (n ': ns) where
  replicateT a =
    let M.Return v = unsafeCoerce $ replicateT @ns a
    in unsafeCoerce $ M.Return (V.replicate @n v)

toTensor :: M.Tensor a -> Tensor '[] a
toTensor = unsafeCoerce

fromTensor :: Tensor '[] a -> M.Tensor a
fromTensor = unsafeCoerce

toVector :: Tensor (n ': ns) a -> Tensor ns (V.Vector n a)
toVector = unsafeCoerce

fromVector :: Tensor ns (V.Vector n a) -> Tensor (n ': ns) a 
fromVector = unsafeCoerce

