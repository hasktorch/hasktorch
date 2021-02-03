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

-- Refence : https://gist.github.com/lotz84/78474ac9ee307d50376e025093316d0f

type family Tensor_ (xs :: [Nat]) = r | r -> xs where
  Tensor_ '[] = Identity
  Tensor_ (n ': ns) = Compose (V.Vector n) (Tensor_ ns)

type Tensor ns = Compose M.Tensor (Tensor_ ns)

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

class Shapely (ns :: [Nat]) where
  replicateT :: (T.TensorLike a) => a -> Tensor ns a

-- toTensor :: M.Tensor a -> Tensor '[] a
-- toTensor v = 

-- instance Shapely '[] where
--  replicateT a = toTensor (return a)

--instance (KnownNat n, Shapely ns, Representable (Tensor ns)) => Shapely (n ': ns) where
--  replicateT a = Compose (replicateT (tabulate (const a)))
