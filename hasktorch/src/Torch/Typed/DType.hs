{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DeriveGeneric #-}

module Torch.Typed.DType where

import           Torch.HList
import           Data.Kind                    (Type)
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           GHC.Generics
import           System.IO.Unsafe

import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Managed.Autograd as LibTorch
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import qualified Torch.Tensor                  as D
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter

class HasToDType dtype' dtype f g | dtype' dtype f -> g, dtype' dtype g -> f where
  -- >>> model <- A.sample (Torch.Typed.NN.LinearSpec @1 @1 @'D.Float @'( 'D.CPU, 0))
  -- >>> :type Torch.Typed.DType.toDType @'D.Double @'D.Float model
  -- Torch.Typed.DType.toDType @'D.Double @'D.Float model
  -- :: Torch.Typed.NN.Linear 1 1 'Double '( 'CPU, 0)
  toDType :: f -> g

-- >>> :kind! PutDType (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) 'D.Double 'D.Float
-- PutDType (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) 'D.Double 'D.Float :: *
-- = Torch.Typed.NN.Linear 1 1 'Double '( 'CPU, 0)
type family PutDType (f :: k) (dtype' :: D.DType) (dtype :: D.DType) :: k where
  PutDType (t dtype) dtype' dtype = t dtype'
  PutDType (t a)     dtype' dtype = (PutDType t dtype' dtype) a
  PutDType t         _      _     = t

instance
  ( g ~ PutDType f dtype' dtype
  , f ~ PutDType g dtype dtype'
  , Generic f
  , Generic g
  , GHasToDType dtype' dtype (Rep f) (Rep g)
  ) => HasToDType dtype' dtype f g where
  toDType = to . gToDType @dtype' @dtype . from

class GHasToDType
  (dtype' :: D.DType)
  (dtype :: D.DType)
  (f :: Type -> Type)
  (g :: Type -> Type) where
  gToDType :: forall a . f a -> g a

instance
  ( GHasToDType dtype' dtype l l'
  , GHasToDType dtype' dtype r r'
  ) => GHasToDType dtype' dtype (l :*: r) (l' :*: r') where
  gToDType (l :*: r) =
    let l' = gToDType @dtype' @dtype l
        r' = gToDType @dtype' @dtype r
    in  l' :*: r'

instance {-# OVERLAPS #-} (KnownDType dtype') => HasToDType dtype' dtype (Tensor device dtype shape) (Tensor device dtype' shape) where
  toDType = Torch.Typed.Tensor.toDType

instance {-# OVERLAPS #-} (KnownDType dtype') => HasToDType dtype' dtype (Parameter device dtype shape) (Parameter device dtype' shape) where
  toDType = Torch.Typed.Parameter.toDType

instance {-# OVERLAPPABLE #-} (HasToDType dtype' dtype f g) => GHasToDType dtype' dtype (K1 i f) (K1 i g) where
  gToDType = K1 . Torch.Typed.DType.toDType @dtype' @dtype . unK1

instance (GHasToDType dtype' dtype f g) => GHasToDType dtype' dtype (M1 i t f) (M1 i t g) where
  gToDType = M1 . gToDType @dtype' @dtype . unM1

instance GHasToDType dtype' dtype U1 U1 where
  gToDType = id
