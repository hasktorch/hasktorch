{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.DType where

import Data.Kind (Type)
import GHC.Generics
import GHC.TypeLits
import GHC.TypeLits.Extra
import System.IO.Unsafe
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Managed.Autograd as LibTorch
import qualified Torch.Tensor as D
import Torch.Typed.Parameter
import Torch.Typed.Tensor

class HasToDType dtype' dtype f g | dtype' dtype f -> g, dtype' dtype g -> f where
  -- >>> model <- A.sample (Torch.Typed.NN.LinearSpec @1 @1 @'D.Float @'( 'D.CPU, 0))
  -- >>> :type Torch.Typed.DType.toDType @'D.Double @'D.Float model
  -- Torch.Typed.DType.toDType @'D.Double @'D.Float model
  -- :: Torch.Typed.NN.Linear 1 1 'Double '( 'CPU, 0)
  toDType :: f -> g

-- In a data type `f` parameterized by zero or more data type variables, replace the given data type `dtype` with the data type `dtype'`.
--
-- >>> :kind! ReplaceDType (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) 'D.Double 'D.Float
-- ReplaceDType (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) 'D.Double 'D.Float :: *
-- = Torch.Typed.NN.Linear 1 1 'Double '( 'CPU, 0)
type family ReplaceDType (f :: k) (dtype' :: D.DType) (dtype :: D.DType) :: k where
  ReplaceDType (t dtype) dtype' dtype = t dtype'
  ReplaceDType (t a) dtype' dtype = (ReplaceDType t dtype' dtype) (ReplaceDType a dtype' dtype)
  ReplaceDType t _ _ = t

-- In a data type `f` parameterized by zero or one data type variables, replace the only occurring data type with the data type `dtype'`.
--
-- >>> :kind! ReplaceDType' (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) 'D.Double
-- ReplaceDType' (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) 'D.Double :: *
-- = Torch.Typed.NN.Linear 1 1 'Double '( 'CPU, 0)
type family ReplaceDType' (f :: k) (dtype' :: D.DType) :: k where
  ReplaceDType' (t (dtype :: D.DType)) dtype' = t dtype'
  ReplaceDType' (t a) dtype' = (ReplaceDType' t dtype') (ReplaceDType' a dtype')
  ReplaceDType' t _ = t

instance
  ( g ~ ReplaceDType f dtype' dtype,
    f ~ ReplaceDType g dtype dtype',
    Generic f,
    Generic g,
    GHasToDType dtype' dtype (Rep f) (Rep g)
  ) =>
  HasToDType dtype' dtype f g
  where
  toDType = to . gToDType @dtype' @dtype . from

class
  GHasToDType
    (dtype' :: D.DType)
    (dtype :: D.DType)
    (f :: Type -> Type)
    (g :: Type -> Type)
  where
  gToDType :: forall a. f a -> g a

instance
  ( GHasToDType dtype' dtype l l',
    GHasToDType dtype' dtype r r'
  ) =>
  GHasToDType dtype' dtype (l :*: r) (l' :*: r')
  where
  gToDType (l :*: r) =
    let l' = gToDType @dtype' @dtype l
        r' = gToDType @dtype' @dtype r
     in l' :*: r'

instance {-# OVERLAPS #-} (KnownDType dtype') => HasToDType dtype' dtype (Tensor device dtype shape) (Tensor device dtype' shape) where
  toDType = Torch.Typed.Tensor.toDType

instance {-# OVERLAPS #-} (KnownDType dtype') => HasToDType dtype' dtype (Parameter device dtype shape) (Parameter device dtype' shape) where
  toDType = Torch.Typed.Parameter.parameterToDType

instance {-# OVERLAPPABLE #-} (HasToDType dtype' dtype f g) => GHasToDType dtype' dtype (K1 i f) (K1 i g) where
  gToDType = K1 . Torch.Typed.DType.toDType @dtype' @dtype . unK1

instance (GHasToDType dtype' dtype f g) => GHasToDType dtype' dtype (M1 i t f) (M1 i t g) where
  gToDType = M1 . gToDType @dtype' @dtype . unM1

instance GHasToDType dtype' dtype U1 U1 where
  gToDType = id

-- >>> :kind! GetDType (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0))
-- GetDType (Torch.Typed.NN.Linear 1 1 'D.Float '( 'D.CPU, 0)) :: Maybe D.DType
-- = 'Just 'D.Float
--
-- >>> :kind! GetDType (Torch.Typed.Tensor.Tensor '( 'D.CUDA, 0) 'D.Float '[1])
-- GetDType (Torch.Typed.Tensor.Tensor '( 'D.CUDA, 0) 'D.Float '[1]) :: Maybe D.DType
-- = 'Just 'D.Float
type family GetDType (f :: k) :: Maybe D.DType where
  GetDType (t (dtype :: D.DType)) = Just dtype
  GetDType (t a) = GetDType t
  GetDType t = Nothing
