{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.Autograd where

import           Data.HList
import           GHC.TypeLits
import           System.IO.Unsafe

import qualified ATen.Cast as ATen
import qualified ATen.Class as ATen
import qualified Torch.Managed.Autograd
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import qualified Torch.Tensor                  as D
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter

type family GradR (parameters :: [a]) :: [a] where
  GradR '[] = '[]
  GradR (Tensor device dtype shape ': parameters) = Tensor device dtype shape ': GradR parameters
  GradR (Parameter device dtype shape ': parameters) = Tensor device dtype shape ': GradR parameters

class HasGrad a b | a -> b where
  grad' :: forall dtype device . Tensor device dtype '[] -> a -> b

instance HasGrad (Tensor dtype device shape) (Tensor dtype device shape) where
  grad' = _undefined

instance HasGrad (Parameter dtype device shape) (Tensor dtype device shape) where
  grad' = _undefined

instance HasGrad (HList '[]) (HList '[])  where
  grad' = _undefined

instance (HasGrad a b, HasGrad (HList as) (HList bs)) => HasGrad (HList (a ': as)) (HList (b ': bs)) where
  grad' = _undefined

-- | calculate gradients of a zero-dimensional tensor with respect to a list of parameters
grad
  :: forall dtype device parameters gradients tensors
   . ( gradients ~ GradR parameters
     , tensors ~ gradients
     , HMap' ToDependent parameters tensors
     , ATen.Castable (HList gradients) [D.ATenTensor]
     )
  => Tensor device dtype '[]
  -> HList parameters
  -> HList gradients
grad loss inputs = unsafePerformIO $ ATen.cast2
  Torch.Managed.Autograd.grad
  loss
  (hmap' ToDependent inputs)
