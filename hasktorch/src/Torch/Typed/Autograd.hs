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

module Torch.Typed.Autograd
  ( Torch.Typed.Autograd.HasGrad
  , Torch.Typed.Autograd.grad
  )
where

import           Data.HList
import           GHC.TypeLits
import           System.IO.Unsafe

import qualified ATen.Cast                     as ATen
import qualified ATen.Class                    as ATen
import qualified Torch.Managed.Autograd
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import qualified Torch.Tensor                  as D
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter

class HasGrad a b | a -> b where
  -- | calculate gradients of a zero-dimensional tensor with respect to a list of parameters
  grad :: forall dtype device . Tensor device dtype '[] -> a -> b

  toDependent :: a -> b

-- instance HasGrad (Tensor dtype device shape) (Tensor dtype device shape) where
--   grad loss input = head . unsafePerformIO $ ATen.cast2
--     Torch.Managed.Autograd.grad
--     loss
--     [Torch.Typed.Autograd.toDependent input]

--   toDependent = id

instance HasGrad (Parameter dtype device shape) (Tensor dtype device shape) where
  grad loss input = head . unsafePerformIO $ ATen.cast2
    Torch.Managed.Autograd.grad
    loss
    [Torch.Typed.Autograd.toDependent input]
  toDependent = Torch.Typed.Parameter.toDependent

instance HasGrad (HList '[]) (HList '[])  where
  grad _ = id

  toDependent = id

instance (HasGrad a b, HasGrad (HList as) (HList bs), ATen.Castable (HList (b ': bs)) [D.ATenTensor]) => HasGrad (HList (a ': as)) (HList (b ': bs)) where
  grad loss inputs = unsafePerformIO $ ATen.cast2
    Torch.Managed.Autograd.grad
    loss
    (Torch.Typed.Autograd.toDependent inputs)

  toDependent (a :. as) =
    Torch.Typed.Autograd.toDependent a :. Torch.Typed.Autograd.toDependent as
