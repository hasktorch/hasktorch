{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}

module Torch.Typed.Autograd where

import           Data.HList
import           GHC.TypeLits
import           System.IO.Unsafe

import qualified LibTorch.ATen.Cast as ATen
import qualified LibTorch.ATen.Class as ATen
import qualified LibTorch.Torch.Managed.Autograd
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import qualified Torch.Tensor                  as D
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter

type family GradR (parameters :: [a]) (dtype :: D.DType) (device :: (D.DeviceType, Nat)) :: [a] where
  GradR '[] _ _ = '[]
  GradR (Parameter device dtype shape ': parameters) dtype device = Tensor device dtype shape ': GradR parameters dtype device

-- | calculate gradients of a zero-dimensional tensor with respect to a list of parameters
grad
  :: forall dtype device parameters gradients tensors
   . ( gradients ~ GradR parameters dtype device
     , tensors ~ gradients
     , HMap' ToDependent parameters tensors
     , ATen.Castable (HList gradients) [D.ATenTensor]
     )
  => Tensor device dtype '[]
  -> HList parameters
  -> HList gradients
grad loss inputs = unsafePerformIO $ ATen.cast2
  LibTorch.Torch.Managed.Autograd.grad
  loss
  (hmap' ToDependent inputs)
