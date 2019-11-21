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

import           ATen.Cast
import qualified Torch.Managed.Autograd
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import qualified Torch.Tensor                  as D
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter

type family GradR (parameters :: [a]) (dtype :: D.DType) (device :: (D.DeviceType, Nat)) :: [a] where
  GradR '[] _ _ = '[]
  GradR (Parameter device dtype shape ': parameters) dtype device = Tensor device dtype shape ': GradR parameters dtype device

grad
  :: forall dtype device parameters gradients
   . ( gradients ~ GradR parameters dtype device
     , HMap ToDependent parameters gradients
     , HFoldrM IO TensorListFold [D.ATenTensor] gradients
     , Apply
         TensorListUnfold
         [D.ATenTensor]
         (HUnfoldMRes IO [D.ATenTensor] gradients)
     , HUnfoldM
         IO
         TensorListUnfold
         (HUnfoldMRes IO [D.ATenTensor] gradients)
         gradients
     )
  => Tensor device dtype '[]
  -> HList parameters
  -> HList gradients
grad y inputs = unsafePerformIO $ cast2
  Torch.Managed.Autograd.grad
  y
  (hmap ToDependent inputs :: HList gradients)
