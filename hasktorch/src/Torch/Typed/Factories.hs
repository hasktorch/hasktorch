{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.Factories where

import Prelude hiding (sin)
import Control.Arrow ((&&&))
import Data.Proxy
import Data.Finite
import Data.Kind (Constraint)
import Data.Reflection
import GHC.TypeLits
import System.IO.Unsafe

import qualified ATen.Managed.Native           as ATen
import           ATen.Cast
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.Functions               as D
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import qualified Torch.TensorOptions           as D
import           Torch.Typed.Aux
import           Torch.Typed.Tensor

zeros
  :: forall shape dtype device
   . (TensorOptions shape dtype device)
  => Tensor shape dtype device
zeros = UnsafeMkTensor $ D.zeros
  (optionsRuntimeShape @shape @dtype @device)
  (D.withDType (optionsRuntimeDType @shape @dtype @device) D.defaultOpts)

ones
  :: forall shape dtype device
   . (TensorOptions shape dtype device)
  => Tensor shape dtype device
ones = UnsafeMkTensor $ D.ones
  (optionsRuntimeShape @shape @dtype @device)
  (D.withDType (optionsRuntimeDType @shape @dtype @device) D.defaultOpts)

rand
  :: forall shape dtype device
   . (TensorOptions shape dtype device)
  => IO (Tensor shape dtype device)
rand = UnsafeMkTensor <$> D.rand
  (optionsRuntimeShape @shape @dtype @device)
  (D.withDType (optionsRuntimeDType @shape @dtype @device) D.defaultOpts)

randn
  :: forall shape dtype device
   . (TensorOptions shape dtype device)
  => IO (Tensor shape dtype device)
randn = UnsafeMkTensor <$> D.randn
  (optionsRuntimeShape @shape @dtype @device)
  (D.withDType (optionsRuntimeDType @shape @dtype @device) D.defaultOpts)

-- | linspace
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Float]) $ linspace @7 0 3
-- (Float,([7],[0.0,0.5,1.0,1.5,2.0,2.5,3.0]))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Float]) $ linspace @3 0 2
-- (Float,([3],[0.0,1.0,2.0]))
linspace
  :: forall steps
   . (KnownNat steps)
  => Float
  -> Float
  -> Tensor '[steps] 'D.Float device
linspace start end =
  unsafePerformIO $ cast3 ATen.linspace_ssl start end (natValI @steps)

eyeSquare
  :: forall n dtype device
   . (KnownNat n, TensorOptions '[n, n] dtype device)
  => Tensor '[n, n] dtype device
eyeSquare = UnsafeMkTensor $ D.eyeSquare
  (natValI @n)
  (D.withDType (optionsRuntimeDType @'[n, n] @dtype @device) D.defaultOpts)
