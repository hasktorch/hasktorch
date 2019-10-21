{-# LANGUAGE KindSignatures #-}
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

import qualified ATen.Managed.Native as ATen
import ATen.Cast
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import qualified Torch.Functions as D
import qualified Torch.DType as D
import qualified Torch.TensorOptions as D
import Torch.Typed

zeros :: forall dtype shape. (TensorOptions dtype shape) => Tensor dtype shape
zeros = UnsafeMkTensor $ D.zeros (optionsRuntimeShape @dtype @shape) (D.withDType (optionsRuntimeDType @dtype @shape) D.defaultOpts)

ones :: forall dtype shape. (TensorOptions dtype shape) => Tensor dtype shape
ones = UnsafeMkTensor $ D.ones (optionsRuntimeShape @dtype @shape) (D.withDType (optionsRuntimeDType @dtype @shape) D.defaultOpts)

rand :: forall dtype shape. (TensorOptions dtype shape) => IO (Tensor dtype shape)
rand = UnsafeMkTensor <$> D.rand (optionsRuntimeShape @dtype @shape) (D.withDType (optionsRuntimeDType @dtype @shape) D.defaultOpts)

randn :: forall dtype shape. (TensorOptions dtype shape) => IO (Tensor dtype shape)
randn = UnsafeMkTensor <$> D.randn (optionsRuntimeShape @dtype @shape) (D.withDType (optionsRuntimeDType @dtype @shape) D.defaultOpts)

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
  -> Tensor 'D.Float '[steps]
linspace start end = unsafePerformIO $ cast3 ATen.linspace_ssl start end (natValI @steps)

eyeSquare
  :: forall dtype n
   . (KnownNat n, TensorOptions dtype '[n, n])
  => Tensor dtype '[n, n]
eyeSquare = UnsafeMkTensor $ D.eyeSquare
  (natValI @n)
  (D.withDType (optionsRuntimeDType @dtype @'[n, n]) D.defaultOpts)
