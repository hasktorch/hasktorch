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

import           Prelude                 hiding ( sin )
import           Control.Arrow                  ( (&&&) )
import           Data.Proxy
import           Data.Finite
import           Data.Kind                      ( Constraint )
import           Data.Reflection
import           GHC.TypeLits
import           System.IO.Unsafe

import           Torch.Internal.Cast
import qualified Torch.Scalar                  as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.Functional              as D
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import qualified Torch.TensorOptions           as D
import           Torch.Typed.Aux
import           Torch.Typed.Tensor

zeros
  :: forall shape dtype device
   . (TensorOptions shape dtype device)
  => Tensor device dtype shape
zeros = UnsafeMkTensor $ D.zeros
  (optionsRuntimeShape @shape @dtype @device)
  ( D.withDevice (optionsRuntimeDevice @shape @dtype @device)
  . D.withDType (optionsRuntimeDType @shape @dtype @device)
  $ D.defaultOpts
  )

ones
  :: forall shape dtype device
   . (TensorOptions shape dtype device)
  => Tensor device dtype shape
ones = UnsafeMkTensor $ D.ones
  (optionsRuntimeShape @shape @dtype @device)
  ( D.withDevice (optionsRuntimeDevice @shape @dtype @device)
  . D.withDType (optionsRuntimeDType @shape @dtype @device)
  $ D.defaultOpts
  )

type family RandDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  RandDTypeIsValid '( 'D.CPU, 0)    dtype = ( DTypeIsNotBool '( 'D.CPU, 0) dtype
                                            , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                            )
  RandDTypeIsValid '( 'D.CUDA, _)   dtype = ()
  RandDTypeIsValid '(deviceType, _) dtype = UnsupportedDTypeForDevice deviceType dtype

rand
  :: forall shape dtype device
   . ( TensorOptions shape dtype device
     , RandDTypeIsValid device dtype
     )
  => IO (Tensor device dtype shape)
rand = UnsafeMkTensor <$> D.rand
  (optionsRuntimeShape @shape @dtype @device)
  ( D.withDevice (optionsRuntimeDevice @shape @dtype @device)
  . D.withDType (optionsRuntimeDType @shape @dtype @device)
  $ D.defaultOpts
  )

randn
  :: forall shape dtype device
   . ( TensorOptions shape dtype device
     , RandDTypeIsValid device dtype
     )
  => IO (Tensor device dtype shape)
randn = UnsafeMkTensor <$> D.randn
  (optionsRuntimeShape @shape @dtype @device)
  ( D.withDevice (optionsRuntimeDevice @shape @dtype @device)
  . D.withDType (optionsRuntimeDType @shape @dtype @device)
  $ D.defaultOpts
  )

randint
  :: forall shape dtype device
   . ( TensorOptions shape dtype device
     , RandDTypeIsValid device dtype
     )
  => Int
  -> Int
  -> IO (Tensor device dtype shape)
randint low high = UnsafeMkTensor <$> (D.randint low high)
  (optionsRuntimeShape @shape @dtype @device)
  ( D.withDevice (optionsRuntimeDevice @shape @dtype @device)
  . D.withDType (optionsRuntimeDType @shape @dtype @device)
  $ D.defaultOpts
  )

-- | linspace
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Float]) $ linspace @7 @'( 'D.CPU, 0) 0 3
-- (Float,([7],[0.0,0.5,1.0,1.5,2.0,2.5,3.0]))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Float]) $ linspace @3 @'( 'D.CPU, 0) 0 2
-- (Float,([3],[0.0,1.0,2.0]))
linspace
  :: forall steps device start end
   . ( D.Scalar start
     , D.Scalar end
     , KnownNat steps
     , TensorOptions '[steps] 'D.Float device
     )
  => start -- ^ start
  -> end -- ^ end
  -> Tensor device 'D.Float '[steps] -- ^ output
linspace start end = UnsafeMkTensor $ D.linspace
  start
  end
  (natValI @steps)
  ( D.withDevice (optionsRuntimeDevice @'[steps] @D.Float @device)
  . D.withDType (optionsRuntimeDType @'[steps] @D.Float @device)
  $ D.defaultOpts
  )

eyeSquare
  :: forall n dtype device
   . ( KnownNat n
     , TensorOptions '[n, n] dtype device
     )
  => Tensor device dtype '[n, n] -- ^ output
eyeSquare = UnsafeMkTensor $ D.eyeSquare
  (natValI @n)
  ( D.withDevice (optionsRuntimeDevice @'[n, n] @dtype @device)
  . D.withDType (optionsRuntimeDType @'[n, n] @dtype @device)
  $ D.defaultOpts
  )
