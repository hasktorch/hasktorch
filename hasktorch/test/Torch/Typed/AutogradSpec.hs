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
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedLists #-}

module Torch.Typed.AutogradSpec
  ( Torch.Typed.AutogradSpec.spec
  )
where

import           Prelude                 hiding ( exp
                                                , cos
                                                , sqrt
                                                )
import           Control.Monad                  ( foldM )
import           Control.Exception.Safe
import           Foreign.Storable
import           Data.HList
import           Data.Kind
import           Data.Proxy
import           Data.Maybe
import           Data.Reflection
import           GHC.Generics
import           GHC.TypeLits
import           GHC.Exts

import           Test.Hspec
import           Test.QuickCheck

import qualified ATen.Cast                     as ATen
import qualified ATen.Class                    as ATen
import qualified ATen.Type                     as ATen
import qualified ATen.Managed.Type.Tensor      as ATen
import qualified ATen.Managed.Type.Context     as ATen
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Functions               as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import qualified Torch.Scalar                  as D
import qualified Torch.NN                      as A
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter
import           Torch.Typed.Native
import           Torch.Typed.Factories
import           Torch.Typed.NN
import           Torch.Typed.Autograd
import           Torch.Typed.Optim
import           Torch.Typed.Serialize
import           Torch.Typed.AuxSpec

data Rastrigin1Spec (features :: Nat)
                    (dtype :: D.DType)
                    (device :: (D.DeviceType, Nat))
  = Rastrigin1Spec deriving (Show, Eq)

data Rastrigin1 (features :: Nat)
                (dtype :: D.DType)
                (device :: (D.DeviceType, Nat))
 where
  Rastrigin1
    :: forall features dtype device
     . { x :: Parameter device dtype '[features] }
    -> Rastrigin1 features dtype device
 deriving (Show, Generic)

rastrigin1
  :: forall features a dtype device
   . ( KnownNat features
     , D.Scalar a
     , KnownDType dtype
     , dtype ~ SumDType dtype
     , SumDTypeIsValid device dtype
     , StandardFloatingPointDTypeValidation device dtype
     , KnownDevice device
     )
  => Rastrigin1 features dtype device
  -> a
  -> Tensor device dtype '[]
rastrigin1 Rastrigin1 {..} a =
  let x' = toDependent x
      n = natValI @features
  in  (cmul a . cmul n $ ones) + sumAll (x' * x' - (cmul a . cos . cmul (2 * pi :: Double)) x')


spec :: Spec
spec = return ()
