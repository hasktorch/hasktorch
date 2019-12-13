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
  , Torch.Typed.AutogradSpec.gradientsTest'
  )
where

import           Prelude                 hiding ( cos
                                                , sin
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
import           System.IO.Unsafe

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

instance
  ( RandDTypeIsValid device dtype
  , KnownNat features
  , KnownDType dtype
  , KnownDevice device
  ) => A.Randomizable (Rastrigin1Spec features dtype device)
                      (Rastrigin1     features dtype device)
 where
  sample _ =
    Rastrigin1 <$> (makeIndependent =<< randn)

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

gradientsRastrigin1 Rastrigin1 {..} a =
  let x' = toDependent x
  in  (cmul (2 :: Int) (x' + (cmul a . cmul (pi :: Double) . sin . cmul (2 * pi :: Double)) x')) :. HNil

gradients model a = grad (rastrigin1 model a) (flattenParameters model)

data Isclose = Isclose

instance Apply' Isclose (Tensor device dtype shape, Tensor device dtype shape) (Tensor device 'D.Bool shape)
  where apply' _ (a, b) = unsafePerformIO $ do
                            print a
                            print b
                            pure $ isclose 1e-05 1e-08 False a b

gradientsTest model a = hZipWith Isclose (gradients model a) (gradientsRastrigin1 model a)

gradientsTest' = do
  model <- A.sample (Rastrigin1Spec @10 @'D.Float @'( 'D.CPU, 0))
  pure $ gradientsTest model (10 :: Int)

spec :: Spec
spec = return ()
