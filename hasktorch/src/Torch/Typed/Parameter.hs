{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DefaultSignatures #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.Parameter
  ( module Torch.Typed.Parameter
  , Torch.NN.Randomizable(..)
  ) where

import           Control.Monad.State.Strict
import           Torch.HList
import           Data.Kind                    (Type)
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           GHC.Generics

import qualified Torch.NN (Randomizable(..), Parameter, sample)
import qualified Torch.Tensor (toDevice, toType)
import qualified Torch.Autograd (IndependentTensor(..), makeIndependent)
import           Torch.Device (DeviceType)
import           Torch.DType (DType)
import           Torch.Typed.Aux
import           Torch.Typed.Factories
import           Torch.Typed.Functional
import           Torch.Typed.Tensor

newtype Parameter (device :: (DeviceType, Nat)) (dtype :: DType) (shape :: [Nat]) = UnsafeMkParameter Torch.Autograd.IndependentTensor
  deriving Show

untypeParam :: Parameter device dtype shape -> Torch.NN.Parameter
untypeParam (UnsafeMkParameter param) = param

toDependent
  :: forall shape dtype device
   . Parameter device dtype shape
  -> Tensor device dtype shape
toDependent (UnsafeMkParameter t) = UnsafeMkTensor $ Torch.Autograd.toDependent t

data ToDependent = ToDependent

instance Apply' ToDependent (Parameter device dtype shape) (Tensor device dtype shape) where
  apply' _ = toDependent

makeIndependent
  :: forall shape dtype device
   . Tensor device dtype shape
  -> IO (Parameter device dtype shape)
makeIndependent t = UnsafeMkParameter <$> Torch.Autograd.makeIndependent (toDynamic t)

data MakeIndependent = MakeIndependent

instance Apply' MakeIndependent (Tensor device dtype shape) (IO (Parameter device dtype shape)) where
  apply' _ = makeIndependent

parameterToDevice
  :: forall device' device dtype shape
   . KnownDevice device'
  => Parameter device  dtype shape
  -> Parameter device' dtype shape
parameterToDevice (UnsafeMkParameter t) = UnsafeMkParameter . Torch.Autograd.IndependentTensor . Torch.Tensor.toDevice (deviceVal @device') . Torch.Autograd.toDependent $ t

parameterToDType
  :: forall dtype' dtype device shape
   . KnownDType dtype'
  => Parameter device dtype  shape
  -> Parameter device dtype' shape
parameterToDType (UnsafeMkParameter t) = UnsafeMkParameter . Torch.Autograd.IndependentTensor . Torch.Tensor.toType (dtypeVal @dtype') . Torch.Autograd.toDependent $ t

class Parameterized
  (f :: Type)
  (as :: [Type]) | f -> as where
  flattenParameters :: f -> HList as
  replaceParameters :: f -> HList as -> f

instance
  ( Generic f
  , GParameterized (Rep f) as
  ) => Parameterized f as where
  flattenParameters f = gFlattenParameters (from f)
  replaceParameters f as = to (gReplaceParameters (from f) as)

class GParameterized
  (f :: Type -> Type)
  (as :: [Type]) | f -> as where
  gFlattenParameters :: forall a . f a -> HList as
  gReplaceParameters :: forall a . f a -> HList as -> f a

instance
  ( GParameterized l as
  , GParameterized r bs
  , HAppendFD as bs cs
  , cs ~ (as ++ bs)
  ) => GParameterized (l :*: r) cs where
  gFlattenParameters (l :*: r) =
    let as = gFlattenParameters l
        bs = gFlattenParameters r
    in  as `happendFD` bs
  gReplaceParameters (l :*: r) cs =
    let (as, bs) = hunappendFD cs
        l'       = gReplaceParameters l as
        r'       = gReplaceParameters r bs
    in  l' :*: r'

instance A.Parameterized (Tensor device dtype shape) where
  flattenParameters _ = []
  replaceOwnParameters = return

instance A.Parameterized (Parameter device dtype shape) where
  flattenParameters (UnsafeMkParameter param) = pure param
  replaceOwnParameters _ = UnsafeMkParameter <$> A.nextParameter

instance {-# OVERLAPS #-} Parameterized (Tensor device dtype shape) '[] where
  flattenParameters _ = HNil
  replaceParameters = const

instance {-# OVERLAPS #-} Parameterized (Parameter device dtype shape) '[Parameter device dtype shape] where
  flattenParameters = (:. HNil)
  replaceParameters _ (parameter :. HNil) = parameter

instance {-# OVERLAPS #-} Parameterized Double '[] where
  flattenParameters _ = HNil
  replaceParameters = const

instance {-# OVERLAPS #-} Parameterized (HList '[]) '[] where
  flattenParameters _ = HNil
  replaceParameters = const

instance {-# OVERLAPS #-}
  ( Parameterized f as
  , Parameterized (HList fs) bs
  , HAppendFD as bs cs
  , cs ~ (as ++ bs)
  ) => Parameterized (HList (f ': fs)) cs
 where
  flattenParameters (f :. fs) = flattenParameters f `happendFD` flattenParameters fs
  replaceParameters (f :. fs) cs =
    let (as, bs) = hunappendFD cs
        f'       = replaceParameters f as
        fs'      = replaceParameters fs bs
    in  f' :. fs'

instance {-# OVERLAPPABLE #-}
  ( Parameterized f as
  ) => GParameterized (K1 i f) as where
  gFlattenParameters = flattenParameters . unK1
  gReplaceParameters (K1 f) = K1 . replaceParameters f

instance (GParameterized f as) => GParameterized (M1 i t f) as where
  gFlattenParameters = gFlattenParameters . unM1
  gReplaceParameters (M1 f) = M1 . gReplaceParameters f

instance GParameterized U1 '[] where
  gFlattenParameters _ = HNil
  gReplaceParameters = const

instance Torch.NN.Randomizable (HList ('[] :: [Type])) (HList ('[] :: [Type])) where
  sample = return

instance (Torch.NN.Randomizable xSpec x, Torch.NN.Randomizable (HList xsSpec) (HList xs))
  => Torch.NN.Randomizable (HList (xSpec ': xsSpec)) (HList (x ': xs))
 where
  sample (xSpec :. xsSpec) = do
    x <- Torch.NN.sample xSpec
    xs <- Torch.NN.sample xsSpec
    return $ x :. xs
