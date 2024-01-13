{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.Parameter
  ( module Torch.Typed.Parameter,
    Torch.NN.Randomizable (..),
  )
where

import Control.Monad.State.Strict
import Data.Kind (Type)
import GHC.Generics
import GHC.TypeLits
import GHC.TypeLits.Extra
import qualified Torch.Autograd (IndependentTensor (..), makeIndependent)
import Torch.DType (DType)
import Torch.Device (DeviceType)
import Torch.HList
import qualified Torch.NN (Parameter, Randomizable (..), sample)
import qualified Torch.Tensor (toType, _toDevice)
import Torch.Typed.Auxiliary
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.Tensor

newtype
  Parameter
    (device :: (DeviceType, Nat))
    (dtype :: DType)
    (shape :: [Nat])
  = UnsafeMkParameter Torch.Autograd.IndependentTensor
  deriving (Show)

untypeParam :: Parameter device dtype shape -> Torch.NN.Parameter
untypeParam (UnsafeMkParameter param) = param

toDependent ::
  forall shape dtype device.
  Parameter device dtype shape ->
  Tensor device dtype shape
toDependent (UnsafeMkParameter t) = UnsafeMkTensor $ Torch.Autograd.toDependent t

data ToDependent = ToDependent

instance Apply' ToDependent (Parameter device dtype shape) (Tensor device dtype shape) where
  apply' _ = toDependent

makeIndependent ::
  forall shape dtype device.
  Tensor device dtype shape ->
  IO (Parameter device dtype shape)
makeIndependent t = UnsafeMkParameter <$> Torch.Autograd.makeIndependent (toDynamic t)

data MakeIndependent = MakeIndependent

instance
  Apply'
    MakeIndependent
    (Tensor device dtype shape)
    (IO (Parameter device dtype shape))
  where
  apply' _ = makeIndependent

parameterToDevice ::
  forall device' device dtype shape.
  KnownDevice device' =>
  Parameter device dtype shape ->
  Parameter device' dtype shape
parameterToDevice (UnsafeMkParameter t) =
  UnsafeMkParameter
    . Torch.Autograd.IndependentTensor
    . Torch.Tensor._toDevice (deviceVal @device')
    . Torch.Autograd.toDependent
    $ t

parameterToDType ::
  forall dtype' dtype device shape.
  KnownDType dtype' =>
  Parameter device dtype shape ->
  Parameter device dtype' shape
parameterToDType (UnsafeMkParameter t) =
  UnsafeMkParameter
    . Torch.Autograd.IndependentTensor
    . Torch.Tensor.toType (dtypeVal @dtype')
    . Torch.Autograd.toDependent
    $ t

class Parameterized (f :: Type) where
  type Parameters f :: [Type]
  type Parameters f = GParameters (Rep f)
  flattenParameters :: f -> HList (Parameters f)
  default flattenParameters ::
    (Generic f, GParameterized (Rep f), Parameters f ~ GParameters (Rep f)) =>
    f ->
    HList (Parameters f)
  flattenParameters f = gFlattenParameters (from f)
  replaceParameters :: f -> HList (Parameters f) -> f
  default replaceParameters ::
    (Generic f, GParameterized (Rep f), Parameters f ~ GParameters (Rep f)) =>
    f ->
    HList (Parameters f) ->
    f
  replaceParameters f as = to (gReplaceParameters (from f) as)

class GParameterized (f :: Type -> Type) where
  type GParameters f :: [Type]
  gFlattenParameters :: forall a. f a -> HList (GParameters f)
  gReplaceParameters :: forall a. f a -> HList (GParameters f) -> f a

instance
  ( GParameterized l,
    GParameterized r,
    HAppendFD (GParameters l) (GParameters r) (GParameters l ++ GParameters r)
  ) =>
  GParameterized (l :*: r)
  where
  type GParameters (l :*: r) = (GParameters l) ++ (GParameters r)
  gFlattenParameters (l :*: r) =
    let as = gFlattenParameters l
        bs = gFlattenParameters r
     in as `happendFD` bs
  gReplaceParameters (l :*: r) cs =
    let (as, bs) = hunappendFD cs
        l' = gReplaceParameters l as
        r' = gReplaceParameters r bs
     in l' :*: r'

instance
  Parameterized f =>
  GParameterized (K1 i f)
  where
  type GParameters (K1 i f) = Parameters f
  gFlattenParameters = flattenParameters . unK1
  gReplaceParameters (K1 f) = K1 . replaceParameters f

instance GParameterized f => GParameterized (M1 i t f) where
  type GParameters (M1 i t f) = GParameters f
  gFlattenParameters = gFlattenParameters . unM1
  gReplaceParameters (M1 f) = M1 . gReplaceParameters f

instance GParameterized U1 where
  type GParameters U1 = '[]
  gFlattenParameters _ = HNil
  gReplaceParameters = const

instance Parameterized (Tensor device dtype shape) where
  type Parameters (Tensor device dtype shape) = '[]
  flattenParameters _ = HNil
  replaceParameters = const

instance Parameterized (Parameter device dtype shape) where
  type Parameters (Parameter device dtype shape) = '[Parameter device dtype shape]
  flattenParameters = (:. HNil)
  replaceParameters _ (parameter :. HNil) = parameter

instance Parameterized Int where
  type Parameters Int = '[]
  flattenParameters _ = HNil
  replaceParameters = const

instance Parameterized Float where
  type Parameters Float = '[]
  flattenParameters _ = HNil
  replaceParameters = const

instance Parameterized Double where
  type Parameters Double = '[]
  flattenParameters _ = HNil
  replaceParameters = const

instance Parameterized (HList '[]) where
  type Parameters (HList '[]) = '[]
  flattenParameters _ = HNil
  replaceParameters = const

instance
  ( Parameterized f,
    Parameterized (HList fs),
    HAppendFD (Parameters f) (Parameters (HList fs)) (Parameters f ++ Parameters (HList fs))
  ) =>
  Parameterized (HList (f ': fs))
  where
  type Parameters (HList (f ': fs)) = Parameters f ++ Parameters (HList fs)
  flattenParameters (f :. fs) = flattenParameters f `happendFD` flattenParameters fs
  replaceParameters (f :. fs) cs =
    let (as, bs) = hunappendFD cs
        f' = replaceParameters f as
        fs' = replaceParameters fs bs
     in f' :. fs'

instance Torch.NN.Randomizable (HList ('[] :: [Type])) (HList ('[] :: [Type])) where
  sample = return

instance
  ( Torch.NN.Randomizable xSpec x,
    Torch.NN.Randomizable (HList xsSpec) (HList xs)
  ) =>
  Torch.NN.Randomizable (HList (xSpec ': xsSpec)) (HList (x ': xs))
  where
  sample (xSpec :. xsSpec) = do
    x <- Torch.NN.sample xSpec
    xs <- Torch.NN.sample xsSpec
    return $ x :. xs
