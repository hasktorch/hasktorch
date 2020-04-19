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
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.Parameter where

import Control.Monad.State.Strict
import Data.Kind (Type)
import GHC.Generics
import GHC.TypeLits
import GHC.TypeLits.Extra
import qualified Torch.Autograd as A
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import qualified Torch.NN as A
import qualified Torch.Tensor as D
import Torch.Typed.Aux
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.Tensor

newtype Parameter (device :: (D.DeviceType, Nat)) (dtype :: D.DType) (shape :: [Nat]) = UnsafeMkParameter A.IndependentTensor
  deriving (Show)

toDependent ::
  forall shape dtype device.
  Parameter device dtype shape ->
  Tensor device dtype shape
toDependent (UnsafeMkParameter t) = UnsafeMkTensor $ A.toDependent t

data ToDependent = ToDependent

instance Apply' ToDependent (Parameter device dtype shape) (Tensor device dtype shape) where
  apply' _ = toDependent

makeIndependent ::
  forall shape dtype device.
  Tensor device dtype shape ->
  IO (Parameter device dtype shape)
makeIndependent t = UnsafeMkParameter <$> A.makeIndependent (toDynamic t)

data MakeIndependent = MakeIndependent

instance Apply' MakeIndependent (Tensor device dtype shape) (IO (Parameter device dtype shape)) where
  apply' _ = makeIndependent

toDevice ::
  forall device' device dtype shape.
  KnownDevice device' =>
  Parameter device dtype shape ->
  Parameter device' dtype shape
toDevice (UnsafeMkParameter t) = UnsafeMkParameter . A.IndependentTensor . D.toDevice (deviceVal @device') . A.toDependent $ t

toDType ::
  forall dtype' dtype device shape.
  KnownDType dtype' =>
  Parameter device dtype shape ->
  Parameter device dtype' shape
toDType (UnsafeMkParameter t) = UnsafeMkParameter . A.IndependentTensor . D.toType (dtypeVal @dtype') . A.toDependent $ t

class
  Parameterized
    (f :: Type)
    (as :: [Type])
    | f -> as where
  flattenParameters :: f -> HList as
  replaceParameters :: f -> HList as -> f

instance
  ( Generic f,
    GParameterized (Rep f) as
  ) =>
  Parameterized f as
  where
  flattenParameters f = gFlattenParameters (from f)
  replaceParameters f as = to (gReplaceParameters (from f) as)

class
  GParameterized
    (f :: Type -> Type)
    (as :: [Type])
    | f -> as where
  gFlattenParameters :: forall a. f a -> HList as
  gReplaceParameters :: forall a. f a -> HList as -> f a

instance
  ( GParameterized l as,
    GParameterized r bs,
    HAppendFD as bs cs,
    cs ~ (as ++ bs)
  ) =>
  GParameterized (l :*: r) cs
  where
  gFlattenParameters (l :*: r) =
    let as = gFlattenParameters l
        bs = gFlattenParameters r
     in as `happendFD` bs
  gReplaceParameters (l :*: r) cs =
    let (as, bs) = hunappendFD cs
        l' = gReplaceParameters l as
        r' = gReplaceParameters r bs
     in l' :*: r'

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

instance
  {-# OVERLAPS #-}
  ( Parameterized f as,
    Parameterized (HList fs) bs,
    HAppendFD as bs cs,
    cs ~ (as ++ bs)
  ) =>
  Parameterized (HList (f ': fs)) cs
  where
  flattenParameters (f :. fs) = flattenParameters f `happendFD` flattenParameters fs
  replaceParameters (f :. fs) cs =
    let (as, bs) = hunappendFD cs
        f' = replaceParameters f as
        fs' = replaceParameters fs bs
     in f' :. fs'

instance
  {-# OVERLAPPABLE #-}
  ( Parameterized f as
  ) =>
  GParameterized (K1 i f) as
  where
  gFlattenParameters = flattenParameters . unK1
  gReplaceParameters (K1 f) = K1 . replaceParameters f

instance (GParameterized f as) => GParameterized (M1 i t f) as where
  gFlattenParameters = gFlattenParameters . unM1
  gReplaceParameters (M1 f) = M1 . gReplaceParameters f

instance GParameterized U1 '[] where
  gFlattenParameters _ = HNil
  gReplaceParameters = const

instance A.Randomizable (HList ('[] :: [Type])) (HList ('[] :: [Type])) where
  sample = return

instance
  (A.Randomizable xSpec x, A.Randomizable (HList xsSpec) (HList xs)) =>
  A.Randomizable (HList (xSpec ': xsSpec)) (HList (x ': xs))
  where
  sample (xSpec :. xsSpec) = do
    x <- A.sample xSpec
    xs <- A.sample xsSpec
    return $ x :. xs
