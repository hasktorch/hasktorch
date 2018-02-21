{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.Tensor.Static.Random
  ( Random.TensorRandom(..)
  , Random.TensorRandomFloating(..)
  , random
  , clampedRandom
  , cappedRandom
  , geometric
  , bernoulli
  , bernoulli_FloatTensor
  , bernoulli_DoubleTensor
  , uniform
  , multivariate_normal
  , normal
  , normal_means
  , normal_stddevs
  , normal_means_stddevs
  , exponential
  , standard_gamma
  , cauchy
  , logNormal
  ) where

import Torch.Class.C.Tensor.Random (TensorRandom, TensorRandomFloating)
import Torch.Class.C.IsTensor (IsTensor)
import Torch.Class.C.Internal (HsAccReal, HsReal, AsDynamic)
import Torch.Core.Tensor.Dim
import THRandomTypes (Generator)
import GHC.Int (Int64)
import qualified Torch.Core.Tensor.Dynamic.Random as Random
import Torch.Core.Tensor.Static.Math (diag1d, TensorMath)
import Torch.Core.Tensor.Static.Math.Infix ((!*!), (^+^))
import Torch.Core.Tensor.Static (newTranspose2d, expand2d, StaticConstraint, StaticConstraint2, StaticConstraint3)

import qualified THFloatTypes as Float
import qualified THDoubleTypes as Double

import Torch.Core.ByteTensor.Static.Random   ()
import Torch.Core.ShortTensor.Static.Random  ()
import Torch.Core.IntTensor.Static.Random    ()
import Torch.Core.LongTensor.Static.Random   ()
import Torch.Core.FloatTensor.Static.Random  ()
import Torch.Core.DoubleTensor.Static.Random ()

import Torch.Core.FloatTensor.Static.Random.Floating ()
import Torch.Core.DoubleTensor.Static.Random.Floating ()

random :: forall t d . (Dimensions d, IsTensor (t d), TensorRandom (t d)) => Generator -> IO (t d)
random = Random.random (dim :: Dim d)

clampedRandom :: forall t d . (Dimensions d, IsTensor (t d), TensorRandom (t d)) => Generator -> Int64 -> Int64 -> IO (t d)
clampedRandom = Random.clampedRandom (dim :: Dim d)

cappedRandom :: forall t d . (Dimensions d, IsTensor (t d), TensorRandom (t d)) => Generator -> Int64 -> IO (t d)
cappedRandom = Random.cappedRandom (dim :: Dim d)

geometric :: forall t d . (Dimensions d, IsTensor (t d), TensorRandom (t d)) => Generator -> Double -> IO (t d)
geometric = Random.geometric (dim :: Dim d)

bernoulli :: forall t d . (Dimensions d, IsTensor (t d), TensorRandom (t d)) => Generator -> Double -> IO (t d)
bernoulli = Random.bernoulli (dim :: Dim d)

bernoulli_FloatTensor :: forall t d . (Dimensions d, IsTensor (t d), TensorRandom (t d)) => Generator -> Float.Tensor d -> IO (t d)
bernoulli_FloatTensor g f = Random.bernoulli_FloatTensor (dim :: Dim d) g (Float.dynamic f)

bernoulli_DoubleTensor :: forall t d . (Dimensions d, IsTensor (t d), TensorRandom (t d)) => Generator -> Double.Tensor d -> IO (t d)
bernoulli_DoubleTensor g d = Random.bernoulli_DoubleTensor (dim :: Dim d) g (Double.dynamic d)

uniform :: forall t d . (Dimensions d, IsTensor (t d), TensorRandomFloating (t d)) => Generator -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
uniform = Random.uniform (dim :: Dim d)


multivariate_normal
  :: forall t p n . (KnownNat2 n p, KnownDim n, KnownDim p) -- , StaticConstraint3 t '[n, p] '[p, p] '[p])
  => (StaticConstraint2 t '[n, p] '[p])
  => (TensorMath (AsDynamic (t '[p])))
  => (StaticConstraint t '[p, p])
  => (HsReal (t '[p, p]) ~ HsReal (AsDynamic (t '[p])))
  => (HsAccReal (t '[p, p]) ~ HsAccReal (AsDynamic (t '[p])))
  => (AsDynamic (t '[p, p]) ~ AsDynamic (t '[p]))
  => (StaticConstraint2 t '[p, n] '[p])
  => (IsTensor (t '[p, n]))
  => (TensorRandomFloating (t '[p, n]))
  => (Num (HsReal (AsDynamic (t '[p]))))
  => (Num (HsAccReal (AsDynamic (t '[p]))))
  => Generator -> t ('[p]::[Nat]) -> t '[p, p] -> t '[p] -> IO (t '[n, p])
multivariate_normal g mu eigvec eigval = do
  offset :: t '[n, p] <- expand2d mu
  samps :: t '[p, n] <- normal g 0 1
  eigvec' <- newTranspose2d eigvec
  eigval' <- diag1d eigval
  let x = (eigvec' !*! eigval') :: t '[p, p]
  let y = x !*! eigvec :: t '[p, p]
  z :: t '[n, p] <- newTranspose2d (y !*! samps)
  let q = z ^+^ offset :: t '[n, p]
  pure q
  -- pure $ ((-- !*! eigvec) !*! samps) ^+^ offset





normal :: forall t d . (Dimensions d, IsTensor (t d), TensorRandomFloating (t d)) => Generator -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
normal = Random.normal (dim :: Dim d)

normal_means :: forall t d . (Dimensions d, IsTensor (t d), TensorRandomFloating (t d)) => Generator -> (t d) -> HsAccReal (t d) -> IO (t d)
normal_means = Random.normal_means (dim :: Dim d)

normal_stddevs :: forall t d . (Dimensions d, IsTensor (t d), TensorRandomFloating (t d)) => Generator -> HsAccReal (t d) -> (t d) -> IO (t d)
normal_stddevs = Random.normal_stddevs (dim :: Dim d)

normal_means_stddevs :: forall t d . (Dimensions d, IsTensor (t d), TensorRandomFloating (t d)) => Generator -> (t d) -> (t d) -> IO (t d)
normal_means_stddevs = Random.normal_means_stddevs (dim :: Dim d)

exponential :: forall t d . (Dimensions d, IsTensor (t d), TensorRandomFloating (t d)) => Generator -> HsAccReal (t d) -> IO (t d)
exponential = Random.exponential (dim :: Dim d)

standard_gamma :: forall t d . (Dimensions d, IsTensor (t d), TensorRandomFloating (t d)) => Generator -> (t d) -> IO (t d)
standard_gamma = Random.standard_gamma (dim :: Dim d)

cauchy :: forall t d . (Dimensions d, IsTensor (t d), TensorRandomFloating (t d)) => Generator -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
cauchy = Random.cauchy (dim :: Dim d)

logNormal :: forall t d . (Dimensions d, IsTensor (t d), TensorRandomFloating (t d)) => Generator -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
logNormal = Random.logNormal (dim :: Dim d)


