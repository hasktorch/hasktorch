module Torch.Indef.TH.Static.Tensor.Random where

import Torch.Indef.Types
import Torch.Dimensions
import Torch.Indef.TH.Dynamic.Tensor.Random ()
import qualified Torch.Class.TH.Tensor.Random.Static as Class
import qualified Torch.Class.TH.Tensor.Random as Dynamic
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.TH.Tensor.Random as Sig
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig

instance Class.THTensorRandom Tensor where
  random_ :: Dimensions d => Tensor d -> Generator -> IO ()
  random_ r = Dynamic.random_ (asDynamic r)

  clampedRandom_ :: Dimensions d => Tensor d -> Generator -> Integer -> Integer -> IO ()
  clampedRandom_ r = Dynamic.clampedRandom_ (asDynamic r)

  cappedRandom_ :: Dimensions d => Tensor d -> Generator -> Integer -> IO ()
  cappedRandom_ r = Dynamic.cappedRandom_ (asDynamic r)

  geometric_ :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
  geometric_ r = Dynamic.geometric_ (asDynamic r)

  bernoulli_ :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
  bernoulli_ r = Dynamic.bernoulli_ (asDynamic r)

  bernoulli_FloatTensor_ :: Dimensions d => Tensor d -> Generator -> TH.FloatTensor d -> IO ()
  bernoulli_FloatTensor_ r g f = Dynamic.bernoulli_FloatTensor_ (asDynamic r) g (TH.floatAsDynamic f)

  bernoulli_DoubleTensor_ :: Dimensions d => Tensor d -> Generator -> TH.DoubleTensor d -> IO ()
  bernoulli_DoubleTensor_ r g d = Dynamic.bernoulli_DoubleTensor_ (asDynamic r) g (TH.doubleAsDynamic d)

  uniform_ :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
  uniform_ r = Dynamic.uniform_ (asDynamic r)

  normal_ :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
  normal_ r = Dynamic.normal_ (asDynamic r)

  normal_means_ :: Dimensions d => Tensor d -> Generator -> Tensor d -> HsAccReal -> IO ()
  normal_means_ r g a = Dynamic.normal_means_ (asDynamic r) g (asDynamic a)

  normal_stddevs_ :: Dimensions d => Tensor d -> Generator -> HsAccReal -> Tensor d -> IO ()
  normal_stddevs_ r g v a = Dynamic.normal_stddevs_ (asDynamic r) g v (asDynamic a)

  normal_means_stddevs_ :: Dimensions d => Tensor d -> Generator -> Tensor d -> Tensor d -> IO ()
  normal_means_stddevs_ r g a b = Dynamic.normal_means_stddevs_ (asDynamic r) g (asDynamic a) (asDynamic b)

  exponential_ :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
  exponential_ r = Dynamic.exponential_ (asDynamic r)

  standard_gamma_ :: Dimensions d => Tensor d -> Generator -> Tensor d -> IO ()
  standard_gamma_ r g t = Dynamic.standard_gamma_ (asDynamic r) g (asDynamic t)

  cauchy_ :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
  cauchy_ r = Dynamic.cauchy_ (asDynamic r)

  logNormal_ :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
  logNormal_ r = Dynamic.logNormal_ (asDynamic r)

  multinomial_ :: Dimensions d => IndexTensor d -> Generator -> Tensor d -> Int -> Int -> IO ()
  multinomial_ r g t = Dynamic.multinomial_ (longAsDynamic r) g (asDynamic t)

  multinomialAliasSetup_ :: Dimensions d => Tensor d -> IndexTensor d -> Tensor d -> IO ()
  multinomialAliasSetup_ r g t = Dynamic.multinomialAliasSetup_ (asDynamic r) (longAsDynamic g) (asDynamic t)

  multinomialAliasDraw_ :: Dimensions d => IndexTensor d -> Generator -> IndexTensor d -> Tensor d -> IO ()
  multinomialAliasDraw_ r g a b = Dynamic.multinomialAliasDraw_ (longAsDynamic r) g (longAsDynamic a) (asDynamic b)

