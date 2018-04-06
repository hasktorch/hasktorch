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
  random :: Dimensions d => Tensor d -> Generator -> IO ()
  random r = Dynamic.random (asDynamic r)

  clampedRandom :: Dimensions d => Tensor d -> Generator -> Integer -> Integer -> IO ()
  clampedRandom r = Dynamic.clampedRandom (asDynamic r)

  cappedRandom :: Dimensions d => Tensor d -> Generator -> Integer -> IO ()
  cappedRandom r = Dynamic.cappedRandom (asDynamic r)

  geometric :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
  geometric r = Dynamic.geometric (asDynamic r)

  bernoulli :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
  bernoulli r = Dynamic.bernoulli (asDynamic r)

  bernoulli_FloatTensor :: Dimensions d => Tensor d -> Generator -> TH.FloatTensor d -> IO ()
  bernoulli_FloatTensor r g f = Dynamic.bernoulli_FloatTensor (asDynamic r) g (TH.floatAsDynamic f)

  bernoulli_DoubleTensor :: Dimensions d => Tensor d -> Generator -> TH.DoubleTensor d -> IO ()
  bernoulli_DoubleTensor r g d = Dynamic.bernoulli_DoubleTensor (asDynamic r) g (TH.doubleAsDynamic d)

  uniform :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
  uniform r = Dynamic.uniform (asDynamic r)

  normal :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
  normal r = Dynamic.normal (asDynamic r)

  normal_means :: Dimensions d => Tensor d -> Generator -> Tensor d -> HsAccReal -> IO ()
  normal_means r g a = Dynamic.normal_means (asDynamic r) g (asDynamic a)

  normal_stddevs :: Dimensions d => Tensor d -> Generator -> HsAccReal -> Tensor d -> IO ()
  normal_stddevs r g v a = Dynamic.normal_stddevs (asDynamic r) g v (asDynamic a)

  normal_means_stddevs :: Dimensions d => Tensor d -> Generator -> Tensor d -> Tensor d -> IO ()
  normal_means_stddevs r g a b = Dynamic.normal_means_stddevs (asDynamic r) g (asDynamic a) (asDynamic b)

  exponential :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
  exponential r = Dynamic.exponential (asDynamic r)

  standard_gamma :: Dimensions d => Tensor d -> Generator -> Tensor d -> IO ()
  standard_gamma r g t = Dynamic.standard_gamma (asDynamic r) g (asDynamic t)

  cauchy :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
  cauchy r = Dynamic.cauchy (asDynamic r)

  logNormal :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
  logNormal r = Dynamic.logNormal (asDynamic r)

  multinomial :: Dimensions d => IndexTensor d -> Generator -> Tensor d -> Int -> Int -> IO ()
  multinomial r g t = Dynamic.multinomial (longAsDynamic r) g (asDynamic t)

  multinomialAliasSetup :: Dimensions d => Tensor d -> IndexTensor d -> Tensor d -> IO ()
  multinomialAliasSetup r g t = Dynamic.multinomialAliasSetup (asDynamic r) (longAsDynamic g) (asDynamic t)

  multinomialAliasDraw :: Dimensions d => IndexTensor d -> Generator -> IndexTensor d -> Tensor d -> IO ()
  multinomialAliasDraw r g a b = Dynamic.multinomialAliasDraw (longAsDynamic r) g (longAsDynamic a) (asDynamic b)

