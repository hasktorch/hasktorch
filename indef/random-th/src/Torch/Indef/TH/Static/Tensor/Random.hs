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
  _random :: Dimensions d => Tensor d -> Generator -> IO ()
  _random r = Dynamic._random (asDynamic r)

  _clampedRandom :: Dimensions d => Tensor d -> Generator -> Integer -> Integer -> IO ()
  _clampedRandom r = Dynamic._clampedRandom (asDynamic r)

  _cappedRandom :: Dimensions d => Tensor d -> Generator -> Integer -> IO ()
  _cappedRandom r = Dynamic._cappedRandom (asDynamic r)

  _geometric :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
  _geometric r = Dynamic._geometric (asDynamic r)

  _bernoulli :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
  _bernoulli r = Dynamic._bernoulli (asDynamic r)

  _bernoulli_FloatTensor :: Dimensions d => Tensor d -> Generator -> TH.FloatTensor d -> IO ()
  _bernoulli_FloatTensor r g f = Dynamic._bernoulli_FloatTensor (asDynamic r) g (TH.floatAsDynamic f)

  _bernoulli_DoubleTensor :: Dimensions d => Tensor d -> Generator -> TH.DoubleTensor d -> IO ()
  _bernoulli_DoubleTensor r g d = Dynamic._bernoulli_DoubleTensor (asDynamic r) g (TH.doubleAsDynamic d)

  _uniform :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
  _uniform r = Dynamic._uniform (asDynamic r)

  _normal :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
  _normal r = Dynamic._normal (asDynamic r)

  _normal_means :: Dimensions d => Tensor d -> Generator -> Tensor d -> HsAccReal -> IO ()
  _normal_means r g a = Dynamic._normal_means (asDynamic r) g (asDynamic a)

  _normal_stddevs :: Dimensions d => Tensor d -> Generator -> HsAccReal -> Tensor d -> IO ()
  _normal_stddevs r g v a = Dynamic._normal_stddevs (asDynamic r) g v (asDynamic a)

  _normal_means_stddevs :: Dimensions d => Tensor d -> Generator -> Tensor d -> Tensor d -> IO ()
  _normal_means_stddevs r g a b = Dynamic._normal_means_stddevs (asDynamic r) g (asDynamic a) (asDynamic b)

  _exponential :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
  _exponential r = Dynamic._exponential (asDynamic r)

  _standard_gamma :: Dimensions d => Tensor d -> Generator -> Tensor d -> IO ()
  _standard_gamma r g t = Dynamic._standard_gamma (asDynamic r) g (asDynamic t)

  _cauchy :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
  _cauchy r = Dynamic._cauchy (asDynamic r)

  _logNormal :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
  _logNormal r = Dynamic._logNormal (asDynamic r)

  _multinomial :: Dimensions d => IndexTensor d -> Generator -> Tensor d -> Int -> Int -> IO ()
  _multinomial r g t = Dynamic._multinomial (longAsDynamic r) g (asDynamic t)

  _multinomialAliasSetup :: Dimensions d => Tensor d -> IndexTensor d -> Tensor d -> IO ()
  _multinomialAliasSetup r g t = Dynamic._multinomialAliasSetup (asDynamic r) (longAsDynamic g) (asDynamic t)

  _multinomialAliasDraw :: Dimensions d => IndexTensor d -> Generator -> IndexTensor d -> Tensor d -> IO ()
  _multinomialAliasDraw r g a b = Dynamic._multinomialAliasDraw (longAsDynamic r) g (longAsDynamic a) (asDynamic b)

