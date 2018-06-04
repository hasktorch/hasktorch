module Torch.Indef.Static.Tensor.Random.THC where

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.Random.THC as Dynamic
import qualified Torch.Types.TH as TH

random :: (Dimensions d) => IO (Tensor d)
random = new >>= \d -> Dynamic._random (asDynamic d) >> pure d

clampedRandom :: (Dimensions d) => Integer -> Integer -> IO (Tensor d)
clampedRandom a b = new >>= \d -> Dynamic._clampedRandom (asDynamic d) a b >> pure d

cappedRandom :: (Dimensions d) => Integer -> IO (Tensor d)
cappedRandom a = new >>= \d -> Dynamic._cappedRandom (asDynamic d) a >> pure d

bernoulli :: (Dimensions d) => HsAccReal -> IO (Tensor d)
bernoulli a = new >>= \d -> Dynamic._bernoulli (asDynamic d) a >> pure d

geometric :: (Dimensions d) => HsAccReal -> IO (Tensor d)
geometric a = new >>= \d -> Dynamic._geometric (asDynamic d) a >> pure d

bernoulli_DoubleTensor :: (Dimensions d) => Tensor d -> IO (Tensor d)
bernoulli_DoubleTensor t = new >>= \r -> Dynamic._bernoulli_DoubleTensor (asDynamic r) (asDynamic t) >> pure r

uniform :: (Dimensions d) => HsAccReal -> HsAccReal -> IO (Tensor d)
uniform a b = new >>= \d -> Dynamic._uniform (asDynamic d) a b >> pure d

normal :: (Dimensions d) => HsAccReal -> HsAccReal -> IO (Tensor d)
normal a b = new >>= \d -> Dynamic._normal (asDynamic d) a b >> pure d

normal_means :: (Dimensions d) => Tensor d -> HsAccReal -> IO (Tensor d)
normal_means a b = new >>= \d -> Dynamic._normal_means (asDynamic d) (asDynamic a) b >> pure d

normal_stddevs :: (Dimensions d) => HsAccReal -> Tensor d -> IO (Tensor d)
normal_stddevs a b = new >>= \d -> Dynamic._normal_stddevs (asDynamic d) a (asDynamic b) >> pure d

normal_means_stddevs :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
normal_means_stddevs a b = new >>= \d ->
  Dynamic._normal_means_stddevs (asDynamic d) (asDynamic a) (asDynamic b) >> pure d

logNormal :: (Dimensions d) => HsAccReal -> HsAccReal -> IO (Tensor d)
logNormal a b = new >>= \d -> Dynamic._logNormal (asDynamic d) a b >> pure d

exponential :: (Dimensions d) => HsAccReal -> IO (Tensor d)
exponential a = new >>= \d -> Dynamic._exponential (asDynamic d) a >> pure d

cauchy :: (Dimensions d) => HsAccReal -> HsAccReal -> IO (Tensor d)
cauchy a b = new >>= \d -> Dynamic._cauchy (asDynamic d) a b >> pure d

rand :: (Dimensions d) => TH.LongStorage -> IO (Tensor d)
rand a = new >>= \d -> Dynamic._rand (asDynamic d) a >> pure d

randn :: (Dimensions d) => TH.LongStorage -> IO (Tensor d)
randn a = new >>= \d -> Dynamic._randn (asDynamic d) a >> pure d


_multinomial d t = Dynamic._multinomial (longAsDynamic d) (asDynamic t)
_multinomialAliasSetup d l t = Dynamic._multinomialAliasSetup (asDynamic d) (longAsDynamic l) (asDynamic t)
_multinomialAliasDraw d a b = Dynamic._multinomialAliasDraw (longAsDynamic d) (longAsDynamic a) (asDynamic b)
