module Torch.Indef.Static.Tensor.Random.THC where

import Torch.Indef.Types
import Torch.Dimensions
import qualified Torch.Indef.Dynamic.Tensor.Random.THC as Dynamic
import qualified Torch.Types.TH as TH

_random r = Dynamic._random (asDynamic r)
_clampedRandom r = Dynamic._clampedRandom (asDynamic r)
_cappedRandom r = Dynamic._cappedRandom (asDynamic r)
_geometric r = Dynamic._geometric (asDynamic r)
_bernoulli r = Dynamic._bernoulli (asDynamic r)
_bernoulli_DoubleTensor r d = Dynamic._bernoulli_DoubleTensor (asDynamic r) (asDynamic r)
_uniform r = Dynamic._uniform (asDynamic r)
_normal r = Dynamic._normal (asDynamic r)
_normal_means r a = Dynamic._normal_means (asDynamic r) (asDynamic a)
_normal_stddevs r v a = Dynamic._normal_stddevs (asDynamic r) v (asDynamic a)
_normal_means_stddevs r a b = Dynamic._normal_means_stddevs (asDynamic r) (asDynamic a) (asDynamic b)
_exponential r = Dynamic._exponential (asDynamic r)
_cauchy r = Dynamic._cauchy (asDynamic r)
_logNormal r = Dynamic._logNormal (asDynamic r)
_multinomial r t = Dynamic._multinomial (longAsDynamic r) (asDynamic t)
_multinomialAliasSetup r l t = Dynamic._multinomialAliasSetup (asDynamic r) (longAsDynamic l) (asDynamic t)
_multinomialAliasDraw r a b = Dynamic._multinomialAliasDraw (longAsDynamic r) (longAsDynamic a) (asDynamic b)
_rand r = Dynamic._rand (asDynamic r)
_randn r = Dynamic._randn (asDynamic r)


random :: (IsTensor t, THCTensorRandom t, Dimensions d) => IO (t d)
random = new >>= \d -> _random d >> pure d

clampedRandom :: (IsTensor t, THCTensorRandom t, Dimensions d) => Integer -> Integer -> IO (t d)
clampedRandom a b = new >>= \d -> _clampedRandom d a b >> pure d

cappedRandom :: (IsTensor t, THCTensorRandom t, Dimensions d) => Integer -> IO (t d)
cappedRandom a = new >>= \d -> _cappedRandom d a >> pure d

bernoulli :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> IO (t d)
bernoulli a = new >>= \d -> _bernoulli d a >> pure d

geometric :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> IO (t d)
geometric a = new >>= \d -> _geometric d a >> pure d

bernoulli_DoubleTensor :: (IsTensor t, THCTensorRandom t, Dimensions d) => t d -> IO (t d)
bernoulli_DoubleTensor t = new >>= \r -> _bernoulli_DoubleTensor r t >> pure r

uniform :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
uniform a b = new >>= \d -> _uniform d a b >> pure d

normal :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
normal a b = new >>= \d -> _normal d a b >> pure d

normal_means :: (IsTensor t, THCTensorRandom t, Dimensions d) => t d -> HsAccReal (t d) -> IO (t d)
normal_means a b = new >>= \d -> _normal_means d a b >> pure d

normal_stddevs :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> t d -> IO (t d)
normal_stddevs a b = new >>= \d -> _normal_stddevs d a b >> pure d

normal_means_stddevs :: (IsTensor t, THCTensorRandom t, Dimensions d) => t d -> t d -> IO (t d)
normal_means_stddevs a b = new >>= \d -> _normal_means_stddevs d a b >> pure d

logNormal :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
logNormal a b = new >>= \d -> _logNormal d a b >> pure d

exponential :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> IO (t d)
exponential a = new >>= \d -> _exponential d a >> pure d

cauchy :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
cauchy a b = new >>= \d -> _cauchy d a b >> pure d

rand :: (IsTensor t, THCTensorRandom t, Dimensions d) => TH.LongStorage -> IO (t d)
rand a = new >>= \d -> _rand d a >> pure d

randn :: (IsTensor t, THCTensorRandom t, Dimensions d) => TH.LongStorage -> IO (t d)
randn a = new >>= \d -> _randn d a >> pure d


