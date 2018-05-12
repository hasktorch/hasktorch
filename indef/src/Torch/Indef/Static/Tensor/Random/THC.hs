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

