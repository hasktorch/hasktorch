module Torch.Indef.THC.Static.Tensor.Random where

import Torch.Indef.Types
import Torch.Dimensions
import Torch.Indef.THC.Dynamic.Tensor.Random ()
import qualified Torch.Class.THC.Tensor.Random.Static as Class
import qualified Torch.Class.THC.Tensor.Random as Dynamic

import qualified Torch.Types.TH as TH

instance Class.THCTensorRandom Tensor where
  random r = Dynamic.random (asDynamic r)
  clampedRandom r = Dynamic.clampedRandom (asDynamic r)
  cappedRandom r = Dynamic.cappedRandom (asDynamic r)
  geometric r = Dynamic.geometric (asDynamic r)
  bernoulli r = Dynamic.bernoulli (asDynamic r)
  bernoulli_DoubleTensor r d = Dynamic.bernoulli_DoubleTensor (asDynamic r) (asDynamic r)
  uniform r = Dynamic.uniform (asDynamic r)
  normal r = Dynamic.normal (asDynamic r)
  normal_means r a = Dynamic.normal_means (asDynamic r) (asDynamic a)
  normal_stddevs r v a = Dynamic.normal_stddevs (asDynamic r) v (asDynamic a)
  normal_means_stddevs r a b = Dynamic.normal_means_stddevs (asDynamic r) (asDynamic a) (asDynamic b)
  exponential r = Dynamic.exponential (asDynamic r)
  cauchy r = Dynamic.cauchy (asDynamic r)
  logNormal r = Dynamic.logNormal (asDynamic r)
  multinomial r t = Dynamic.multinomial (longAsDynamic r) (asDynamic t)
  multinomialAliasSetup r l t = Dynamic.multinomialAliasSetup (asDynamic r) (longAsDynamic l) (asDynamic t)
  multinomialAliasDraw r a b = Dynamic.multinomialAliasDraw (longAsDynamic r) (longAsDynamic a) (asDynamic b)
  rand r = Dynamic.rand (asDynamic r)
  randn r = Dynamic.randn (asDynamic r)

