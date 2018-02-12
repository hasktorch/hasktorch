{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Core.Tensor.Static.Random
  ( random
  , clampedRandom
  , cappedRandom
  , geometric
  , bernoulli
  , bernoulli_FloatTensor
  , bernoulli_DoubleTensor
  ) where

import Torch.Class.C.Internal (AsDynamic)
import GHC.Int (Int64)
import THTypes (CTHGenerator)
import Foreign (Ptr)

import Torch.Core.Tensor.Static
import Torch.Core.Tensor.Dim
import Torch.Core.Tensor.Dynamic.Random (TensorRandom)
import qualified Torch.Core.Tensor.Dynamic as Dynamic

-- FIXME: (stites) - I think we can replace all of these with the derived dynamic instance and just implement the bernoulli_Double-/Float-Tensors.
type RandomConstraint t d = (StaticConstraint (t d), TensorRandom (AsDynamic (t d)), Dimensions d)

random :: RandomConstraint t d => Ptr CTHGenerator -> IO (t d)
random g = withInplace (`Dynamic.random` g)

clampedRandom :: RandomConstraint t d => Ptr CTHGenerator -> Int64 -> Int64 -> IO (t d)
clampedRandom g a b = withInplace $ \res -> Dynamic.clampedRandom res g a b

cappedRandom :: RandomConstraint t d => Ptr CTHGenerator -> Int64 -> IO (t d)
cappedRandom g a = withInplace $ \res -> Dynamic.cappedRandom res g a

geometric :: RandomConstraint t d => Ptr CTHGenerator -> Double -> IO (t d)
geometric g a = withInplace $ \res -> Dynamic.geometric res g a

bernoulli :: RandomConstraint t d => Ptr CTHGenerator -> Double -> IO (t d)
bernoulli g a = withInplace $ \res -> Dynamic.bernoulli res g a

-- (stites): I think these functions take a distribution as input, but I'm not sure how the dimensions need to line up.
-- TODO: use static tensors / singletons to encode distribution information.
bernoulli_FloatTensor :: RandomConstraint t d => Ptr CTHGenerator -> Dynamic.FloatTensor -> IO (t d)
bernoulli_FloatTensor g a = withInplace $ \res -> Dynamic.bernoulli_FloatTensor res g a

bernoulli_DoubleTensor :: RandomConstraint t d => Ptr CTHGenerator -> Dynamic.DoubleTensor -> IO (t d)
bernoulli_DoubleTensor g a = withInplace $ \res -> Dynamic.bernoulli_DoubleTensor res g a

