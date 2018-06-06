-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed
  ( neg_, Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed.neg
  , abs_, Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed.abs
  ) where

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor
import qualified Torch.Sig.Tensor.Math.Pointwise.Signed as Sig

_abs r t = with2DynamicState r t Sig.c_abs
_neg r t = with2DynamicState r t Sig.c_neg

-- | Return a new tensor flipping the sign on every element.
neg :: Dynamic -> IO Dynamic
neg  t = withEmpty t $ \r -> _neg r t

-- | Inplace version of 'neg'
neg_ :: Dynamic -> IO ()
neg_ t = _neg t t

-- | Return a new tensor applying the absolute function to all elements.
abs :: Dynamic -> IO Dynamic
abs t = withEmpty t $ \r -> _abs r t

-- | Inplace version of 'abs'
abs_ :: Dynamic -> IO ()
abs_ t = _abs t t

