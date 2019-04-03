-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed
  ( neg_, Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed.neg
  , abs_, Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed.abs
  ) where

import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor
import qualified Torch.Sig.Tensor.Math.Pointwise.Signed as Sig

_abs r t = withLift $ Sig.c_abs
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t

_neg r t = withLift $ Sig.c_neg
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t

-- | Return a new tensor flipping the sign on every element.
neg :: Dynamic -> Dynamic
neg  t = unsafeDupablePerformIO $ let r = empty in _neg r t >> pure r
{-# NOINLINE neg #-}

-- | Inplace version of 'neg'
neg_ :: Dynamic -> IO ()
neg_ t = _neg t t

-- | Return a new tensor applying the absolute function to all elements.
abs :: Dynamic -> Dynamic
abs t = unsafeDupablePerformIO $ let r = empty in _abs r t >> pure r
{-# NOINLINE abs #-}

-- | Inplace version of 'abs'
abs_ :: Dynamic -> IO ()
abs_ t = _abs t t

