-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Copy
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Functions to copy (and cast) tensors into different types.
-- This is a pure module.
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Copy where

import Torch.Types.TH
import Torch.Indef.Types (Tensor, asDynamic, asStatic)
import qualified Torch.Indef.Dynamic.Tensor.Copy as Dynamic

-- | copy a tensor
copy :: Tensor d -> Tensor d
copy = asStatic . Dynamic.copy . asDynamic

-- | copy a tensor to a byte tensor. *Use at your own discresion*
copyByte :: Tensor d -> ByteTensor d
copyByte = byteAsStatic . Dynamic.copyByte . asDynamic

-- | copy a tensor to a char tensor. *Use at your own discresion*
copyChar :: Tensor d -> CharTensor d
copyChar = charAsStatic . Dynamic.copyChar . asDynamic

-- | copy a tensor to a short tensor. *Use at your own discresion*
copyShort :: Tensor d -> ShortTensor d
copyShort = shortAsStatic . Dynamic.copyShort . asDynamic

-- | copy a tensor to a int tensor. *Use at your own discresion*
copyInt :: Tensor d -> IntTensor d
copyInt = intAsStatic . Dynamic.copyInt . asDynamic

-- | copy a tensor to a long tensor. *Use at your own discresion*
copyLong :: Tensor d -> LongTensor d
copyLong = longAsStatic . Dynamic.copyLong . asDynamic

-- | copy a tensor to a float tensor. *Use at your own discresion*
copyFloat :: Tensor d -> FloatTensor d
copyFloat = floatAsStatic . Dynamic.copyFloat . asDynamic

-- | copy a tensor to a double tensor. *Use at your own discresion*
copyDouble :: Tensor d -> DoubleTensor d
copyDouble = doubleAsStatic . Dynamic.copyDouble . asDynamic


