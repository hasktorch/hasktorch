{-# LANGUAGE InstanceSigs #-}
module Torch.Core.Tensor.Dynamic.Math.Negative where

import qualified TensorMathNegative as Sig
import qualified Torch.Class.Tensor.Math as Class

import Torch.Core.Types

instance Class.TensorMathNegative Tensor where
  neg :: Tensor -> Tensor -> IO ()
  neg = with2Tensors Sig.c_neg

  abs :: Tensor -> Tensor -> IO ()
  abs = with2Tensors Sig.c_abs

