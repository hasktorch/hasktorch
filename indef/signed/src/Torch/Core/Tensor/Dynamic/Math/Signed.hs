{-# LANGUAGE InstanceSigs #-}
module Torch.Core.Tensor.Dynamic.Math.Signed where

import qualified TensorMathSigned as Sig
import qualified Torch.Class.Tensor.Math as Class

import Torch.Core.Types

instance Class.TensorMathSigned Tensor where
  neg :: Tensor -> Tensor -> IO ()
  neg = with2Tensors Sig.c_neg

  abs :: Tensor -> Tensor -> IO ()
  abs = with2Tensors Sig.c_abs

