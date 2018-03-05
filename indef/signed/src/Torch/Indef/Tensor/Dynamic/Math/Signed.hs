{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Indef.Tensor.Dynamic.Math.Signed where

import qualified Torch.Signature.Tensor.MathSigned as Sig
import qualified Torch.Class.Tensor.Math as Class

import Torch.Indef.Types

instance Class.TensorMathSigned Tensor where
  neg_ :: Tensor -> Tensor -> IO ()
  neg_ = with2Tensors Sig.c_neg

  abs_ :: Tensor -> Tensor -> IO ()
  abs_ = with2Tensors Sig.c_abs

