{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Dynamic.Math.Signed where

import qualified TensorMathSigned as Sig
import qualified Torch.Class.C.Tensor.Math as Class

import Torch.Core.Types

instance Class.TensorMathSigned Tensor where
  neg_ :: Tensor -> Tensor -> IO ()
  neg_ = with2Tensors Sig.c_neg

  abs_ :: Tensor -> Tensor -> IO ()
  abs_ = with2Tensors Sig.c_abs

