{-# LANGUAGE InstanceSigs #-}
module Torch.Core.Tensor.Dynamic.Random where

import Foreign
import Foreign.C.Types
import GHC.Int
import qualified TensorRandom as Sig
import qualified Torch.Class.C.Tensor.Random as Class
import THTypes

import Torch.Core.Types

instance Class.TensorRandom Tensor where
  random :: Tensor -> Ptr CTHGenerator -> IO ()
  random t g = _withTensor t $ \ t' -> Sig.c_random t' g

  clampedRandom :: Tensor -> Ptr CTHGenerator -> Int64 -> Int64 -> IO ()
  clampedRandom t g l0 l1 = _withTensor t $ \ t' -> Sig.c_clampedRandom t' g (CLLong l0) (CLLong l1)

  cappedRandom :: Tensor -> Ptr CTHGenerator -> Int64 -> IO ()
  cappedRandom t g l0 = _withTensor t $ \ t' -> Sig.c_cappedRandom t' g (CLLong l0)

  geometric :: Tensor -> Ptr CTHGenerator -> Double -> IO ()
  geometric t g ar = _withTensor t $ \ t' -> Sig.c_geometric t' g (realToFrac ar)

  bernoulli :: Tensor -> Ptr CTHGenerator -> Double -> IO ()
  bernoulli t g d = _withTensor t $ \ t' -> Sig.c_bernoulli t' g (realToFrac d)

  bernoulli_FloatTensor :: Tensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()
  bernoulli_FloatTensor t g ft = _withTensor t $ \ t' -> Sig.c_bernoulli_FloatTensor t' g ft

  bernoulli_DoubleTensor :: Tensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()
  bernoulli_DoubleTensor t g dt = _withTensor t $ \ t' -> Sig.c_bernoulli_DoubleTensor t' g dt


