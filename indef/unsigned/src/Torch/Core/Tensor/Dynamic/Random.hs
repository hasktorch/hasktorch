{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Dynamic.Random where

import Foreign
import Foreign.C.Types
import GHC.Int
import qualified TensorRandom as Sig
import qualified Torch.Class.C.Tensor.Random as Class
import THTypes
import THRandomTypes
import qualified THDoubleTypes as D
import qualified THFloatTypes  as F

import Torch.Core.Types

withTensorAndRNG :: Tensor -> Generator -> (Ptr CTensor -> Ptr CTHGenerator -> IO ()) -> IO ()
withTensorAndRNG t g op =
  _withTensor t $ \t' ->
    withForeignPtr (rng g) $ \g' ->
      op t' g'

instance Class.TensorRandom Tensor where
  random_ :: Tensor -> Generator -> IO ()
  random_ t g = withTensorAndRNG t g Sig.c_random

  clampedRandom_ :: Tensor -> Generator -> Int64 -> Int64 -> IO ()
  clampedRandom_ t g l0 l1 = withTensorAndRNG t g $ \ t' g' -> Sig.c_clampedRandom t' g' (CLLong l0) (CLLong l1)

  cappedRandom_ :: Tensor -> Generator -> Int64 -> IO ()
  cappedRandom_ t g l0 = withTensorAndRNG t g $ \ t' g' -> Sig.c_cappedRandom t' g' (CLLong l0)

  geometric_ :: Tensor -> Generator -> Double -> IO ()
  geometric_ t g ar = withTensorAndRNG t g $ \ t' g' -> Sig.c_geometric t' g' (realToFrac ar)

  bernoulli_ :: Tensor -> Generator -> Double -> IO ()
  bernoulli_ t g d = withTensorAndRNG t g $ \ t' g' -> Sig.c_bernoulli t' g' (realToFrac d)

  bernoulli_FloatTensor_ :: Tensor -> Generator -> F.DynTensor -> IO ()
  bernoulli_FloatTensor_ t g ft =
    withTensorAndRNG t g $ \ t' g' ->
      withForeignPtr (F.tensor ft) $ \ ft' ->
        Sig.c_bernoulli_FloatTensor t' g' ft'

  bernoulli_DoubleTensor_ :: Tensor -> Generator -> D.DynTensor -> IO ()
  bernoulli_DoubleTensor_ t g dt =
    withTensorAndRNG t g $ \ t' g' ->
      withForeignPtr (D.tensor dt) $ \ dt' ->
        Sig.c_bernoulli_DoubleTensor t' g' dt'


