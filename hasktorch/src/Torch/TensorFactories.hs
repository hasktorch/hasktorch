{-# LANGUAGE FlexibleContexts #-}

module Torch.TensorFactories where

import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified ATen.Const as ATen
import qualified ATen.Managed.Native as ATen
import qualified ATen.Managed.Type.TensorOptions as ATen
import qualified ATen.Type as ATen
import qualified Torch.Managed.Native as LibTorch
import ATen.Managed.Cast
import ATen.Class (Castable(..))
import ATen.Cast

import Torch.Tensor
import Torch.TensorOptions
import Torch.Scalar

-- XXX: We use the torch:: constructors, not at:: constructures, because
--      otherwise we cannot use libtorch's AD.

type FactoryType = ForeignPtr ATen.IntArray
                    -> ForeignPtr ATen.TensorOptions
                    -> IO (ForeignPtr ATen.Tensor)

mkFactory :: FactoryType -> [Int] -> TensorOptions -> IO Tensor
mkFactory aten_impl shape opts = (cast2 aten_impl) shape opts

mkFactoryUnsafe :: FactoryType -> [Int] -> TensorOptions -> Tensor
mkFactoryUnsafe f shape opts = unsafePerformIO $ mkFactory f shape opts

mkDefaultFactory :: ([Int] -> TensorOptions -> a) -> [Int] -> a
mkDefaultFactory non_default shape = non_default shape defaultOpts

-------------------- Factories --------------------

ones :: [Int] -> TensorOptions -> Tensor
ones = mkFactoryUnsafe LibTorch.ones_lo

zeros :: [Int] -> TensorOptions -> Tensor
zeros = mkFactoryUnsafe LibTorch.zeros_lo

rand :: [Int] -> TensorOptions -> IO Tensor
rand = mkFactory LibTorch.rand_lo

randn :: [Int] -> TensorOptions -> IO Tensor
randn = mkFactory LibTorch.randn_lo

linspace :: (Scalar a, Scalar b) => a -> b -> Int -> TensorOptions -> Tensor
linspace start end steps opts = unsafePerformIO $ (cast4 LibTorch.linspace_sslo) start end steps opts

logspace :: (Scalar a, Scalar b) => a -> b -> Int -> Double -> TensorOptions -> Tensor
logspace start end steps base opts = unsafePerformIO $ (cast5 LibTorch.logspace_ssldo) start end steps base opts

-- https://github.com/hasktorch/ffi-experimental/pull/57#discussion_r301062033
-- empty :: [Int] -> TensorOptions -> Tensor
-- empty = mkFactoryUnsafe LibTorch.empty_lo

eye :: Int -> TensorOptions -> Tensor
eye dim opts = unsafePerformIO $ (cast2 LibTorch.eye_lo) dim opts

eye2 :: Int -> Int -> TensorOptions -> Tensor
eye2 nrows ncols opts = unsafePerformIO $ (cast3 LibTorch.eye_llo) nrows ncols opts

full :: Scalar a => [Int] -> a -> TensorOptions -> Tensor
full shape value opts = unsafePerformIO $ (cast3 LibTorch.full_lso) shape value opts

-------------------- Factories with default type --------------------

ones' :: [Int] -> Tensor
ones' = mkDefaultFactory ones

zeros' :: [Int] -> Tensor
zeros' = mkDefaultFactory zeros

rand' :: [Int] -> IO Tensor
rand' = mkDefaultFactory rand

randn' :: [Int] -> IO Tensor
randn' = mkDefaultFactory randn

linspace' :: (Scalar a, Scalar b) => a -> b -> Int -> Tensor
linspace' start end steps = linspace start end steps defaultOpts

logspace' :: (Scalar a, Scalar b) => a -> b -> Int -> Double -> Tensor
logspace' start end steps base = logspace start end steps base defaultOpts

eye' :: Int -> Tensor
eye' dim =  eye dim defaultOpts

eye2' :: Int -> Int -> Tensor
eye2' nrows ncols =  eye2 nrows ncols defaultOpts

full' :: Scalar a => [Int] -> a -> Tensor
full' shape value = full shape value defaultOpts
