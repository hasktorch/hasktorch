{-# LANGUAGE FlexibleContexts #-}

module Torch.TensorFactories where

import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified ATen.Const as ATen
import qualified ATen.Managed.Native as ATen
import qualified ATen.Managed.Type.Tensor as ATen
import qualified ATen.Managed.Type.TensorOptions as ATen
import qualified ATen.Type as ATen
import qualified Torch.Managed.Native as LibTorch
import qualified Torch.Managed.Autograd as LibTorch
import ATen.Managed.Cast
import ATen.Class (Castable(..))
import ATen.Cast

import Torch.Tensor
import Torch.TensorOptions
import Torch.Scalar
import Torch.Dimname

-- XXX: We use the torch:: constructors, not at:: constructures, because
--      otherwise we cannot use libtorch's AD.

type FactoryType = ForeignPtr ATen.IntArray
                    -> ForeignPtr ATen.TensorOptions
                    -> IO (ForeignPtr ATen.Tensor)

type FactoryTypeWithDimnames = ForeignPtr ATen.IntArray
                                -> ForeignPtr ATen.DimnameList
                                -> ForeignPtr ATen.TensorOptions
                                -> IO (ForeignPtr ATen.Tensor)

mkFactory :: FactoryType -> [Int] -> TensorOptions -> IO Tensor
mkFactory aten_impl shape opts = (cast2 aten_impl) shape opts

mkFactoryUnsafe :: FactoryType -> [Int] -> TensorOptions -> Tensor
mkFactoryUnsafe f shape opts = unsafePerformIO $ mkFactory f shape opts

mkFactoryWithDimnames :: FactoryTypeWithDimnames -> [(Int,Dimname)] -> TensorOptions -> IO Tensor
mkFactoryWithDimnames aten_impl shape opts = (cast3 aten_impl) (map fst shape) (map snd shape) opts

mkFactoryUnsafeWithDimnames :: FactoryTypeWithDimnames -> [(Int,Dimname)] -> TensorOptions -> Tensor
mkFactoryUnsafeWithDimnames f shape opts = unsafePerformIO $ mkFactoryWithDimnames f shape opts

mkDefaultFactory :: ([Int] -> TensorOptions -> a) -> [Int] -> a
mkDefaultFactory non_default shape = non_default shape defaultOpts

mkDefaultFactoryWithDimnames :: ([(Int,Dimname)] -> TensorOptions -> a) -> [(Int,Dimname)] -> a
mkDefaultFactoryWithDimnames non_default shape = non_default shape defaultOpts

-------------------- Factories --------------------

ones :: [Int] -> TensorOptions -> Tensor
ones = mkFactoryUnsafe LibTorch.ones_lo

zeros :: [Int] -> TensorOptions -> Tensor
zeros = mkFactoryUnsafe LibTorch.zeros_lo

rand :: [Int] -> TensorOptions -> IO Tensor
rand = mkFactory LibTorch.rand_lo

randn :: [Int] -> TensorOptions -> IO Tensor
randn = mkFactory LibTorch.randn_lo

randnLike :: Tensor -> IO Tensor
randnLike = cast1 ATen.randn_like_t

fullLike :: Tensor -> Float -> TensorOptions -> IO Tensor
fullLike input _fill_value opt = (cast3 LibTorch.full_like_tso) input _fill_value opt

randLike :: Tensor -> TensorOptions -> IO Tensor
randLike input opt = (cast2 LibTorch.rand_like_to) input opt

onesWithDimnames :: [(Int,Dimname)] -> TensorOptions -> Tensor
onesWithDimnames = mkFactoryUnsafeWithDimnames LibTorch.ones_lNo

zerosWithDimnames :: [(Int,Dimname)] -> TensorOptions -> Tensor
zerosWithDimnames = mkFactoryUnsafeWithDimnames LibTorch.zeros_lNo

randWithDimnames :: [(Int,Dimname)] -> TensorOptions -> IO Tensor
randWithDimnames = mkFactoryWithDimnames LibTorch.rand_lNo

randnWithDimnames :: [(Int,Dimname)] -> TensorOptions -> IO Tensor
randnWithDimnames = mkFactoryWithDimnames LibTorch.randn_lNo

linspace :: (Scalar a, Scalar b) => a -> b -> Int -> TensorOptions -> Tensor
linspace start end steps opts = unsafePerformIO $ (cast4 LibTorch.linspace_sslo) start end steps opts

logspace :: (Scalar a, Scalar b) => a -> b -> Int -> Double -> TensorOptions -> Tensor
logspace start end steps base opts = unsafePerformIO $ (cast5 LibTorch.logspace_ssldo) start end steps base opts

-- https://github.com/hasktorch/ffi-experimental/pull/57#discussion_r301062033
-- empty :: [Int] -> TensorOptions -> Tensor
-- empty = mkFactoryUnsafe LibTorch.empty_lo

eyeSquare :: Int -> TensorOptions -> Tensor
eyeSquare dim opts = unsafePerformIO $ (cast2 LibTorch.eye_lo) dim opts

eye :: Int -> Int -> TensorOptions -> Tensor
eye nrows ncols opts = unsafePerformIO $ (cast3 LibTorch.eye_llo) nrows ncols opts

full :: Scalar a => [Int] -> a -> TensorOptions -> Tensor
full shape value opts = unsafePerformIO $ (cast3 LibTorch.full_lso) shape value opts

sparseCooTensor :: Tensor -> Tensor -> [Int] -> TensorOptions -> Tensor
sparseCooTensor indices values size opts =  unsafePerformIO $ (cast4 sparse_coo_tensor_ttlo) indices values size opts
  where
    sparse_coo_tensor_ttlo indices' values' size' opts' = do
      i' <- LibTorch.dropVariable indices'
      v' <- LibTorch.dropVariable values'
      LibTorch.sparse_coo_tensor_ttlo i' v' size' opts'

-------------------- Factories with default type --------------------

ones' :: [Int] -> Tensor
ones' = mkDefaultFactory ones

zeros' :: [Int] -> Tensor
zeros' = mkDefaultFactory zeros

rand' :: [Int] -> IO Tensor
rand' = mkDefaultFactory rand

randn' :: [Int] -> IO Tensor
randn' = mkDefaultFactory randn

randLike' :: Tensor -> IO Tensor
randLike' t = randLike t defaultOpts

onesWithDimnames' :: [(Int,Dimname)] -> Tensor
onesWithDimnames' = mkDefaultFactoryWithDimnames onesWithDimnames

zerosWithDimnames' :: [(Int,Dimname)] -> Tensor
zerosWithDimnames' = mkDefaultFactoryWithDimnames zerosWithDimnames

randWithDimnames' :: [(Int,Dimname)] -> IO Tensor
randWithDimnames' = mkDefaultFactoryWithDimnames randWithDimnames

randnWithDimnames' :: [(Int,Dimname)] -> IO Tensor
randnWithDimnames' = mkDefaultFactoryWithDimnames randnWithDimnames

linspace' :: (Scalar a, Scalar b) => a -> b -> Int -> Tensor
linspace' start end steps = linspace start end steps defaultOpts

logspace' :: (Scalar a, Scalar b) => a -> b -> Int -> Double -> Tensor
logspace' start end steps base = logspace start end steps base defaultOpts

eyeSquare' :: Int -> Tensor
eyeSquare' dim =  eyeSquare dim defaultOpts

eye' :: Int -> Int -> Tensor
eye' nrows ncols =  eye nrows ncols defaultOpts

full' :: Scalar a => [Int] -> a -> Tensor
full' shape value = full shape value defaultOpts

sparseCooTensor' :: Tensor -> Tensor -> [Int] -> Tensor
sparseCooTensor' indices values size = sparseCooTensor indices values size defaultOpts
