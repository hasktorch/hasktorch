{-# LANGUAGE FlexibleContexts #-}

module Torch.TensorFactories where

import Foreign.ForeignPtr
import System.IO.Unsafe
import Torch.Dimname
import Torch.Internal.Cast
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Managed.Autograd as LibTorch
import Torch.Internal.Managed.Cast
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.TensorFactories as LibTorch
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Type.TensorOptions as ATen
import qualified Torch.Internal.Type as ATen
import Torch.Scalar
import Torch.Tensor
import Torch.TensorOptions

-- XXX: We use the torch:: constructors, not at:: constructures, because
--      otherwise we cannot use libtorch's AD.

type FactoryType =
  ForeignPtr ATen.IntArray ->
  ForeignPtr ATen.TensorOptions ->
  IO (ForeignPtr ATen.Tensor)

type FactoryTypeWithDimnames =
  ForeignPtr ATen.IntArray ->
  ForeignPtr ATen.DimnameList ->
  ForeignPtr ATen.TensorOptions ->
  IO (ForeignPtr ATen.Tensor)

mkFactory :: FactoryType -> [Int] -> TensorOptions -> IO Tensor
mkFactory aten_impl shape opts = (cast2 aten_impl) shape opts

mkFactoryUnsafe :: FactoryType -> [Int] -> TensorOptions -> Tensor
mkFactoryUnsafe f shape opts = unsafePerformIO $ mkFactory f shape opts

mkFactoryWithDimnames :: FactoryTypeWithDimnames -> [(Int, Dimname)] -> TensorOptions -> IO Tensor
mkFactoryWithDimnames aten_impl shape opts = (cast3 aten_impl) (map fst shape) (map snd shape) opts

mkFactoryUnsafeWithDimnames :: FactoryTypeWithDimnames -> [(Int, Dimname)] -> TensorOptions -> Tensor
mkFactoryUnsafeWithDimnames f shape opts = unsafePerformIO $ mkFactoryWithDimnames f shape opts

mkDefaultFactory :: ([Int] -> TensorOptions -> a) -> [Int] -> a
mkDefaultFactory non_default shape = non_default shape defaultOpts

mkDefaultFactoryWithDimnames :: ([(Int, Dimname)] -> TensorOptions -> a) -> [(Int, Dimname)] -> a
mkDefaultFactoryWithDimnames non_default shape = non_default shape defaultOpts

-------------------- Factories --------------------

-- | Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.
ones ::
  -- | sequence of integers defining the shape of the output tensor.
  [Int] ->
  -- | configures the data type, device, layout and other properties of the resulting tensor.
  TensorOptions ->
  -- | output
  Tensor
ones = mkFactoryUnsafe LibTorch.ones_lo

-- TODO - ones_like from Native.hs is redundant with this

-- | Returns a tensor filled with the scalar value 1, with the same size as input tensor
onesLike ::
  -- | input
  Tensor ->
  -- | output
  Tensor
onesLike self = unsafePerformIO $ (cast1 ATen.ones_like_t) self

-- | Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
zeros ::
  -- | sequence of integers defining the shape of the output tensor.
  [Int] ->
  -- | configures the data type, device, layout and other properties of the resulting tensor.
  TensorOptions ->
  -- | output
  Tensor
zeros = mkFactoryUnsafe LibTorch.zeros_lo

-- | Returns a tensor filled with the scalar value 0, with the same size as input tensor
zerosLike ::
  -- | input
  Tensor ->
  -- | output
  Tensor
zerosLike self = unsafePerformIO $ (cast1 ATen.zeros_like_t) self

-- | Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1)
randIO ::
  -- | sequence of integers defining the shape of the output tensor.
  [Int] ->
  -- | configures the data type, device, layout and other properties of the resulting tensor.
  TensorOptions ->
  -- | output
  IO Tensor
randIO = mkFactory LibTorch.rand_lo

-- | Returns a tensor filled with random numbers from a standard normal distribution.
randnIO ::
  -- | sequence of integers defining the shape of the output tensor.
  [Int] ->
  -- | configures the data type, device, layout and other properties of the resulting tensor.
  TensorOptions ->
  -- | output
  IO Tensor
randnIO = mkFactory LibTorch.randn_lo

-- | Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
randintIO ::
  -- | lowest integer to be drawn from the distribution. Default: 0.
  Int ->
  -- | one above the highest integer to be drawn from the distribution.
  Int ->
  -- | the shape of the output tensor.
  [Int] ->
  -- | configures the data type, device, layout and other properties of the resulting tensor.
  TensorOptions ->
  -- | output
  IO Tensor
randintIO low high = mkFactory (LibTorch.randint_lllo (fromIntegral low) (fromIntegral high))

-- | Returns a tensor with the same size as input that is filled with random numbers from standard normal distribution.
randnLikeIO ::
  -- | input
  Tensor ->
  -- | output
  IO Tensor
randnLikeIO = cast1 ATen.randn_like_t

-- | Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0,1).
randLikeIO ::
  -- | input
  Tensor ->
  -- | configures the data type, device, layout and other properties of the resulting tensor.
  TensorOptions ->
  -- | output
  IO Tensor
randLikeIO input opt = cast2 LibTorch.rand_like_to input opt

fullLike :: Tensor -> Float -> TensorOptions -> IO Tensor
fullLike input _fill_value opt = cast3 LibTorch.full_like_tso input _fill_value opt

onesWithDimnames :: [(Int, Dimname)] -> TensorOptions -> Tensor
onesWithDimnames = mkFactoryUnsafeWithDimnames LibTorch.ones_lNo

zerosWithDimnames :: [(Int, Dimname)] -> TensorOptions -> Tensor
zerosWithDimnames = mkFactoryUnsafeWithDimnames LibTorch.zeros_lNo

randWithDimnames :: [(Int, Dimname)] -> TensorOptions -> IO Tensor
randWithDimnames = mkFactoryWithDimnames LibTorch.rand_lNo

randnWithDimnames :: [(Int, Dimname)] -> TensorOptions -> IO Tensor
randnWithDimnames = mkFactoryWithDimnames LibTorch.randn_lNo

-- | Returns a one-dimensional tensor of steps equally spaced points between start and end.
linspace ::
  (Scalar a, Scalar b) =>
  -- | @start@
  a ->
  -- | @end@
  b ->
  -- | @steps@
  Int ->
  -- | configures the data type, device, layout and other properties of the resulting tensor.
  TensorOptions ->
  -- | output
  Tensor
linspace start end steps opts = unsafePerformIO $ (cast4 LibTorch.linspace_sslo) start end steps opts

logspace :: (Scalar a, Scalar b) => a -> b -> Int -> Double -> TensorOptions -> Tensor
logspace start end steps base opts = unsafePerformIO $ (cast5 LibTorch.logspace_ssldo) start end steps base opts

-- https://github.com/hasktorch/ffi-experimental/pull/57#discussion_r301062033
-- empty :: [Int] -> TensorOptions -> Tensor
-- empty = mkFactoryUnsafe LibTorch.empty_lo

eyeSquare :: Int -> TensorOptions -> Tensor
eyeSquare dim opts = unsafePerformIO $ (cast2 LibTorch.eye_lo) dim opts

-- | Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
eye ::
  -- | the number of rows
  Int ->
  -- | the number of columns
  Int ->
  -- | configures the data type, device, layout and other properties of the resulting tensor.
  TensorOptions ->
  -- | output
  Tensor
eye nrows ncols opts = unsafePerformIO $ (cast3 LibTorch.eye_llo) nrows ncols opts

-- | Returns a tensor of given size filled with fill_value.
full ::
  Scalar a =>
  -- | the shape of the output tensor.
  [Int] ->
  -- | the number to fill the output tensor with
  a ->
  -- | configures the data type, device, layout and other properties of the resulting tensor.
  TensorOptions ->
  -- | output
  Tensor
full shape value opts = unsafePerformIO $ (cast3 LibTorch.full_lso) shape value opts

-- | Constructs a sparse tensors in COO(rdinate) format with non-zero elements at the given indices with the given values.
sparseCooTensor ::
  -- | The indices are the coordinates of the non-zero values in the matrix
  Tensor ->
  -- | Initial values for the tensor.
  Tensor ->
  -- | the shape of the output tensor.
  [Int] ->
  -- |
  TensorOptions ->
  -- | output
  Tensor
sparseCooTensor indices values size opts = unsafePerformIO $ (cast4 sparse_coo_tensor_ttlo) indices values size opts
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

randIO' :: [Int] -> IO Tensor
randIO' = mkDefaultFactory randIO

randnIO' :: [Int] -> IO Tensor
randnIO' = mkDefaultFactory randnIO

randintIO' :: Int -> Int -> [Int] -> IO Tensor
randintIO' low high = mkDefaultFactory (randintIO low high)

randLikeIO' :: Tensor -> IO Tensor
randLikeIO' t = randLikeIO t defaultOpts

onesWithDimnames' :: [(Int, Dimname)] -> Tensor
onesWithDimnames' = mkDefaultFactoryWithDimnames onesWithDimnames

zerosWithDimnames' :: [(Int, Dimname)] -> Tensor
zerosWithDimnames' = mkDefaultFactoryWithDimnames zerosWithDimnames

randWithDimnames' :: [(Int, Dimname)] -> IO Tensor
randWithDimnames' = mkDefaultFactoryWithDimnames randWithDimnames

randnWithDimnames' :: [(Int, Dimname)] -> IO Tensor
randnWithDimnames' = mkDefaultFactoryWithDimnames randnWithDimnames

linspace' :: (Scalar a, Scalar b) => a -> b -> Int -> Tensor
linspace' start end steps = linspace start end steps defaultOpts

logspace' :: (Scalar a, Scalar b) => a -> b -> Int -> Double -> Tensor
logspace' start end steps base = logspace start end steps base defaultOpts

eyeSquare' :: Int -> Tensor
eyeSquare' dim = eyeSquare dim defaultOpts

eye' :: Int -> Int -> Tensor
eye' nrows ncols = eye nrows ncols defaultOpts

full' :: Scalar a => [Int] -> a -> Tensor
full' shape value = full shape value defaultOpts

sparseCooTensor' :: Tensor -> Tensor -> [Int] -> Tensor
sparseCooTensor' indices values size = sparseCooTensor indices values size defaultOpts
