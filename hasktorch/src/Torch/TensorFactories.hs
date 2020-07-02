{-# LANGUAGE FlexibleContexts #-}

module Torch.TensorFactories where

import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Type.TensorOptions as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.Internal.Managed.TensorFactories as LibTorch
import qualified Torch.Internal.Managed.Autograd as LibTorch
import Torch.Internal.Managed.Cast
import Torch.Internal.Class (Castable(..))
import Torch.Internal.Cast

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

-- | Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.
ones 
  :: [Int] -- ^ sequence of integers defining the shape of the output tensor.
  -> TensorOptions -- ^ configures the data type, device, layout and other properties of the resulting tensor.
  -> Tensor -- ^ output
ones = mkFactoryUnsafe LibTorch.ones_lo

-- TODO - ones_like from Native.hs is redundant with this

-- | Returns a tensor filled with the scalar value 1, with the same size as input tensor
onesLike 
  :: Tensor -- ^ input
  -> Tensor -- ^ output
onesLike self = unsafePerformIO $ (cast1 ATen.ones_like_t) self

-- | Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
zeros 
  :: [Int] -- ^ sequence of integers defining the shape of the output tensor.
  -> TensorOptions -- ^ configures the data type, device, layout and other properties of the resulting tensor.
  -> Tensor -- ^ output
zeros = mkFactoryUnsafe LibTorch.zeros_lo

-- | Returns a tensor filled with the scalar value 0, with the same size as input tensor
zerosLike 
  :: Tensor -- ^ input
  -> Tensor -- ^ output
zerosLike self = unsafePerformIO $ (cast1 ATen.zeros_like_t) self

-- | Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1)
randIO 
  :: [Int] -- ^ sequence of integers defining the shape of the output tensor.
  -> TensorOptions -- ^ configures the data type, device, layout and other properties of the resulting tensor.
  -> IO Tensor -- ^ output
randIO = mkFactory LibTorch.rand_lo

-- | Returns a tensor filled with random numbers from a standard normal distribution.
randnIO 
  :: [Int] -- ^ sequence of integers defining the shape of the output tensor.
  -> TensorOptions -- ^ configures the data type, device, layout and other properties of the resulting tensor.
  -> IO Tensor -- ^ output
randnIO = mkFactory LibTorch.randn_lo

-- | Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
randintIO 
  :: Int -- ^ lowest integer to be drawn from the distribution. Default: 0.
  -> Int -- ^ one above the highest integer to be drawn from the distribution.
  -> [Int] -- ^ the shape of the output tensor.
  -> TensorOptions -- ^ configures the data type, device, layout and other properties of the resulting tensor.
  -> IO Tensor -- ^ output
randintIO low high = mkFactory (LibTorch.randint_lllo (fromIntegral low) (fromIntegral high))

-- | Returns a tensor with the same size as input that is filled with random numbers from standard normal distribution. 
randnLikeIO 
  :: Tensor -- ^ input
  -> IO Tensor -- ^ output
randnLikeIO = cast1 ATen.randn_like_t

-- | Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0,1).
randLikeIO 
  :: Tensor -- ^ input 
  -> TensorOptions -- ^ configures the data type, device, layout and other properties of the resulting tensor.
  -> IO Tensor -- ^ output
randLikeIO input opt = cast2 LibTorch.rand_like_to input opt

fullLike :: Tensor -> Float -> TensorOptions -> IO Tensor
fullLike input _fill_value opt = cast3 LibTorch.full_like_tso input _fill_value opt

onesWithDimnames :: [(Int,Dimname)] -> TensorOptions -> Tensor
onesWithDimnames = mkFactoryUnsafeWithDimnames LibTorch.ones_lNo

zerosWithDimnames :: [(Int,Dimname)] -> TensorOptions -> Tensor
zerosWithDimnames = mkFactoryUnsafeWithDimnames LibTorch.zeros_lNo

randWithDimnames :: [(Int,Dimname)] -> TensorOptions -> IO Tensor
randWithDimnames = mkFactoryWithDimnames LibTorch.rand_lNo

randnWithDimnames :: [(Int,Dimname)] -> TensorOptions -> IO Tensor
randnWithDimnames = mkFactoryWithDimnames LibTorch.randn_lNo

-- | Returns a one-dimensional tensor of steps equally spaced points between start and end.
linspace 
  :: (Scalar a, Scalar b)
  => a -- ^ @start@
  -> b -- ^ @end@ 
  -> Int -- ^ @steps@
  -> TensorOptions -- ^ configures the data type, device, layout and other properties of the resulting tensor.
  -> Tensor -- ^ output
linspace start end steps opts = unsafePerformIO $ (cast4 LibTorch.linspace_sslo) start end steps opts

logspace :: (Scalar a, Scalar b) => a -> b -> Int -> Double -> TensorOptions -> Tensor
logspace start end steps base opts = unsafePerformIO $ (cast5 LibTorch.logspace_ssldo) start end steps base opts

-- https://github.com/hasktorch/ffi-experimental/pull/57#discussion_r301062033
-- empty :: [Int] -> TensorOptions -> Tensor
-- empty = mkFactoryUnsafe LibTorch.empty_lo

eyeSquare :: Int -> TensorOptions -> Tensor
eyeSquare dim opts = unsafePerformIO $ (cast2 LibTorch.eye_lo) dim opts

-- | Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
eye 
  :: Int -- ^ the number of rows
  -> Int -- ^ the number of columns
  -> TensorOptions -- ^ configures the data type, device, layout and other properties of the resulting tensor. 
  -> Tensor -- ^ output
eye nrows ncols opts = unsafePerformIO $ (cast3 LibTorch.eye_llo) nrows ncols opts

-- | Returns a tensor of given size filled with fill_value.
full 
  :: Scalar a 
  => [Int] -- ^ the shape of the output tensor. 
  -> a -- ^ the number to fill the output tensor with
  -> TensorOptions -- ^ configures the data type, device, layout and other properties of the resulting tensor. 
  -> Tensor -- ^ output
full shape value opts = unsafePerformIO $ (cast3 LibTorch.full_lso) shape value opts

-- | Constructs a sparse tensors in COO(rdinate) format with non-zero elements at the given indices with the given values.
sparseCooTensor 
  :: Tensor -- ^ The indices are the coordinates of the non-zero values in the matrix
  -> Tensor -- ^ Initial values for the tensor.
  -> [Int] -- ^ the shape of the output tensor. 
  -> TensorOptions -- ^  
  -> Tensor -- ^ output
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

randIO' :: [Int] -> IO Tensor
randIO' = mkDefaultFactory randIO

randnIO' :: [Int] -> IO Tensor
randnIO' = mkDefaultFactory randnIO

randintIO' :: Int -> Int -> [Int] -> IO Tensor
randintIO' low high = mkDefaultFactory (randintIO low high)

randLikeIO' :: Tensor -> IO Tensor
randLikeIO' t = randLikeIO t defaultOpts

bernoulliIO'
  :: Tensor
  -> IO Tensor
bernoulliIO' t =
  (cast1 ATen.bernoulli_t) t

bernoulliIO
  :: Tensor
  -> Double
  -> IO Tensor
bernoulliIO t p =
  (cast2 ATen.bernoulli_td) t p

poissonIO
  :: Tensor
  -> IO Tensor
poissonIO t =
  (cast1 ATen.poisson_t) t

multinomialIO'
  :: Tensor
  -> Int
  -> IO Tensor
multinomialIO' t num_samples =
  (cast2 ATen.multinomial_tl) t num_samples

multinomialIO
  :: Tensor
  -> Int
  -> Bool
  -> IO Tensor
multinomialIO t num_samples replacement =
  (cast3 ATen.multinomial_tlb) t num_samples replacement

normalIO'
  :: Tensor
  -> IO Tensor
normalIO' _mean =
  (cast1 ATen.normal_t) _mean

normalIO
  :: Tensor
  -> Tensor
  -> IO Tensor
normalIO _mean _std =
  (cast2 ATen.normal_tt) _mean _std

normalScalarIO
  :: Tensor
  -> Double
  -> IO Tensor
normalScalarIO _mean _std =
  (cast2 ATen.normal_td) _mean _std

normalScalarIO'
  :: Double
  -> Tensor
  -> IO Tensor
normalScalarIO' _mean _std =
  (cast2 ATen.normal_dt) _mean _std

normalWithSizeIO
  :: Double
  -> Double
  -> Int
  -> IO Tensor
normalWithSizeIO _mean _std _size =
  (cast3 ATen.normal_ddl) _mean _std _size

rreluIO'''
  :: Tensor
  -> IO Tensor
rreluIO''' t =
  (cast1 ATen.rrelu_t) t

rreluIO''
  :: Scalar a 
  => Tensor
  -> a
  -> IO Tensor
rreluIO'' t _upper =
  (cast2 ATen.rrelu_ts) t _upper

rreluIO'
  :: Scalar a 
  => Tensor
  -> a
  -> a
  -> IO Tensor
rreluIO' t _lower _upper =
  (cast3 ATen.rrelu_tss) t _lower _upper

rreluIO
  :: Scalar a 
  => Tensor
  -> a
  -> a
  -> Bool
  -> IO Tensor
rreluIO t _lower _upper _training =
  (cast4 ATen.rrelu_tssb) t _lower _upper _training

rreluWithNoiseIO'''
  :: Tensor
  -> Tensor
  -> IO Tensor
rreluWithNoiseIO''' t _noise =
  (cast2 ATen.rrelu_with_noise_tt) t _noise

rreluWithNoiseIO''
  :: Scalar a 
  => Tensor
  -> Tensor
  -> a
  -> IO Tensor
rreluWithNoiseIO'' t _noise _upper =
  (cast3 ATen.rrelu_with_noise_tts) t _noise _upper

rreluWithNoiseIO'
  :: Scalar a 
  => Tensor
  -> Tensor
  -> a
  -> a
  -> IO Tensor
rreluWithNoiseIO' t _noise _lower _upper =
  (cast4 ATen.rrelu_with_noise_ttss) t _noise _lower _upper

rreluWithNoiseIO
  :: Scalar a 
  => Tensor
  -> Tensor
  -> a
  -> a
  -> Bool
  -> IO Tensor
rreluWithNoiseIO t _noise _lower _upper _training =
  (cast5 ATen.rrelu_with_noise_ttssb) t _noise _lower _upper _training

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
