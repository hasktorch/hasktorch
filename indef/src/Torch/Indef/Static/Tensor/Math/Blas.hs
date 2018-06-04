{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.Math.Blas where

import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math

import qualified Torch.Indef.Dynamic.Tensor.Math.Blas as Dynamic

blasOp
  :: (Dimensions4 d d' d'' d''')
  => (Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ())
  -> Tensor d -> HsReal -> Tensor d' -> HsReal -> Tensor d'' -> Tensor d''' -> IO ()
blasOp fn r a x b y z = fn (asDynamic r) a (asDynamic x) b (asDynamic y) (asDynamic z)

-- | beta * t + alpha * (src1 #> src2)
_addmv
  :: KnownDim2 r c
  => Tensor '[r] -> HsReal
  -> Tensor '[r] -> HsReal
  -> Tensor '[r, c]
  -> Tensor '[c] -> IO ()
_addmv  = blasOp Dynamic._addmv

-- | inplace 'addmv' (pure, dupable)
addmv
  :: (KnownDim2 r c)
  => HsReal -> Tensor '[r]
  -> HsReal -> Tensor '[r, c]
  -> Tensor '[c] -> Tensor '[r]
addmv a b c d e = unsafeDupablePerformIO $ withEmpty $ \r -> _addmv r a b c d e
{-# NOINLINE addmv #-}

-- | added simplified use of addmv: src1 #> src2
mv
  :: (KnownDim2 r c)
  => Tensor '[r, c] -> Tensor '[c] -> Tensor '[r]
mv m v = addmv 0 (constant 0) 1 m v

-- | inline version of 'mv'
(!*) :: (KnownDim2 r c) => Tensor '[r, c] -> Tensor '[c] -> Tensor '[r]
(!*) a b = mv a b

-- | only matrix-matrix multiplication:
-- https://github.com/torch/torch7/blob/aed31711c6b8846b8337a263a7f9f998697994e7/doc/maths.md#res-torchaddmmres-v1-m-v2-mat1-mat2
_addmm
  :: KnownDim3 a b c
  => Tensor '[a, c] -> HsReal
  -> Tensor '[a, c] -> HsReal
  -> Tensor '[a, b]
  -> Tensor '[b, c] -> IO ()
_addmm  = blasOp Dynamic._addmm

-- | pure version of '_addmm', matrix-matrix multiplication, adding a constant matrix
addmm
  :: KnownDim3 a b c
  => HsReal
  -> Tensor '[a, c]
  -> HsReal
  -> Tensor '[a, b]
  -> Tensor '[b, c]
  -> Tensor '[a, c]
addmm a m b x y = unsafePerformIO . withEmpty $ \r -> _addmm r a m b x y
{-# NOINLINE addmm #-}

-- | simplified wrapper of 'addmm'
--
-- FIXME: see if we can pass a null pointer in as the constant value (which might eliminate a noop linear pass).
mmult
  :: KnownDim3 a b c
  => Tensor '[a, b]
  -> Tensor '[b, c]
  -> Tensor '[a, c]
mmult x y = addmm 1 (constant 0) 1 x y

(!*!) :: (KnownDim3 a b c) => Tensor '[a, b] -> Tensor '[b, c] -> Tensor '[a, c]
(!*!) = mmult


-- outer product between a 1D tensor and a 1D tensor:
-- https://github.com/torch/torch7/blob/aed31711c6b8846b8337a263a7f9f998697994e7/doc/maths.md#res-torchaddrres-v1-mat-v2-vec1-vec2
--
-- res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
_addr
  :: KnownDim2 r c
  => Tensor '[r, c] -> HsReal
  -> Tensor '[r,c] -> HsReal
  -> Tensor '[r]
  -> Tensor '[c] -> IO ()
_addr r a x b y z = Dynamic._addr (asDynamic r) a (asDynamic x) b (asDynamic y) (asDynamic z)

-- pure version of 'addr'
addr
  :: (KnownDim2 r c)
  => HsReal -> Tensor '[r,c] -> HsReal -> Tensor '[r] -> Tensor '[c] -> IO (Tensor '[r, c])
addr a t b x y = withEmpty $ \r -> _addr r a t b x y

-- 'addr' with the parameters for an outer product filled in.
outer
  :: forall t r c . (KnownDim2 r c)
  => Tensor '[r] -> Tensor '[c] -> Tensor '[r, c]
outer v1 v2 = unsafePerformIO $ do
  t :: Tensor '[r, c] <- zerosLike
  addr 0 t 1 v1 v2
{-# NOINLINE outer #-}



-- | Batch matrix matrix product of matrices stored in batch1 and batch2, with a reduced add step
-- (all matrix multiplications get accumulated in a single place).
--
-- batch1 and batch2 must be 3D Tensors each containing the same number of matrices. If batch1
-- is a b × n × m Tensor, batch2 a b × m × p Tensor, res will be a n × p Tensor.
--
-- In other words,
--
--     res = (v1 * M) + (v2 * sum(batch1_i * batch2_i, i = 1, b))
--
_addbmm
  :: KnownDim4 n p b m
  => Tensor '[n, p] -> HsReal
  -> Tensor '[n, p] -> HsReal
  -> Tensor '[b, n, m] -> Tensor '[b, m, p] -> IO ()
_addbmm = blasOp Dynamic._addbmm

-- | Batch matrix matrix product of matrices stored in batch1 and batch2, with batch add.
--
-- batch1 and batch2 must be 3D Tensors each containing the same number of matrices. If batch1
-- is a b × n × m Tensor, batch2 a b × m × p Tensor, res will be a b × n × p Tensor.
--
-- In other words,
--
--     res_i = (v1 * M_i) + (v2 * batch1_i * batch2_i)
--
_baddbmm
  :: KnownDim4 n p b m
  => Tensor '[b, n, p] -> HsReal
  -> Tensor '[b, n, p] -> HsReal
  -> Tensor '[b, n, m] -> Tensor '[b, m, p] -> IO ()
_baddbmm = blasOp Dynamic._baddbmm


-- Performs the dot product between tensor1 and tensor2. The number of elements must match: both 
-- Tensors are seen as a 1D vector.
dot :: Dimensions2 d d' => Tensor d -> Tensor d' -> IO HsAccReal
dot a b = Dynamic.dot (asDynamic a) (asDynamic b)

(<.>)
  :: (Dimensions2 d d')
  => Tensor d
  -> Tensor d'
  -> HsAccReal
(<.>) a b = unsafePerformIO $ dot a b
{-# NOINLINE (<.>) #-}

