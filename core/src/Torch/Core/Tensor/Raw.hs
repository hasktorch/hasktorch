{-# LANGUAGE LambdaCase #-}
module Torch.Core.Tensor.Raw where

import Foreign (Ptr)
import Foreign.C.Types (CLong, CDouble, CInt)
import Torch.Core.Tensor.Index (TIdx(..))
import THTypes (CTHDoubleTensor, CTHGenerator)
import TensorTypes (TensorDim(..), (^.), _1, _2, _3, _4)

import qualified THDoubleTensor as T


-- | flatten a CTHDoubleTensor into a list
toList :: Ptr CTHDoubleTensor -> [CDouble]
toList tensor =
  case length size of
    0 -> mempty
    1 -> fmap (\t -> T.c_THDoubleTensor_get1d tensor (t ^. _1)) indexes
    2 -> fmap (\t -> T.c_THDoubleTensor_get2d tensor (t ^. _1) (t ^. _2)) indexes
    3 -> fmap (\t -> T.c_THDoubleTensor_get3d tensor (t ^. _1) (t ^. _2) (t ^. _3)) indexes
    4 -> fmap (\t -> T.c_THDoubleTensor_get4d tensor (t ^. _1) (t ^. _2) (t ^. _3) (t ^. _4)) indexes
    _ -> undefined
  where
    size :: [Int]
    size = fmap (fromIntegral . T.c_THDoubleTensor_size tensor) [0 .. T.c_THDoubleTensor_nDimension tensor - 1]

    indexes :: [TIdx CLong]
    indexes = idx size

    range :: Integral i => Int -> [i]
    range mx = [0 .. fromIntegral mx - 1]

    idx :: [Int] -> [TIdx CLong]
    idx = \case
      []               -> [I0]
      [nx]             -> [I1 x | x <- range nx ]
      [nx, ny]         -> [I2 x y | x <- range nx, y <- range ny ]
      [nx, ny, nz]     -> [I3 x y z | x <- range nx, y <- range ny, z <- range nz]
      [nx, ny, nz, nq] -> [I4 x y z q | x <- range nx, y <- range ny, z <- range nz, q <- range nq]
      _ -> error "should not be run"


