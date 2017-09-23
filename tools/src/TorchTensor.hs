{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

module TorchTensor (
  TensorDouble(..),
  TensorInt(..),
  size
  ) where

import Foreign
import Foreign.C.Types
import THTypes

import THDoubleTensor
import THDoubleTensorMath
import THIntTensor
import THIntTensorMath
-- import THDoubleTensorRandom

{-

Experimental abstracted interfaces into tensor

-}

import THTypes
import THDoubleStorage
import THDoubleTensor

data TensorDouble = TensorDouble {
  val_double :: Ptr CTHDoubleTensor
  } deriving (Eq, Show)

data TensorInt = TensorInt {
  val_int :: Ptr CTHIntTensor
  } deriving (Eq, Show)

-- TODO generalize these to multiple tensor types

-- tensorNew :: [Int] -> Maybe TensorDouble
tensorNew dims
  | ndim == 0 = Just c_THDoubleTensor_new
  | ndim == 1 = Just $ c_THDoubleTensor_newWithSize1d $ head dims
  | ndim == 2 = Just $ c_THDoubleTensor_newWithSize2d (cdims !! 0) (cdims !! 1)
  | ndim == 3 = Just $ c_THDoubleTensor_newWithSize3d (cdims !! 0) (cdims !! 1) (cdims !! 2)
  | ndim == 4 = Just $ c_THDoubleTensor_newWithSize4d (cdims !! 0) (cdims !! 1) (cdims !! 2) (cdims !! 3)
  | otherwise = Nothing
  where
    ndim = length dims
    cdims = (\x -> (fromIntegral x) :: CLong) <$> dims

size :: (Ptr CTHDoubleTensor) -> [Int]
size t =
  fmap f [0..maxdim]
  where
    maxdim = (c_THDoubleTensor_nDimension t) - 1
    f x = fromIntegral (c_THDoubleTensor_size t x) :: Int

main = do
  putStrLn "Done"
