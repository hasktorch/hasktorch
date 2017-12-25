{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE LambdaCase #-}
module Torch.Core.Tensor.Raw
  ( dimFromRaw
  , dispRaw
  , fillRaw
  , fillLongRaw
  , fillRaw0
  , randInitRaw
  , tensorRaw
  , tensorLongRaw
  , toList
  , invlogit
  , TensorDoubleRaw
  , TensorLongRaw
  ) where

import Foreign (Ptr)
import Foreign.C.Types (CLLong, CLong, CDouble, CInt)
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr, mallocForeignPtrArray, newForeignPtr)
import Numeric (showGFloat)
import Numeric.Dimensions
import System.IO.Unsafe (unsafePerformIO)
import Control.Monad (forM_)

import GHC.Ptr (FunPtr)

import Torch.Core.Internal (onDims)
import Torch.Core.Tensor.Dim
import THTypes (CTHDoubleTensor, CTHGenerator)
import Torch.Core.Tensor.Types (TensorDoubleRaw, TensorLongRaw, (^.), _1, _2, _3, _4)

import qualified THDoubleTensor as T
import qualified THLongTensor as T
import qualified THDoubleTensorMath as M (c_THDoubleTensor_sigmoid, c_THDoubleTensor_fill)
import qualified THLongTensorMath as M (c_THLongTensor_fill)
import qualified THDoubleTensorRandom as R (c_THDoubleTensor_uniform)
import qualified THRandom as R (c_THGenerator_new)


-- | flatten a CTHDoubleTensor into a list
toList :: Ptr CTHDoubleTensor -> [CDouble]
toList tensor =
  case size of
    [] -> mempty
    [nx] ->
      T.c_THDoubleTensor_get1d tensor
        <$> range nx
    [nx, ny] ->
      T.c_THDoubleTensor_get2d tensor
        <$> range nx
        <*> range ny
    [nx, ny, nz] ->
      T.c_THDoubleTensor_get3d tensor
        <$> range nx
        <*> range ny
        <*> range nz
    [nx, ny, nz, nq] ->
      T.c_THDoubleTensor_get4d tensor
        <$> range nx
        <*> range ny
        <*> range nz
        <*> range nq
    _ -> undefined
  where
    size :: [Int]
    size = map (fromIntegral . T.c_THDoubleTensor_size tensor) [0 .. T.c_THDoubleTensor_nDimension tensor - 1]

    range :: Integral i => Int -> [i]
    range mx = [0 .. fromIntegral mx - 1]

-- |displaying raw tensor values
dispRaw :: Ptr CTHDoubleTensor -> IO ()
dispRaw tensor
  | (length sz) == 0 = putStrLn "Empty Tensor"
  | (length sz) == 1 = do
      putStrLn ""
      let indexes = [ fromIntegral idx :: CLLong
                    | idx <- [0..(sz !! 0 - 1)] ]
      putStr "[ "
      mapM_ (\idx -> putStr $
                     (showLim $ T.c_THDoubleTensor_get1d tensor idx) ++ " ")
        indexes
      putStrLn "]\n"
  | (length sz) == 2 = do
      putStrLn ""
      let pairs = [ ((fromIntegral r) :: CLLong,
                     (fromIntegral c) :: CLLong)
                  | r <- [0..(sz !! 0 - 1)], c <- [0..(sz !! 1 - 1)] ]
      putStr ("[ " :: String)
      mapM_ (\(r, c) -> do
                let val = T.c_THDoubleTensor_get2d tensor r c
                if c == fromIntegral (sz !! 1) - 1
                  then do
                  putStrLn (((showLim val) ++ " ]") :: String)
                  putStr (if (fromIntegral r :: Int) < (sz !! 0 - 1)
                          then "[ " :: String
                          else "")
                  else
                  putStr $ ((showLim val) ++ " " :: String)
            ) pairs
  | otherwise = putStrLn "Can't print this yet."
  where
    size :: Ptr CTHDoubleTensor -> [Int]
    size t =
      fmap f [0..maxdim]
      where
        maxdim = (T.c_THDoubleTensor_nDimension t) - 1
        f x = fromIntegral (T.c_THDoubleTensor_size t x) :: Int

    showLim :: RealFloat a => a -> String
    showLim x = showGFloat (Just 2) x ""

    sz :: [Int]
    sz = size tensor

-- |randomly initialize a tensor with uniform random values from a range
-- TODO - finish implementation to handle sizes correctly
randInitRaw
  :: Ptr CTHGenerator
  -> Dim (dims :: [k])
  -> CDouble
  -> CDouble
  -> IO TensorDoubleRaw
randInitRaw gen dims lower upper = do
  t <- tensorRaw dims 0.0
  R.c_THDoubleTensor_uniform t gen lower upper
  pure t


-- |Create a new (double) tensor of specified dimensions and fill it with 0
-- safe version
tensorRaw :: Dim (ns::[k]) -> Double -> IO TensorDoubleRaw
tensorRaw dims value = do
  newPtr <- go dims
  fillRaw value newPtr
  pure newPtr
  where
    go :: Dim (ns::[k]) -> IO TensorDoubleRaw
    go = onDims fromIntegral
      T.c_THDoubleTensor_new
      T.c_THDoubleTensor_newWithSize1d
      T.c_THDoubleTensor_newWithSize2d
      T.c_THDoubleTensor_newWithSize3d
      T.c_THDoubleTensor_newWithSize4d

-- |Create a new (Long) tensor of specified dimensions and fill it with 0
-- safe version
tensorLongRaw :: Dim (ns::[k]) -> Int -> IO TensorLongRaw
tensorLongRaw dims value = do
  newPtr <- go dims
  fillLongRaw value newPtr
  pure newPtr
  where
    go :: Dim (ns::[k]) -> IO TensorLongRaw
    go = onDims fromIntegral
      T.c_THLongTensor_new
      T.c_THLongTensor_newWithSize1d
      T.c_THLongTensor_newWithSize2d
      T.c_THLongTensor_newWithSize3d
      T.c_THLongTensor_newWithSize4d

-- |apply a tensor transforming function to a tensor
applyRaw
  :: (TensorDoubleRaw -> TensorDoubleRaw -> IO ())
  -> TensorDoubleRaw
  -> IO TensorDoubleRaw
applyRaw f t1 = do
  r_ <- T.c_THDoubleTensor_new
  f r_ t1
  pure r_

-- |apply inverse logit to all values of a tensor
invlogit :: TensorDoubleRaw -> IO TensorDoubleRaw
invlogit = applyRaw M.c_THDoubleTensor_sigmoid

-- |Dimensions of a raw tensor as a list
sizeRaw :: Ptr CTHDoubleTensor -> [Int]
sizeRaw t =
  fmap f [0..maxdim]
  where
    maxdim :: CInt
    maxdim = (T.c_THDoubleTensor_nDimension t) - 1

    f :: CInt -> Int
    f x = fromIntegral (T.c_THDoubleTensor_size t x)

-- |Dimensions of a raw tensor as a TensorDim value
dimFromRaw :: TensorDoubleRaw -> DimView
dimFromRaw raw =
  case length sz of
    0 -> D0
    1 -> D1 (at 0)
    2 -> D2 (at 0) (at 1)
    3 -> D3 (at 0) (at 1) (at 2)
    4 -> D4 (at 0) (at 1) (at 2) (at 3)
    5 -> D5 (at 0) (at 1) (at 2) (at 3) (at 5)
    _ -> undefined -- TODO - make this safe
  where
    sz :: [Int]
    sz = sizeRaw raw

    at :: Int -> Int
    at n = fromIntegral (sz !! n)

-- |Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
fillRaw :: Real a => a -> TensorDoubleRaw -> IO ()
fillRaw value = flip M.c_THDoubleTensor_fill (realToFrac value)

-- |Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
fillLongRaw :: Int -> TensorLongRaw -> IO ()
fillLongRaw value = flip M.c_THLongTensor_fill (fromIntegral value)

-- |Fill a raw Double tensor with 0.0
fillRaw0 :: TensorDoubleRaw -> IO TensorDoubleRaw
fillRaw0 tensor = fillRaw 0.0 tensor >> pure tensor

