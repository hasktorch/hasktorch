{-# LANGUAGE LambdaCase #-}
module Torch.Core.Tensor.Raw
  ( dimFromRaw
  , dispRaw
  , fillRaw
  , fillRaw0
  , randInitRaw
  , tensorRaw
  , toList
  , invlogit
  , TensorDoubleRaw
  , TensorLongRaw
  ) where

import Foreign (Ptr)
import Foreign.C.Types (CLLong, CLong, CDouble, CInt)
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr, mallocForeignPtrArray, newForeignPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)
import Control.Monad (forM_)

import GHC.Ptr (FunPtr)

import Torch.Core.Internal (w2cll, onDims)
import Torch.Core.Tensor.Index (TIdx(..))
import THTypes (CTHDoubleTensor, CTHGenerator)
import Torch.Core.Tensor.Types (TensorDim(..), TensorDoubleRaw, TensorLongRaw, (^.), _1, _2, _3, _4)

import qualified THDoubleTensor as T
import qualified THDoubleTensorMath as M (c_THDoubleTensor_sigmoid, c_THDoubleTensor_fill)
import qualified THDoubleTensorRandom as R (c_THDoubleTensor_uniform)
import qualified THRandom as R (c_THGenerator_new)


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

    indexes :: [TIdx CLLong]
    indexes = idx size

    range :: Integral i => Int -> [i]
    range mx = [0 .. fromIntegral mx - 1]

    idx :: [Int] -> [TIdx CLLong]
    idx = \case
      []               -> [I0]
      [nx]             -> [I1 x | x <- range nx ]
      [nx, ny]         -> [I2 x y | x <- range nx, y <- range ny ]
      [nx, ny, nz]     -> [I3 x y z | x <- range nx, y <- range ny, z <- range nz]
      [nx, ny, nz, nq] -> [I4 x y z q | x <- range nx, y <- range ny, z <- range nz, q <- range nq]
      _ -> error "should not be run"



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
  -> TensorDim Word
  -> CDouble
  -> CDouble
  -> IO TensorDoubleRaw
randInitRaw gen dims lower upper = do
  t <- tensorRaw dims 0.0
  R.c_THDoubleTensor_uniform t gen lower upper
  pure t


-- |Create a new (double) tensor of specified dimensions and fill it with 0
-- safe version
tensorRaw :: TensorDim Word -> Double -> IO TensorDoubleRaw
tensorRaw dims value = do
  newPtr <- go dims
  fillRaw value newPtr
  pure newPtr
  where
    go :: TensorDim Word -> IO TensorDoubleRaw
    go = onDims w2cll
      T.c_THDoubleTensor_new
      T.c_THDoubleTensor_newWithSize1d
      T.c_THDoubleTensor_newWithSize2d
      T.c_THDoubleTensor_newWithSize3d
      T.c_THDoubleTensor_newWithSize4d


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
dimFromRaw :: TensorDoubleRaw -> TensorDim Word
dimFromRaw raw =
  case (length sz) of 0 -> D0
                      1 -> D1 (getN 0)
                      2 -> D2 ((getN 0), (getN 1))
                      3 -> D3 ((getN 0), (getN 1), (getN 2))
                      4 -> D4 ((getN 0), (getN 1), (getN 2), (getN 3))
                      _ -> undefined -- TODO - make this safe
  where
    sz :: [Int]
    sz = sizeRaw raw

    getN :: Int -> Word
    getN n = fromIntegral (sz !! n)

-- |Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
fillRaw :: Real a => a -> TensorDoubleRaw -> IO ()
fillRaw value = (flip M.c_THDoubleTensor_fill) (realToFrac value)

-- |Fill a raw Double tensor with 0.0
fillRaw0 :: TensorDoubleRaw -> IO TensorDoubleRaw
fillRaw0 tensor = fillRaw 0.0 tensor >> pure tensor

