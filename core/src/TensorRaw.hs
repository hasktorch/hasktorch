module TensorRaw (
  dispRaw,
  fillRaw,
  fillRaw0,
  randInitRaw,
  randInitRawTest,
  tensorRaw,

  TensorDoubleRaw,
  TensorLongRaw
  ) where

import Data.Maybe (fromJust)

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

-- import TensorTypes
import THTypes
import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorRandom
import THRandom

import TensorTypes

type TensorDoubleRaw = Ptr CTHDoubleTensor
type TensorLongRaw = Ptr CTHLongTensor

-- |displaying raw tensor values
dispRaw :: Ptr CTHDoubleTensor -> IO ()
dispRaw tensor
  | (length sz) == 0 = putStrLn "Empty Tensor"
  | (length sz) == 1 = do
      putStrLn ""
      let indexes = [ fromIntegral idx :: CLong
                    | idx <- [0..(sz !! 0 - 1)] ]
      putStr "[ "
      mapM_ (\idx -> putStr $
                     (showLim $ c_THDoubleTensor_get1d tensor idx) ++ " ")
        indexes
      putStrLn "]\n"
  | (length sz) == 2 = do
      putStrLn ""
      let pairs = [ ((fromIntegral r) :: CLong,
                     (fromIntegral c) :: CLong)
                  | r <- [0..(sz !! 0 - 1)], c <- [0..(sz !! 1 - 1)] ]
      putStr ("[ " :: String)
      mapM_ (\(r, c) -> do
                let val = c_THDoubleTensor_get2d tensor r c
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
    --size :: (Ptr CTHDoubleTensor) -> [Int]
    size t =
      fmap f [0..maxdim]
      where
        maxdim = (c_THDoubleTensor_nDimension t) - 1
        f x = fromIntegral (c_THDoubleTensor_size t x) :: Int
    -- showLim :: RealFloat a => a -> String
    showLim x = showGFloat (Just 2) x ""
    sz = size tensor

-- |randomly initialize a tensor with uniform random values from a range
-- TODO - finish implementation to handle sizes correctly
randInitRaw gen dims lower upper = do
  t <- tensorRaw dims 0.0
  c_THDoubleTensor_uniform t gen lower upper
  pure t

randInitRawTest = do
  gen <- c_THGenerator_new
  mapM_ (\_ -> dispRaw =<< (randInitRaw gen (D2 2 2) (-1.0) 3.0)) [0..10]

w2cl = fromIntegral

-- |Create a new (double) tensor of specified dimensions and fill it with 0
-- safe version
tensorRaw :: TensorDim Word -> Double -> IO TensorDoubleRaw
tensorRaw dims value = do
  newPtr <- go dims
  -- fillPtr <- fill0 newPtr
  fillRaw value newPtr
  pure newPtr
  where
    go D0 = c_THDoubleTensor_new
    go (D1 d1) = c_THDoubleTensor_newWithSize1d $ w2cl d1
    go (D2 d1 d2) = c_THDoubleTensor_newWithSize2d
                    (w2cl d1) (w2cl d2)
    go (D3 d1 d2 d3) = c_THDoubleTensor_newWithSize3d
                       (w2cl d1) (w2cl d2) (w2cl d3)
    go (D4 d1 d2 d3 d4) = c_THDoubleTensor_newWithSize4d
                          (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)


-- |apply a tensor transforming function to a tensor
applyRaw ::
  (TensorDoubleRaw -> TensorDoubleRaw -> IO ())
  -> TensorDoubleRaw -> IO TensorDoubleRaw
applyRaw f t1 = do
  r_ <- c_THDoubleTensor_new
  f r_ t1
  pure r_

-- |apply inverse logit to all values of a tensor
invlogit :: TensorDoubleRaw -> IO TensorDoubleRaw
invlogit = applyRaw c_THDoubleTensor_sigmoid

-- |Dimensions of a tensor as a list
sizeRaw :: (Ptr CTHDoubleTensor) -> [Int]
sizeRaw t =
  fmap f [0..maxdim]
  where
    maxdim = (c_THDoubleTensor_nDimension t) - 1
    f x = fromIntegral (c_THDoubleTensor_size t x) :: Int

-- |Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
fillRaw :: Real a => a -> TensorDoubleRaw -> IO ()
fillRaw value = (flip c_THDoubleTensor_fill) (realToFrac value)

-- |Fill a raw Double tensor with 0.0
fillRaw0 :: TensorDoubleRaw -> IO (TensorDoubleRaw)
fillRaw0 tensor = fillRaw 0.0 tensor >> pure tensor

testRaw = do
  tmp <- tensorRaw (D1 5) 25.0
  dispRaw tmp
  t2 <- invlogit =<< tensorRaw (D1 5) 25.0
  dispRaw t2
  dispRaw tmp
