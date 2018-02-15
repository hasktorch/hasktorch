{-# LANGUAGE LambdaCase #-}

module PlaygroundRawUtils where

import Foreign
import Foreign.C.Types
import Foreign.ForeignPtr (ForeignPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import THTypes
import qualified THDoubleTensorMath as DM (c_fill)
import qualified THLongTensorMath as LM (c_fill)
import THDoubleTensor as DT
import qualified THLongTensor as LT

type TensorDoubleRaw = Ptr CTHDoubleTensor
type TensorLongRaw = Ptr CTHLongTensor

w2cll :: Word -> CLLong
w2cll = fromIntegral

-- | simple helper to clean up common pattern matching on TensorDim
onDims
  :: (a0 -> a)
  -> b
  -> ( a -> b )
  -> ( a -> a -> b )
  -> ( a -> a -> a -> b )
  -> ( a -> a -> a -> a -> b )
  -> TensorDim a0
  -> b
onDims ap f0 f1 f2 f3 f4 = \case
  D0 -> f0
  D1 d1_ -> f1 (ap d1_)
  D2 (d1_, d2_) -> f2 (ap d1_) (ap d2_)
  D3 (d1_, d2_, d3_) -> f3 (ap d1_) (ap d2_) (ap d3_)
  D4 (d1_, d2_, d3_, d4_) -> f4 (ap d1_) (ap d2_) (ap d3_) (ap d4_)

data TensorDim a =
  D0
  | D1 { d1 :: a }
  | D2 { d2 :: (a, a) }
  | D3 { d3 :: (a, a, a) }
  | D4 { d4 :: (a, a, a, a) }
  deriving (Eq, Show)

data TensorDouble = TensorDouble {
  tdTensor :: !(ForeignPtr CTHDoubleTensor),
  tdDim :: !(TensorDim Word)
  } deriving (Eq, Show)

-- |Dimensions of a raw tensor as a list
sizeRaw :: Ptr CTHDoubleTensor -> [Int]
sizeRaw t =
  fmap f [0..maxdim]
  where
    maxdim :: CInt
    maxdim = unsafePerformIO (DT.c_nDimension t) - 1

    f :: CInt -> Int
    f x = fromIntegral (unsafePerformIO $ DT.c_size t x)

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

-- |Create a new (Long) tensor of specified dimensions and fill it with 0
-- safe version
tensorLongRaw :: TensorDim Word -> Int -> IO TensorLongRaw
tensorLongRaw dims value = do
  newPtr <- go dims
  fillLongRaw value newPtr
  pure newPtr
  where
    go :: TensorDim Word -> IO TensorLongRaw
    go = onDims w2cll
      LT.c_new
      LT.c_newWithSize1d
      LT.c_newWithSize2d
      LT.c_newWithSize3d
      LT.c_newWithSize4d

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
      DT.c_new
      DT.c_newWithSize1d
      DT.c_newWithSize2d
      DT.c_newWithSize3d
      DT.c_newWithSize4d

-- |Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
fillRaw :: Real a => a -> TensorDoubleRaw -> IO ()
fillRaw value = (flip DM.c_fill) (realToFrac value)

-- |Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
fillLongRaw :: Int -> TensorLongRaw -> IO ()
fillLongRaw value = (flip LM.c_fill) (fromIntegral value)

-- |Fill a raw Double tensor with 0.0
fillRaw0 :: TensorDoubleRaw -> IO ()
fillRaw0 tensor = fillRaw (0.0 :: Double) tensor >> pure ()

-- |displaying raw tensor values
dispRaw :: Ptr CTHDoubleTensor -> IO ()
dispRaw tensor
  | (length sz) == 0 = putStrLn "Empty Tensor"
  | (length sz) == 1 = do
      putStrLn ""
      let indexes = [ fromIntegral idx :: CLLong
                    | idx <- [0..(sz !! 0 - 1)] ]
      putStr "[ "
      mapM_ (\idx -> do
        x <- DT.c_get1d tensor idx
        putStr (showLim x ++ " "))
        indexes
      putStrLn "]\n"
  | (length sz) == 2 = do
      putStrLn ""
      let pairs = [ ((fromIntegral r) :: CLLong,
                     (fromIntegral c) :: CLLong)
                  | r <- [0..(sz !! 0 - 1)], c <- [0..(sz !! 1 - 1)] ]
      putStr ("[ " :: String)
      mapM_ (\(r, c) -> do
                val <- DT.c_get2d tensor r c
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
        maxdim = unsafePerformIO (DT.c_nDimension t) - 1
        f x = fromIntegral (unsafePerformIO $ DT.c_size t x) :: Int

    showLim :: RealFloat a => a -> String
    showLim x = showGFloat (Just 2) x ""

    sz :: [Int]
    sz = size tensor


