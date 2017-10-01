{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

module TorchTensor (
  TensorDim(..),
  TensorDouble(..),
  apply,
  disp,
  fill,
  fill0,
  mvSimple,
  (#>),
  size,
  tensorNew,
  ) where

import Data.Maybe  (fromJust)
import Data.Monoid ((<>))
import Data.Word
import Numeric (showGFloat)
import qualified Data.Text as T

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)

import System.IO.Unsafe (unsafePerformIO)

import THTypes
import THRandom
import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorRandom
import THFloatTensor
import THFloatTensorMath
import THIntTensor
import THIntTensorMath

{-

Experimental abstracted interfaces into tensor
TODO generalize these to multiple tensor types

-}

import THTypes
import THDoubleStorage
import THDoubleTensor

type TensorDouble = Ptr CTHDoubleTensor

data TensorDim a =
  D0
  | D1 { d1_1 :: a }
  | D2 { d2_1 :: a, d2_2 :: a }
  | D3 { d3_1 :: a, d3_2 :: a, d3_3 :: a }
  | D4 { d4_1 :: a, d4_2 :: a, d4_3 :: a, d4_4 :: a }
  deriving (Eq, Show)

instance Functor TensorDim where
  fmap f D0 = D0
  fmap f (D1 d1) = D1 (f d1)
  fmap f (D2 d1 d2) = D2 (f d1) (f d2)
  fmap f (D3 d1 d2 d3) = D3 (f d1) (f d2) (f d3)
  fmap f (D4 d1 d2 d3 d4) = D4 (f d1) (f d2) (f d3) (f d4)

data TensorDouble_ = TensorDouble_ {
  tdTensor :: !(ForeignPtr CTHDoubleTensor)
  } deriving (Eq, Show)

-- |apply a tensor transforming function to a tensor
apply ::
  (TensorDouble -> TensorDouble -> IO ())
  -> TensorDouble -> IO TensorDouble
apply f t1 = do
  -- r_ <- fromJust $ tensorNew (size t1)
  r_ <- c_THDoubleTensor_new
  f r_ t1
  pure r_

-- TODO: need bindings to THStorage to use c_THDoubleTensor_resize
initialize values sz = do
  tensor <- c_THDoubleTensor_newWithSize1d nel
  mapM_
    (\(idx, value) -> c_THDoubleTensor_set1d tensor idx value)
    (zip [1..nel], values)
  pure tensor
  where
    nel = product sz

-- |matrix vector multiplication, no error checking for now
-- |tag: unsafe
-- TODO - determine how to deal with resource allocation
(#>) :: TensorDouble -> TensorDouble -> TensorDouble
mat #> vec = unsafePerformIO $ do
  res <- fromJust $ tensorNew $ [nrows mat]
  c_THDoubleTensor_addmv res 1.0 res 1.0 mat vec
  pure res

-- |simplified matrix vector multiplication
mvSimple :: TensorDouble -> TensorDouble -> IO TensorDouble
mvSimple mat vec = do
  res <- fromJust $ tensorNew $ [nrows mat]
  zero <- fromJust $ tensorNew $ [nrows mat]
  print $ "dimension check matrix:" <>
    show (c_THDoubleTensor_nDimension mat == 2)
  print $ "dimension check vector:" <>
    show (c_THDoubleTensor_nDimension vec == 1)
  c_THDoubleTensor_addmv res 1.0 zero 1.0 mat vec
  pure res

-- |number of rows of a tensor (unsafe)
nrows tensor = (size tensor) !! 0

-- |number of cols of a tensor (unsafe)
ncols tensor = (size tensor) !! 1

-- |Show a real value with limited precision (convenience function)
showLim :: RealFloat a => a -> String
showLim x = showGFloat (Just 2) x ""

-- |displaying tensor values
disp :: Ptr CTHDoubleTensor -> IO ()
disp tensor
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
      putStrLn ""
  | otherwise = putStrLn "Can't print this yet."
  where
    sz = size tensor

-- |Dimensions of a tensor as a list
size :: (Ptr CTHDoubleTensor) -> [Int]
size t =
  fmap f [0..maxdim]
  where
    maxdim = (c_THDoubleTensor_nDimension t) - 1
    f x = fromIntegral (c_THDoubleTensor_size t x) :: Int

-- |Word to CLong conversion
w2cl :: Word -> CLong
w2cl = fromIntegral

-- |Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
fill :: Real a => a -> Ptr CTHDoubleTensor -> IO ()
fill value = (flip c_THDoubleTensor_fill) (realToFrac value)

-- |Fill a raw Double tensor with 0.0
fill0 :: Ptr CTHDoubleTensor -> IO (Ptr CTHDoubleTensor)
fill0 tensor = fill 0.0 tensor >> pure tensor

-- |Create a new (double) tensor of specified dimensions and fill it with 0
tensorNew :: [Int] -> Maybe (IO TensorDouble)
tensorNew dims
  | ndim == 0 = Just $ c_THDoubleTensor_new
  | ndim == 1 = Just $ (c_THDoubleTensor_newWithSize1d $ head cdims) >>= fill0
  | ndim == 2 = Just $ c_THDoubleTensor_newWithSize2d (cdims !! 0) (cdims !! 1) >>= fill0
  | ndim == 3 = Just $ (c_THDoubleTensor_newWithSize3d
                        (cdims !! 0) (cdims !! 1) (cdims !! 2)) >>= fill0
  | ndim == 4 = Just $ (c_THDoubleTensor_newWithSize4d
                        (cdims !! 0) (cdims !! 1) (cdims !! 2) (cdims !! 3)
                        >>= fill0)
  | otherwise = Nothing
  where
    ndim = length dims
    cdims = (\x -> (fromIntegral x) :: CLong) <$> dims

main = do
  -- disp =<< (fromJust $ tensorNew [4,3])
  putStrLn "Done"
