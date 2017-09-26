{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

module TorchTensor (
  TensorDouble(..),
  TensorInt(..),
  disp,
  size,
  tensorNew
  ) where

import Data.Maybe  (fromJust)
import Numeric (showGFloat)
import qualified Data.Text as T

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
TODO generalize these to multiple tensor types

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

-- str :: Ptr CTHDoubleTensor -> Text
-- str tensor
--   | (length sz) == 0 = "Empty Tensor"
--   | (length sz) == 1 =
--       let indexes = [ fromIntegral idx :: CLong
--                     | idx <- [0..(sz !! 0 - 1)] ] in
--         ("[ " <>
--          foldr (\idx -> (show $ c_THDoubleTensor_get1d tensor idx) <> " ")
--         indexes
--       putStrLn "]"
--   | (length sz) == 2 = do
--       let pairs = [ ((fromIntegral r) :: CLong,
--                      (fromIntegral c) :: CLong)
--                   | r <- [0..(sz !! 0 - 1)], c <- [0..(sz !! 1 - 1)] ]
--       ("[ " :: Text)
--       mapM_ (\(r, c) -> do
--                 let val = c_THDoubleTensor_get2d tensor r c
--                 if c == fromIntegral (sz !! 1) - 1
--                   then do
--                   putStrLn (((show val) ++ " ]") :: String)
--                   putStr (if (fromIntegral r :: Int) < (sz !! 0 - 1)
--                           then "[ " :: String
--                           else "")
--                   else
--                   putStr $ ((show val) ++ " " :: String)
--             ) pairs
--   | otherwise = putStrLn "Can't print this yet."
--   where
--     sz = size tensor

showLim x = showGFloat (Just 2) x ""

disp :: Ptr CTHDoubleTensor -> IO ()
disp tensor
  | (length sz) == 0 = putStrLn "Empty Tensor"
  | (length sz) == 1 = do
      let indexes = [ fromIntegral idx :: CLong
                    | idx <- [0..(sz !! 0 - 1)] ]
      putStr "[ "
      mapM_ (\idx -> putStr $
                     (showLim $ c_THDoubleTensor_get1d tensor idx) ++ " ")
        indexes
      putStrLn "]"
  | (length sz) == 2 = do
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
    sz = size tensor

tensorNew :: [Int] -> Maybe (IO (Ptr CTHDoubleTensor))
tensorNew dims
  | ndim == 0 = Just c_THDoubleTensor_new
  | ndim == 1 = Just $ (c_THDoubleTensor_newWithSize1d $ head cdims) >>= fill
  | ndim == 2 = Just $ c_THDoubleTensor_newWithSize2d (cdims !! 0) (cdims !! 1) >>= fill
  | ndim == 3 = Just $ (c_THDoubleTensor_newWithSize3d
                        (cdims !! 0) (cdims !! 1) (cdims !! 2)) >>= fill
  | ndim == 4 = Just $ (c_THDoubleTensor_newWithSize4d
                        (cdims !! 0) (cdims !! 1) (cdims !! 2) (cdims !! 3)
                        >>= fill)
  | otherwise = Nothing
  where
    ndim = length dims
    cdims = (\x -> (fromIntegral x) :: CLong) <$> dims
    create = (flip c_THDoubleTensor_fill) 0.0
    fill x = create x >> pure x

size :: (Ptr CTHDoubleTensor) -> [Int]
size t =
  fmap f [0..maxdim]
  where
    maxdim = (c_THDoubleTensor_nDimension t) - 1
    f x = fromIntegral (c_THDoubleTensor_size t x) :: Int

main = do
  disp =<< (fromJust $ tensorNew [4,3])
  putStrLn "Done"
