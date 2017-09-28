{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

module TorchTensor (
  TensorDouble(..),
  TensorInt(..),
  apply,
  invlogit,
  disp,
  mvSimple,
  size,
  tensorNew,
  tensorFloatNew,
  tensorIntNew
  ) where

import Data.Maybe  (fromJust)
import Data.Monoid ((<>))
import Numeric (showGFloat)
import qualified Data.Text as T

import Foreign
import Foreign.C.Types
import THTypes

import THDoubleTensor
import THDoubleTensorMath
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

apply ::
  (TensorDouble -> TensorDouble -> IO ())
  -> TensorDouble -> IO TensorDouble
apply f t1 = do
  -- r_ <- fromJust $ tensorNew (size t1)
  r_ <- c_THDoubleTensor_new
  f r_ t1
  pure r_

apply2 ::
  (TensorDouble -> TensorDouble -> TensorDouble -> IO ())
  -> TensorDouble -> TensorDouble -> IO TensorDouble
apply2 f t1 t2 = do
  r_ <- c_THDoubleTensor_new
  f r_ t1 t2
  pure r_

apply3 ::
  (TensorDouble -> TensorDouble -> TensorDouble -> TensorDouble -> IO ())
  -> TensorDouble -> TensorDouble -> TensorDouble -> IO TensorDouble
apply3 f t1 t2 t3 = do
  r_ <- c_THDoubleTensor_new
  f r_ t1 t2 t3
  pure r_

invlogit :: TensorDouble -> IO TensorDouble
invlogit = apply c_THDoubleTensor_sigmoid

nrows tensor = (size tensor) !! 0

ncols tensor = (size tensor) !! 1

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

-- initialize values tensor = do
--   [(r, c) |
--     x <- [1..(nrows tensor)],
--     y <- [1..(ncols tensor)]]
--   mapM_ ((r, c) -> c_THDoubleTensor_set2d
--                    (fromIntegral r)
--                    (fromIntegral c)
--   where
--     idx = product . size $ tensor

-- TODO: need bindings to THStorage to use c_THDoubleTensor_resize
initialize values sz = do
  tensor <- c_THDoubleTensor_newWithSize1d nel
  mapM_
    (\(idx, value) -> c_THDoubleTensor_set1d tensor idx value)
    (zip [1..nel], values)
  pure tensor
  where
    nel = product sz

type TensorDouble = Ptr CTHDoubleTensor

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

tensorNew :: [Int] -> Maybe (IO (Ptr CTHDoubleTensor))
tensorNew dims
  | ndim == 0 = Just $ c_THDoubleTensor_new
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

tensorFloatNew :: [Int] -> Maybe (IO (Ptr CTHFloatTensor))
tensorFloatNew dims
  | ndim == 0 = Just c_THFloatTensor_new
  | ndim == 1 = Just $ (c_THFloatTensor_newWithSize1d $ head cdims) >>= fill
  | ndim == 2 = Just $ c_THFloatTensor_newWithSize2d (cdims !! 0) (cdims !! 1) >>= fill
  | ndim == 3 = Just $ (c_THFloatTensor_newWithSize3d
                        (cdims !! 0) (cdims !! 1) (cdims !! 2)) >>= fill
  | ndim == 4 = Just $ (c_THFloatTensor_newWithSize4d
                        (cdims !! 0) (cdims !! 1) (cdims !! 2) (cdims !! 3)
                        >>= fill)
  | otherwise = Nothing
  where
    ndim = length dims
    cdims = (\x -> (fromIntegral x) :: CLong) <$> dims
    create = (flip c_THFloatTensor_fill) 0.0
    fill x = create x >> pure x

tensorIntNew :: [Int] -> Maybe (IO (Ptr CTHIntTensor))
tensorIntNew dims
  | ndim == 0 = Just c_THIntTensor_new
  | ndim == 1 = Just $ (c_THIntTensor_newWithSize1d $ head cdims) >>= fill
  | ndim == 2 = Just $ c_THIntTensor_newWithSize2d (cdims !! 0) (cdims !! 1) >>= fill
  | ndim == 3 = Just $ (c_THIntTensor_newWithSize3d
                        (cdims !! 0) (cdims !! 1) (cdims !! 2)) >>= fill
  | ndim == 4 = Just $ (c_THIntTensor_newWithSize4d
                        (cdims !! 0) (cdims !! 1) (cdims !! 2) (cdims !! 3)
                        >>= fill)
  | otherwise = Nothing
  where
    ndim = length dims
    cdims = (\x -> (fromIntegral x) :: CLong) <$> dims
    create = (flip c_THIntTensor_fill) 0
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
