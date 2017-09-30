{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

module TorchTensor (
  TensorDim(..),
  TensorByte(..),
  TensorDouble(..),
  TensorFloat(..),
  TensorInt(..),
  apply,
  apply2,
  apply3,
  disp,
  invlogit,
  mvSimple,
  (#>),
  randInit,
  size,
  tensorNew,
  tensorNew_,
  tensorFloatNew,
  tensorIntNew
  ) where

import Data.Maybe  (fromJust)
import Data.Monoid ((<>))
import Data.Word
import Numeric (showGFloat)
import qualified Data.Text as T

import Foreign
import Foreign.C.Types
import Foreign.ForeignPtr
import Foreign.Ptr
import GHC.Ptr

-- import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
--                            newForeignPtr )
import THTypes

import System.IO.Unsafe

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

type TensorByte = Ptr CTHByteTensor
type TensorDouble = Ptr CTHDoubleTensor
type TensorFloat = Ptr CTHFloatTensor
type TensorInt = Ptr CTHIntTensor

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

data TensorDouble_ =
  TensorDouble_ !(ForeignPtr (Ptr CTHDoubleTensor))
  deriving (Eq, Show)

foo :: (Ptr ()) -> IO ()
foo = undefined

-- |apply a tensor transforming function to a tensor
apply ::
  (TensorDouble -> TensorDouble -> IO ())
  -> TensorDouble -> IO TensorDouble
apply f t1 = do
  -- r_ <- fromJust $ tensorNew (size t1)
  r_ <- c_THDoubleTensor_new
  f r_ t1
  pure r_

-- |apply an operation on 2 tensors
apply2 ::
  (TensorDouble -> TensorDouble -> TensorDouble -> IO ())
  -> TensorDouble -> TensorDouble -> IO TensorDouble
apply2 f t1 t2 = do
  r_ <- c_THDoubleTensor_new
  f r_ t1 t2
  pure r_

-- |apply an operation on 3 tensors
apply3 ::
  (TensorDouble -> TensorDouble -> TensorDouble -> TensorDouble -> IO ())
  -> TensorDouble -> TensorDouble -> TensorDouble -> IO TensorDouble
apply3 f t1 t2 t3 = do
  r_ <- c_THDoubleTensor_new
  f r_ t1 t2 t3
  pure r_

-- |apply inverse logit to all values of a tensor
invlogit :: TensorDouble -> IO TensorDouble
invlogit = apply c_THDoubleTensor_sigmoid

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

-- initialize values tensor = do
--   [(r, c) |
--     x <- [1..(nrows tensor)],
--     y <- [1..(ncols tensor)]]
--   mapM_ ((r, c) -> c_THDoubleTensor_set2d
--                    (fromIntegral r)
--                    (fromIntegral c)
--   where
--     idx = product . size $ tensor

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

-- |Show a real value with limited precision (convenience function)
showLim :: RealFloat a => a -> String
showLim x = showGFloat (Just 2) x ""

-- |display a tensor
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

-- |randomly initialize a tensor with uniform random values from a range
-- TODO - finish implementation to handle sizes correctly
randInit sz lower upper = do
  gen <- c_THGenerator_new
  t <- fromJust $ tensorNew sz
  mapM_ (\x -> do
            c_THDoubleTensor_uniform t gen lower upper
            disp t
        ) [0..3]

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

-- foreign import ccall unsafe "THTensor.h &THDoubleTensor_free"
--   p_THDoubleTensor_free :: FunPtr ((Ptr CTHDoubleTensor) -> IO ())

-- fr :: FunPtr (Ptr a -> IO ())
-- fr = FunPtr (Ptr )
-- fr = FunPtr $ c_THDoubleTensor_free

-- finalizer = FunPtr go
--   where
--     go x = c_THDoubleTensor_free x

-- getPtr = newForeignPtr p_THDoubleTensor_f8ree
-- newForeignPtr :: FinalizerPtr a -> Ptr a -> IO (ForeignPtr a)

-- |Create a new (double) tensor of specified dimensions and fill it with 0
-- |tag: unsafe
tensorNew_ :: TensorDim Word -> Double -> TensorDouble
tensorNew_ dims value = unsafePerformIO $ fill =<< (go dims)
  where
    create = (flip c_THDoubleTensor_fill) (realToFrac value)
    fill x = create x >> pure x
    -- go dims = undefined
    go D0 = c_THDoubleTensor_new
    go (D1 d1) = c_THDoubleTensor_newWithSize1d $ w2cl d1
    go (D2 d1 d2) = c_THDoubleTensor_newWithSize2d
                    (w2cl d1) (w2cl d2)
    go (D3 d1 d2 d3) = c_THDoubleTensor_newWithSize3d
                       (w2cl d1) (w2cl d2) (w2cl d3)
    go (D4 d1 d2 d3 d4) = c_THDoubleTensor_newWithSize4d
                          (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)

-- |Create a new (double) tensor of specified dimensions and fill it with 0
tensorNew :: [Int] -> Maybe (IO TensorDouble)
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

main = do
  disp =<< (fromJust $ tensorNew [4,3])
  putStrLn "Done"
