{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

module TorchTensor (
  TensorDouble(..),
  -- fill,
  -- fill0,
  -- mvSimple,
  -- (#>),
  size,
  -- tensorNew,
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

data TensorDouble_ = TensorDouble_ {
  tdTensor :: !(ForeignPtr CTHDoubleTensor)
  } deriving (Eq, Show)

-- TODO: need bindings to THStorage to use c_THDoubleTensor_resize
initialize values sz = do
  tensor <- c_THDoubleTensor_newWithSize1d nel
  mapM_
    (\(idx, value) -> c_THDoubleTensor_set1d tensor idx value)
    (zip [1..nel], values)
  pure tensor
  where
    nel = product sz

-- -- |matrix vector multiplication, no error checking for now
-- -- |tag: unsafe
-- -- TODO - determine how to deal with resource allocation
-- (#>) :: TensorDouble -> TensorDouble -> TensorDouble
-- mat #> vec = unsafePerformIO $ do
--   res <- fromJust $ tensorNew $ [nrows mat]
--   c_THDoubleTensor_addmv res 1.0 res 1.0 mat vec
--   pure res

-- -- |simplified matrix vector multiplication
-- mvSimple :: TensorDouble -> TensorDouble -> IO TensorDouble
-- mvSimple mat vec = do
--   res <- fromJust $ tensorNew $ [nrows mat]
--   zero <- fromJust $ tensorNew $ [nrows mat]
--   print $ "dimension check matrix:" <>
--     show (c_THDoubleTensor_nDimension mat == 2)
--   print $ "dimension check vector:" <>
--     show (c_THDoubleTensor_nDimension vec == 1)
--   c_THDoubleTensor_addmv res 1.0 zero 1.0 mat vec
--   pure res

-- |number of rows of a tensor (unsafe)
nrows tensor = (size tensor) !! 0

-- |number of cols of a tensor (unsafe)
ncols tensor = (size tensor) !! 1

-- |Show a real value with limited precision (convenience function)
showLim :: RealFloat a => a -> String
showLim x = showGFloat (Just 2) x ""

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

-- -- |Create a new (double) tensor of specified dimensions and fill it with 0
-- tensorNew :: [Int] -> Maybe (IO TensorDouble)
-- tensorNew dims
--   | ndim == 0 = Just $ c_THDoubleTensor_new
--   | ndim == 1 = Just $ (c_THDoubleTensor_newWithSize1d $ head cdims) >>= fill0
--   | ndim == 2 = Just $ c_THDoubleTensor_newWithSize2d (cdims !! 0) (cdims !! 1) >>= fill0
--   | ndim == 3 = Just $ (c_THDoubleTensor_newWithSize3d
--                         (cdims !! 0) (cdims !! 1) (cdims !! 2)) >>= fill0
--   | ndim == 4 = Just $ (c_THDoubleTensor_newWithSize4d
--                         (cdims !! 0) (cdims !! 1) (cdims !! 2) (cdims !! 3)
--                         >>= fill0)
--   | otherwise = Nothing
--   where
--     ndim = length dims
--     cdims = (\x -> (fromIntegral x) :: CLong) <$> dims

main = do
  -- disp =<< (fromJust $ tensorNew [4,3])
  putStrLn "Done"
