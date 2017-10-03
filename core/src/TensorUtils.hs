module TensorUtils (
  disp,
  disp_,
  w2cl
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import TensorTypes

import THDoubleTensor
import THDoubleTensorMath
import THTypes

import TensorDouble
import TensorDoubleMath

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

-- |Display memory managed tensor
disp_ tensor =
  (withForeignPtr(tdTensor tensor) disp)

-- |Dimensions of a tensor as a list
size :: (Ptr CTHDoubleTensor) -> [Int]
size t =
  fmap f [0..maxdim]
  where
    maxdim = (c_THDoubleTensor_nDimension t) - 1
    f x = fromIntegral (c_THDoubleTensor_size t x) :: Int

-- |Show a real value with limited precision (convenience function)
showLim :: RealFloat a => a -> String
showLim x = showGFloat (Just 2) x ""

-- |Word to CLong conversion
w2cl :: Word -> CLong
w2cl = fromIntegral
