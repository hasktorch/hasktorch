{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Core.Tensor.Static.Byte (
  tbs_new,
  tbs_cloneDim,
  tbs_init,
  tbs_p
  ) where

import Data.Singletons
-- import Data.Singletons.Prelude
import Data.Singletons.TypeLits

import Foreign (Ptr)
import Foreign.C.Types (CLLong)
import Foreign.ForeignPtr ( ForeignPtr, withForeignPtr, newForeignPtr )
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Types
import THByteTensor
import THByteTensorMath
import THTypes

class StaticByteTensor t where
  -- |tensor dimensions
  -- |create tensor
  tbs_new :: t
  -- |create tensor of the same dimensions
  tbs_cloneDim :: t -> t -- takes unused argument, gets dimensions by matching types
  -- |create and initialize tensor
  tbs_init :: Int -> t
  -- |Display tensor
  tbs_p ::  t -> IO ()

data TensorByteStatic (d :: [Nat]) = TBS {
  tbsTensor :: !(ForeignPtr CTHByteTensor)
  } deriving (Show)

type TBS = TensorByteStatic

w2cl = fromIntegral

-- |Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
-- fillRaw :: Real a => a -> TensorByteRaw -> IO ()
fillRaw value = (flip c_THByteTensor_fill) (fromIntegral value)

-- |Create a new (double) tensor of specified dimensions and fill it with 0
-- safe version
tensorRaw :: TensorDim Word -> Int -> IO TensorByteRaw
tensorRaw dims value = do
  newPtr <- go dims
  fillRaw value newPtr
  pure newPtr
  where
    go D0 = c_THByteTensor_new
    go (D1 d1) = c_THByteTensor_newWithSize1d $ w2cl d1
    go (D2 (d1, d2)) = c_THByteTensor_newWithSize2d
                       (w2cl d1) (w2cl d2)
    go (D3 (d1, d2, d3)) = c_THByteTensor_newWithSize3d
                           (w2cl d1) (w2cl d2) (w2cl d3)
    go (D4 (d1, d2, d3, d4)) = c_THByteTensor_newWithSize4d
                               (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)

list2dim :: (Num a2, Integral a1) => [a1] -> TensorDim a2
list2dim lst  = case (length lst) of
  0 -> D0
  1 -> D1 (d !! 0)
  2 -> D2 ((d !! 0), (d !! 1))
  3 -> D3 ((d !! 0), (d !! 1), (d !! 2))
  4 -> D4 ((d !! 0), (d !! 1), (d !! 2), (d !! 3))
  _ -> error "Tensor type signature has invalid dimensions"
  where
    d = fromIntegral <$> lst -- cast as needed for tensordim


-- |Make a foreign pointer from requested dimensions
mkTHelper
  :: TensorDim Word
     -> (TensorDim Word -> ForeignPtr CTHByteTensor -> a) -> Int -> a
mkTHelper dims makeStatic value = unsafePerformIO $ do
  newPtr <- mkPtr dims value
  fPtr <- newForeignPtr p_THByteTensor_free newPtr
  pure $ makeStatic dims fPtr
  where
    mkPtr dim value = tensorRaw dim value

instance SingI d => StaticByteTensor (TensorByteStatic (d :: [Nat]))  where
  tbs_init initVal = mkTHelper dims makeStatic initVal
    where
      dims = list2dim $ fromSing (sing :: Sing d)
      makeStatic dims fptr = (TBS fptr) :: TBS d
  tbs_new = tbs_init 0
  tbs_cloneDim _ = tbs_new :: TBS d
  tbs_p tensor = (withForeignPtr(tbsTensor tensor) dispByteRaw)

-- |displaying raw tensor values
dispByteRaw :: Ptr CTHByteTensor -> IO ()
dispByteRaw tensor
  | (length sz) == 0 = putStrLn "Empty Tensor"
  | (length sz) == 1 = do
      putStrLn ""
      let indexes = [ fromIntegral idx :: CLLong
                    | idx <- [0..(sz !! 0 - 1)] ]
      putStr "[ "
      mapM_ (\idx -> putStr $
                     (showLim $ c_THByteTensor_get1d tensor idx) ++ " ")
        indexes
      putStrLn "]\n"
  | (length sz) == 2 = do
      putStrLn ""
      let pairs = [ ((fromIntegral r) :: CLLong,
                     (fromIntegral c) :: CLLong)
                  | r <- [0..(sz !! 0 - 1)], c <- [0..(sz !! 1 - 1)] ]
      putStr ("[ " :: String)
      mapM_ (\(r, c) -> do
                let val = c_THByteTensor_get2d tensor r c
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
    --size :: (Ptr CTHByteTensor) -> [Int]
    size t =
      fmap f [0..maxdim]
      where
        maxdim = (c_THByteTensor_nDimension t) - 1
        f x = fromIntegral (c_THByteTensor_size t x) :: Int
    showLim x = show x
    sz = size tensor

test = tbs_p (tbs_init 3 :: TBS '[4,2])
