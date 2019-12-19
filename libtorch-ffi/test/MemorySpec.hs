{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
module MemorySpec (main, spec) where

import Test.Hspec
import Control.Exception (bracket)
import Control.Monad (forM_,forM)
import Data.Int
import Foreign
import Torch.Internal.Const
import Torch.Internal.Type
import Torch.Internal.Managed.Type.TensorOptions
import Torch.Internal.Managed.Type.Tensor
import Torch.Internal.Managed.Type.IntArray
import Torch.Internal.Managed.Type.Context
import Torch.Internal.Managed.Native

import System.Mem ()

-- |Confirm that memory is deallocated (works)
main :: IO ()
main = hspec spec

--type SomeDims = IntArray

spec :: Spec
spec = do
  describe "MemorySpec" $ do
    it "scenario: memoryTestMinimal" memoryTestMinimal


fromList :: [Int64] -> IO (ForeignPtr IntArray)
fromList dims = do
  ary <- newIntArray
  forM_ dims $ intArray_push_back_l ary
  return ary

newTensor_zeros :: (ForeignPtr IntArray) -> IO (ForeignPtr Tensor)
newTensor_zeros dims = do
  flag <- hasCUDA
  to <- device_D $ if flag == 0 then kCPU else kCUDA
  tod <- tensorOptions_dtype_s to kByte
  zeros_lo dims tod

totalDim :: (ForeignPtr IntArray) -> IO Int64
totalDim dims = do
  size <- intArray_size dims
  dims' <- forM [0..(size-1)] $ \i -> intArray_at_s dims i
  return $ sum dims'

iterator :: (ForeignPtr IntArray) -> Int -> IO ()
iterator = iteratorBracket

-- |Leaks memory
iteratorAssign :: (ForeignPtr IntArray) -> Int -> IO ()
iteratorAssign d niter = do
  size <- memSizeGB d
  putStrLn $ show size ++ " GB per allocation x " ++ show niter
  forM_ [1..niter] $ \iter -> do
    putStr ("Iteration : " ++ show iter ++ " / ")
    x <- newTensor_zeros d
    v <- tensor_dim x
    putStr $ "Printing dummy value: " ++ show v ++ "\r"
  putStrLn "Done"

-- |Releases memory on OSX (but not consistently on linux)
iteratorMonadic :: (ForeignPtr IntArray) -> Int -> IO ()
iteratorMonadic d niter = do
  size <- memSizeGB d
  putStrLn $ show size ++ " GB per allocation x " ++ show niter
  forM_ [1..niter] $ \iter -> do
    putStr ("Iteration : " ++ show iter ++ " / ")
    x <- newTensor_zeros d
    v <- tensor_dim x
    putStr $ "Printing dummy value: " ++ show v ++ "\r"
  putStrLn "Done"

-- |Releases memory
iteratorBracket :: (ForeignPtr IntArray) -> Int -> IO ()
iteratorBracket d niter = do
  size <- memSizeGB d
  putStrLn $ show size ++ " GB per allocation x " ++ show niter
  forM_ [1..niter] $ \iter ->
    bracket (pure iter)
    (\iter -> do
       putStr ("Iteration : " ++ show iter ++ " / ")
       x <- newTensor_zeros d
       v <- tensor_dim x
       putStr $ "Printing dummy value: " ++ show v ++ "\r"
    )
    (const (pure ()))
  putStrLn "Done"


-- |getDim' size per allocation
memSizeGB :: (ForeignPtr IntArray) -> IO Double
memSizeGB d = do
  td <- totalDim d
  return $ (fromIntegral td * 8) / 1000000000.0

memoryTestLarge :: IO ()
memoryTestLarge = do
  dims <- fromList [200, 200, 200, 200]
  iterator dims 1000000 -- 12.8 GB x 1M = 12M GB

memoryTestSmall :: IO ()
memoryTestSmall = do
  dims <- fromList [100, 100, 100, 7] 
  iterator dims 300 -- 50 MB x 300 = 15 GB

memoryTestFast :: IO ()
memoryTestFast = do
  dims <- fromList [50, 50, 50, 5]
  iterator dims 10000 -- 5 MB x 1000 = 5 GB

memoryTestMinimal :: IO ()
memoryTestMinimal = do
  dims <- fromList [50, 50, 50, 5]
  iterator dims 100 -- 5 MB x 100 = 500 MB
