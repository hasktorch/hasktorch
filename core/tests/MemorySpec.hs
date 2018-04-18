{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
module MemorySpec (spec) where

import Test.Hspec
import Control.Exception (bracket)
import Control.Monad (forM_)
import Torch.Dynamic as Dynamic

import System.Mem ()

-- |Confirm that memory is deallocated (works)
main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  it "scenario: memoryTestMinimal" memoryTestMinimal

headIdx :: Dim '[0, 0, 0, 0]
headIdx = dim

headIdx' :: SomeDims
headIdx' = SomeDims (dim :: Dim '[0, 0, 0, 0])

iterator :: SomeDims -> Int -> IO ()
iterator = iteratorBracket

-- |Leaks memory
iteratorAssign :: SomeDims -> Int -> IO ()
iteratorAssign dim niter = do
  putStrLn $ show (memSizeGB dim) ++ " GB per allocation x " ++ show niter
  forM_ [1..niter] $ \iter -> do
    putStr ("Iteration : " ++ show iter ++ " / ")
    x <- (`getDim` headIdx) =<< (Dynamic.new' dim :: IO DoubleDynamic)
    putStrLn $ "Printing dummy value: " ++ show x
  putStrLn "Done"

-- |Releases memory on OSX (but not consistently on linux)
iteratorMonadic :: SomeDims -> Int -> IO ()
iteratorMonadic dim niter = do
  putStrLn $ show (memSizeGB dim) ++ " GB per allocation x " ++ show niter
  forM_ [1..niter] $ \iter -> do
    putStr ("Iteration : " ++ show iter ++ " / ")
    x <- (`getDim` headIdx) =<< (Dynamic.new' dim :: IO DoubleDynamic)
    putStrLn $ "Printing dummy value: " ++ show x
  putStrLn "Done"

-- |Releases memory
iteratorBracket :: SomeDims -> Int -> IO ()
iteratorBracket dim niter = do
  putStrLn $ show (memSizeGB dim) ++ " GB per allocation x " ++ show niter
  forM_ [1..niter] $ \iter ->
    bracket (pure iter)
    (\iter -> do
       putStr ("Iteration : " ++ show iter ++ " / ")
       x <- (`getDim` headIdx) =<< (Dynamic.new' dim :: IO DoubleDynamic)
       putStrLn $ "Printing dummy value: " ++ show x
    )
    (const (pure ()))
  putStrLn "Done"

manualAlloc1 :: IO ()
manualAlloc1 = do
  putStrLn   "Allocating"
  !(t :: DoubleDynamic) <- new (dim :: Dim '[200, 200, 200, 200])
  x <- getDim t headIdx 
  putStrLn $ "Printing dummy value: " ++ show x

manualAlloc2 :: Double -> IO (DoubleDynamic)
manualAlloc2 v = do
  putStrLn "Allocating"
  !(t :: DoubleDynamic) <- constant (dim :: Dim '[200, 200, 100, 100]) v
  x <- getDim' t headIdx'
  putStrLn $ "Printing dummy value: " ++ show x
  pure t

pr :: DoubleDynamic -> IO ()
pr t = do
  v <- getDim' t headIdx'
  putStrLn $ "Printing dummy value: " ++ show v

-- |getDim' size per allocation
memSizeGB :: SomeDims -> Double
memSizeGB dim = fromIntegral (product' dim * 8) / 1000000000.0

memoryTestLarge :: IO ()
memoryTestLarge = iterator (SomeDims (dim :: Dim '[200, 200, 200, 200])) 1000000 -- 12.8 GB x 1M = 12M GB

memoryTestSmall :: IO ()
memoryTestSmall = iterator (SomeDims (dim :: Dim '[100, 100, 100, 7])) 300 -- 50 MB x 300 = 15 GB

memoryTestFast :: IO ()
memoryTestFast = iterator (SomeDims (dim :: Dim '[50, 50, 50, 5])) 10000 -- 5 MB x 1000 = 5 GB

memoryTestMinimal :: IO ()
memoryTestMinimal = iterator (SomeDims (dim :: Dim '[50, 50, 50, 5])) 100 -- 5 MB x 100 = 500 MB
