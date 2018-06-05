{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
module MemorySpec (spec) where

import Test.Hspec
import Control.Exception (bracket)
import Control.Monad (forM_)
import Torch.Double.Dynamic as Dynamic

import System.Mem ()

-- |Confirm that memory is deallocated (works)
main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  it "scenario: memoryTestMinimal" memoryTestMinimal

headIdx :: Dims '[0, 0, 0, 0]
headIdx = dims

headIdx' :: SomeDims
headIdx' = SomeDims (dims :: Dims '[0, 0, 0, 0])

iterator :: SomeDims -> Int -> IO ()
iterator = iteratorBracket

-- |Leaks memory
iteratorAssign :: SomeDims -> Int -> IO ()
iteratorAssign d niter = do
  putStrLn $ show (memSizeGB d) ++ " GB per allocation x " ++ show niter
  forM_ [1..niter] $ \iter -> do
    putStr ("Iteration : " ++ show iter ++ " / ")
    x <- (`getDim` headIdx) =<< (Dynamic.new' d :: IO DoubleDynamic)
    putStrLn $ "Printing dummy value: " ++ show x
  putStrLn "Done"

-- |Releases memory on OSX (but not consistently on linux)
iteratorMonadic :: SomeDims -> Int -> IO ()
iteratorMonadic d niter = do
  putStrLn $ show (memSizeGB d) ++ " GB per allocation x " ++ show niter
  forM_ [1..niter] $ \iter -> do
    putStr ("Iteration : " ++ show iter ++ " / ")
    x <- (`getDim` headIdx) =<< (Dynamic.new' d :: IO DoubleDynamic)
    putStrLn $ "Printing dummy value: " ++ show x
  putStrLn "Done"

-- |Releases memory
iteratorBracket :: SomeDims -> Int -> IO ()
iteratorBracket d niter = do
  putStrLn $ show (memSizeGB d) ++ " GB per allocation x " ++ show niter
  forM_ [1..niter] $ \iter ->
    bracket (pure iter)
    (\iter -> do
       putStr ("Iteration : " ++ show iter ++ " / ")
       x <- (`getDim` headIdx) =<< (Dynamic.new' d :: IO DoubleDynamic)
       putStrLn $ "Printing dummy value: " ++ show x
    )
    (const (pure ()))
  putStrLn "Done"

manualAlloc1 :: IO ()
manualAlloc1 = do
  putStrLn   "Allocating"
  !(t :: DoubleDynamic) <- new (dims :: Dims '[200, 200, 200, 200])
  x <- getDim t headIdx
  putStrLn $ "Printing dummy value: " ++ show x

manualAlloc2 :: Double -> IO (DoubleDynamic)
manualAlloc2 v = do
  putStrLn "Allocating"
  let !(t :: DoubleDynamic) = constant (dims :: Dims '[200, 200, 100, 100]) v
  x <- getDim' t headIdx'
  putStrLn $ "Printing dummy value: " ++ show x
  pure t

pr :: DoubleDynamic -> IO ()
pr t = do
  v <- getDim' t headIdx'
  putStrLn $ "Printing dummy value: " ++ show v

-- |getDim' size per allocation
memSizeGB :: SomeDims -> Double
memSizeGB d = fromIntegral (product' d * 8) / 1000000000.0

memoryTestLarge :: IO ()
memoryTestLarge = iterator (SomeDims (dims :: Dims '[200, 200, 200, 200])) 1000000 -- 12.8 GB x 1M = 12M GB

memoryTestSmall :: IO ()
memoryTestSmall = iterator (SomeDims (dims :: Dims '[100, 100, 100, 7])) 300 -- 50 MB x 300 = 15 GB

memoryTestFast :: IO ()
memoryTestFast = iterator (SomeDims (dims :: Dims '[50, 50, 50, 5])) 10000 -- 5 MB x 1000 = 5 GB

memoryTestMinimal :: IO ()
memoryTestMinimal = iterator (SomeDims (dims :: Dims '[50, 50, 50, 5])) 100 -- 5 MB x 100 = 500 MB
