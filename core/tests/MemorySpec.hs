{-# LANGUAGE BangPatterns #-}

module MemorySpec (spec) where

import Control.Exception (bracket)
import Control.Monad (forM_)
import Torch.Core.Tensor.Dynamic.Double
import Torch.Core.Tensor.Types

import Torch.Prelude.Extras
import System.Mem

-- |Confirm that memory is deallocated (works)
main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  it "scenario: memoryTestMinimal" memoryTestMinimal

iterator = iteratorBracket

-- |Leaks memory
iteratorAssign :: TensorDim Word -> Int -> IO ()
iteratorAssign dim niter = do
  putStrLn $ show (memSizeGB dim) ++ " GB per allocation x " ++ show niter
  forM_ [1..niter] $ \iter -> do
    putStr ("Iteration : " ++ show iter ++ " / ")
    x <- td_get (D4 (0, 0, 0, 0)) (td_new dim)
    putStrLn $ "Printing dummy value: " ++ show x
  putStrLn "Done"

-- |Releases memory on OSX (but not consistently on linux)
iteratorMonadic :: TensorDim Word -> Int -> IO ()
iteratorMonadic dim niter = do
  putStrLn $ show (memSizeGB dim) ++ " GB per allocation x " ++ show niter
  forM_ [1..niter] $ \iter -> do
    putStr ("Iteration : " ++ show iter ++ " / ")
    x <- td_get (D4 (0, 0, 0, 0)) =<< td_new_ dim
    putStrLn $ "Printing dummy value: " ++ show x
  putStrLn "Done"

-- |Releases memory
iteratorBracket :: TensorDim Word -> Int -> IO ()
iteratorBracket dim niter = do
  putStrLn $ show (memSizeGB dim) ++ " GB per allocation x " ++ show niter
  forM_ [1..niter] $ \iter ->
    bracket (pure iter)
    (\iter -> do
       putStr ("Iteration : " ++ show iter ++ " / ")
       x <- td_get (D4 (0, 0, 0, 0)) (td_new dim)
       putStrLn $ "Printing dummy value: " ++ show x
    )
    (\x -> pure ())
  putStrLn "Done"

manualAlloc1 :: IO ()
manualAlloc1 = do
  putStrLn $ "Allocating"
  let !t = td_new (D4 (200, 200, 200, 200))
  x <- td_get (D4 (0, 0, 0, 0)) t
  putStrLn $ "Printing dummy value: " ++ show x

manualAlloc2 :: Double -> IO (TensorDouble)
manualAlloc2 v = do
  putStrLn $ "Allocating"
  let !t = td_init (D4 (200, 200, 100, 100)) v
  x <- td_get (D4 (0, 0, 0, 0)) t
  putStrLn $ "Printing dummy value: " ++ show x
  pure t

pr :: TensorDouble -> IO ()
pr t = do
  x <- td_get (D4 (0, 0, 0, 0)) t
  putStrLn $ "Printing dummy value: " ++ show x

-- |Get size per allocation
memSizeGB :: TensorDim Word -> Double
memSizeGB dim = fromIntegral (product dim * 8) / 1000000000.0

memoryTestLarge :: IO ()
memoryTestLarge = iterator (D4 (200, 200, 200, 200)) 1000000 -- 12.8 GB x 1M = 12M GB

memoryTestSmall :: IO ()
memoryTestSmall = iterator (D4 (100, 100, 100, 7)) 300 -- 50 MB x 300 = 15 GB

memoryTestFast :: IO ()
memoryTestFast = iterator (D4 (50, 50, 50, 5)) 10000 -- 5 MB x 1000 = 5 GB

memoryTestMinimal :: IO ()
memoryTestMinimal = iterator (D4 (50, 50, 50, 5)) 100 -- 5 MB x 100 = 500 MB
