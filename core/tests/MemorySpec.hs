{-# LANGUAGE BangPatterns #-}

module MemorySpec (spec) where

import Control.Monad (forM_)
import Torch.Core.Tensor.Dynamic.Double
import Torch.Core.Tensor.Types

import Torch.Prelude.Extras

-- |Confirm that memory is deallocated (works)
main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  it "scenario: memoryTestSmall" memoryTestSmall

-- |Iteration - allocate a tensor, print a value, allocate another tensor... etc.
memoryTest :: TensorDim Word -> Int -> IO ()
memoryTest dim niter = do
  putStrLn $ show (memSizeGB dim) ++ " GB per allocation x " ++ show niter

  forM_ [1..niter] $ \iter -> do
    putStr ("Iteration : " ++ show iter ++ " / ")
    !t <- td_new_ dim
    !x <- td_get (D4 (0, 0, 0, 0)) t
    putStrLn $ "Printing dummy value: " ++ show x -- Need some IO with value

  putStrLn "Done"


-- |Get size per allocation
memSizeGB :: TensorDim Word -> Double
memSizeGB dim = fromIntegral (product dim * 8) / 1000000000.0

memoryTestLarge :: IO ()
memoryTestLarge = memoryTest (D4 (200, 200, 200, 200)) 1000000 -- 12.8 GB x 1M = 12M GB

memoryTestSmall :: IO ()
memoryTestSmall = memoryTest (D4 (100, 100, 100, 7)) 300 -- 50 MB x 300 = 15 GB

memoryTestFast :: IO ()
memoryTestFast = memoryTest (D4 (50, 50, 50, 5)) 10000 -- 5 MB x 1000 = 5 GB

