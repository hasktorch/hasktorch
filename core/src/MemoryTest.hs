module Main where

import Tensor
import TensorTypes

-- |Iteration - allocate a tensor, print a value, allocate another tensor... etc.
memoryTest :: TensorDim Word -> Int -> IO ()
memoryTest dim niter = do
  putStrLn $ (show $ memSizeGB dim) ++ " GB per allocation x " ++ (show niter)
  mapM_ (\iter -> do
            putStr ("Iteration : " ++ show iter ++ " / ")
            let x = tensorNew_ dim
            x <- get_ (D4 0 0 0 0) x
            putStrLn $ "Printing dummy value: " ++
              (show x) -- Need some IO with value
            pure ()
        ) [1..niter]
  putStrLn "Done"


-- |Get size per allocation
memSizeGB :: (TensorDim Word) -> Double
memSizeGB dim = fromIntegral((foldr (*) 1 dim) * 8) / 1000000000.0

memoryTestLarge =
  memoryTest (D4 200 200 200 200) 1000000 -- 12.8 GB x 1M = 12M GB

memoryTestSmall =
  memoryTest (D4 100 100 100 7) 300 -- 50 MB x 300 = 15 GB

-- |Confirm that memory is deallocated (works)
main = memoryTestSmall
