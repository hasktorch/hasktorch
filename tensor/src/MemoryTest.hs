module Main where

import Tensor
import TensorTypes

memoryTest = do
  mapM_ (\iter -> do
            putStr (show iter ++ " ")
            let x = tensorNew_ (D4 200 200 200 200)
            x <- get_ (D4 5 5 5 5) x
            putStrLn (show x) -- Need some IO with value
            pure ()
        ) [1..1000000]
  putStrLn "Done"

-- |confirm that memory is deallocated - seems to work
main = memoryTest
