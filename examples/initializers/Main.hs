module Main where

import Initializers

main :: IO ()
main = do
    putStrLn "\nKaiming Uniform"
    x <- kaimingUniform' [4, 5]
    print x
    putStrLn "\nKaiming Normal"
    x <- kaimingNormal' [4, 5]
    print x
    putStrLn "\nXavier Uniform"
    x <- xavierUniform' [4, 5]
    print x
    putStrLn "\nXavier Normal"
    x <- xavierNormal' [4, 5]
    print x
    putStrLn "Done"
