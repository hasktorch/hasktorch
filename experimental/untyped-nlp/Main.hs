import Torch

main = do
    let x = embedding' (asTensor ([1,2,3] :: [Float])) (asTensor ([0,0,1] :: [Int]))
    print x
    putStrLn "Done"
