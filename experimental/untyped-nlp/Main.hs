import Torch

convTest = do
    input <- randnIO' [1, 2, 5] -- input: minibatch, channels, input width
    -- weights: out channels, in channels, kernel width
    let weights = asTensor ([[[0, 1, 0], [0, 1, 0]],
                             [[0, 1, 0], [0, 0, 1]]
                           ] :: [[[Float]]])
    let bias = zeros' [2] -- bias: out channels
    let output = conv1d' weights bias 1 1 input
    putStrLn "input"
    print $ squeezeAll input
    putStrLn "kernel"
    print $ squeezeAll weights
    putStrLn "output"
    print $ squeezeAll output

embedTest = do
    let dic = asTensor ([[1,2,3], [4,5,6]] :: [[Float]])
    let indices = asTensor ([0,0,1,0,1] :: [Int])
    let x = embedding' dic indices
    print x

main = do
    convTest
    embedTest
