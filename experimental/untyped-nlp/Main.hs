import Torch

convTest = do
    -- input: minibatch, channels, input width
    input <- randnIO' [1, 2, 5] 
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

convTest' input = do
    -- weights: (2 output channels, 3 input channels, 3 width kernel)
    let weights = asTensor ([[[0, 1, 0], [0, 1, 0], [0, 1, 0]],
                             [[0, 1, 0], [0, 0, 1], [0, 1, 0]]
                            ] :: [[[Float]]])
    let bias = zeros' [2] -- bias: out channels
    let output = conv1d' weights bias 1 1 input
    putStrLn "input"
    print $ squeezeAll input
    putStrLn "kernel"
    print $ squeezeAll weights
    putStrLn "output"
    print $ squeezeAll output
    pure output

embedTest :: IO Tensor
embedTest = do
    let dic = asTensor ([[1,2,3], [4,5,6]] :: [[Float]])
    let indices = asTensor ([0,0,1,0,1] :: [Int])
    let x = embedding' dic indices
    -- this results in 5 x 3 where
    -- 5 = input width, 3 = # channels
    pure x

main = do
    let dic = asTensor ([[1,2,3], [4,5,6]] :: [[Float]])
    putStrLn "Dictionary"
    print dic
    let indices = asTensor ([0,0,1,0,1] :: [Int])
    putStrLn "Indices"
    print indices
    x' <- embedTest
    let x = reshape [1, 3, 5] $ transpose2D x'
    -- let x = reshape [1, 5, 3] $ embedding' dic indices
    putStrLn "Embeddings"
    print x
    putStrLn "Embeddings Shape"
    print $ shape x
    outputs <- convTest' x
    print outputs
