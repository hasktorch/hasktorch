module Main where

import Torch
import Torch.NN.Recurrent.Cell.Elman

convTest = do
  -- input: minibatch, channels, input width
  input <- randnIO' [1, 2, 5]
  -- weights: out channels, in channels, kernel width
  let weights =
        asTensor
          ( [ [[0, 1, 0], [0, 1, 0]],
              [[0, 1, 0], [0, 0, 1]]
            ] ::
              [[[Float]]]
          )
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
  let weights =
        asTensor
          ( [ [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
              [[0, 1, 0], [0, 0, 1], [0, 1, 0]]
            ] ::
              [[[Float]]]
          )
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
  let dic = asTensor ([[1, 2, 3], [4, 5, 6]] :: [[Float]])
  let indices = asTensor ([0, 0, 1, 0, 1] :: [Int])
  let x = embedding' dic indices
  -- this results in 5 x 3 where
  -- 5 = input width, 3 = # channels
  pure $ reshape [1, 3, 5] $ transpose2D x

rnnTest :: IO Tensor
rnnTest = do
  let hx = zeros' [hsz]
  let wih = zeros' [hsz, isz]
  let whh = zeros' [hsz, hsz]
  let bih = zeros' [hsz]
  let bhh = zeros' [hsz]
  input <- randnIO' [isz]
  pure $ rnnReluCell wih whh bih bhh hx input
  where
    hsz = 5 -- hidden dimensions
    isz = 3 -- input dimensions

lstmTest :: IO (Tensor, Tensor)
lstmTest = do
  let hx = (zeros' [bsz, hsz], zeros' [bsz, hsz])
  let wih = full' [hsz * 4, isz] (1.0 :: Float)
  let whh = full' [hsz * 4, hsz] (1.0 :: Float)
  let bih = full' [hsz * 4] (1.0 :: Float)
  let bhh = full' [hsz * 4] (1.0 :: Float)
  let input = full' [bsz, isz] (1.0 :: Float)
  pure $ lstmCell wih whh bih bhh hx input
  where
    bsz = 3 -- batch size
    hsz = 2 -- hidden dimensions
    isz = 5 -- input dimensions

main = do
  -- Embeddings
  x <- embedTest
  putStrLn "Embeddings Shape"
  print $ shape x

  -- Convolutions
  putStrLn "\nConvolution"
  outputs <- convTest' x
  print outputs

  -- RNN Cells
  putStrLn "\nElman"
  print =<< rnnTest
  putStrLn "\nLSTM"
  print =<< lstmTest

-- cosineSimilarity
