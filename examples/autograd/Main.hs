module Main where

import Torch
import Torch.Autograd
import Torch.Functional
import Torch.TensorFactories

printTensor :: String -> Tensor -> IO ()
printTensor s t = do
  putStr $ s ++ "\n" ++ (show t) ++ "\n\n"

main :: IO ()
main = do
  xi <- makeIndependent $ asTensor ([[-1], [2], [-3], [4]] :: [[Float]])
  wi1 <- makeIndependent $ ones' [4, 3]
  wi2 <- makeIndependent $ ones' [3, 1]
  let x = toDependent xi
      w1 = toDependent wi1
      w2 = toDependent wi2
      h1 = relu $ (transpose2D x) `matmul` w1
      h2 = sigmoid $ h1 `matmul` w2
      loss = (h2 - 0) ^ 2
  let gradients = grad loss [wi1, wi2]

  printTensor "The input features:" x
  printTensor "The output of the first layer:" h1
  printTensor "The output of the model :" h2
  printTensor "The gradient of weights between input and hidden layer:" (gradients !! 0)
  printTensor "The gradient of weights between hidden and output layer:" (gradients !! 1)
