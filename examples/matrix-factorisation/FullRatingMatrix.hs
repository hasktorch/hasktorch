{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (foldM, when)
import MF
import Torch.Autograd (grad, makeIndependent, toDependent)
import Torch.DType (DType (Float))
import Torch.Functional (matmul, mse_loss)
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import Torch.NN
  ( sample,
    sgd,
  )
import Torch.Tensor (Tensor, asTensor, shape)

r =
  asTensor
    ( [ [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
      ] ::
        [[Float]]
    )

lossMF :: MatrixFact -> Tensor
lossMF mf = mse_loss (mulMF mf) r

printParams :: MatrixFact -> IO ()
printParams (MatrixFact u v) = do
  putStrLn "\nR:"
  print r
  putStrLn "\nU*V:"
  print (matmul u' v')
  putStrLn "\n|U*V - R|:"
  print $ abs ((matmul u' v') - r)
  where
    u' = toDependent u
    v' = toDependent v

num_iters = 10000

main :: IO ()
main = do
  manual_seed_L 123
  -- The rank of r is 2 and we therefore choose k equal to 2.
  -- The gradient descent does not converge for k equal to 1, as expected.
  init <- sample $ (MatrixFactSpec n m k)
  trained <- foldLoop init num_iters gdBlock
  printParams trained
  where
    n = shape r !! 0
    m = shape r !! 1
    k = 2
    foldLoop x count block = foldM block x [1 .. count]
    gdBlock state i = do
      let loss = lossMF state
          flat_parameters = flattenParameters state
          gradients = grad loss flat_parameters
      when (i `mod` 500 == 0) do
        putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
      new_flat_parameters <- mapM makeIndependent $ sgd 5e-3 flat_parameters gradients
      return $ replaceParameters state $ new_flat_parameters
