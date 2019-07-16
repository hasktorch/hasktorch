{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Control.Monad (foldM)

import Prelude hiding (exp)

import Torch.Tensor
import Torch.DType (DType (Float))
import Torch.TensorFactories (eye', ones', rand', randn')
import Torch.Functions
import Torch.Autograd
import Torch.NN


-- | 1-dimensional radial basis function kernel
kernel1d_rbf :: Double -> Double -> Tensor -> Tensor -> Tensor
kernel1d_rbf sigma length t t' =
    (sigma'^2) * exp eterm
    where
        sigma' = asTensor sigma
        eterm = undefined
        -- eterm = (-(t - t')^2) / (2 * length^2)

-- | Multivariate 0-mean normal via cholesky decomposition
mvnCholesky :: Tensor -> Int -> IO Tensor
mvnCholesky cov n = do
    samples <- randn' [2, n]
    let l = cholesky cov Upper
    putStrLn $ show l
    putStrLn $ show samples
    pure $ matmul l samples

condition muX muY covXX covXY covYY y =
    (postMu, postCov)
    where
        covYX = transpose2D covXY
        invY = inverse covYY
        postMu = muX + (matmul covXY (matmul invY (y - muY)))
        postCov = covXX - (matmul covXY (matmul invY covYX))

main = do
    let cov = asTensor ([[1.2, 0.4], [0.4, 1.2]] :: [[Float]])
    vals <- mvnCholesky cov 30
    putStrLn "samples:"
    putStrLn $ show $ transpose2D vals
    pure ()
