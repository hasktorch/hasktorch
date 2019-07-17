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

-- | construct pairs of points on the axis
makeAxis axis1 axis2 = do
    let t = asTensor (fst <$> rngPairs)
    let t' = asTensor (snd <$> rngPairs)
    pure (t, t')
    where 
        pairs axis1' axis2' = [(t, t') | t <- axis1', t' <- axis2']
        rngPairs = pairs axis1 axis2

-- | 1-dimensional radial basis function kernel
kernel1d_rbf :: Double -> Double -> Tensor -> Tensor -> Tensor
kernel1d_rbf sigma length t t' =
    (sigma'^2) * exp eterm
    where
        sigma' = asTensor sigma
        eterm = cmul (- (pow (t - t') (2 :: Int))) (1 / 2 * length^2)

-- | derive a covariance matrix from the kernel for points on the axis
makeCovmatrix axis1 axis2 = do
    (t, t') <- makeAxis axis1 axis2
    let result = kernel1d_rbf 1.0 1.0 t t'
    pure $ result

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
