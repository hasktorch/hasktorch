{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Control.Monad (foldM)

import Prelude hiding (exp)

import Torch.Tensor
import Torch.DType (DType (Float))
import Torch.TensorFactories (eye', ones', rand', randn', zeros')
import Torch.Functions
import Torch.Autograd
import Torch.NN

-- Axis points
axisDim = 7
tRange = (*) scale <$> (fromIntegral <$> [0 .. (axisDim - 1)])
    where scale = 0.1

-- Observed data points
dataDim = 3
dataPredictors = [0.1, 0.3, 0.6] :: [Float]
dataValues = [-2.3, 1.5, -4] :: [Float]

-- | construct pairs of points on the axis
makeAxis :: [Float] -> [Float] -> (Tensor, Tensor)
makeAxis axis1 axis2 = (t, t')
    where 
        t = asTensor (fst <$> rngPairs)
        t' = asTensor (snd <$> rngPairs)
        pairs axis1' axis2' = [(t, t') | t <- axis1', t' <- axis2']
        rngPairs = pairs axis1 axis2

-- | 1-dimensional radial basis function kernel
kernel1d_rbf :: Double -> Double -> Tensor -> Tensor -> Tensor
kernel1d_rbf sigma length t t' = (sigma'^2) * exp eterm
    where
        sigma' = asTensor sigma
        eterm = cmul (- (pow (t - t') (2 :: Int))) (1 / 2 * length^2)

-- | derive a covariance matrix from the kernel for points on the axis
makeCovmatrix :: [Float] -> [Float] -> Tensor
makeCovmatrix axis1 axis2 = reshape (kernel1d_rbf 1.0 1.0 t t') [length axis1, length axis2]
    where
      (t, t') = makeAxis axis1 axis2

-- | Multivariate 0-mean normal via cholesky decomposition
mvnCholesky :: Tensor -> Int -> IO Tensor
mvnCholesky cov n = do
    samples <- randn' [axisDim, n]
    pure $ matmul l samples
    where 
      l = cholesky cov Upper

condition muX muY covXX covXY covYY y =
    (postMu, postCov)
    where
        covYX = transpose2D covXY
        invY = inverse covYY
        postMu = muX + (matmul covXY (matmul invY (y - muY)))
        postCov = covXX - (matmul covXY (matmul invY covYX))


computePosterior dataPredictors dataValues = do

    -- multivariate normal parameters for axis locations
    let priorMuAxis = zeros' [axisDim, 1]
    let priorCov = makeCovmatrix tRange tRange

    -- multivariate normal parameters for observation locations
    let priorMuData = zeros' [dataDim, 1]
    let obsCov = makeCovmatrix dataPredictors dataPredictors
    putStrLn $ "\nObservation coordinates covariance\n" ++ show obsCov

    -- cross-covariance terms
    let crossCov = makeCovmatrix tRange dataPredictors
    putStrLn $ "\nCross covariance\n" ++ show crossCov

    -- conditional distribution
    let obsVals = reshape (asTensor dataValues) [dataDim, 1]
    let (postMu, postCov) = 
            condition
                priorMuAxis priorMuData 
                priorCov crossCov obsCov
                obsVals

    pure $ (postMu, postCov)

main = do

    -- Setup prediction axis
    let cov =  makeCovmatrix tRange tRange
    putStrLn $ "Predictor values\n" ++ show tRange
    putStrLn $ "\nCovariance based on radial basis function\n" ++ show cov

    putStrLn "prior"
    -- Prior GP, take 4 example samples
    let reg = 0.01 * (eye'  axisDim axisDim) -- regularization
    mvnSampPrior <- mvnCholesky (cov + reg) 4
    putStrLn $ "\nGP Samples (prior,  rows = values, cols = realizations)\n"
        ++ show mvnSampPrior

    -- Observations
    putStrLn $ "\nObservations: predictor coordinates\n" ++ show dataPredictors
    putStrLn $ "\nObservations: values\n" ++ show dataValues
    (postMu, postCov) <- computePosterior dataPredictors dataValues

    -- Conditional GP
    putStrLn $ "\nConditional mu (posterior)\n" ++ show postMu
    putStrLn $ "\nConditional covariance (posterior)\n" ++ show postCov
    mvnSampPost <- mvnCholesky (postCov + reg) 1
    putStrLn "\nGP Conditional Samples (posterior, rows = values, cols = realizations)"
    print (postMu + mvnSampPost)
