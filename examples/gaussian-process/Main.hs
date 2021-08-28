{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (foldM)
import Torch
import Prelude hiding (exp)

newtype MeanVector = MeanVector Tensor deriving (Show)

newtype CovMatrix = CovMatrix Tensor deriving (Show)

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
kernel1d_rbf sigma length t t' = (sigma' ^ 2) * exp eterm
  where
    sigma' = asTensor sigma
    eterm = mulScalar (1 / (2 * length ^ 2)) (- (pow (2 :: Int) (t - t')))

-- | derive a covariance matrix from the kernel for points on the axis
makeCovmatrix :: [Float] -> [Float] -> CovMatrix
makeCovmatrix axis1 axis2 =
  CovMatrix (reshape [length axis1, length axis2] (kernel1d_rbf 1.0 1.0 t t'))
  where
    (t, t') = makeAxis axis1 axis2

-- | Multivariate 0-mean normal via cholesky decomposition
mvnCholesky :: CovMatrix -> Int -> Int -> IO Tensor
mvnCholesky (CovMatrix cov) axisDim n = do
  samples <- randnIO' [axisDim, n]
  pure $ matmul l samples
  where
    l = cholesky Upper cov

-- | Compute posterior mean and covariance parameters based on observed data y
condition ::
  -- | mean of unobserved points X
  MeanVector ->
  -- | mean of observed points Y
  MeanVector ->
  -- | covariance of unobserved points X
  CovMatrix ->
  -- | cross-covariance between observed and unobserved points X <-> Y
  CovMatrix ->
  -- | covariance of observed points Y
  CovMatrix ->
  -- | values of observed points Y
  Tensor ->
  -- | mean and covariance of unobserved points X
  (MeanVector, CovMatrix)
condition (MeanVector muX) (MeanVector muY) (CovMatrix covXX) (CovMatrix covXY) (CovMatrix covYY) y =
  (MeanVector postMu, CovMatrix postCov)
  where
    covYX = transpose2D covXY
    invY = inverse covYY
    postMu = muX + (matmul covXY (matmul invY (y - muY)))
    postCov = covXX - (matmul covXY (matmul invY covYX))

-- | Add small values on the diagonal of a covariance matrix
regularize :: CovMatrix -> CovMatrix
regularize (CovMatrix cov) = CovMatrix (cov + reg)
  where
    axisDim = shape cov !! 0
    reg = 0.01 * (eye' axisDim axisDim) -- regularization

-- | Given observations + points of interest derive covariance terms and condition on observation
computePosterior :: [Float] -> [Float] -> [Float] -> IO (MeanVector, CovMatrix)
computePosterior dataPredictors dataValues tRange = do
  let dataDim = length dataPredictors
  let axisDim = length tRange

  -- multivariate normal parameters for axis locations
  let priorMuAxis = MeanVector $ zeros' [axisDim, 1]
  let priorCov = makeCovmatrix tRange tRange

  -- multivariate normal parameters for observation locations
  let priorMuData = MeanVector $ zeros' [dataDim, 1]
  let obsCov = makeCovmatrix dataPredictors dataPredictors
  putStrLn $ "\nObservation coordinates covariance\n" ++ show obsCov

  -- cross-covariance terms
  let crossCov = makeCovmatrix tRange dataPredictors
  putStrLn $ "\nCross covariance\n" ++ show crossCov

  -- conditional distribution
  let obsVals = reshape [dataDim, 1] (asTensor dataValues)
  let (postMu, postCov) =
        condition
          priorMuAxis
          priorMuData
          priorCov
          crossCov
          obsCov
          obsVals
  pure $ (postMu, regularize postCov)

addMean :: MeanVector -> Tensor -> Tensor
addMean (MeanVector meanVec) x = meanVec + x

main :: IO ()
main = do
  -- Setup prediction axis
  let cov = regularize $ makeCovmatrix tRange tRange
  putStrLn $ "Predictor values\n" ++ show tRange
  putStrLn $ "\nCovariance based on radial basis function\n" ++ show cov

  -- Prior GP, take 4 example samples
  putStrLn "prior"
  mvnSampPrior <- mvnCholesky cov axisDim 4
  putStrLn $
    "\nGP Samples (prior,  rows = values, cols = realizations)\n"
      ++ show mvnSampPrior

  -- Observations
  putStrLn $ "\nObservations: predictor coordinates\n" ++ show dataPredictors
  putStrLn $ "\nObservations: values\n" ++ show dataValues
  (postMu, postCov) <- computePosterior dataPredictors dataValues tRange

  -- Conditional GP
  putStrLn $ "\nConditional mu (posterior)\n" ++ show postMu
  putStrLn $ "\nConditional covariance (posterior)\n" ++ show postCov
  mvnSampPost <- mvnCholesky postCov (length tRange) 1
  putStrLn "\nGP Conditional Samples (posterior, rows = values, cols = realizations)"
  print $ addMean postMu mvnSampPost
  where
    -- Axis points
    scale = 0.1
    axisDim = 7
    tRange = (*) scale <$> (fromIntegral <$> [0 .. (axisDim - 1)])
    -- Observed data points
    dataPredictors = [0.1, 0.3, 0.6] :: [Float]
    dataValues = [-2.3, 1.5, -4] :: [Float]
