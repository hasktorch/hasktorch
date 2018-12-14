{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE GADTs #-}

module Main where

import Data.Proxy (Proxy(..))
import Data.Singletons.Prelude.List (Product)
import GHC.TypeLits (natVal, KnownNat)
import Kernels (kernel1d_rbf)
import Prelude as P
import Torch.Double as T
import qualified Torch.Core.Random as RNG

-- Define an axis
type GridDim = 5
type GridSize = GridDim * GridDim
tRange = (*) scale <$> ([0 .. (gridDim - 1)] :: [HsReal])
    where scale = 0.2
          gridDim = fromIntegral $ natVal (Proxy :: Proxy GridDim)

-- Observed data points
type DataDim = 3
dataPredictors = [0.1, 0.3, 0.6]
dataValues = [-2.3, 1.5, -4]

-- | Cartesian product of axis coordinates
-- d represents the total number of elements and should be |axis1| x |axis2|
makeGrid :: (KnownDim d, KnownNat d) 
         => [Double] -> [Double] -> IO (Tensor '[d], Tensor '[d])
makeGrid axis1 axis2 = do
    t :: Tensor '[d] <- unsafeVector (fst <$> rngPairs)
    t' :: Tensor '[d] <- unsafeVector (snd <$> rngPairs)
    pure (t, t')
    where 
        pairs axis1' axis2' = [(t, t') | t <- axis1', t' <- axis2']
        rngPairs = pairs axis1 axis2

makeCovmatrix
 :: (All KnownDim [d1, d2, d3], All KnownNat [d1, d2, d3], d3 ~ Product [d1, d2]) 
 => [Double] -> [Double] -> IO (Tensor '[d1, d2])
makeCovmatrix axis1 axis2 = do
    (t, t') :: (Tensor '[d3], Tensor '[d3]) <- makeGrid axis1 axis2
    pure $ resizeAs $ kernel1d_rbf 1.0 1.0 t t'

-- | Multivariate 0-mean normal via cholesky decomposition
mvnCholesky :: (KnownDim b, KnownDim c) =>
    Generator -> Tensor '[b, b] -> IO (Tensor '[b, c])
mvnCholesky gen cov = do
    let Just sd = positive 1.0
    samples <- normal gen 0.0 sd
    let l = potrf cov Upper
    pure $ l !*! samples

-- | Conditional multivariate normal parameters for Predicted (X) | Observed (Y)
condition muX muY covXX covXY covYY y =
    (postMu, postCov) 
    where
        covYX = transpose2d covXY
        y' = resizeAs y
        postMu = muX + covXY !*! (getri covYY) !*! (y' - muY)
        postCov = covXX - covXY !*! (getri covYY) !*! covYX

-- | Compute GP conditioned on observed points
computePosterior :: IO (Tensor '[GridDim, 1], Tensor '[GridDim, GridDim])
computePosterior = do
    -- multivariate normal parameters for axis locations
    let priorMuAxis = constant 0 :: Tensor [GridDim, 1]
    priorCov <-  makeCovmatrix tRange tRange

    -- multivariate normal parameters for observation locations
    let priorMuData = (constant 0 :: Tensor '[DataDim, 1])
    obsCov :: Tensor '[DataDim, DataDim] <- makeCovmatrix dataPredictors dataPredictors
    putStrLn $ "\nObservation coordinates covariance\n" ++ show obsCov

    -- cross-covariance terms
    crossCov :: Tensor '[GridDim, DataDim] <- makeCovmatrix tRange dataPredictors
    putStrLn $ "\nCross covariance\n" ++ show crossCov

    -- conditional distribution
    obsVals :: Tensor '[DataDim] <- unsafeVector dataValues
    let (postMu, postCov) = 
            condition
                priorMuAxis priorMuData 
                priorCov crossCov obsCov
                obsVals -- observed y
    pure $ (postMu, postCov)
    
main :: IO ()
main = do
    -- Setup prediction axis
    cov :: Tensor [GridDim, GridDim] <- makeCovmatrix tRange tRange
    putStrLn $ "Predictor values\n" ++ show tRange
    putStrLn $ "\nCovariance based on radial basis function\n" ++ show cov

    -- Prior GP, take 3 samples
    let reg = 0.01 * eye -- regularization
    gen <- newRNG
    mvnSamp :: Tensor '[GridDim, 3] <- mvnCholesky gen (cov + reg)
    putStrLn $ "\nGP Samples (prior,  rows = values, cols = realizations)\n"
        ++ show mvnSamp

    -- Observations
    putStrLn $ "\nObservations: predictor coordinates\n" ++ show dataPredictors
    putStrLn $ "\nObservations: values\n" ++ show dataValues
    (postMu, postCov) <- computePosterior

    -- Conditional GP
    putStrLn $ "\nConditional mu (posterior)\n" ++ show postMu
    putStrLn $ "\nConditional covariance (posterior)\n" ++ show postCov
    gen <- newRNG
    mvnSamp :: Tensor '[GridDim, 1] <- mvnCholesky gen (postCov + reg)
    putStrLn "\nGP Conditional Samples (posterior, rows = values, cols = realizations)"
    print (postMu + mvnSamp)

    putStrLn "Done"