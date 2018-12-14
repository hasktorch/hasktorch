{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import Data.List (tails)
import Data.Proxy (Proxy(..))
import GHC.TypeLits (natVal)
import Graphics.Vega.VegaLite
import Kernels (kernel1d_rbf)
import Prelude as P
import Torch.Double as T
import qualified Torch.Core.Random as RNG

-- Define an axis
type GridDim = 5
type GridSize = GridDim * GridDim
xRange = (*) scale <$> ([-halfwidth .. halfwidth] :: [HsReal])
    where scale = 0.2
          gridDim = natVal (Proxy :: Proxy GridDim)
          halfwidth = fromIntegral (P.div gridDim 2)

-- Observed data points
type DataDim = 3
dataPredictors = [-0.3, 0.0, 0.4]
dataValues = [-2.3, 1.5, -4]

-- | Cartesian product of axis coordinates with itself
makeGrid :: IO (Tensor '[GridSize], Tensor '[GridSize])
makeGrid = do
    t :: Tensor '[GridSize] <- unsafeVector (fst <$> rngPairs)
    t' :: Tensor '[GridSize] <- unsafeVector (snd <$> rngPairs)
    pure (t, t')
    where 
        pairs l = [(t, t') | t <- l, t' <- l]
        rngPairs = pairs xRange

-- | Multivariate 0-mean normal via cholesky decomposition
mvnCholesky :: (KnownDim b, KnownDim c) =>
    Generator -> Tensor '[b, b] -> IO (Tensor '[b, c])
mvnCholesky gen cov = do
    let Just sd = positive 1.0
    samples <- normal gen 0.0 sd
    let l = potrf cov Upper
    pure $ l !*! samples

-- | conditional distribution parameters for X|Y
--  Y are data points, X are predicted points
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
    -- instantiate tensors from data lists
    y :: Tensor '[DataDim] <- unsafeVector dataPredictors
    vals :: Tensor '[DataDim] <- unsafeVector dataValues

    -- covariance terms for predictions
    (t, t') <- makeGrid
    let rbf = kernel1d_rbf 1.0 1.0 t t' 
    let priorMu = constant 0 :: Tensor [GridDim, 1]
    let priorCov = (resizeAs rbf) :: Tensor [GridDim, GridDim]

    -- covariance terms for data
    let pairs = [(y, y') | y <- dataPredictors, y' <- dataPredictors]
    t :: Tensor '[DataDim * DataDim] <- unsafeVector (fst <$> pairs)
    t' :: Tensor '[DataDim * DataDim] <- unsafeVector (snd <$> pairs)
    let obsCov :: Tensor '[DataDim, DataDim] = 
            resizeAs $ kernel1d_rbf 1.0 1.0 t t'
    putStrLn $ "\nObservation coordinates covariance\n" ++ show obsCov

    -- cross-covariance terms
    let pairs = [(x, y) | x <- xRange, y <- dataPredictors]
    t :: Tensor '[GridDim * DataDim] <- unsafeVector (fst <$> pairs)
    t' :: Tensor '[GridDim * DataDim] <- unsafeVector (snd <$> pairs)
    let crossCov :: Tensor '[GridDim, DataDim] = 
            resizeAs $ kernel1d_rbf 1.0 1.0 t t'
    putStrLn $ "\nCross covariance\n" ++ show crossCov

    -- conditional distribution
    let (postMu, postCov) = 
            condition
                priorMu (constant 0 :: Tensor '[DataDim, 1]) -- mean
                priorCov crossCov obsCov -- covariance matrix terms
                vals -- observed y
    pure $ (postMu, postCov)
    
main :: IO ()
main = do
    -- Setup prediction axis
    (t, t') <- makeGrid
    let rbf = kernel1d_rbf 1.0 1.0 t t'
        cov = resizeAs rbf :: Tensor [GridDim, GridDim]
        mu = constant 0 :: Tensor [GridDim, GridDim]
    putStrLn $ "Predictor values\n" ++ show xRange
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