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

-- Function predicted value locations (on a grid)
type GridDim = 5
type NSamp = 3
type GridSize = GridDim * GridDim

-- Construct an axis w/ function values of size GridDim
xRange = (*) scale <$> ([-halfwidth .. halfwidth] :: [HsReal])
    where
        scale = 0.2
        gridDim = natVal (Proxy :: Proxy GridDim)
        halfwidth = fromIntegral (P.div gridDim 2)

-- Observed data
type DataDim = 3
type DataSize = DataDim * DataDim
dataPredictors = [-0.3, 0.3, 0.5]
dataValues = [-2.3, 1.5, -4]

-- Cross-covariance dimensions
type CrossDim = DataDim * GridDim
type CrossSize = GridDim * DataDim

-- | Cartesian product of predictor axes coordinates
makeGrid :: IO (Tensor '[GridSize], Tensor '[GridSize])
makeGrid = do
    t :: Tensor '[GridSize] <- unsafeVector (fst <$> rngPairs)
    t' :: Tensor '[GridSize] <- unsafeVector (snd <$> rngPairs)
    pure (t, t')
    where 
        pairs l = [(t, t') | t <- l, t' <- l]
        rngPairs = pairs xRange

-- | multivariate 0-mean normal via cholesky decomposition
mvnCholesky :: (KnownDim b, KnownDim c) =>
    Generator -> Tensor '[b, b] -> IO (Tensor '[b, c])
mvnCholesky gen cov = do
    let Just sd = positive 1.0
    samples <- normal gen 0.0 sd
    let l = potrf cov Upper
    let mvnSamp = l !*! samples
    pure mvnSamp

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
    y :: Tensor '[DataDim] <- unsafeVector dataPredictors
    vals :: Tensor '[DataDim] <- unsafeVector dataValues

    -- covariance terms for predictions
    (t, t') <- makeGrid
    let rbf = kernel1d_rbf 1.0 1.0 t t' 
    let priorMu = constant 0 :: Tensor [GridDim, 1]
    let priorCov = (resizeAs rbf) :: Tensor [GridDim, GridDim]

    -- covariance terms for data
    let pairs = [(y, y') | y <- dataPredictors, y' <- dataPredictors]
    t :: Tensor '[DataSize] <- unsafeVector (fst <$> pairs)
    t' :: Tensor '[DataSize] <- unsafeVector (snd <$> pairs)
    let obsCov :: Tensor '[DataDim, DataDim] = 
            resizeAs $ kernel1d_rbf 1.0 1.0 t t'

    -- cross-covariance terms
    let pairs = [(x, y) | x <- xRange, y <- dataPredictors]
    t :: Tensor '[CrossSize] <- unsafeVector (fst <$> pairs)
    t' :: Tensor '[CrossSize] <- unsafeVector (snd <$> pairs)
    let crossCov :: Tensor '[GridDim, DataDim] = 
            resizeAs $ kernel1d_rbf 1.0 1.0 t t'

    -- conditional distribution
    let (postMu, postCov) = 
            condition
                priorMu (constant 0 :: Tensor '[DataDim, 1]) -- mean
                priorCov crossCov obsCov -- covariance matrix terms
                vals -- observed y

    putStrLn "\nObservation coordinates covariance"
    print obsCov
    putStrLn "\nCross covariance"
    print crossCov
    pure $ (postMu, postCov)
    
main :: IO ()
main = do
    (t, t') <- makeGrid
    let rbf = kernel1d_rbf 1.0 1.0 t t'
    let mu = constant 0 :: Tensor [GridDim, GridDim]
    let cov = resizeAs rbf :: Tensor [GridDim, GridDim]
    let regularizationScale = 0.01
    let reg = regularizationScale * eye

    -- Setup prediction axis
    putStrLn "Predictor values"
    print xRange
    putStrLn "\nCovariance based on radial basis function"
    print cov

    -- Prior GP
    putStrLn "\nGP Samples (prior,  rows = values, cols = realizations)"
    gen <- newRNG
    mvnSamp :: Tensor '[GridDim, NSamp] <- mvnCholesky gen (cov + reg)
    print mvnSamp

    -- Observations
    putStrLn "\nObservations: predictor coordinates"
    print dataPredictors 
    putStrLn "\nObservations: values"
    print dataValues
    (postMu, postCov) <- computePosterior

    -- Conditional GP
    putStrLn "\nConditional mu (posterior)"
    print postMu
    putStrLn "\nConditional covariance (posterior)"
    print postCov
    putStrLn "\nGP Conditional Samples (posterior, rows = values, cols = realizations)"
    gen <- newRNG
    mvnSamp :: Tensor '[GridDim, 1] <- mvnCholesky gen (postCov + reg)
    print (postMu + mvnSamp)

    putStrLn "Done"