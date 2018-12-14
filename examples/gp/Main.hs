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
import Graphics.Vega.VegaLite
import Prelude as P
import Torch.Double as T
import qualified Torch.Core.Random as RNG
import GHC.TypeLits (natVal)
import Data.Proxy (Proxy(..))


import Kernels (kernel1d_rbf)

-- type GridDim = 9
-- type GridSize = 81
-- type NSamp = 5

-- type GridDim = 7
-- type GridSize = 49
-- type NSamp = 5

-- Function predicted value locations (on a grid)
type GridDim = 5
type GridSize = GridDim * GridDim
type NSamp = 3

xRange = (*) (0.2) <$> ([-halfwidth .. halfwidth] :: [HsReal])
    where
        gridDim = natVal (Proxy :: Proxy GridDim)
        halfwidth = fromIntegral (P.div gridDim 2)

-- Observed data
type DataDim = 2
type DataSize = DataDim * DataDim
dataPredictors = [-0.3, 0.3]
dataValues = [-2.3, 1.5]

-- Cross-covariance dimensions
type CrossDim = DataDim * GridDim
type CrossSize = GridDim * DataDim

{- Helper functions -}

-- | cartesian product of all predictor values
makeGrid :: IO (Tensor '[GridSize], Tensor '[GridSize])
makeGrid = do
    x :: Tensor '[GridSize] <- unsafeVector (fst <$> rngPairs)
    x' :: Tensor '[GridSize] <- unsafeVector (snd <$> rngPairs)
    pure (x, x')
    where 
        pairs l = [(x, x') | x <- l, x' <- l]
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
conditionalXY (muX :: Tensor '[GridDim, 1]) (muY :: Tensor '[DataDim, 1]) covXX covXY covYY y =
    (postMu, postCov) 
    where
        covYX = transpose2d covXY
        y' = resizeAs y
        postMu = muX + covXY !*! (getri covYY) !*! (y' - muY)
        postCov = covXX - covXY !*! (getri covYY) !*! covYX

{- Main -}

-- | produce observation data
-- addObservations :: IO ()
addObservations = do
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
            conditionalXY 
                priorMu (constant 0 :: Tensor '[DataDim, 1]) -- mean
                priorCov crossCov obsCov -- covariance matrix terms
                vals -- observed y
        
    putStrLn "\nObservations: predictor coordinates"
    print dataPredictors 

    putStrLn "\nObservations: values"
    print dataValues

    putStrLn "\nObservation coordinates covariance"
    print obsCov

    putStrLn "\nCross covariance"
    print crossCov

    pure $ (postMu, postCov)

main = do
    (x, y) <- makeGrid
    let rbf = kernel1d_rbf 1.0 1.0 x y
    let mu = constant 0 :: Tensor [GridDim, GridDim]
    let cov = resizeAs rbf :: Tensor [GridDim, GridDim]

    putStrLn "Predictor values"
    print [x | x <- xRange]

    putStrLn "\nCovariance based on radial basis function"
    print cov

    putStrLn "\nGP Samples (prior,  rows = values, cols = realizations)"
    gen <- newRNG
    mvnSamp :: Tensor '[GridDim, NSamp] <- mvnCholesky gen cov
    print mvnSamp

    (postMu, postCov) <- addObservations

    putStrLn "\nConditional mu (posterior)"
    print postMu

    putStrLn "\nConditional covariance (posterior)"
    print postCov

    putStrLn "\nGP Conditional Samples (posterior, rows = values, cols = realizations)"
    gen <- newRNG
    let regularizationScale = 1.0
    let reg = regularizationScale * eye
    mvnSamp :: Tensor '[GridDim, 1] <- mvnCholesky gen (postCov + reg)
    print (postMu + mvnSamp)

    putStrLn "Done"

    pure ()