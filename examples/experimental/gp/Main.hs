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

import Kernels (kernel1d_rbf)

{- Helper functions -}

-- type GridDim = 9
-- type GridSize = 81
-- type NSamp = 5
-- xRange = [-4..4]

-- type GridDim = 7
-- type GridSize = 49
-- type NSamp = 5
-- xRange = [-3..3]

type GridDim = 5
type GridSize = GridDim * GridDim
type NSamp = 3
xRange = [-2..2]

xScale = 0.2

type DataDim = 2
type DataSize = DataDim * DataDim
type CrossDim = DataDim * GridDim
dataPredictors = [-0.3, 0.3]
dataValues = [-2.3, 1.5]

type CrossSize = GridDim * DataDim


data DataModel = DataModel {
    predTensor :: Tensor '[DataDim],
    valTensor :: Tensor '[DataDim],
    dataCov :: Tensor '[DataDim, DataDim],
    crossCov :: Tensor '[GridDim, DataDim]
}

-- | cartesian product of all predictor values
makeGrid :: IO (Tensor '[GridSize], Tensor '[GridSize])
makeGrid = do
    x :: Tensor '[GridSize] <- unsafeVector (fst <$> rngPairs)
    x' :: Tensor '[GridSize] <- unsafeVector (snd <$> rngPairs)
    pure (x, x')
    where 
        pairs l = [(x * xScale ,x' * xScale) | x <- l, x' <- l]
        rngPairs = pairs xRange

-- | multivariate 0-mean normal via cholesky decomposition
mvnCholesky gen cov = do
    let Just sd = positive 1.0
    samples <- normal gen 0.0 sd
    let l = potrf cov Upper
    let mvnSamp = l !*! samples
    pure mvnSamp

-- | conditional distribution parameters for X|Y
conditionalXY (muX :: Tensor '[GridDim, 1]) (muY :: Tensor '[DataDim, 1]) covXX covXY covYY y = (postMu, postCov) 
    where
        covYX = transpose2d covXY
        y' = resizeAs y
        postMu = muX + covXY !*! (getri covYY) !*! (y' - muY)
        postCov = covXX - covXY !*! (getri covYY) !*! covYX

{- Main -}

-- | produce observation data
addObservations :: IO DataModel
addObservations = do
    y :: Tensor '[DataDim] <- unsafeVector dataPredictors
    vals :: Tensor '[DataDim] <- unsafeVector dataValues

    -- covariance terms for predictions
    (t, t') <- makeGrid
    let rbf = kernel1d_rbf 1.0 1.0 t t' 
    let mu = constant 0 :: Tensor [GridDim, GridDim]
    let predCov = (resizeAs rbf) :: Tensor [GridDim, GridDim]

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
    let (postMu, postCov) = conditionalXY (constant 0 :: Tensor '[GridDim, 1]) (constant 0 :: Tensor '[DataDim, 1]) predCov crossCov obsCov vals
        
    -- putStrLn "\nPrediction coordinates covariance"
    -- print predCov

    putStrLn "\nObservations: predictor coordinates"
    print dataPredictors 

    putStrLn "\nObservations: values"
    print dataValues

    putStrLn "\nObservation coordinates covariance"
    print obsCov

    putStrLn "\nCross covariance"
    print crossCov

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

    pure $ DataModel y vals obsCov crossCov

-- | condition on observations
condition :: IO ()
condition = do
    -- let combinedX = cat1d x obsX
    -- let crossCov = kernel1d_rbf 1.0 1.0 x obsX
    --postMu = mu + (transpose2d crossCov) !*! (getri )
    pure undefined

main = do
    (x, y) <- makeGrid
    let rbf = kernel1d_rbf 1.0 1.0 x y
    let mu = constant 0 :: Tensor [GridDim, GridDim]
    let cov = resizeAs rbf :: Tensor [GridDim, GridDim]

    putStrLn "Predictor values"
    print [x * xScale | x <- xRange]

    -- putStrLn "Radial Basis Function Kernel Values"
    -- print rbf

    putStrLn "\nCovariance based on radial basis function"
    print cov

    -- let invCov = getri cov :: Tensor '[GridDim, GridDim]
    -- putStrLn "\nInv Covariance"
    -- print invCov

    -- putStrLn "\nCheck Inversion Operation (should recover identity matrix)"
    -- print $ invCov !*! cov

    putStrLn "\nGP Samples (prior,  rows = values, cols = realizations)"
    gen <- newRNG
    mvnSamp :: Tensor '[GridDim, NSamp] <- mvnCholesky gen cov
    print mvnSamp

    addObservations

    putStrLn "Done"

    pure ()