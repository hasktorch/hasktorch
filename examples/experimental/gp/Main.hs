{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

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
type GridSize = 25
type NSamp = 3
xRange = [-2..2]

-- | cartesian product of all predictor values
makeGrid :: IO (Tensor '[GridSize], Tensor '[GridSize])
makeGrid = do
    x :: Tensor '[GridSize] <- unsafeVector (fst <$> rngPairs)
    x' :: Tensor '[GridSize] <- unsafeVector (snd <$> rngPairs)
    pure (x, x')
    where 
        pairs l = [(x * 0.1 ,x' * 0.1) | x <- l, x' <- l]
        rngPairs = pairs xRange

-- | multivariate 0-mean normal via cholesky decomposition
mvnCholesky gen cov = do
    let Just sd = positive 1.0
    samples <- normal gen 0.0 sd
    let l = potrf cov Upper
    let mvnSamp = l !*! samples
    pure mvnSamp

-- | conditional distribution parameters for X|Y
conditionalXY muX muY covXX covXY covYX covYY x y = 
    (postMu, postCov)
    where
        postMu = muX + covXY !*! (getri covYY) !*! (y - muY)
        postCov = covXX - covXY !*! (getri covYY) !*! covYX

-- | conditional distribution parameters for X|Y
conditionalYX muX muY covXX covXY covYX covYY x y = 
    (postMu, postCov)
    where
        postMu = muY + covYX !*! (getri covXX) !*! (x - muX)
        postCov = covYY - covYX !*! (getri covXX) !*! covXY

{- Lapack sanity checks -}

testGesv = do
    putStrLn "\n\ngesv test\n\n"
    Just (t :: Tensor '[3, 3]) <- fromList [2, 4, 6, 0, -1, -8, 0, 0, 96]
    let trg = eye :: Tensor '[3, 3]
    let (invT, invTLU) = gesv (eye :: Tensor '[3, 3]) t
    print t
    print trg
    print invT
    print invTLU
    print (t !*! invT)

testGetri = do
    putStrLn "\n\ngetri test\n\n"
    Just (t :: Tensor '[3, 3]) <- fromList [2, 4, 6, 0, -1, -8, 0, 0, 96]
    let invT = (getri t) :: Tensor '[3, 3]
    print t
    print invT
    print (t !*! invT)

{- Main -}

-- | produce observation data
addObservations :: IO (Tensor '[2], Tensor '[2], Tensor '[2, 2])
addObservations = do
    let xList = [-1.4, 0.3]
    obsX :: Tensor '[2] <- unsafeVector xList
    obsY :: Tensor '[2] <- unsafeVector [0.4, 1.8]
    let pairs = [(x,x') | x <- xList, x' <- xList]
    x :: Tensor '[4] <- unsafeVector (fst <$> pairs)
    x' :: Tensor '[4] <- unsafeVector (snd <$> pairs)
    let obsCov :: Tensor '[2, 2] = resizeAs $ kernel1d_rbf 1.0 1.0 x x'
    -- print obsCov
    pure (obsX, obsY, obsCov)

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
    putStrLn "Radial Basis Function Kernel Values"
    print rbf
    let mu = constant 0 :: Tensor [GridDim, GridDim]
    let cov = resizeAs rbf :: Tensor [GridDim, GridDim]

    putStrLn "\nReshaped as a Covariance Matrix"
    print cov

    let invCov = getri cov :: Tensor '[GridDim, GridDim]
    putStrLn "\nInv Covariance"
    print invCov

    putStrLn "\nCheck Inversion Operation (should recover identity matrix)"
    print $ invCov !*! cov

    putStrLn "\nGP Samples (cols = realizations)"
    gen <- newRNG
    mvnSamp :: Tensor '[GridDim, NSamp] <- mvnCholesky gen cov
    print mvnSamp

    addObservations

    putStrLn "Done"

    pure ()