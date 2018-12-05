{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import Data.List (tails)
import Prelude as P
import Torch.Double as T
import qualified Torch.Core.Random as RNG

-- import Torch.Indef.Dynamic.Tensor.Math (new') -- for testing

import Kernels (kernel1d_rbf)

{- Helper functions -}

-- type GridDim = 9
-- type GridSize = 81
-- type NSamp = 5
-- xRange = [-4..4]

type GridDim = 5
type GridSize = 25
type NSamp = 3
xRange = [-2..2]

makeGrid :: IO (DoubleTensor '[GridSize], DoubleTensor '[GridSize])
makeGrid = do
    x :: DoubleTensor '[GridSize] <- unsafeVector (fst <$> rngPairs)
    x' :: DoubleTensor '[GridSize] <- unsafeVector (snd <$> rngPairs)
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
    Just (t :: DoubleTensor '[3, 3]) <- fromList [2, 4, 6, 0, -1, -8, 0, 0, 96]
    let trg = eye :: DoubleTensor '[3, 3]
    let (invT, invTLU) = gesv (eye :: DoubleTensor '[3, 3]) t
    print t
    print trg
    print invT
    print invTLU
    print (t !*! invT)

testGetri = do
    putStrLn "\n\ngetri test\n\n"
    Just (t :: DoubleTensor '[3, 3]) <- fromList [2, 4, 6, 0, -1, -8, 0, 0, 96]
    let invT = (getri t) :: DoubleTensor '[3, 3]
    print t
    print invT
    print (t !*! invT)

{- Main -}

main = do
    (x, y) <- makeGrid
    let rbf = kernel1d_rbf 1.0 1.0 x y
    print rbf
    let mu = constant 0 :: DoubleTensor [GridDim, GridDim]
    let cov = resizeAs rbf :: DoubleTensor [GridDim, GridDim]

    putStrLn "\nCovariance Matrix"
    print cov

    putStrLn "\nGP Samples (cols = realizations)"
    gen <- newRNG
    mvnSamp :: DoubleTensor '[GridDim, NSamp] <- mvnCholesky gen cov
    print mvnSamp

    -- let (invX, invLU) = gesv (eye :: DoubleTensor '[GridDim, GridDim]) cov
    -- putStrLn "\nCovariance Inverse X"
    -- print invX
    -- putStrLn "\nCovariance Inverse LU"
    -- print invLU

    let invCov = getri cov :: DoubleTensor '[GridDim, GridDim]
    putStrLn "\nInv Covariance"
    print invCov

    putStrLn "\nCheck"
    print $ invCov !*! cov

    putStrLn "Done"
    pure ()