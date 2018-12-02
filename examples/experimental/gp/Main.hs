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

import Kernels (kernel1d_rbf)

{- Helper functions -}

-- type D = 11

type GridDim = 9
type GridSize = 81
type NSamp = 5

makeGrid :: IO (DoubleTensor '[GridSize], DoubleTensor '[GridSize])
makeGrid = do
    x :: DoubleTensor '[GridSize] <- unsafeVector (fst <$> rngPairs)
    x' :: DoubleTensor '[GridSize] <- unsafeVector (snd <$> rngPairs)
    pure (x, x')
    where 
        pairs l = [(x * 0.1 ,x' * 0.1) | x <- l, x' <- l]
        rngPairs = pairs [-4..4]

-- | multivariate 0-mean normal via cholesky decomposition
mvnCholesky gen cov = do
    let Just sd = positive 1.0
    samples <- normal gen 0.0 sd
    let l = potrf cov Upper
    let mvnSamp = l !*! samples
    pure mvnSamp

main = do
    (x, y) <- makeGrid
    let rbf = kernel1d_rbf 1.0 1.0 x y
    print rbf
    let mu = constant 0 :: DoubleTensor [GridDim, GridDim]
    let cov = resizeAs rbf :: DoubleTensor [GridDim, GridDim]
    -- result :: Tensor '[10, n] <- multivariate_normal gen mu eigenvector eigenvalue

    putStrLn "\nCovariance Matrix"
    print cov

    putStrLn "\nGP Samples (cols = realizations)"
    gen <- newRNG
    mvnSamp :: DoubleTensor '[GridDim, NSamp] <- mvnCholesky gen cov
    print mvnSamp

    putStrLn "Done"