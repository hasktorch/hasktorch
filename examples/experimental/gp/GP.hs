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

-- Kernels

m1 = do
    Just (mat :: DoubleTensor '[5]) <- fromList [1, 2, 3, 4, 5]
    pure mat

m2 = do
    Just (mat :: DoubleTensor '[5]) <- fromList [1.1, 2, 3, 4, 5]
    pure mat

{- Helper functions -}

makeGrid :: IO (DoubleTensor '[55], DoubleTensor '[55])
makeGrid = do
    x :: DoubleTensor '[55] <- unsafeVector (fst <$> rngPairs)
    y :: DoubleTensor '[55] <- unsafeVector (snd <$> rngPairs)
    pure (x, y)
    where 
        pairs l = [(x * 0.1 ,y * 0.1) | (x:ys) <- tails l, y <- ys]
        rngPairs = pairs [-5..5]

main = do
    (x, y) <- makeGrid
    let rbf = kernel1d_rbf 1.0 1.0 x y
    print rbf
    putStrLn "Done"