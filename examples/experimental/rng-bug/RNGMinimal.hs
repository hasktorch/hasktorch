{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Torch.Double as Torch
import qualified Torch.Core.Random as RNG
-- import Torch.Indef.Dynamic.Tensor (_normal)
-- import qualified Torch.Indef.Dynamic.Tensor as Dyn

genRand1 :: Generator -> IO (Tensor '[2])
genRand1 gen = do
    vals :: Tensor '[2] <- normal gen 0.0 sd 
    pure vals
    where
        Just sd = positive 1.0

genRand2 :: IO (Generator) -> IO (Tensor '[2])
genRand2 gen = do
    currGen <- gen
    vals :: Tensor '[2] <- normal currGen 0.0 sd 
    pure vals
    where
        Just sd = positive 1.0

genRand3 :: Generator -> IO (Tensor '[2])
genRand3 gen = do
    vals :: Tensor '[2] <- normal gen 0.0 sd 
    pure vals
    where
        Just sd = positive 1.0
{-# NOINLINE genRand3 #-}

genRand4 :: Generator -> IO (Tensor '[2])
genRand4 gen = do
    vals :: Tensor '[2] <- normal gen 0.0 sd 
    putStrLn "genRand4:"
    print vals
    pure vals
    where
        Just sd = positive 1.0

genRand5 :: Generator -> IO (Tensor '[2])
genRand5 gen = do
    !(vals :: Tensor '[2]) <- (normal gen 0.0 sd)
    putStrLn "genRand5:"
    print vals
    pure $ vals
    where
        Just sd = positive 1.0

genRand6 :: IO ()
genRand6 = do
    gen <- newRNG
    let Just sd = positive 1.0

    !(vals1 :: Tensor '[2]) <- normal gen 0.0 sd 
    print vals1
    !(vals2 :: Tensor '[2]) <- normal gen 0.0 sd 
    print vals2
    print vals1
    -- print [vals1]
    -- print [vals2]
    putStrLn ""
    print vals1
    print vals2
    putStrLn ""
    let !result = [vals1, vals2]
    print result

genRand7 :: IO ()
genRand7 = do
    gen <- newRNG
    let Just sd = positive 1.0
    !(vals1 :: Tensor '[2]) <- normal gen 0.0 sd 
    print vals1
    !(vals2 :: Tensor '[2]) <- normal gen 0.0 sd 
    print vals2
    putStrLn ""
    print vals1
    print vals2


-- genRand8 :: IO ()
-- genRand8 = do
--     gen <- newRNG
--     let Just sd = positive 1.0
--     let
--     vals1 <- Dyn.new (Dims :: '[3])
--     _normal vals1 gen 0.0 sd
--     print vals1
--     vals2 <- Dyn.new (Dims :: '[3])
--     _normal vals2 gen 0.0 sd
--     print vals2
--     putStrLn ""
--     print vals1
--     print vals2
   

main = do

    let iter = 3

    putStrLn "\nV1 ========================================\n"
    gen <- newRNG
    test <- mapM (\_ -> genRand1 gen)   ([0..iter] :: [Integer])
    print test

    putStrLn "\nV2 ========================================\n"
    let gen = newRNG
    test <- mapM (\_ -> genRand2 gen)   ([0..iter] :: [Integer])
    print test

    putStrLn "\nV3 ========================================\n"
    gen <- newRNG
    test <- mapM (\_ -> genRand3 gen)   ([0..iter] :: [Integer])
    print test

    putStrLn "\nV4 ========================================\n"
    gen <- newRNG
    test <- mapM (\_ -> genRand4 gen)   ([0..iter] :: [Integer])
    print test

    putStrLn "\nV5 ========================================\n"
    gen <- newRNG
    test <- mapM (\_ -> genRand5 gen)   ([0..iter] :: [Integer])
    print test

    putStrLn "\nV6 ========================================\n"

    genRand6

    putStrLn "\nV7 ========================================\n"

    genRand7

    putStrLn "\nV8 ========================================\n"

    -- genRand8