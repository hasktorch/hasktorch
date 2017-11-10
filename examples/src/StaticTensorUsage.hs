{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Random
import StaticTensorDouble
import StaticTensorDoubleMath
import StaticTensorDoubleRandom

matrixVectorMultiplication = do
  putStrLn "Matrix Vector Multiplication"
  putStrLn "----------------------------"
  gen <- newRNG
  randMat :: TDS '[2, 2] <- tds_uniform tds_new gen (-1.0) (1.0)
  let constVec = tds_init 2.0 :: TDS '[2]
  let result = randMat !* constVec
  putStrLn "\nRandom matrix:"
  tds_p randMat
  putStrLn "\nConstant vector:"
  tds_p constVec
  putStrLn "\nResult:"
  tds_p result
  pure ()

main = do
  putStrLn "Statically Typed Tensors Example Usage"
  putStrLn "======================================\n"
  matrixVectorMultiplication
  pure ()
