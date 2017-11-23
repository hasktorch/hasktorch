{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Torch.Core.Random (newRNG)
import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleMath
import Torch.Core.Tensor.Static.DoubleRandom (tds_uniform)

transformations = do
  putStrLn "\nExample bulk tensor transformations"
  putStrLn "-----------------------------------"
  gen <- newRNG
  randMat :: TDS '[4, 4] <- tds_uniform gen (1.0) (3.0)
  putStrLn "\nRandom matrix:"
  tds_p randMat
  putStrLn "\nNegated:"
  tds_p $ tds_neg randMat
  putStrLn "\nSigmoid:"
  tds_p $ tds_sigmoid randMat
  putStrLn "\nTanh:"
  tds_p $ tds_tanh randMat
  putStrLn "\nLog:"
  tds_p $ tds_log randMat
  putStrLn "\nRound:"
  tds_p $ tds_round randMat

matrixVectorOps = do
  putStrLn "\nMatrix/vector operations"
  putStrLn "------------------------"
  gen <- newRNG
  randMat :: TDS '[2, 2] <- tds_uniform gen (-1.0) (1.0)
  let constVec = tds_init 2.0 :: TDS '[2]
  let result = randMat !* constVec
  putStrLn "\nRandom matrix:"
  tds_p randMat
  putStrLn "\nConstant vector:"
  tds_p constVec
  putStrLn "\nMatrix x vector:"
  tds_p $ randMat !* constVec
  putStrLn "\nVector outer product:"
  tds_p $ constVec `tds_outer` constVec
  putStrLn "\nVector dot product:"
  print $ constVec <.> constVec
  putStrLn "\nMatrix trace:"
  print $ tds_trace randMat
  pure ()

main = do
  putStrLn "Statically Typed Tensors Example Usage"
  putStrLn "======================================\n"
  matrixVectorOps
  transformations
  pure ()
