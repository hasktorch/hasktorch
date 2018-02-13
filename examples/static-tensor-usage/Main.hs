{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import qualified Torch.Core.Random as R (new)
import Torch.Core.Tensor.Static
import Torch.Core.Tensor.Static.Math
import Torch.Core.Tensor.Static.Random (uniform)

initialization :: IO ()
initialization = do
  putStrLn "\nInitialization"
  putStrLn "--------------"

  putStrLn "\nZeros:"
  let zeroMat = tds_new :: TDS '[3,2]
  tds_p zeroMat

  putStrLn "\nConstant:"
  let constVec = tds_init 2.0 :: TDS '[2]
  tds_p constVec

  putStrLn "\nInitialize 1D vector from list:"
  let listVec = tds_fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: TDS '[6]
  tds_p listVec

  putStrLn "\nResize 1D vector as 2D matrix:"
  let asMat = tds_resize listVec :: TDS '[3, 2]
  -- let asMat = tds_resize listVec :: TDS '[3, 3] -- won't type check
  tds_p asMat

  putStrLn "\nInitialize arbitrary dimensions directly from list:"
  let listVec2 = tds_fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: TDS '[3, 2]
  tds_p listVec2

  putStrLn "\nRandom values:"
  gen <- newRNG
  randMat :: TDS '[4, 4] <- tds_uniform gen (1.0) (3.0)
  tds_p randMat

valueTransformations :: IO ()
valueTransformations = do
  putStrLn "\nBatch tensor value transformations"
  putStrLn "-----------------------------------"
  gen <- newRNG

  putStrLn "\nRandom matrix:"
  randMat :: TDS '[4, 4] <- tds_uniform gen (1.0) (3.0)
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

matrixVectorOps :: IO ()
matrixVectorOps = do
  putStrLn "\nMatrix/vector operations"
  putStrLn "------------------------"
  gen <- newRNG

  putStrLn "\nRandom matrix:"
  randMat :: TDS '[2, 2] <- tds_uniform gen (-1.0) (1.0)
  tds_p randMat

  putStrLn "\nConstant vector:"
  let constVec = tds_init 2.0 :: TDS '[2]
  tds_p constVec

  putStrLn "\nMatrix x vector:"
  tds_p $ randMat !* constVec

  putStrLn "\nVector outer product:"
  tds_p $ constVec `tds_outer` constVec

  putStrLn "\nVector dot product:"
  print $ constVec <.> constVec

  putStrLn "\nMatrix trace:"
  print $ tds_trace randMat

main :: IO ()
main = do
  putStrLn "\nExample Usage of Typed Tensors"
  putStrLn "=============================="
  initialization
  matrixVectorOps
  valueTransformations
  pure ()
