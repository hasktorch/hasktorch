{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import qualified Torch.Core.Random as RNG (new)
import Torch.Core.Tensor.Static
import           Torch.Core.Tensor.Static.Math
import qualified Torch.Core.Tensor.Static.Math as Math
import qualified Torch.Core.Tensor.Static.Random as R (uniform)


header :: String -> IO ()
header h = do
  putStrLn "--------------"
  putStrLn h
  putStrLn "--------------"

section :: forall d . String -> IO (DoubleTensor d) -> IO (DoubleTensor d)
section title tensor = do
  putStrLn ("\n" ++ title ++ ":")
  t <- tensor
  printTensor t
  pure t

initialization :: IO ()
initialization = do
  header "Initialization"

  section "Zeros" $ do
    zeroMat :: DoubleTensor '[3, 2] <- new
    pure zeroMat

  section "Constant" $ do
    constVec :: DoubleTensor '[2] <- constant 2
    pure constVec

  listVec :: DoubleTensor '[6] <-
    section "Initialize 1D vector from list" $
      fromList1d [1, 2, 3, 4, 5, 6]

  section "Resize 1D vector as 2D matrix" $ do
    asMat :: DoubleTensor '[3, 2] <- resizeAs listVec
    pure asMat

  section "Initialize arbitrary dimensions directly from list" $ do
    listVec2 :: DoubleTensor '[3, 2] <- fromList [1, 2, 3, 4, 5, 6]
    pure listVec2

  section "Random values" $ do
    gen <- RNG.new
    randMat :: DoubleTensor '[4, 4] <- uniform gen 1 2
    pure randMat

  pure ()

valueTransformations :: IO ()
valueTransformations = do
  putStrLn "\nBatch tensor value transformations"
  putStrLn "-----------------------------------"
  gen <- RNG.new

  putStrLn "\nRandom matrix:"
  randMat :: DoubleTensor '[4, 4] <- uniform gen (1.0) (3.0)
  printTensor randMat

  putStrLn "\nNegated:"
  neg randMat >>= printTensor

  putStrLn "\nSigmoid:"
  -- sig :: DoubleTensor '[4, 4] <- Math.sigmoid randMat
  -- printTensor sig

  -- putStrLn "\nTanh:"
  -- Math.tanh randMat >>= printTensor

  -- putStrLn "\nLog:"
  -- Math.log randMat >>= printTensor

  -- putStrLn "\nRound:"
  -- Math.round randMat >>= printTensor

matrixVectorOps :: IO ()
matrixVectorOps = do
  putStrLn "\nMatrix/vector operations"
  putStrLn "------------------------"
  gen <- RNG.new

  putStrLn "\nRandom matrix:"
  randMat :: DoubleTensor '[2, 2] <- uniform gen (-1) 1
  printTensor randMat

  putStrLn "\nConstant vector:"
  constVec :: DoubleTensor '[2] <- constant 2
  printTensor constVec

  putStrLn "\nMatrix x vector:"
  randMat !* constVec >>= printTensor

  putStrLn "\nVector outer product:"
  constVec `outer` constVec >>= printTensor

  putStrLn "\nVector dot product:"
  constVec <.> constVec >>= print

  putStrLn "\nMatrix trace:"
  trace randMat >>= print

main :: IO ()
main = do
  putStrLn "\nExample Usage of Typed Tensors"
  putStrLn "=============================="
  initialization
  matrixVectorOps
  valueTransformations

