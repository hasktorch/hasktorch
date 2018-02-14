{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Control.Monad (void)

import qualified Torch.Core.Random as RNG (new)
import Torch.Core.Tensor.Static
import Torch.Core.Tensor.Static.Math as Math
import qualified Torch.Core.Tensor.Static.Random as R (uniform)

main :: IO ()
main = do
  h1 "Example Usage of Typed Tensors"
  initialization
  matrixVectorOps
  valueTransformations

initialization :: IO ()
initialization = void $ do
  h2 "Initialization"

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

matrixVectorOps :: IO ()
matrixVectorOps = do
  h2 "Matrix/vector operations"
  gen <- RNG.new

  randMat :: DoubleTensor '[2, 2] <-
    section "Random matrix" $
      uniform gen (-1) 1

  constVec :: DoubleTensor '[2] <-
    section "Constant vector" $
      constant 2

  section "Matrix x vector" $
    randMat !* constVec

  section "Vector outer product" $
    constVec `outer` constVec

  putStrLn "\nVector dot product:"
  constVec <.> constVec >>= print

  putStrLn "\nMatrix trace:"
  trace randMat >>= print


valueTransformations :: IO ()
valueTransformations = do
  h2 "Batch tensor value transformations"

  gen <- RNG.new

  randMat :: DoubleTensor '[4, 4] <-
    section "Random matrix" $ do
      uniform gen (1.0) (3.0)

  section "Negated" $
    neg randMat

  putStrLn "\nSigmoid:"
  -- sig :: DoubleTensor '[4, 4] <- Math.sigmoid randMat
  -- printTensor sig

  -- putStrLn "\nTanh:"
  -- Math.tanh randMat >>= printTensor

  -- putStrLn "\nLog:"
  -- Math.log randMat >>= printTensor

  -- putStrLn "\nRound:"
  -- Math.round randMat >>= printTensor

-- ========================================================================= --
-- helpers

h2 :: String -> IO ()
h2 = header '-'

h1 :: String -> IO ()
h1 = header '='

header :: Char -> String -> IO ()
header c h = do
  putStrLn ""
  putStrLn $ replicate (length h) c
  putStrLn h
  putStrLn $ replicate (length h) c



section :: forall d . String -> IO (DoubleTensor d) -> IO (DoubleTensor d)
section title tensor = do
  putStrLn ("\n" ++ title ++ ":")
  t <- tensor
  printTensor t
  pure t


