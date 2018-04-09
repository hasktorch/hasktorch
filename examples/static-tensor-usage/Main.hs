{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Control.Monad (void)

import Torch
import qualified Torch.Core.Random as RNG (new)
import qualified Torch as Math
-- import Torch.Core.Tensor.Static.Math as Math
-- import Torch.Core.Tensor.Static.Math.Infix

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
    section' "Initialize 1D vector from list" $
      fromList1d [1, 2, 3, 4, 5, 6]

  section "Resize 1D vector as 2D matrix" $ do
    asMat :: DoubleTensor '[3, 2] <- resizeAs listVec
    pure asMat

  section "Initialize arbitrary dimensions directly from list" $ do
    listVec2 :: DoubleTensor '[3, 2] <- fromList [1, 2, 3, 4, 5, 6]
    pure listVec2

  section "Random values" $ do
    gen :: Generator <- RNG.new
    randMat :: DoubleTensor '[4, 4] <- uniform gen 1 2
    pure randMat

matrixVectorOps :: IO ()
matrixVectorOps = void $ do
  h2 "Matrix/vector operations"
  gen <- RNG.new

  randMat :: DoubleTensor '[2, 2] <-
    section' "Random matrix" $
      uniform gen (-1) 1

  constVec :: DoubleTensor '[2] <-
    section' "Constant vector" $
      constant 2

  section "Matrix x vector" $
    pure $ randMat !* constVec

  section "Vector outer product" $
    outer constVec constVec

  showSection "Vector dot product" $
    pure $ constVec <.> constVec

  showSection "Matrix trace" $
    trace randMat


valueTransformations :: IO ()
valueTransformations = void $ do
  h2 "Batch tensor value transformations"

  gen <- RNG.new

  randMat :: DoubleTensor '[4, 4] <-
    section' "Random matrix" $ do
      uniform gen (1.0) (3.0)

  section "Negated" $
    neg randMat

  section "Sigmoid" $ do
    sig :: DoubleTensor '[4, 4] <- Math.sigmoid randMat
    pure sig

  section "Tanh" $ Math.tanh randMat

  section "Log" $ Math.log randMat

  section "Round" $ Math.round randMat

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

section :: Dimensions d => String -> IO (DoubleTensor d) -> IO ()
section a b = _section printTensor a b >> pure ()

section' :: Dimensions d => String -> IO (DoubleTensor d) -> IO (DoubleTensor d)
section' = _section printTensor

showSection :: Show x => String -> IO x -> IO ()
showSection a b = _section print a b >> pure ()

_section :: (x -> IO ()) -> String -> IO x -> IO x
_section printer title buildit = do
  putStrLn ("\n" ++ title ++ ":")
  t <- buildit
  printer t
  pure t



