{-# LANGUAGE DataKinds #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Control.Monad (void)

#ifdef CUDA
import Torch.Cuda.Double
import qualified Torch.Cuda.Double as Math
#else
import Torch.Double
import qualified Torch.Core.Random as RNG (newRNG)
import qualified Torch.Double as Math
#endif


#ifdef CUDA
tensortype = "CUDA Tensors"
#else
tensortype = "CPU Tensors"
#endif

main :: IO ()
main = do
  h1 $ "Example Usage of Typed " ++ tensortype
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
    let constVec :: DoubleTensor '[2] = constant 2
    pure constVec

  listVec :: DoubleTensor '[6] <-
    section' "Initialize 1D vector from list" $ do
      let Right v = vector [1, 2, 3, 4, 5, 6]
      pure v

  section "Resize 1D vector as 2D matrix" $ do
    let asMat :: Tensor '[3, 2] = resizeAs listVec
    pure asMat

  section "Initialize arbitrary dimensions directly from list" $ do
    let Just (listVec2 :: DoubleTensor '[3, 2]) = fromList [1, 2, 3, 4, 5, 6]
    pure listVec2

  section "Random values" $ do
    let Just p = ord2Tuple (1, 2)
#ifdef CUDA
    randMat <- uniform p
#else
    randMat <- (`uniform` p) =<< RNG.newRNG
#endif
    pure (randMat :: DoubleTensor '[4, 4])

matrixVectorOps :: IO ()
matrixVectorOps = void $ do
  h2 "Matrix/vector operations"

  randMat :: DoubleTensor '[2, 2] <-
    section' "Random matrix" $ do
      let Just p = ord2Tuple (-1, 1)
#ifdef CUDA
      uniform p
#else
      RNG.newRNG >>= (`uniform` p)
#endif

  constVec :: DoubleTensor '[2] <-
    section' "Constant vector" $
      pure (constant 2)

  section "Matrix x vector" $
    pure $ randMat !* constVec

  section "Vector outer product" $
    pure $ outer constVec constVec

  showSection "Vector dot product" $
    pure $ constVec <.> constVec

  showSection "Matrix trace" $
    ttrace randMat


valueTransformations :: IO ()
valueTransformations = void $ do
  h2 "Batch tensor value transformations"

  randMat :: DoubleTensor '[4, 4] <-
    section' "Random matrix" $ do
      let Just p = ord2Tuple (1, 3)
#ifdef CUDA
      uniform p
#else
      RNG.newRNG >>= (`uniform` p)
#endif

  section "Negated" $
    neg randMat

  section "Sigmoid" $ do
    let sig :: DoubleTensor '[4, 4] = Math.sigmoid randMat
    pure sig

  section "Tanh" $ pure $ Math.tanh randMat

  section "Log" $ pure $ Math.log randMat

  section "Round" $ pure $ Math.round randMat

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
section a b = _section print a b >> pure ()

section' :: Dimensions d => String -> IO (DoubleTensor d) -> IO (DoubleTensor d)
section' = _section print

showSection :: Show x => String -> IO x -> IO ()
showSection a b = _section print a b >> pure ()

_section :: (x -> IO ()) -> String -> IO x -> IO x
_section printer title buildit = do
  putStrLn ("\n" ++ title ++ ":")
  t <- buildit
  printer t
  pure t



