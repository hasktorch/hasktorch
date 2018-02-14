{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Foreign
import THTypes
import qualified THRandom as TH
import qualified THDoubleTensor as TH
import qualified THDoubleTensorMath as TH

main :: IO ()
main = do
  putStrLn "\nExample Usage of Typed Tensors"
  putStrLn "=============================="
  initialization
  -- matrixVectorOps
  -- valueTransformations

initialization :: IO ()
initialization = do
  header "Initialization"

  section "Zeros" $ do
    ptr <- TH.c_new
    pure ptr

--   putStrLn ""
--   putStrLn "Constant:"
--   constVec :: DoubleTensor '[2] <- constant 2
--   printTensor constVec
--
--   putStrLn ""
--   putStrLn "Initialize 1D vector from list:"
--   listVec :: DoubleTensor '[6] <- fromList1d [1, 2, 3, 4, 5, 6]
--   printTensor listVec
--
--   putStrLn ""
--   putStrLn "Resize 1D vector as 2D matrix:"
--   asMat :: DoubleTensor '[3, 2] <- resizeAs listVec
--   printTensor asMat
--
--   putStrLn "\nInitialize arbitrary dimensions directly from list:"
--   listVec2 :: DoubleTensor '[3, 2] <- fromList [1, 2, 3, 4, 5, 6]
--   printTensor listVec2
--
--   putStrLn "\nRandom values:"
--   gen <- RNG.new
--   randMat :: DoubleTensor '[4, 4] <- uniform gen 1 2
--   printTensor randMat
--
-- valueTransformations :: IO ()
-- valueTransformations = do
--   putStrLn "\nBatch tensor value transformations"
--   putStrLn "-----------------------------------"
--   gen <- RNG.new
--
--   putStrLn "\nRandom matrix:"
--   randMat :: DoubleTensor '[4, 4] <- uniform gen (1.0) (3.0)
--   printTensor randMat
--
--   putStrLn "\nNegated:"
--   neg randMat >>= printTensor
--
--   putStrLn "\nSigmoid:"
--   -- sig :: DoubleTensor '[4, 4] <- Math.sigmoid randMat
--   -- printTensor sig
--
--   -- putStrLn "\nTanh:"
--   -- Math.tanh randMat >>= printTensor
--
--   -- putStrLn "\nLog:"
--   -- Math.log randMat >>= printTensor
--
--   -- putStrLn "\nRound:"
--   -- Math.round randMat >>= printTensor
--
-- matrixVectorOps :: IO ()
-- matrixVectorOps = do
--   putStrLn "\nMatrix/vector operations"
--   putStrLn "------------------------"
--   gen <- RNG.new
--
--   putStrLn "\nRandom matrix:"
--   randMat :: DoubleTensor '[2, 2] <- uniform gen (-1) 1
--   printTensor randMat
--
--   putStrLn "\nConstant vector:"
--   constVec :: DoubleTensor '[2] <- constant 2
--   printTensor constVec
--
--   putStrLn "\nMatrix x vector:"
--   randMat !* constVec >>= printTensor
--
--   putStrLn "\nVector outer product:"
--   constVec `outer` constVec >>= printTensor
--
--   putStrLn "\nVector dot product:"
--   constVec <.> constVec >>= print
--
--   putStrLn "\nMatrix trace:"
--   trace randMat >>= print
header :: String -> IO ()
header h = do
  putStrLn "--------------"
  putStrLn h
  putStrLn "--------------"

section :: forall d . String -> IO (Ptr CTHDoubleTensor) -> IO ()
section title tensor = do
  putStrLn ("\n" ++ title ++ ":")
  -- tensor >>= printTensor


