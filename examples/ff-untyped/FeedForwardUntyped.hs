{-# LANGUAGE DataKinds, KindSignatures, TypeFamilies, TypeOperators #-}
{-# LANGUAGE GADTs #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}
module Main where

import Torch.Double.Dynamic hiding (DoubleTensor)

type DoubleTensor = DoubleDynamic

data Weights = W
  { biases :: DoubleTensor
  , nodes :: DoubleTensor
  } deriving (Eq, Show)

{- Simple FF neural network, dynamically typed version, based on JL's example -}

data Network :: * where
  O :: Weights -> Network
  (:~) :: Weights -> Network -> Network
  deriving Show

infixr 5 :~

dispW :: Weights -> IO ()
dispW w = do
  putStrLn "Biases:"
  print (biases w)
  putStrLn "Weights:"
  print (nodes w)

dispN :: Network -> IO ()
dispN (O w) = dispW w
dispN (w :~ n') = putStrLn "Current Layer ::::\n" >> dispW w >> dispN n'

randomWeights :: Word -> Word -> IO Weights
randomWeights i o = do
  gen <- newRNG
  let d1 = someDimsVal [fromIntegral o]
  let d2 = someDimsVal [fromIntegral o, fromIntegral i]
  b <- uniform' d1 gen (-1) 1
  w <- uniform' d2 gen (-1) 1
  pure W { biases = b, nodes = w }

randomData :: Word -> IO DoubleTensor
randomData i = do
  gen <- newRNG
  let someD1 = someDimsVal [fromIntegral i]
  uniform' someD1 gen (-1.0) (1.0)

randomNet :: Word -> [Word] -> Word -> IO Network
randomNet i [] o = O <$> randomWeights i o
randomNet i (h:hs) o = (:~) <$> randomWeights i h <*> randomNet h hs o

runLayer :: Weights -> DoubleTensor -> IO DoubleTensor
runLayer (W wB wN) v = do
  putStrLn "++++++"
  putStrLn "x"
  print wB
  putStrLn ""
  print wN
  putStrLn ""
  print v
  putStrLn ""
  -- print (wN !* v)
  putStrLn "========"
  dt <- addmv 1 wB 1 wN v
  putStrLn "y"
  print dt
  pure dt

runNet :: Network -> DoubleTensor -> IO DoubleTensor
runNet (O w) v     = runLayer w v >>= sigmoid
runNet (w :~ n') v = runLayer w v >>= sigmoid >>= runNet n'


main :: IO ()
main = do
  net <- randomNet 5 [3, 2, 4, 2, 3] 2
  dat <- randomData 5
  putStrLn "Data"
  putStrLn "--------"
  print dat

  putStrLn "Network"
  putStrLn "--------"
  dispN net

  putStrLn "=============================="
  putStrLn "Running the network"
  putStrLn "=============================="
  result <- runNet net dat

  putStrLn "Result"
  putStrLn "--------"
  print result

  putStrLn "Done"
