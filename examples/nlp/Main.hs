-------------------------------------------------------------------------------
-- |
-- Module    :  Main
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
--
-- This tutorial is intended to familiarize a new user of haskell and deep learning
-- with the current state of haskell bindings to torch, a deep learning library.
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds, ScopedTypeVariables, GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# OPTIONS_GHC -fno-cse #-}
module Main where

import Torch.Cuda.Double
import Numeric.Backprop
import System.IO.Unsafe
import Prelude hiding ((!!))

-- You can make tensors from lists:
tensorIntro = do
  let Just (v :: Tensor '[2]) = fromList [1..2]
  print v
  let Just (m :: Tensor '[2, 3]) = fromList [1..2*3]
  print m
  let Just (t :: Tensor '[2, 3, 2]) = fromList [1..2*3*2]
  print t
  let Just (r'4:: Tensor '[2, 3, 2, 3]) = fromList [1..2*3*2*3]
  print r'4

-- keep in mind that `fromList` will check element size
tensorEdgecases = do
  print (fromList []     :: Maybe (Tensor '[2, 3]))
  print (fromList [1..5] :: Maybe (Tensor '[2   ]))

-- * Indexing

-- You can index them like so:

tensorIndexing = do
  let Just (v :: Tensor '[2]) = fromList [1..2]
  print v
  print (v !! 0 :: Tensor '[1])
  let Just (m :: Tensor '[2, 3]) = fromList [1..2*3]
  print m
  print (m !! 1 :: Tensor '[3])
  let Just (t :: Tensor '[2, 3, 2]) = fromList [1..2*3*2]
  print t
  print (t !! 2  :: Tensor '[2,3])
  let Just (r'4 :: Tensor '[2, 3, 2, 3]) = fromList [1..2*3*2*3]
  print (r'4 !! 0  :: Tensor '[3,2,3])
  print (r'4 !! 1  :: Tensor '[2,2,3])
  print (r'4 !! 2  :: Tensor '[2,3,3])
  print (r'4 !! 3  :: Tensor '[2,3,2])

-- Also, we can make tensors with random inputs

tensorRandom = do
  t :: Tensor '[3,5] <- random
  print t
  c :: Tensor '[3,5] <- uniform 0 1
  print c
  n :: Tensor '[3,5] <- normal 5 1
  print n

-- You can do normal operations on them:
tensorOps = do
  let Just (a :: Tensor '[3]) = fromList [1,2,3]
  let Just (b :: Tensor '[3]) = fromList [4,5,6]
  print (a + b)
  cat1d a b >>= print

  r :: Tensor '[6] <- catArray (asDynamic <$> [a, b]) 0
  print r

  let Just (a':: Tensor '[3,3])  = fromList [1..9]
  let Just (b':: Tensor '[3,2])  = fromList [10..15]
  cat2d1 a' b' >>= print

  let Just (a'' :: Tensor '[3,2]) = fromList [1..6]
  let Just (b'' :: Tensor '[2,2]) = fromList [7..10]
  cat2d0 a'' b'' >>= print

  r :: Tensor '[5,2] <- catArray ([asDynamic a', asDynamic b', asDynamic a'']) (-1)
  print r

  r :: Tensor '[3,7] <- catArray ([asDynamic a'', asDynamic b'', asDynamic b']) 0
  print r

-- but this will fail:
  -- catArray (asDynamic <$> [a', b']) (-2) >>= \(r :: Tensor '[6,2]) -> print r


-- reshaping is what you would expect
tensorReshape = do
  x :: Tensor '[2,3,4] <- uniform 0 10
  print x
  r :: Tensor '[2,12] <- view x
  print r


---- Autodiff is...
----   ...currently offloaded onto backprop!
tensorAutoDiff = do
  -- There is no notion of a 'Variable' from pytorch < 0.4
  -- instead we rely on backprop's `BVar`
  --
  -- Here we take an input, pass it into the backprop'd identity function
  -- and evaluate it for its output value (itself)
  print (evalBP id x)

  -- Alternatively, we can do the same thing but only inspect it's gradient:
  print $ gradBP id x

  -- Both the output and gradient can be extracted at the same time with
  -- 'backprop'
  let (o, g) = backprop id x
  print o
  print g

  -- We can perform math using the `Num`, `Floating`, and `Fractional`
  -- instances of tensors:
  let (out, (gx, gy)) = backprop2 (+) x y
  print out
  print gx
  print gy

  -- We can also write our own backprop-able functions.
  let (out, (gx, gy)) = backprop2 sse x y
  print out
  print gx
  print gy

 where
  x, y :: Tensor '[3]
  Just x = vector [1,2,3]
  Just y = vector [4,5,6]

  -- backprop will handle management of all backward passes if
  -- given @BVar@s
  sse
    :: Reifies s W
    => BVar s (Tensor '[3]) -> BVar s (Tensor '[3]) -> BVar s Double
  sse x y = sumallBP ((y - x) ** 2)


  -- However, here we have a library-specific function, @sumall@,
  -- where we must specify how to compute the derivative.
  sumallBP :: Reifies s W => BVar s (Tensor '[3]) -> BVar s Double
  sumallBP = liftOp1 . op1 $ \t ->
    (sumall t, unsafePerformIO . constant)
  {-# NOINLINE sumallBP #-}


-- running everything in a main loop:
main :: IO ()
main = do
  section "Intro" tensorIntro
  section "Edgecases" tensorEdgecases
  section "Indexing" tensorIndexing
  section "Random" tensorRandom
  section "Ops" tensorOps
  section "Reshape" tensorReshape
  section "AutoDiff" tensorAutoDiff
 where
  section :: String -> IO () -> IO ()
  section t action = do
    putStrLn "================"
    putStrLn t
    putStrLn "================"
    action
    putStrLn ""

