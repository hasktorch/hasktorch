{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

{- Optimizer implementations run on standard test functions -}

import Control.Monad (foldM, forM_, when)
import Data.Default.Class
import GHC.Generics
import GHC.TypeNats
import TestFunctions
import Text.Printf (printf)
import Torch.Optim (foldLoop)
import Torch.Typed hiding (runStep)
import Torch.Typed.Optim.CppOptim
import Prelude hiding (sqrt)

-- import Data.IORef

-- | show output after n iterations
showLog :: (Show a) => Int -> Int -> Int -> Tensor dev 'Float '[] -> a -> IO ()
showLog n i maxIter lossValue state =
  when (i == 0 || mod i n == 0 || i == maxIter -1) do
    putStrLn
      ( "Iter: " ++ printf "%6d" i
          ++ " | Loss:"
          ++ printf "%05.4f" (toFloat lossValue)
          ++ " | Parameters: "
          ++ show state
      )

-- | Optimize convex quadratic with specified optimizer
optConvQuad ::
  forall dev n opt.
  ( KnownDevice dev,
    KnownNat n,
    CppOptimizer opt,
    RandDTypeIsValid dev 'Float,
    DotDTypeIsValid dev 'Float
  ) =>
  Int ->
  opt ->
  IO ()
optConvQuad numIter optInit = do
  let a = eyeSquare
      b = zeros
  paramInit <- sample $ ConvQuadSpec @dev @'Float @n
  putStrLn ("Initial :" ++ show paramInit)
  optimizer <- initOptimizer optInit paramInit
  trained <- foldLoop (paramInit, optimizer) numIter $ \(paramState, optState) i -> do
    let lossValue = (lossConvQuad a b) paramState
    showLog 1000 i numIter lossValue paramState
    runStep paramState optState lossValue
  pure ()

-- | Optimize Rosenbrock function with specified optimizer
optRosen ::
  forall dev opt.
  ( KnownDevice dev,
    CppOptimizer opt,
    RandDTypeIsValid dev 'Float,
    DotDTypeIsValid dev 'Float
  ) =>
  Int ->
  opt ->
  IO ()
optRosen numIter optInit = do
  paramInit <- sample $ RosenSpec @dev @'Float
  putStrLn ("Initial :" ++ show paramInit)
  optimizer <- initOptimizer optInit paramInit
  trained <- foldLoop (paramInit, optimizer) numIter $ \(paramState, optState) i -> do
    let lossValue = lossRosen paramState
    showLog 1000 i numIter lossValue paramState
    runStep paramState optState lossValue
  pure ()

-- | Optimize Rosenbrock function with specified optimizer
optLinear ::
  forall dev n opt.
  ( KnownDevice dev,
    KnownNat n,
    CppOptimizer opt,
    RandDTypeIsValid dev 'Float,
    DotDTypeIsValid dev 'Float,
    _
  ) =>
  Int ->
  opt ->
  IO ()
optLinear numIter optInit = do
  paramInit <- sample $ LinearSpec @10 @1 @'Float @dev
  putStrLn ("Initial :" ++ show paramInit)
  optimizer <- initOptimizer optInit paramInit
  trained <- foldLoop (paramInit, optimizer) numIter $ \(linear, optState) i -> do
    rt <- randn
    let lossValue = linearForward linear rt
    showLog 1000 i numIter lossValue linear
    runStep linear optState lossValue
  pure ()

--
-- -- | Optimize Ackley function with specified optimizer
-- optAckley :: (CppOptimizer opt) => Int -> opt -> IO ()
-- optAckley numIter optInit = do
--     paramInit <- sample AckleySpec
--     putStrLn ("Initial :" ++ show paramInit)
--     optimizer <- initOptimizer optInit paramInit
--     trained <- foldLoop (paramInit, optimizer) numIter $ \(paramState, optState) i -> do
--         let lossValue = lossAckley paramState
--         showLog 1000 i numIter lossValue paramState
--         runStep paramState optState lossValue 5e-4
--     pure ()
--
-- -- | Check global minimum point for Rosenbrock
checkGlobalMinRosen :: IO ()
checkGlobalMinRosen = do
  putStrLn "\nCheck Actual Global Minimum (at 1, 1):"
  print $ rosenbrock' @'(CPU, 0) 1.0 1.0 -- (asTensor (1.0 :: Float)) (asTensor (1.0 :: Float))

-- | Check global minimum point for Convex Quadratic
checkGlobalMinConvQuad :: IO ()
checkGlobalMinConvQuad = do
  putStrLn "\nCheck Actual Global Minimum (at 0, 0):"
  let a = eyeSquare @2
      b = zeros
  print $ convexQuadratic @'(CPU, 0) @'Float a b zeros

-- -- | Check global minimum point for Ackley
-- checkGlobalMinAckley :: IO ()
-- checkGlobalMinAckley = do
--     putStrLn "\nCheck Actual Global Minimum (at 0, 0):"
--     print $ ackley' (zeros' [2])

main :: IO ()
main = do
  let numIter = 20000
      adamOpt =
        ( def
            { adamLr = 1e-4,
              adamBetas = (0.9, 0.999),
              adamEps = 1e-8,
              adamWeightDecay = 0,
              adamAmsgrad = False
            } ::
            AdamOptions
        )

  -- Convex Quadratic w/ GD, GD+Momentum, Adam
  putStrLn "\nConvex Quadratic\n================"
  putStrLn "\nGD"
  optConvQuad @'(CPU, 0) @2 numIter (def :: SGDOptions)
  putStrLn "\nAdam"
  optConvQuad @'(CPU, 0) @2 numIter adamOpt
  checkGlobalMinConvQuad

  -- 2D Rosenbrock w/ GD, GD+Momentum, Adam
  putStrLn "\n2D Rosenbrock\n================"
  putStrLn "\nGD"
  optRosen @'(CPU, 0) numIter (def :: SGDOptions)
  putStrLn "\nAdam"
  optRosen @'(CPU, 0) numIter adamOpt
  checkGlobalMinRosen

-- -- Ackley w/ GD, GD+Momentum, Adam
-- putStrLn "\nAckley (Gradient methods fail)\n================"
-- putStrLn "\nGD"
-- optAckley numIter (def :: SGDOptions)
-- putStrLn "\nAdam"
-- optAckley numIter adamOpt
-- checkGlobalMinAckley
