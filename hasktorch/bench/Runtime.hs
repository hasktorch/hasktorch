-----------------------------------------------------------------------------
-- |
-- Source      : https://github.com/Magalame/fastest-matrices
-- Copyright   : (c) 2019 Magalame
--
-- License     : BSD3
-- Maintainer  : Junji Hashimoto<junji.hashimoto@gmail.com>
-- Stability   : experimental
-- Portability : GHC
--
-----------------------------------------------------------------------------

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ExtendedDefaultRules #-}


module Main where

import qualified Data.Vector.Unboxed         as U
import           Data.Vector.Unboxed         (Vector)
import qualified Data.Vector as V
import           Control.Monad.Primitive
import qualified Data.Vector.Generic         as G
import           Control.DeepSeq
import           System.IO.Unsafe
import           Foreign

-- hmatrix
import qualified Numeric.LinearAlgebra as H

-- hasktorch
import qualified Torch as T
import qualified Torch.Functional.Internal as TI
import qualified Torch.Internal.Unmanaged.Type.Tensor as TIU
import qualified Torch.Internal.Unmanaged.Type.Extra as TIU
import qualified Torch.Internal.Managed.Type.Tensor as TIM
import qualified Torch.Internal.Managed.Type.Extra as TIM
import qualified Torch.Jit as T

import qualified System.Random.MWC as Mwc

import qualified Criterion.Main as C

#define N 10
#define N2 100
#define N3 1000


instance NFData T.Tensor
  where
    rnf (T.Unsafe _) = ()

instance NFData (ForeignPtr a)
  where
    rnf v = v `seq` ()


n :: Int
n = N

uniformVector :: (PrimMonad m, Mwc.Variate a, G.Vector v a)
              => Mwc.Gen (PrimState m) -> Int -> m (v a)
uniformVector gen n = G.replicateM n (Mwc.uniform gen)

vectorGen :: IO (Vector Double)
vectorGen =  do 
    gen <- Mwc.create
    uniformVector gen (n*n)

matrixH :: IO (H.Matrix H.R)
matrixH = do
    vec <- vectorGen
    return $ (n H.>< n) $ U.toList $ vec 

identH :: Int -> H.Matrix Double
identH = H.ident

elemZero :: Double -> Double
elemZero = const 0

elemSqr :: Double -> Double
elemSqr x = x*x

mapH :: (Double -> Double) -> H.Matrix Double -> H.Matrix Double
mapH = H.cmap

main :: IO ()
main = do 
    vDLA' <- vectorGen
    uDLA' <- vectorGen

    let 
    --
      vList = U.toList vDLA'
      uList = U.toList uDLA'
    
    --
      aH' = (n H.>< n) vList
      bH' = (n H.>< n) uList

      subH' = H.fromList . take n $ vList
      vH' = H.fromList vList

    --
      to2d [] = []
      to2d xs = take n xs : to2d (drop n xs)
      aT' = T.asTensor $ to2d vList
      bT' = T.asTensor $ to2d uList

      subT' = T.asTensor . take n $ vList
      vT' = T.asTensor vList

    cache <- T.newScriptCache

    let jit :: (T.Tensor -> T.Tensor) -> T.Tensor -> T.Tensor
        jit func input = let [r] = T.jit cache (\[v] -> [func v]) [input] in r

    C.defaultMain [
        C.env (pure (aH', bH', subH', vH')) $ \ ~(aH, bH, subH, vH) ->
            C.bgroup "Hmatrix" [ 
                             C.bench "multiplication" $ C.nf ((<>) aH) bH,
                             C.bench "repeated multiplication" $ C.nf ( H.sumElements . flip (H.?) [1] . (<>) bH . (<>) aH . (<>) aH) bH,
                             C.bench "multiplicationV" $ C.nf ((H.#>) aH) subH,
                             -- C.bench "qr factorization" $ C.nf H.qr aH,
                             C.bench "transpose" $ C.nf H.tr aH,
                             C.bench "norm" $ C.nf H.norm_2 vH,
                             C.bench "row" $ C.nf ((H.?) aH) [0],
                             C.bench "column" $ C.nf ((H.¿) aH) [0], 
                             C.bench "identity" $ C.nf identH n,
                             C.bench "diag" $ C.nf H.diag subH,
                             C.bench "map const 0" $ C.nf (mapH elemZero) aH,
                             C.bench "map sqr" $ C.nf (mapH elemSqr) aH,
                             C.bench "size" $ C.nf H.size aH
                           ],

        C.env (pure (aT', bT', subT', vT')) $ \ ~(aT, bT, subT, vT) ->
            C.bgroup "Hasktorch" [ 
                             C.bench "multiplication" $ C.nf (T.matmul aT) bT,
                             C.bench "repeated multiplication" $ C.nf (T.sumAll . T.matmul bT . T.matmul aT . T.matmul aT) bT,
                             C.bench "repeated multiplication with JIT" $ C.nf (jit $ T.sumAll . T.matmul bT . T.matmul aT . T.matmul aT) bT,
                             C.bench "multiplicationV" $ C.nf (T.matmul aT) subT,
                             -- C.bench "qr factorization" $ C.nf (\v -> TI.qr v True) aT,
                             C.bench "transpose" $ C.nf TI.t aT,
                             C.bench "norm" $ C.nf (\v -> TI.normAll v 2) vT,
                             C.bench "row" $ C.nf ((T.!) aT) (0::Int),
                             C.bench "column" $ C.nf ((T.!) aT) [T.slice|...,0|],
                             C.bench "identity" $ C.nf (\i -> T.eye' i i) n,
                             C.bench "diag" $ C.nf (T.diag (T.Diag 0)) subT,
                             C.bench "map const 0" $ C.nf (\v -> T.maskedFill v  [T.slice|...|] 0) aT,
                             C.bench "map sqr" $ C.nf (\v -> v * v) aT,
                             C.bench "shape" $ C.nf T.shape aT,
                             C.bench "shape(managed)" $ C.nf (\(T.Unsafe v) -> unsafePerformIO $ TIM.tensor_sizes v) aT,
                             C.bench "shape(unmanaged)" $ C.nf (\(T.Unsafe v) -> unsafePerformIO $ withForeignPtr v $ \ptr -> TIU.tensor_sizes ptr) aT,
                             C.bench "dim" $ C.nf T.dim aT,
                             C.bench "dim(managed)" $ C.nf (\(T.Unsafe v) -> unsafePerformIO $ TIM.tensor_dim v) aT,
                             C.bench "dim(unmanaged)" $ C.nf (\(T.Unsafe v) -> unsafePerformIO $ withForeignPtr v $ \ptr -> TIU.tensor_dim ptr) aT,
                             C.bench "dim(unsafe)" $ C.nf T.dimUnsafe aT,
                             C.bench "dim(unsafe/managed)" $ C.nf (\(T.Unsafe v) -> unsafePerformIO $ TIM.tensor_dim_unsafe v) aT,
                             C.bench "dim(unsafe/unmanaged)" $ C.nf (\(T.Unsafe v) -> unsafePerformIO $ withForeignPtr v $ \ptr -> TIU.tensor_dim_unsafe ptr) aT,
                             C.bench "dim(unsafe-c)" $ C.nf T.dimCUnsafe aT,
                             C.bench "dim(unsafe-c/managed)" $ C.nf (\(T.Unsafe v) -> unsafePerformIO $ TIM.tensor_dim_c_unsafe v) aT,
                             C.bench "dim(unsafe-c/unmanaged)" $ C.nf (\(T.Unsafe v) -> unsafePerformIO $ withForeignPtr v $ \ptr -> TIU.tensor_dim_c_unsafe ptr) aT
                           ]
               ]

