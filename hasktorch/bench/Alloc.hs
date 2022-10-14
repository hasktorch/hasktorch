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
import qualified Torch.Internal.Managed.Type.Tensor as TIM

import qualified System.Random.MWC as Mwc

import qualified Weigh as W


n :: Int
n = 100

instance NFData T.Tensor
  where
    rnf (T.Unsafe _) = ()

instance NFData (ForeignPtr a)
  where
    rnf v = v `seq` ()

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
    vDLA <- vectorGen
    uDLA <- vectorGen

    let 
    --
      vList = U.toList vDLA
      uList = U.toList uDLA
    
    --
      aH = (n H.>< n) vList
      bH = (n H.>< n) uList
      vH = H.fromList vList

    --
      to2d [] = []
      to2d xs = take n xs : to2d (drop n xs)
      aT = T.asTensor $ to2d vList
      bT = T.asTensor $ to2d uList
      vT = T.asTensor vList
    
    W.mainWith (do 
               W.func "Hmatrix - multiplication" ((<>) aH) bH
               W.func "Hmatrix - qr factorization" H.qr aH
               W.func "Hmatrix - transpose" H.tr aH
               W.func "Hmatrix - norm" H.norm_2 vH
               W.func "Hmatrix - row" ((H.?) aH) [0]
               W.func "Hmatrix - column" ((H.¿) aH) [0]
               W.func "Hmatrix - identity" identH n
               
               W.func "Hasktorch - multiplication" (T.matmul aT) bT
               W.func "Hasktorch - qr factorization" (\v -> TI.qr v True) aT
               W.func "Hasktorch - transpose" TI.t aT
               W.func "Hasktorch - norm" (\v -> TI.normAll v 2) vT
               W.func "Hasktorch - row" ((T.!) aT) (0::Int)
               W.func "Hasktorch - column" ((T.!) aT) [T.slice|...,0|]
               W.func "Hasktorch - identity" (\i -> T.eye' i i) n
               )
