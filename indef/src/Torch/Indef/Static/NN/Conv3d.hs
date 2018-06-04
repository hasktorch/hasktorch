-------------------------------------------------------------------------------
-- |
-- Module    : Torch.Indef.Static.NN.Conv3d
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Volumetric (3D) Convolutions
-------------------------------------------------------------------------------


module Torch.Indef.Static.NN.Conv3d where

import Data.Kind (Type)

_volumetricConvolution_updateOutput      :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricConvolution_updateGradInput   :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricConvolution_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()

_volumetricFullConvolution_updateOutput      :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricFullConvolution_updateGradInput   :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricFullConvolution_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()

_volumetricDilatedConvolution_updateOutput      :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricDilatedConvolution_updateGradInput   :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricDilatedConvolution_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()

_volumetricFullDilatedConvolution_updateOutput      :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricFullDilatedConvolution_updateGradInput   :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricFullDilatedConvolution_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
