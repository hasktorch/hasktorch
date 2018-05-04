module Torch.Class.NN.Static.Conv
  ( module X
  , Convolutions(..)
  ) where

import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Dimensions
import Torch.Class.NN.Static.Conv1d as X
import Data.Kind (Type)

class
  ( IsTensor t
  , TemporalConvolutions t
  , SpatialConvolutions t
  , VolumetricConvolutions t
  ) => Convolutions (t :: [Nat] -> Type) where


-- | Spatial (2D) Convolutions
class IsTensor t => SpatialConvolutions (t :: [Nat] -> Type) where
  _spatialConvolutionMM_updateOutput      :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  _spatialConvolutionMM_updateGradInput   :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  _spatialConvolutionMM_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()

  _spatialConvolutionLocal_updateOutput      :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Integer -> Integer -> Integer -> Integer -> IO ()
  _spatialConvolutionLocal_updateGradInput   :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Integer -> Integer -> Integer -> Integer -> IO ()
  _spatialConvolutionLocal_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Integer -> Integer -> Integer -> Integer -> Double -> IO ()

  spatialFullConvolution_updateOutput      :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialFullConvolution_updateGradInput   :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialFullConvolution_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()

  spatialDilatedConvolution_updateOutput      :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialDilatedConvolution_updateGradInput   :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialDilatedConvolution_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()

  spatialFullDilatedConvolution_updateOutput      :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialFullDilatedConvolution_updateGradInput   :: t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialFullDilatedConvolution_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()


-- | Volumetric (3D) Convolutions
class IsTensor t => VolumetricConvolutions (t :: [Nat] -> Type) where
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
