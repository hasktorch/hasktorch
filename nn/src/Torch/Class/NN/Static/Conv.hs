{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeInType #-}
module Torch.Class.NN.Static.Conv where

import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Dimensions
-- import Numeric.Backprop hiding ((:>))
import Data.Singletons.Prelude.Num
import Data.Singletons.Prelude.Ord
import Data.Kind

class
  ( IsTensor t
  , TemporalConvolutions t
  , SpatialConvolutions t
  , VolumetricConvolutions t
  ) => Convolutions (t :: [Nat] -> *) where

-- | Temporal (1D) Convolutions
class IsTensor t => TemporalConvolutions (t :: [Nat] -> *) where
  _temporalConvolution_updateOutput         :: t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  _temporalConvolution_updateGradInput      :: t d -> t d -> t d -> t d -> Int -> Int -> IO ()
  _temporalConvolution_accGradParameters    :: t d -> t d -> t d -> t d -> Int -> Int -> Double -> IO ()

  _temporalRowConvolution_updateOutput      :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Bool -> IO ()
  _temporalRowConvolution_updateGradInput   :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Bool -> IO ()
  _temporalRowConvolution_accGradParameters :: t d -> t d -> t d -> t d -> t d -> t d -> Int -> Int -> Int -> Bool -> Double -> IO ()

type TemporalConvC t s f kW dW = (TemporalConvolutions t, (s :> kW) ~ 'True, (kW :> 0) ~ 'True, (dW :> 0) ~ 'True)

temporalConvolution_forward
  :: TemporalConvC  t s f kW dW
  => t '[s,f]  -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> t d       -- ^ weight
  -> t d       -- ^ bias
  -> Proxy kW        -- ^ kW
  -> Proxy dW        -- ^ dW
  -> Int       -- ^ inputFrameSize
  -> Int       -- ^ outputFrameSize
  -> IO (t d)  -- ^ output
temporalConvolution_forward = undefined

-- temporalConvolution_forwardBatch
--   :: TemporalConvC  t s f kW dW
--   => t '[b,s,f]          -- ^ input
--   -> t d                 -- ^ weight
--   -> t d                 -- ^ bias
--   -> StrictPositive Int  -- ^ kW
--   -> StrictPositive Int  -- ^ dW
--   -> Int                 -- ^ inputFrameSize
--   -> Int                 -- ^ outputFrameSize
--   -> IO (t d)            -- ^ output



-- | Spatial (2D) Convolutions
class IsTensor t => SpatialConvolutions (t :: [Nat] -> *) where
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
class IsTensor t => VolumetricConvolutions (t :: [Nat] -> *) where
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
