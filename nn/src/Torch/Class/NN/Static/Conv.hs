{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Torch.Class.NN.Static.Conv where

import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Dimensions
-- import Numeric.Backprop hiding ((:>))
import Data.Kind (Type)

class
  ( IsTensor t
  , TemporalConvolutions t
  , SpatialConvolutions t
  , VolumetricConvolutions t
  ) => Convolutions (t :: [Nat] -> Type) where

-- | Temporal (1D) Convolutions
class IsTensor t => TemporalConvolutions (t :: [Nat] -> Type) where
  -- | Applies a 1D convolution over an input sequence composed of nInputFrame frames. The input tensor in forward(input) is expected to be a 2D tensor (nInputFrame x inputFrameSize) or a 3D tensor (nBatchFrame x nInputFrame x inputFrameSize).
  _temporalConvolution_updateOutput         :: t d -> t d' -> t d'' -> t d''' -> Int -> Int -> Int -> Int -> IO ()
  _temporalConvolution_updateGradInput      :: t d -> t d' -> t d'' -> t d''' -> Int -> Int -> IO ()
  _temporalConvolution_accGradParameters    :: t d -> t d' -> t d'' -> t d''' -> Int -> Int -> Double -> IO ()

  _temporalRowConvolution_updateOutput      :: t d -> t d' -> t d'' -> t d''' -> t d -> t d -> Int -> Int -> Int -> Bool -> IO ()
  _temporalRowConvolution_updateGradInput   :: t d -> t d' -> t d'' -> t d''' -> t d -> t d -> Int -> Int -> Int -> Bool -> IO ()
  _temporalRowConvolution_accGradParameters :: t d -> t d' -> t d'' -> t d''' -> t d -> t d -> Int -> Int -> Int -> Bool -> Double -> IO ()

type TemporalConvC t s f kW dW o =
  ( TemporalConvolutions t
  , (s > kW) ~ 'True
  , (kW > 0) ~ 'True
  , (dW > 0) ~ 'True
  -- , o ~ ((Div (s - kW) dW) + 1)
  )

-- If the input sequence is a 2D tensor of dimension (nInputFrame x inputFrameSize), the
-- output sequence will be (nOutputFrame x outputFrameSize) where
--
--    nOutputFrame = (nInputFrame - kW) / dW + 1
--
conv1d_forward
  :: forall t s f kW dW o b
  .  KnownNat5 s f o kW dW
  => TemporalConvC t s f kW dW o
  => t '[s, f]                         -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> t '[o, f*kW]                      -- ^ weight
  -> t '[o]                            -- ^ bias
  -> Proxy '(kW, dW)                   -- ^ kW: The kernel width of the convolution
                                       --   dW: The step of the convolution. Default is 1 in C.
  -> IO (t '[s, o])                    -- ^ output
conv1d_forward = _conv1d_forward

conv1d_forwardBatch
  :: forall t s f kW dW o b
  .  KnownNat5 s f o kW dW
  => TemporalConvC t s f kW dW o
  => t '[b,s,f]                        -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> t '[o, f*kW]                      -- ^ weight
  -> t '[o]                            -- ^ bias
  -> Proxy '(kW, dW)                   -- ^ kW: The kernel width of the convolution
                                       --   dW: The step of the convolution. Default is 1 in C.
  -> IO (t '[b,s,o])                   -- ^ output
conv1d_forwardBatch = _conv1d_forward

_conv1d_forward
  :: forall  t f o kW dW d d'
  .  KnownNat4 f o kW dW
  => TemporalConvolutions t
  => t d
  -> t '[o, f*kW]
  -> t '[o]
  -> Proxy '(kW, dW)
  -> IO (t d')
_conv1d_forward inp weight bias _ = do
  out <- empty
  _temporalConvolution_updateOutput inp out weight bias
    (fromIntegral $ natVal (Proxy :: Proxy kW))
    (fromIntegral $ natVal (Proxy :: Proxy dW))
    (fromIntegral $ natVal (Proxy :: Proxy f))
    (fromIntegral $ natVal (Proxy :: Proxy o))
  pure out


conv1d_backward
  :: forall t s f kW dW o b
  .  KnownNat5 s f o kW dW
  => TemporalConvC t s f kW dW o
  => t '[s, f]                         -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> t '[s, o]                         -- ^ grad output
  -> t '[o, f*kW]                      -- ^ weight
  -> Proxy '(kW, dW)                   -- ^ kW: The kernel width of the convolution
                                       --   dW: The step of the convolution. Default is 1 in C.
  -> IO (t '[s, f])                    -- ^ output
conv1d_backward = _conv1d_backwardBatch

conv1d_backwardBatch
  :: forall t s f kW dW o b
  .  KnownNat5 s f o kW dW
  => TemporalConvC t s f kW dW o
  => t '[b, s, f]                      -- ^ input: s for 'sequence dimension', f for 'feature dimension'
  -> t '[b, s, o]                      -- ^ grad output
  -> t '[o, f*kW]                      -- ^ weight
  -> Proxy '(kW, dW)                   -- ^ kW: The kernel width of the convolution
                                       --   dW: The step of the convolution. Default is 1 in C.
  -> IO (t '[b, s, f])                 -- ^ output
conv1d_backwardBatch = _conv1d_backwardBatch

_conv1d_backwardBatch
  :: forall t f o kW dW d d'
  .  KnownNat4 f o kW dW
  => TemporalConvolutions t
  => t d
  -> t d'
  -> t '[o, f*kW]
  -> Proxy '(kW, dW)
  -> IO (t d)
_conv1d_backwardBatch input gradOut weight _ = do
  gradIn <- empty
  _temporalConvolution_updateGradInput input gradOut gradIn weight
    (fromIntegral $ natVal (Proxy :: Proxy kW))
    (fromIntegral $ natVal (Proxy :: Proxy dW))
  pure gradIn



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
