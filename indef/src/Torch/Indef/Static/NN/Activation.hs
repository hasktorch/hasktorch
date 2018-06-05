{-# LANGUAGE FlexibleContexts #-}
module Torch.Indef.Static.NN.Activation where

import Numeric.Backprop
import Control.Monad
import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.NN.Backprop ()

import qualified Torch.Indef.Dynamic.NN.Activation as Dynamic

_pReLU_updateOutput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_pReLU_updateOutput a0 a1 a2 = Dynamic._pReLU_updateOutput (asDynamic a0) (asDynamic a1) (asDynamic a2)

_pReLU_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_pReLU_updateGradInput a0 a1 a2 a3 =
  Dynamic._pReLU_updateGradInput (asDynamic a0) (asDynamic a1) (asDynamic a2) (asDynamic a3)

_pReLU_accGradParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> IO ()
_pReLU_accGradParameters a0 a1 a2 a3 a4 =
  Dynamic._pReLU_accGradParameters (asDynamic a0) (asDynamic a1) (asDynamic a2) (asDynamic a3) (asDynamic a4)

_rReLU_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Double -> Double -> Bool -> Bool -> Generator -> IO ()
_rReLU_updateOutput t0 t1 t2 d0 d1 b0 b1 g =
  Dynamic._rReLU_updateOutput
    (asDynamic t0) (asDynamic t1) (asDynamic t2)
    (d0) (d1)
    (b0) (b1)
    g

_rReLU_updateGradInput   :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> Double -> Bool -> Bool -> IO ()
_rReLU_updateGradInput t0 t1 t2 t3 d0 d1 b0 b1 =
  Dynamic._rReLU_updateGradInput
    (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)
    (d0) (d1)
    (b0) (b1)

_eLU_updateOutput :: Tensor d -> Tensor d -> Double -> Double -> Bool -> IO ()
_eLU_updateOutput t0 t1 d0 d1 b0 =
  Dynamic._eLU_updateOutput
    (asDynamic t0) (asDynamic t1)
    (d0) (d1)
    (b0)

_eLU_updateGradInput :: Tensor d -> Tensor d' -> Tensor d'' -> Double -> Double -> IO ()
_eLU_updateGradInput t0 t1 t2 d0 d1 =
  Dynamic._eLU_updateGradInput
    (asDynamic t0) (asDynamic t1) (asDynamic t2)
    (d0) (d1)

_leakyReLU_updateOutput :: Tensor d -> Tensor d -> Double -> Bool -> IO ()
_leakyReLU_updateOutput t0 t1 d0 b0 =
  Dynamic._leakyReLU_updateOutput
    (asDynamic t0) (asDynamic t1)
    (d0) (b0)


_leakyReLU_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Double -> Bool -> IO ()
_leakyReLU_updateGradInput t0 t1 t2 d0 b0 =
  Dynamic._leakyReLU_updateGradInput
    (asDynamic t0) (asDynamic t1) (asDynamic t2)
    (d0) (b0)

-- | ReLU activation function
relu :: Reifies s W => Dimensions d => BVar s (Tensor d) -> BVar s (Tensor d)
relu = threshold 0 0

-- | run a threshold function againts two BVar variables
threshold
  :: Reifies s W
  => Dimensions d
  => Double               -- ^ threshold
  -> Double               -- ^ replacement value
  -> BVar s (Tensor d)    -- ^ input
  -> BVar s (Tensor d)    -- ^ output
threshold thr value = liftOp1 . op1 $ \inp ->
  (unsafePerformIO (_threshold_updateOutput thr value False inp), \gout ->
    unsafePerformIO (_threshold_updateGradInput thr value False inp gout))
  where
    _threshold_updateOutput
      :: Double              -- ^ threshold
      -> Double              -- ^ replacement value
      -> Bool                -- ^ inplace
      -> Tensor d            -- ^ input
      -> IO (Tensor d)       -- ^ output
    _threshold_updateOutput thr val inplace input = do
      out <- empty
      Dynamic._threshold_updateOutput
        (asDynamic input) (asDynamic out)
        thr val
        inplace
      pure out

    _threshold_updateGradInput
      :: Double        -- ^ threshold
      -> Double        -- ^ replacement value
      -> Bool          -- ^ inplace
      -> Tensor d      -- ^ input
      -> Tensor d      -- ^ gradient output
      -> IO (Tensor d) -- ^ gradient input
    _threshold_updateGradInput thr val inplace input gout = do
      gin <- empty
      Dynamic._threshold_updateGradInput
        (asDynamic input) (asDynamic gout) (asDynamic gin)
        thr val
        inplace
      pure gin

