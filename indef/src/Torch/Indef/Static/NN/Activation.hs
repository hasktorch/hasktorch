module Torch.Indef.Static.NN.Activation where

import Torch.Dimensions
import Control.Monad
import Torch.Indef.Types

import qualified Torch.Indef.Dynamic.NN as Dynamic

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


_leakyReLU_updateGradInput:: Tensor d -> Tensor d -> Tensor d -> Double -> Bool -> IO ()
_leakyReLU_updateGradInput t0 t1 t2 d0 b0 =
  Dynamic._leakyReLU_updateGradInput
    (asDynamic t0) (asDynamic t1) (asDynamic t2)
    (d0) (b0)


_threshold_updateOutput :: Tensor d -> Tensor d -> Double -> Double -> Bool -> IO ()
_threshold_updateOutput t0 t1 d0 d1 b0 =
  Dynamic._threshold_updateOutput
    (asDynamic t0) (asDynamic t1)
    (d0) (d1)
    (b0)

_threshold_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Double -> Double -> Bool -> IO ()
_threshold_updateGradInput t0 t1 t2 d0 d1 b0 =
  Dynamic._threshold_updateGradInput
    (asDynamic t0) (asDynamic t1) (asDynamic t2)
    (d0) (d1)
    (b0)
