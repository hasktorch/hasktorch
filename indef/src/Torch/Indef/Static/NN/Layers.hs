module Torch.Indef.Static.NN.Layers where

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.NN as Dynamic

_sparseLinear_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_sparseLinear_updateOutput t0 t1 t2 t3 = Dynamic._sparseLinear_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)
_sparseLinear_accGradParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> Double -> IO ()
_sparseLinear_accGradParameters t0 t1 t2 t3 t4 t5 = Dynamic._sparseLinear_accGradParameters (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5)

_sparseLinear_zeroGradParameters :: Tensor d -> Tensor d -> Tensor d -> IO ()
_sparseLinear_zeroGradParameters t0 t1 t2 = Dynamic._sparseLinear_zeroGradParameters (asDynamic t0) (asDynamic t1) (asDynamic t2)
_sparseLinear_updateParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> IO ()
_sparseLinear_updateParameters t0 t1 t2 t3 t4 = Dynamic._sparseLinear_updateParameters (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)

_gatedLinear_updateOutput :: Tensor d -> Tensor d -> Int -> IO ()
_gatedLinear_updateOutput t0 t1 = Dynamic._gatedLinear_updateOutput (asDynamic t0) (asDynamic t1)
_gatedLinear_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> IO ()
_gatedLinear_updateGradInput t0 t1 t2 = Dynamic._gatedLinear_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

_gRUFused_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_gRUFused_updateOutput t0 t1 t2 t3 t4 t5 t6 = Dynamic._gRUFused_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5) (asDynamic t6)
_gRUFused_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_gRUFused_updateGradInput t0 t1 t2 t3 t4 = Dynamic._gRUFused_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)

_lSTMFused_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_lSTMFused_updateOutput t0 t1 t2 t3 t4 t5 t6 = Dynamic._lSTMFused_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5) (asDynamic t6)
_lSTMFused_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_lSTMFused_updateGradInput t0 t1 t2 t3 t4 t5 t6 = Dynamic._lSTMFused_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5) (asDynamic t6)


