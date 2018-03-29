module Torch.Class.Tensor.Math.Compare where

import Torch.Class.Types

class TensorMathCompare t where
  ltValue_ :: MaskTensor t -> t -> HsReal t -> IO ()
  leValue_ :: MaskTensor t -> t -> HsReal t -> IO ()
  gtValue_ :: MaskTensor t -> t -> HsReal t -> IO ()
  geValue_ :: MaskTensor t -> t -> HsReal t -> IO ()
  neValue_ :: MaskTensor t -> t -> HsReal t -> IO ()
  eqValue_ :: MaskTensor t -> t -> HsReal t -> IO ()

  ltValueT_ :: t -> t -> HsReal t -> IO ()
  leValueT_ :: t -> t -> HsReal t -> IO ()
  gtValueT_ :: t -> t -> HsReal t -> IO ()
  geValueT_ :: t -> t -> HsReal t -> IO ()
  neValueT_ :: t -> t -> HsReal t -> IO ()
  eqValueT_ :: t -> t -> HsReal t -> IO ()


