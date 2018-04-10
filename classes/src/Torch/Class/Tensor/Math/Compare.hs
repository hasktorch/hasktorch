module Torch.Class.Tensor.Math.Compare where

import Torch.Class.Types

class TensorMathCompare t where
  _ltValue :: MaskDynamic t -> t -> HsReal t -> IO ()
  _leValue :: MaskDynamic t -> t -> HsReal t -> IO ()
  _gtValue :: MaskDynamic t -> t -> HsReal t -> IO ()
  _geValue :: MaskDynamic t -> t -> HsReal t -> IO ()
  _neValue :: MaskDynamic t -> t -> HsReal t -> IO ()
  _eqValue :: MaskDynamic t -> t -> HsReal t -> IO ()

  _ltValueT :: t -> t -> HsReal t -> IO ()
  _leValueT :: t -> t -> HsReal t -> IO ()
  _gtValueT :: t -> t -> HsReal t -> IO ()
  _geValueT :: t -> t -> HsReal t -> IO ()
  _neValueT :: t -> t -> HsReal t -> IO ()
  _eqValueT :: t -> t -> HsReal t -> IO ()


