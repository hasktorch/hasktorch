module Torch.Class.Tensor.Math.Compare.Static where

import Torch.Class.Types
import Torch.Dimensions

class TensorMathCompare t where
  _ltValue :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  _leValue :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  _gtValue :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  _geValue :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  _neValue :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  _eqValue :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()

  _ltValueT :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  _leValueT :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  _gtValueT :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  _geValueT :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  _neValueT :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  _eqValueT :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()


