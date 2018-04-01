module Torch.Class.Tensor.Math.Compare.Static where

import Torch.Class.Types
import Torch.Dimensions

class TensorMathCompare t where
  ltValue_ :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  leValue_ :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  gtValue_ :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  geValue_ :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  neValue_ :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  eqValue_ :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()

  ltValueT_ :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  leValueT_ :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  gtValueT_ :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  geValueT_ :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  neValueT_ :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  eqValueT_ :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()


