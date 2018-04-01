module Torch.Class.Tensor.Math.Compare.Static where

import Torch.Class.Types
import GHC.TypeLits

class TensorMathCompare t (d::[Nat]) (n::[Nat]) where
  ltValue_ :: MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  leValue_ :: MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  gtValue_ :: MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  geValue_ :: MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  neValue_ :: MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()
  eqValue_ :: MaskTensor (t d) n -> t d -> HsReal (t d) -> IO ()

  ltValueT_ :: t d -> t d -> HsReal (t d) -> IO ()
  leValueT_ :: t d -> t d -> HsReal (t d) -> IO ()
  gtValueT_ :: t d -> t d -> HsReal (t d) -> IO ()
  geValueT_ :: t d -> t d -> HsReal (t d) -> IO ()
  neValueT_ :: t d -> t d -> HsReal (t d) -> IO ()
  eqValueT_ :: t d -> t d -> HsReal (t d) -> IO ()


