module Torch.Class.Tensor.Math.CompareT.Static where

import Torch.Class.Types
import GHC.TypeLits

class TensorMathCompareT t (d::[Nat]) (n::[Nat]) where
  ltTensor_ :: MaskTensor (t d) n -> t d -> t d -> IO ()
  leTensor_ :: MaskTensor (t d) n -> t d -> t d -> IO ()
  gtTensor_ :: MaskTensor (t d) n -> t d -> t d -> IO ()
  geTensor_ :: MaskTensor (t d) n -> t d -> t d -> IO ()
  neTensor_ :: MaskTensor (t d) n -> t d -> t d -> IO ()
  eqTensor_ :: MaskTensor (t d) n -> t d -> t d -> IO ()

  ltTensorT_ :: t d -> t d -> t d -> IO ()
  leTensorT_ :: t d -> t d -> t d -> IO ()
  gtTensorT_ :: t d -> t d -> t d -> IO ()
  geTensorT_ :: t d -> t d -> t d -> IO ()
  neTensorT_ :: t d -> t d -> t d -> IO ()
  eqTensorT_ :: t d -> t d -> t d -> IO ()


