module Torch.Class.Tensor.Math.CompareT.Static where

import Torch.Class.Types
import GHC.TypeLits
import Torch.Dimensions

class TensorMathCompareT t where
  ltTensor_ :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> t d -> IO ()
  leTensor_ :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> t d -> IO ()
  gtTensor_ :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> t d -> IO ()
  geTensor_ :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> t d -> IO ()
  neTensor_ :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> t d -> IO ()
  eqTensor_ :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> t d -> IO ()

  ltTensorT_ :: Dimensions d => t d -> t d -> t d -> IO ()
  leTensorT_ :: Dimensions d => t d -> t d -> t d -> IO ()
  gtTensorT_ :: Dimensions d => t d -> t d -> t d -> IO ()
  geTensorT_ :: Dimensions d => t d -> t d -> t d -> IO ()
  neTensorT_ :: Dimensions d => t d -> t d -> t d -> IO ()
  eqTensorT_ :: Dimensions d => t d -> t d -> t d -> IO ()


