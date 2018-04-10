module Torch.Class.Tensor.Math.CompareT.Static where

import Torch.Class.Types
import GHC.TypeLits
import Torch.Dimensions

class TensorMathCompareT t where
  _ltTensor :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> t d -> IO ()
  _leTensor :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> t d -> IO ()
  _gtTensor :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> t d -> IO ()
  _geTensor :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> t d -> IO ()
  _neTensor :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> t d -> IO ()
  _eqTensor :: (Dimensions d, Dimensions n) => MaskTensor (t d) n -> t d -> t d -> IO ()

  _ltTensorT :: Dimensions d => t d -> t d -> t d -> IO ()
  _leTensorT :: Dimensions d => t d -> t d -> t d -> IO ()
  _gtTensorT :: Dimensions d => t d -> t d -> t d -> IO ()
  _geTensorT :: Dimensions d => t d -> t d -> t d -> IO ()
  _neTensorT :: Dimensions d => t d -> t d -> t d -> IO ()
  _eqTensorT :: Dimensions d => t d -> t d -> t d -> IO ()


