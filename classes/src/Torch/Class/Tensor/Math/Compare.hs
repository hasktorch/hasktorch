module Torch.Class.Tensor.Math.Compare where

import Torch.Class.Types

class TensorMathCompare t where
  ltValue_ :: MaskTensor t -> t -> HsReal t -> io ()
  leValue_ :: MaskTensor t -> t -> HsReal t -> io ()
  gtValue_ :: MaskTensor t -> t -> HsReal t -> io ()
  geValue_ :: MaskTensor t -> t -> HsReal t -> io ()
  neValue_ :: MaskTensor t -> t -> HsReal t -> io ()
  eqValue_ :: MaskTensor t -> t -> HsReal t -> io ()

  ltValueT_ :: t -> t -> HsReal t -> io ()
  leValueT_ :: t -> t -> HsReal t -> io ()
  gtValueT_ :: t -> t -> HsReal t -> io ()
  geValueT_ :: t -> t -> HsReal t -> io ()
  neValueT_ :: t -> t -> HsReal t -> io ()
  eqValueT_ :: t -> t -> HsReal t -> io ()


