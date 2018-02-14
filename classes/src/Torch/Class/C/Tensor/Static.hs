module Torch.Class.C.Tensor.Static
  ( IsStatic(..)
  )
  where

import Torch.Class.C.Internal

class IsStatic t where
  asDynamic :: t -> AsDynamic t
  asStatic :: AsDynamic t -> t

