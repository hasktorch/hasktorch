module Torch.Class.Tensor.Static
  ( IsStatic(..)
  )
  where

import Torch.Class.Internal

class IsStatic t where
  asDynamic :: t -> AsDynamic t
  asStatic :: AsDynamic t -> t

