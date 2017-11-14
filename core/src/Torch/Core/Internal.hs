module Torch.Core.Internal
  ( w2cl
  ) where

import Foreign (Word)
import Foreign.C.Types (CLong)

w2cl :: Word -> CLong
w2cl = fromIntegral




