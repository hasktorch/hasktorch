module Torch.Core.Internal
  ( showLim
  , impossible
  ) where

-- import Foreign (Word, Ptr)
-- import Foreign.C.Types (CLLong, CLong, CDouble, CShort, CLong, CChar, CInt, CFloat)
import Numeric (showGFloat)

-- | Show a real value with limited precision
showLim :: RealFloat a => a -> String
showLim x = showGFloat (Just 2) x ""

impossible :: String -> a
impossible = error

