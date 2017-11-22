module Torch.Core.Exceptions
  ( TorchException(..)
  , module X
  ) where

import Control.Exception.Safe as X
-- import Control.Exception.Base as X (catch)
import Data.Typeable (Typeable)
import Data.Text (Text)

data TorchException
  = MathException Text
  deriving (Show, Typeable)

instance Exception TorchException

