module Torch.Core.Storage
  ( ByteStorage
  , ShortStorage
  , module X
  ) where

import Torch.Class.Storage as X (IsStorage(..))
import Torch.Core.Storage.Copy as X (UserStorageCopy(..))

import qualified Torch.Core.ByteStorage as B
import qualified Torch.Core.ShortStorage as S

type ByteStorage = B.Storage
type ShortStorage = S.Storage
