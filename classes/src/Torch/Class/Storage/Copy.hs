{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
module Torch.Class.Storage.Copy where

import Torch.Class.Types
import Foreign (Ptr)
import Torch.Types.TH
import Control.Monad.IO.Class

import Torch.Types.TH.Byte   as B hiding (HsReal)
import Torch.Types.TH.Char   as C hiding (HsReal)
import Torch.Types.TH.Short  as S hiding (HsReal)
import Torch.Types.TH.Int    as I hiding (HsReal)
import Torch.Types.TH.Long   as L hiding (HsReal)
import Torch.Types.TH.Float  as F hiding (HsReal)
import Torch.Types.TH.Double as D hiding (HsReal)
-- FIXME: reintroduce half
-- import Torch.Types.TH.Half   as H

class MonadIO io => StorageCopyRaw io t where
  rawCopy    :: t -> io [HsReal t]
  copy       :: t -> io t
  copyByte   :: t -> io (Ptr B.CStorage)
  copyChar   :: t -> io (Ptr C.CStorage)
  copyShort  :: t -> io (Ptr S.CStorage)
  copyInt    :: t -> io (Ptr I.CStorage)
  copyLong   :: t -> io (Ptr L.CStorage)
  copyFloat  :: t -> io (Ptr F.CStorage)
  copyDouble :: t -> io (Ptr D.CStorage)
  -- FIXME: reintroduce half
  -- copyHalf   :: t -> io H.Storage

class MonadIO io => GPUStorageCopy io gpu cpu | gpu -> io cpu where
  thCopyCuda :: cpu -> io gpu
  copyCuda   :: gpu -> io gpu
  copyCPU    :: gpu -> io cpu


