{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
module Torch.Class.Storage.Copy where

import Torch.Class.Types
import Foreign (Ptr)
import Torch.Types.TH

import Torch.Types.TH.Byte   as B hiding (HsReal)
import Torch.Types.TH.Char   as C hiding (HsReal)
import Torch.Types.TH.Short  as S hiding (HsReal)
import Torch.Types.TH.Int    as I hiding (HsReal)
import Torch.Types.TH.Long   as L hiding (HsReal)
import Torch.Types.TH.Float  as F hiding (HsReal)
import Torch.Types.TH.Double as D hiding (HsReal)
-- FIXME: reintroduce half
-- import Torch.Types.TH.Half   as H

class StorageCopy t where
  rawCopy    :: t -> io [HsReal t]
  copy       :: t -> io t
  copyByte   :: t -> io B.Storage
  copyChar   :: t -> io C.Storage
  copyShort  :: t -> io S.Storage
  copyInt    :: t -> io I.Storage
  copyLong   :: t -> io L.Storage
  copyFloat  :: t -> io F.Storage
  copyDouble :: t -> io D.Storage
  -- FIXME: reintroduce half
  -- copyHalf   :: t -> io H.Storage

class GPUStorageCopy gpu cpu | gpu -> cpu where
  thCopyCuda :: cpu -> io gpu
  copyCuda   :: gpu -> io gpu
  copyCPU    :: gpu -> io cpu


