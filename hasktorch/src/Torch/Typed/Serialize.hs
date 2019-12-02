{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}

module Torch.Typed.Serialize where

import Data.HList
import           Foreign.ForeignPtr

import qualified Torch.Managed.Serialize as S
import Torch.Typed.Tensor
import qualified ATen.Class                    as ATen
import qualified ATen.Cast                     as ATen
import qualified ATen.Type                     as ATen

save
  :: forall tensors
   . (ATen.Castable (HList tensors) (ForeignPtr ATen.TensorList))
  => HList tensors
  -> FilePath
  -> IO ()
save inputs file = ATen.cast2 S.save inputs file

load
  :: forall tensors
   . (ATen.Castable (HList tensors) (ForeignPtr ATen.TensorList))
  => FilePath
  -> IO (HList tensors)
load file = ATen.cast1 S.load file
