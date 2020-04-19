{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Typed.Serialize where

import Torch.HList
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Managed.Serialize as S
import qualified Torch.Internal.Type as ATen
import qualified Torch.Tensor as D
import Torch.Typed.Tensor

-- | save list of tensors to file
save ::
  forall tensors.
  ATen.Castable (HList tensors) [D.ATenTensor] =>
  -- | list of input tensors
  HList tensors ->
  -- | file
  FilePath ->
  IO ()
save = ATen.cast2 S.save

-- | load list of tensors from file
load ::
  forall tensors.
  ATen.Castable (HList tensors) [D.ATenTensor] =>
  -- | file
  FilePath ->
  IO (HList tensors)
load = ATen.cast1 S.load
