{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Typed.Serialize where

import           Torch.HList

import qualified Torch.Internal.Class             as ATen
import qualified Torch.Internal.Cast              as ATen
import qualified Torch.Internal.Type              as ATen
import qualified Torch.Internal.Managed.Serialize as S
import qualified Torch.Tensor                     as D
import           Torch.Typed.Tensor

-- | save list of tensors to file
save
  :: forall tensors
   . ATen.Castable (HList tensors) [D.ATenTensor]
  => HList tensors -- ^ list of input tensors
  -> FilePath -- ^ file
  -> IO ()
save = ATen.cast2 S.save

-- | load list of tensors from file
load
  :: forall tensors
   . ATen.Castable (HList tensors) [D.ATenTensor]
  => FilePath -- ^ file
  -> IO (HList tensors)
load = ATen.cast1 S.load
