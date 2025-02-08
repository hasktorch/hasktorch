{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Torch.Typed.Serialize where

import Torch.HList
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Managed.Serialize as S
import qualified Torch.Internal.Type as ATen
import qualified Torch.Tensor as D
import Torch.Typed.Tensor
import Torch.Typed.Parameter
import Torch.Typed.NN
import Torch.Typed.Autograd

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

saveParameters ::
  forall model parameters tensors dtype device.
  ( Parameterized model,
    parameters ~ Parameters model,
    HMap' ToDependent parameters tensors,
    HMapM' IO MakeIndependent tensors parameters,
    HFoldrM IO TensorListFold [D.ATenTensor] tensors [D.ATenTensor],
    Apply TensorListUnfold [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] tensors),
    HUnfoldM IO TensorListUnfold (HUnfoldMRes IO [D.ATenTensor] tensors) tensors
  ) =>
  model ->
  FilePath ->
  IO ()
saveParameters model filePath = save (hmap' ToDependent . flattenParameters $ model) filePath

loadParameters ::
  forall model parameters tensors dtype device.
  ( Parameterized model,
    parameters ~ Parameters model,
    HMap' ToDependent parameters tensors,
    HMapM' IO MakeIndependent tensors parameters,
    HFoldrM IO TensorListFold [D.ATenTensor] tensors [D.ATenTensor],
    Apply TensorListUnfold [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] tensors),
    HUnfoldM IO TensorListUnfold (HUnfoldMRes IO [D.ATenTensor] tensors) tensors
  ) =>
  model ->
  FilePath ->
  IO model
loadParameters model filePath = do
  tensors <- load @tensors filePath
  params <- hmapM' MakeIndependent tensors
  pure $ replaceParameters model params
