module Torch.Serialize where

import Torch.Autograd
import Torch.Internal.Cast
import qualified Torch.Internal.Managed.Serialize as S
import Torch.NN
import Torch.Tensor

save ::
  -- | inputs
  [Tensor] ->
  -- | file
  FilePath ->
  -- | output
  IO ()
save = cast2 S.save

load ::
  -- | file
  FilePath ->
  -- | output
  IO [Tensor]
load = cast1 S.load

saveParams ::
  Parameterized f =>
  -- | model
  f ->
  -- | filepath
  FilePath ->
  -- | output
  IO ()
saveParams model filePath = do
  let params = map toDependent $ flattenParameters model
  save params filePath

loadParams ::
  Parameterized b =>
  -- | model
  b ->
  -- | filepath
  FilePath ->
  -- | output
  IO b
loadParams model filePath = do
  tensors <- load filePath
  let params = map IndependentTensor tensors
  pure $ replaceParameters model params
