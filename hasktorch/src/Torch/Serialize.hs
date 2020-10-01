module Torch.Serialize where

import Torch.Autograd
import Torch.Internal.Cast
import qualified Torch.Internal.Managed.Serialize as S
import Torch.NN
import Torch.Tensor

save :: [Tensor] -> FilePath -> IO ()
save = cast2 S.save

load :: FilePath -> IO [Tensor]
load = cast1 S.load

saveParams :: Parameterized f => f -> FilePath -> IO ()
saveParams model filePath = do
  let params = map toDependent $ flattenParameters model
  save params filePath

loadParams :: Parameterized b => b -> FilePath -> IO b
loadParams model filePath = do
  tensors <- load filePath
  let params = map IndependentTensor tensors
  pure $ replaceParameters model params
