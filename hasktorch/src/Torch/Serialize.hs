module Torch.Serialize where

import qualified LibTorch.Torch.Managed.Serialize as S
import Torch.Tensor
import LibTorch.ATen.Cast

save :: [Tensor] -> FilePath -> IO ()
save inputs file = cast2 S.save inputs file

load :: FilePath -> IO [Tensor]
load file = cast1 S.load file
