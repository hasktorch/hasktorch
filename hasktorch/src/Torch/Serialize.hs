module Torch.Serialize where

import qualified Torch.Managed.Serialize as S
import Torch.Tensor
import ATen.Cast

save :: [Tensor] -> FilePath -> IO ()
save inputs file = cast2 S.save inputs file

load :: FilePath -> IO [Tensor]
load file = cast1 S.load file
