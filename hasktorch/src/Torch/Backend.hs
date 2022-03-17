{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Backend where

import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Type as ATen

data Backend = CPU | CUDA | HIP | SparseCPU | SparseCUDA | XLA
  deriving (Eq, Show)

instance Castable Backend ATen.Backend where
  cast CPU f = f ATen.bCPU
  cast CUDA f = f ATen.bCUDA
  cast HIP f = f ATen.bHIP
  cast SparseCPU f = f ATen.bSparseCPU
  cast SparseCUDA f = f ATen.bSparseCUDA
  cast XLA f = f ATen.bXLA

  uncast x f
    | x == ATen.bCPU = f CPU
    | x == ATen.bCUDA = f CUDA
    | x == ATen.bHIP = f HIP
    | x == ATen.bSparseCPU = f SparseCPU
    | x == ATen.bSparseCUDA = f SparseCUDA
    | x == ATen.bXLA = f XLA
