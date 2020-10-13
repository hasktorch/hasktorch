{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Layout where

import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Type as ATen

data Layout = Strided | Sparse | Mkldnn
  deriving (Eq, Show)

instance Castable Layout ATen.Layout where
  cast Strided f = f ATen.kStrided
  cast Sparse f = f ATen.kSparse
  cast Mkldnn f = f ATen.kMkldnn

  uncast x f
    | x == ATen.kStrided = f Strided
    | x == ATen.kSparse = f Sparse
    | x == ATen.kMkldnn = f Mkldnn
