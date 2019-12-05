{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Layout where

import LibTorch.ATen.Class (Castable(..))
import qualified LibTorch.ATen.Const as ATen
import qualified LibTorch.ATen.Type as ATen

data Layout = Strided | Sparse | Mkldnn
  deriving (Eq, Show)

instance Castable Layout ATen.Layout where
  cast Strided f = f ATen.kStrided
  cast Sparse f  = f ATen.kSparse
  cast Mkldnn f  = f ATen.kMkldnn

  uncast x f
    | x == ATen.kStrided = f Strided
    | x == ATen.kSparse  = f Sparse
    | x == ATen.kMkldnn  = f Mkldnn
