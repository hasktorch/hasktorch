module Torch.Indef.Static.Tensor.Index where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Index.Static as Class
import qualified Torch.Class.Tensor.Index as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Index ()
import Torch.Indef.Static.Tensor ()

instance Class.TensorIndex Tensor where
  _indexCopy :: Tensor d -> Int -> IndexTensor '[n] -> Tensor d' -> IO ()
  _indexCopy r x ix t = Dynamic._indexCopy (asDynamic r) x (longAsDynamic ix) (asDynamic t)

  _indexAdd :: Tensor d -> Int -> IndexTensor '[n] -> Tensor d' -> IO ()
  _indexAdd r x ix t = Dynamic._indexAdd (asDynamic r) x (longAsDynamic ix) (asDynamic t)

  _indexFill :: Tensor d -> Int -> IndexTensor '[n] -> HsReal -> IO ()
  _indexFill r x ix v = Dynamic._indexFill (asDynamic r) x (longAsDynamic ix) v

  _indexSelect :: Tensor d -> Tensor d' -> Int -> IndexTensor '[n] -> IO ()
  _indexSelect r t d ix = Dynamic._indexSelect (asDynamic r) (asDynamic t) d (longAsDynamic ix)

  _take :: Tensor d -> Tensor d' -> IndexTensor '[n] -> IO ()
  _take r t ix = Dynamic._take (asDynamic r) (asDynamic t) (longAsDynamic ix)

  _put :: Tensor d -> IndexTensor '[n] -> Tensor d' -> Int -> IO ()
  _put r ix t d = Dynamic._put (asDynamic r) (longAsDynamic ix) (asDynamic t) d

