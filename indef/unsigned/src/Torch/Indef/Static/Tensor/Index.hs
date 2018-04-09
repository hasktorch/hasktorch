module Torch.Indef.Static.Tensor.Index where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Index.Static as Class
import qualified Torch.Class.Tensor.Index as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Index ()
import Torch.Indef.Static.Tensor ()

instance Class.TensorIndex Tensor where
  indexCopy_ :: Tensor d -> Int -> IndexTensor '[n] -> Tensor d' -> IO ()
  indexCopy_ r x ix t = Dynamic.indexCopy_ (asDynamic r) x (longAsDynamic ix) (asDynamic t)

  indexAdd_ :: Tensor d -> Int -> IndexTensor '[n] -> Tensor d' -> IO ()
  indexAdd_ r x ix t = Dynamic.indexAdd_ (asDynamic r) x (longAsDynamic ix) (asDynamic t)

  indexFill_ :: Tensor d -> Int -> IndexTensor '[n] -> HsReal -> IO ()
  indexFill_ r x ix v = Dynamic.indexFill_ (asDynamic r) x (longAsDynamic ix) v

  indexSelect_ :: Tensor d -> Tensor d' -> Int -> IndexTensor '[n] -> IO ()
  indexSelect_ r t d ix = Dynamic.indexSelect_ (asDynamic r) (asDynamic t) d (longAsDynamic ix)

  take_ :: Tensor d -> Tensor d' -> IndexTensor '[n] -> IO ()
  take_ r t ix = Dynamic.take_ (asDynamic r) (asDynamic t) (longAsDynamic ix)

  put_ :: Tensor d -> IndexTensor '[n] -> Tensor d' -> Int -> IO ()
  put_ r ix t d = Dynamic.put_ (asDynamic r) (longAsDynamic ix) (asDynamic t) d

