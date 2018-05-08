{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.Index where

import GHC.Natural
import Control.Exception.Safe
import Torch.Dimensions
import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Index as Ix
import qualified Torch.Indef.Dynamic.Tensor as Dynamic
import qualified Torch.Indef.Dynamic.Tensor.Index as Dynamic

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


-- retrieves a single row
getRow
  :: forall t n m . (KnownNatDim2 n m)
  => Tensor '[n, m] -> Natural -> IO (Tensor '[1, m])
getRow t r
  | r > fromIntegral (natVal (Proxy :: Proxy n)) = throwString "Row out of bounds"
  | otherwise = do
      res <- Dynamic.new (dim :: Dim '[1, m])
      let ixs = Ix.indexDyn [ fromIntegral r ]
      Dynamic._indexSelect res (asDynamic t) 0 ixs
      pure (asStatic res)

-- retrieves a single column
getColumn
  :: forall t n m . (KnownNatDim2 n m)
  => Tensor '[n, m] -> Natural -> IO (Tensor '[n, 1])
getColumn t r
  | r > fromIntegral (natVal (Proxy :: Proxy m)) = throwString "Column out of bounds"
  | otherwise = do
      res <- new
      let ixs = Ix.indexDyn [ fromIntegral r ]
      _indexSelect res t 1 (longAsStatic ixs :: IndexTensor '[n])
      pure res


