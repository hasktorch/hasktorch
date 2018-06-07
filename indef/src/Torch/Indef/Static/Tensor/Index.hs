-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Index
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.Index where

import Numeric.Dimensions
import GHC.TypeLits
import GHC.Natural
import Control.Exception.Safe
import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Index as Ix
import qualified Torch.Indef.Dynamic.Tensor as Dynamic
import qualified Torch.Indef.Dynamic.Tensor.Index as Dynamic

-- | Static call to 'Dynamic._indexCopy'
_indexCopy :: Tensor d -> Int -> IndexTensor '[n] -> Tensor d' -> IO ()
_indexCopy r x ix t = Dynamic._indexCopy (asDynamic r) x (longAsDynamic ix) (asDynamic t)

-- | Static call to 'Dynamic._indexAdd'
_indexAdd :: Tensor d -> Int -> IndexTensor '[n] -> Tensor d' -> IO ()
_indexAdd r x ix t = Dynamic._indexAdd (asDynamic r) x (longAsDynamic ix) (asDynamic t)

-- | Static call to 'Dynamic._indexFill'
_indexFill :: Tensor d -> Int -> IndexTensor '[n] -> HsReal -> IO ()
_indexFill r x ix v = Dynamic._indexFill (asDynamic r) x (longAsDynamic ix) v

-- | Static call to 'Dynamic._indexSelect'
_indexSelect :: Tensor d -> Tensor d' -> Int -> IndexTensor '[n] -> IO ()
_indexSelect r t d ix = Dynamic._indexSelect (asDynamic r) (asDynamic t) d (longAsDynamic ix)

-- | Static call to 'Dynamic._take'
_take :: Tensor d -> Tensor d' -> IndexTensor '[n] -> IO ()
_take r t ix = Dynamic._take (asDynamic r) (asDynamic t) (longAsDynamic ix)

-- | Static call to 'Dynamic._put'
_put :: Tensor d -> IndexTensor '[n] -> Tensor d' -> Int -> IO ()
_put r ix t d = Dynamic._put (asDynamic r) (longAsDynamic ix) (asDynamic t) d


-- | Retrieve a single row from a matrix
--
-- FIXME: Use 'Idx' and remove the 'throwString' function
getRow
  :: forall t n m . (All KnownDim '[n, m], KnownNat m)
  => Tensor '[n, m] -> Natural -> IO (Tensor '[1, m])
getRow t r
  | r > fromIntegral (dimVal (dim :: Dim n)) = throwString "Row out of bounds"
  | otherwise = do
      res <- Dynamic.new (dims :: Dims '[1, m])
      let ixs = Ix.indexDyn [ fromIntegral r ]
      Dynamic._indexSelect res (asDynamic t) 0 ixs
      pure (asStatic res)

-- | Retrieve a single column from a matrix
--
-- FIXME: Use 'Idx' and remove the 'throwString' function
getColumn
  :: forall t n m . (All KnownDim '[n, m], KnownNat n)
  => Tensor '[n, m] -> Natural -> IO (Tensor '[n, 1])
getColumn t r
  | r > fromIntegral (dimVal (dim :: Dim m)) = throwString "Column out of bounds"
  | otherwise = do
      res <- new
      let ixs = Ix.indexDyn [ fromIntegral r ]
      _indexSelect res t 1 (longAsStatic ixs :: IndexTensor '[n])
      pure res


