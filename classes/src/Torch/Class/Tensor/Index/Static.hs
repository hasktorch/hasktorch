{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
module Torch.Class.Tensor.Index.Static where

import Torch.Class.Types
import Torch.Class.Tensor.Static
import qualified Torch.Class.Tensor as Dynamic
import qualified Torch.Class.Tensor.Index as Dynamic
import GHC.TypeLits
import Data.Proxy
import GHC.Natural
import Torch.Dimensions
import Control.Exception.Safe


class IsTensor t => TensorIndex t where
  _indexCopy   :: (Dimensions2 d d') => t d -> Int -> IndexTensor t '[(n::Nat)] -> t d -> IO ()
  _indexAdd    :: (Dimensions2 d d') => t d -> Int -> IndexTensor t '[(n::Nat)] -> t d -> IO ()
  _indexFill   :: (Dimensions2 d d') => t d -> Int -> IndexTensor t '[(n::Nat)] -> HsReal (t d) -> IO ()
  _indexSelect :: (Dimensions2 d d', KnownNatDim n) => t d -> t d' -> Int -> IndexTensor t '[n] -> IO ()
  _take        :: (Dimensions2 d d') => t d -> t d' -> IndexTensor t '[(n::Nat)] -> IO ()
  _put         :: (Dimensions2 d d') => t d -> IndexTensor t '[(n::Nat)] -> t d -> Int -> IO ()


-- retrieves a single row
getRow
  :: forall t n m . (KnownNatDim2 n m)
  => (Dynamic.TensorIndex (AsDynamic t))
  => (Dynamic.IsTensor (AsDynamic t))
  => Dynamic.IsTensor (IndexDynamic (AsDynamic t))
  => (IsStatic t, Num (HsReal (IndexDynamic (AsDynamic t))))
  => t '[n, m] -> Natural -> IO (t '[1, m])
getRow t r
  | r > fromIntegral (natVal (Proxy :: Proxy n)) = throwString "Row out of bounds"
  | otherwise = do
      res <- Dynamic.new (dim :: Dim '[1, m])
      ixs <- Dynamic.fromList1d [ fromIntegral r ]
      Dynamic._indexSelect res (asDynamic t) 0 ixs
      pure (asStatic res)

-- retrieves a single column
getColumn
  :: forall t n m . (KnownNatDim2 n m)
  => (AsDynamic (IndexTensor t) ~ IndexDynamic (AsDynamic t))
  => (IsStatic t, IsTensor t)
  => (Dynamic.IsTensor (IndexDynamic (AsDynamic t)))
  => TensorIndex t
  => (Num (HsReal (IndexDynamic (AsDynamic t))))
  => (IsStatic (IndexTensor t))
  => t '[n, m] -> Natural -> IO (t '[n, 1])
getColumn t r
  | r > fromIntegral (natVal (Proxy :: Proxy m)) = throwString "Column out of bounds"
  | otherwise = do
      res <- new
      ixs :: AsDynamic (IndexTensor t) <- Dynamic.fromList1d [ fromIntegral r ]
      _indexSelect res t 1 (asStatic ixs :: IndexTensor t '[n])
      pure res


