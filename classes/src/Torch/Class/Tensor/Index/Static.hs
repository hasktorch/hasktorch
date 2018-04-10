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
  _indexCopy   :: (Dimensions d, Dimensions d') => t d -> Int -> IndexTensor (t d) '[(n::Nat)] -> t d -> IO ()
  _indexAdd    :: (Dimensions d, Dimensions d') => t d -> Int -> IndexTensor (t d) '[(n::Nat)] -> t d -> IO ()
  _indexFill   :: (Dimensions d, Dimensions d') => t d -> Int -> IndexTensor (t d) '[(n::Nat)] -> HsReal (t d) -> IO ()
  _indexSelect :: (Dimensions d, Dimensions d', KnownNatDim n) => t d -> t d' -> Int -> IndexTensor (t d) '[n] -> IO ()
  _take        :: (Dimensions d, Dimensions d') => t d -> t d' -> IndexTensor (t d) '[(n::Nat)] -> IO ()
  _put         :: (Dimensions d, Dimensions d') => t d -> IndexTensor (t d) '[(n::Nat)] -> t d -> Int -> IO ()


-- retrieves a single row
getRow
  :: forall t n m . (KnownNatDim2 n m)
  => (Dynamic.TensorIndex (AsDynamic (t '[1, m])))
  => Dynamic.IsTensor (IndexDynamic (AsDynamic (t '[1, m])))
  => (Static2 t '[1, m] '[n, m])
  => t '[n, m] -> Natural -> IO (t '[1, m])
getRow t r
  | r > fromIntegral (natVal (Proxy :: Proxy n)) = throwString "Row out of bounds"
  | otherwise = do
      res <- Dynamic.new (dim :: Dim '[1, m])
      ixs <- Dynamic.fromList1d [ fromIntegral r ]
      Dynamic._indexSelect res (asDynamic t) 0 ixs
      pure (asStatic res)

-- -- retrieves a single column
-- getColumn
--   :: forall t n m . (KnownNat2 n m)
--   -- => (MathConstraint2 t '[n, m] '[n, 1])
--   => t '[n, m] -> Natural -> IO (t '[n, 1])
-- getColumn t r
--   | r > fromIntegral (natVal (Proxy :: Proxy m)) = throwString "Column out of bounds"
--   | otherwise = do
--       res <- Dynamic.new (dim :: Dim '[n, 1])
--       ixs :: Dynamic.LongTensor <- Dynamic.fromList1d [ fromIntegral r ]
--       Dynamic._indexSelect res (asDynamic t) 1 ixs
      -- pure (asStatic res)


