{-# LANGUAGE FlexibleContexts #-}
module Torch.Class.Tensor.Math.Compare.Static where

import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Dimensions

class TensorMathCompare t where
  _ltValue :: Dimensions d => MaskTensor t d -> t d -> HsReal (t d) -> IO ()
  _leValue :: Dimensions d => MaskTensor t d -> t d -> HsReal (t d) -> IO ()
  _gtValue :: Dimensions d => MaskTensor t d -> t d -> HsReal (t d) -> IO ()
  _geValue :: Dimensions d => MaskTensor t d -> t d -> HsReal (t d) -> IO ()
  _neValue :: Dimensions d => MaskTensor t d -> t d -> HsReal (t d) -> IO ()
  _eqValue :: Dimensions d => MaskTensor t d -> t d -> HsReal (t d) -> IO ()

  _ltValueT :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  _leValueT :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  _gtValueT :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  _geValueT :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  _neValueT :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  _eqValueT :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()

ltValue, leValue, gtValue, geValue, neValue, eqValue
  :: (Dimensions d, IsTensor (MaskTensor t), TensorMathCompare t)
  => t d -> HsReal (t d) -> IO (MaskTensor t d)
ltValue a b = withEmpty $ \r -> _ltValue r a b
leValue a b = withEmpty $ \r -> _leValue r a b
gtValue a b = withEmpty $ \r -> _gtValue r a b
geValue a b = withEmpty $ \r -> _geValue r a b
neValue a b = withEmpty $ \r -> _neValue r a b
eqValue a b = withEmpty $ \r -> _eqValue r a b

ltValueT, leValueT, gtValueT, geValueT, neValueT, eqValueT
  :: (IsTensor t, Dimensions d, TensorMathCompare t)
  => t d -> HsReal (t d) -> IO (t d)
ltValueT a b = withEmpty $ \r -> _ltValueT r a b
leValueT a b = withEmpty $ \r -> _leValueT r a b
gtValueT a b = withEmpty $ \r -> _gtValueT r a b
geValueT a b = withEmpty $ \r -> _geValueT r a b
neValueT a b = withEmpty $ \r -> _neValueT r a b
eqValueT a b = withEmpty $ \r -> _eqValueT r a b

ltValueT_, leValueT_, gtValueT_, geValueT_, neValueT_, eqValueT_
  :: (Dimensions d, TensorMathCompare t)
  => t d -> HsReal (t d) -> IO (t d)
ltValueT_ a b = _ltValueT a a b >> pure a
leValueT_ a b = _leValueT a a b >> pure a
gtValueT_ a b = _gtValueT a a b >> pure a
geValueT_ a b = _geValueT a a b >> pure a
neValueT_ a b = _neValueT a a b >> pure a
eqValueT_ a b = _eqValueT a a b >> pure a

