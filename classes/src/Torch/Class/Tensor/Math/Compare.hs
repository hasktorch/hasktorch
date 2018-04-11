{-# LANGUAGE FlexibleContexts #-}
module Torch.Class.Tensor.Math.Compare where

import Torch.Class.Types
import Torch.Class.Tensor

class IsTensor t => TensorMathCompare t where
  _ltValue :: MaskDynamic t -> t -> HsReal t -> IO ()
  _leValue :: MaskDynamic t -> t -> HsReal t -> IO ()
  _gtValue :: MaskDynamic t -> t -> HsReal t -> IO ()
  _geValue :: MaskDynamic t -> t -> HsReal t -> IO ()
  _neValue :: MaskDynamic t -> t -> HsReal t -> IO ()
  _eqValue :: MaskDynamic t -> t -> HsReal t -> IO ()

  _ltValueT :: t -> t -> HsReal t -> IO ()
  _leValueT :: t -> t -> HsReal t -> IO ()
  _gtValueT :: t -> t -> HsReal t -> IO ()
  _geValueT :: t -> t -> HsReal t -> IO ()
  _neValueT :: t -> t -> HsReal t -> IO ()
  _eqValueT :: t -> t -> HsReal t -> IO ()

ltValue, leValue, gtValue, geValue, neValue, eqValue
  :: IsTensor (MaskDynamic t) => TensorMathCompare t => t -> HsReal t -> IO (MaskDynamic t)
ltValue  a b = withEmpty $ \r -> _ltValue r a b
leValue  a b = withEmpty $ \r -> _leValue r a b
gtValue  a b = withEmpty $ \r -> _gtValue r a b
geValue  a b = withEmpty $ \r -> _geValue r a b
neValue  a b = withEmpty $ \r -> _neValue r a b
eqValue  a b = withEmpty $ \r -> _eqValue r a b

ltValueT, leValueT, gtValueT, geValueT, neValueT, eqValueT
  :: TensorMathCompare t => t -> HsReal t -> IO t
ltValueT  a b = withEmpty $ \r -> _ltValueT r a b
leValueT  a b = withEmpty $ \r -> _leValueT r a b
gtValueT  a b = withEmpty $ \r -> _gtValueT r a b
geValueT  a b = withEmpty $ \r -> _geValueT r a b
neValueT  a b = withEmpty $ \r -> _neValueT r a b
eqValueT  a b = withEmpty $ \r -> _eqValueT r a b

ltValueT_, leValueT_, gtValueT_, geValueT_, neValueT_, eqValueT_
  :: TensorMathCompare t => t -> HsReal t -> IO t
ltValueT_ a b = _ltValueT a a b >> pure a
leValueT_ a b = _leValueT a a b >> pure a
gtValueT_ a b = _gtValueT a a b >> pure a
geValueT_ a b = _geValueT a a b >> pure a
neValueT_ a b = _neValueT a a b >> pure a
eqValueT_ a b = _eqValueT a a b >> pure a

