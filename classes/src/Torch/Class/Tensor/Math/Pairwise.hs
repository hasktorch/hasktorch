module Torch.Class.Tensor.Math.Pairwise where

import Torch.Class.Tensor
import Torch.Class.Types

class IsTensor t => TensorMathPairwise t where
  _add :: t -> t -> HsReal t -> IO ()
  _sub :: t -> t -> HsReal t -> IO ()
  _add_scaled :: t -> t -> HsReal t -> HsReal t -> IO ()
  _sub_scaled :: t -> t -> HsReal t -> HsReal t -> IO ()
  _mul :: t -> t -> HsReal t -> IO ()
  _div :: t -> t -> HsReal t -> IO ()
  _lshift :: t -> t -> HsReal t -> IO ()
  _rshift :: t -> t -> HsReal t -> IO ()
  _fmod :: t -> t -> HsReal t -> IO ()
  _remainder :: t -> t -> HsReal t -> IO ()
  _bitand :: t -> t -> HsReal t -> IO ()
  _bitor :: t -> t -> HsReal t -> IO ()
  _bitxor :: t -> t -> HsReal t -> IO ()
  equal :: t -> t -> IO Bool

add_ :: TensorMathPairwise t => t -> HsReal t -> IO t
add_ t v = twice t $ \r t' -> _add r t' v

sub_ :: TensorMathPairwise t => t -> HsReal t -> IO t
sub_ t v = twice t $ \r t' -> _sub r t' v

add_scaled_ :: TensorMathPairwise t => t -> HsReal t -> HsReal t -> IO t
add_scaled_ t v0 v1 = twice t $ \r t' -> _add_scaled r t' v0 v1

sub_scaled_ :: TensorMathPairwise t => t -> HsReal t -> HsReal t -> IO t
sub_scaled_ t v0 v1 = twice t $ \r t' -> _sub_scaled r t' v0 v1

mul_ :: TensorMathPairwise t => t -> HsReal t -> IO t
mul_ t v = twice t $ \r t' -> _mul r t' v

div_ :: TensorMathPairwise t => t -> HsReal t -> IO t
div_ t v = twice t $ \r t' -> _div r t' v

lshift_ :: TensorMathPairwise t => t -> HsReal t -> IO t
lshift_ t v = twice t $ \r t' -> _lshift r t' v

rshift_ :: TensorMathPairwise t => t -> HsReal t -> IO t
rshift_ t v = twice t $ \r t' -> _rshift r t' v

fmod_ :: TensorMathPairwise t => t -> HsReal t -> IO t
fmod_ t v = twice t $ \r t' -> _fmod r t' v

remainder_ :: TensorMathPairwise t => t -> HsReal t -> IO t
remainder_ t v = twice t $ \r t' -> _remainder r t' v

bitand_ :: TensorMathPairwise t => t -> HsReal t -> IO t
bitand_ t v = twice t $ \r t' -> _bitand r t' v

bitor_ :: TensorMathPairwise t => t -> HsReal t -> IO t
bitor_ t v = twice t $ \r t' -> _bitor r t' v

bitxor_ :: TensorMathPairwise t => t -> HsReal t -> IO t
bitxor_ t v = twice t $ \r t' -> _bitxor r t' v


