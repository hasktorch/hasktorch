{-# OPTIONS_GHC -fno-cse #-}
module Torch.Class.Tensor.Math.Pairwise.Static where

import System.IO.Unsafe

import Torch.Class.Tensor.Static
import Torch.Class.Types
import Torch.Dimensions

class IsTensor t => TensorMathPairwise t where
  _add :: t d -> t d -> HsReal (t d) -> IO ()
  _sub :: t d -> t d -> HsReal (t d) -> IO ()
  _add_scaled :: t d -> t d -> HsReal (t d) -> HsReal (t d) -> IO ()
  _sub_scaled :: t d -> t d -> HsReal (t d) -> HsReal (t d) -> IO ()
  _mul :: t d -> t d -> HsReal (t d) -> IO ()
  _div :: t d -> t d -> HsReal (t d) -> IO ()
  _lshift :: t d -> t d -> HsReal (t d) -> IO ()
  _rshift :: t d -> t d -> HsReal (t d) -> IO ()
  _fmod :: t d -> t d -> HsReal (t d) -> IO ()
  _remainder :: t d -> t d -> HsReal (t d) -> IO ()
  _bitand :: t d -> t d -> HsReal (t d) -> IO ()
  _bitor :: t d -> t d -> HsReal (t d) -> IO ()
  _bitxor :: t d -> t d -> HsReal (t d) -> IO ()
  equal :: t d -> t d -> IO Bool

add_, add :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> IO (t d)
add_ t v = withInplace t $ \r t' -> _add r t' v
add  t v = withEmpty     $ \r    -> _add r t  v
(^+) :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> (t d)
(^+) a b = unsafePerformIO $ add a b
{-# NOINLINE (^+) #-}
(+^) :: TensorMathPairwise t => Dimensions d => HsReal (t d) -> t d -> (t d)
(+^) = flip (^+)

sub_, sub :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> IO (t d)
sub_ t v = withInplace t $ \r t' -> _sub r t' v
sub  t v = withEmpty     $ \r    -> _sub r t  v
(^-) :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> (t d)
(^-) a b = unsafePerformIO $ sub a b
{-# NOINLINE (^-) #-}
(-^) :: TensorMathPairwise t => Dimensions d => HsReal (t d) -> t d -> (t d)
(-^) = flip (^-)

add_scaled_, add_scaled :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> HsReal (t d) -> IO (t d)
add_scaled_ t v0 v1 = withInplace t $ \r t' -> _add_scaled r t' v0 v1
add_scaled  t v0 v1 = withEmpty     $ \r    -> _add_scaled r t  v0 v1

sub_scaled_, sub_scaled :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> HsReal (t d) -> IO (t d)
sub_scaled_ t v0 v1 = withInplace t $ \r t' -> _sub_scaled r t' v0 v1
sub_scaled  t v0 v1 = withEmpty     $ \r    -> _sub_scaled r t  v0 v1

mul_, mul :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> IO (t d)
mul_ t v = withInplace t $ \r t' -> _mul r t' v
mul  t v = withEmpty     $ \r    -> _mul r t  v
(^*) :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> (t d)
(^*) a b = unsafePerformIO $ mul a b
{-# NOINLINE (^*) #-}
(*^) :: TensorMathPairwise t => Dimensions d => HsReal (t d) -> t d -> (t d)
(*^) = flip (^*)

div_, div :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> IO (t d)
div_ t v = withInplace t $ \r t' -> _div r t' v
div  t v = withEmpty     $ \r    -> _div r t  v
(^/) :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> (t d)
(^/) a b = unsafePerformIO $ Torch.Class.Tensor.Math.Pairwise.Static.div a b
{-# NOINLINE (^/) #-}
(/^) :: TensorMathPairwise t => Dimensions d => HsReal (t d) -> t d -> (t d)
(/^) = flip (^/)

lshift_, lshift :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> IO (t d)
lshift_ t v = withInplace t $ \r t' -> _lshift r t' v
lshift  t v = withEmpty     $ \r    -> _lshift r t  v

rshift_, rshift :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> IO (t d)
rshift_ t v = withInplace t $ \r t' -> _rshift r t' v
rshift  t v = withEmpty     $ \r    -> _rshift r t  v

fmod_, fmod :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> IO (t d)
fmod_ t v = withInplace t $ \r t' -> _fmod r t' v
fmod  t v = withEmpty     $ \r    -> _fmod r t  v

remainder_, remainder :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> IO (t d)
remainder_ t v = withInplace t $ \r t' -> _remainder r t' v
remainder  t v = withEmpty     $ \r    -> _remainder r t  v

bitand_, bitand :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> IO (t d)
bitand_ t v = withInplace t $ \r t' -> _bitand r t' v
bitand  t v = withEmpty     $ \r    -> _bitand r t  v

bitor_, bitor :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> IO (t d)
bitor_ t v = withInplace t $ \r t' -> _bitor r t' v
bitor  t v = withEmpty     $ \r    -> _bitor r t  v

bitxor_, bitxor :: TensorMathPairwise t => Dimensions d => t d -> HsReal (t d) -> IO (t d)
bitxor_ t v = withInplace t $ \r t' -> _bitxor r t' v
bitxor  t v = withEmpty     $ \r    -> _bitxor r t  v


