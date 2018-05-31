{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.Tensor.Math.Pairwise where

import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.Math.Pairwise as Dynamic


_add r t v = Dynamic._add (asDynamic r) (asDynamic t) v
_sub r t v = Dynamic._sub (asDynamic r) (asDynamic t) v
_add_scaled r t v0 v1 = Dynamic._add_scaled (asDynamic r) (asDynamic t) v0 v1
_sub_scaled r t v0 v1 = Dynamic._sub_scaled (asDynamic r) (asDynamic t) v0 v1
_mul r t v = Dynamic._mul (asDynamic r) (asDynamic t) v
_div r t v = Dynamic._div (asDynamic r) (asDynamic t) v
_lshift r t v = Dynamic._lshift (asDynamic r) (asDynamic t) v
_rshift r t v = Dynamic._rshift (asDynamic r) (asDynamic t) v
_fmod r t v = Dynamic._fmod (asDynamic r) (asDynamic t) v
_remainder r t v = Dynamic._remainder (asDynamic r) (asDynamic t) v
_bitand r t v = Dynamic._bitand (asDynamic r) (asDynamic t) v
_bitor r t v = Dynamic._bitor (asDynamic r) (asDynamic t) v
_bitxor r t v = Dynamic._bitxor (asDynamic r) (asDynamic t) v
equal r t = Dynamic.equal (asDynamic r) (asDynamic t)


add_, add :: Dimensions d => Tensor d -> HsReal -> IO (Tensor d)
add_ t v = withInplace t $ \r t' -> _add r t' v
add  t v = withEmpty     $ \r    -> _add r t  v
(^+) :: Dimensions d => Tensor d -> HsReal -> (Tensor d)
(^+) a b = unsafePerformIO $ add a b
{-# NOINLINE (^+) #-}
(+^) :: Dimensions d => HsReal -> Tensor d -> (Tensor d)
(+^) = flip (^+)

sub_, sub :: Dimensions d => Tensor d -> HsReal -> IO (Tensor d)
sub_ t v = withInplace t $ \r t' -> _sub r t' v
sub  t v = withEmpty     $ \r    -> _sub r t  v
(^-) :: Dimensions d => Tensor d -> HsReal -> (Tensor d)
(^-) a b = unsafePerformIO $ sub a b
{-# NOINLINE (^-) #-}
(-^) :: Dimensions d => HsReal -> Tensor d -> (Tensor d)
(-^) = flip (^-)

add_scaled_, add_scaled :: Dimensions d => Tensor d -> HsReal -> HsReal -> IO (Tensor d)
add_scaled_ t v0 v1 = withInplace t $ \r t' -> _add_scaled r t' v0 v1
add_scaled  t v0 v1 = withEmpty     $ \r    -> _add_scaled r t  v0 v1

sub_scaled_, sub_scaled :: Dimensions d => Tensor d -> HsReal -> HsReal -> IO (Tensor d)
sub_scaled_ t v0 v1 = withInplace t $ \r t' -> _sub_scaled r t' v0 v1
sub_scaled  t v0 v1 = withEmpty     $ \r    -> _sub_scaled r t  v0 v1

mul_, mul :: Dimensions d => Tensor d -> HsReal -> IO (Tensor d)
mul_ t v = withInplace t $ \r t' -> _mul r t' v
mul  t v = withEmpty     $ \r    -> _mul r t  v
(^*) :: Dimensions d => Tensor d -> HsReal -> (Tensor d)
(^*) a b = unsafePerformIO $ mul a b
{-# NOINLINE (^*) #-}
(*^) :: Dimensions d => HsReal -> Tensor d -> (Tensor d)
(*^) = flip (^*)

div_, div :: Dimensions d => Tensor d -> HsReal -> IO (Tensor d)
div_ t v = withInplace t $ \r t' -> _div r t' v
div  t v = withEmpty     $ \r    -> _div r t  v
(^/) :: Dimensions d => Tensor d -> HsReal -> (Tensor d)
(^/) a b = unsafePerformIO $ Torch.Indef.Static.Tensor.Math.Pairwise.div a b
{-# NOINLINE (^/) #-}
(/^) :: Dimensions d => HsReal -> Tensor d -> (Tensor d)
(/^) = flip (^/)

lshift_, lshift :: Dimensions d => Tensor d -> HsReal -> IO (Tensor d)
lshift_ t v = withInplace t $ \r t' -> _lshift r t' v
lshift  t v = withEmpty     $ \r    -> _lshift r t  v

rshift_, rshift :: Dimensions d => Tensor d -> HsReal -> IO (Tensor d)
rshift_ t v = withInplace t $ \r t' -> _rshift r t' v
rshift  t v = withEmpty     $ \r    -> _rshift r t  v

fmod_, fmod :: Dimensions d => Tensor d -> HsReal -> IO (Tensor d)
fmod_ t v = withInplace t $ \r t' -> _fmod r t' v
fmod  t v = withEmpty     $ \r    -> _fmod r t  v

remainder_, remainder :: Dimensions d => Tensor d -> HsReal -> IO (Tensor d)
remainder_ t v = withInplace t $ \r t' -> _remainder r t' v
remainder  t v = withEmpty     $ \r    -> _remainder r t  v

bitand_, bitand :: Dimensions d => Tensor d -> HsReal -> IO (Tensor d)
bitand_ t v = withInplace t $ \r t' -> _bitand r t' v
bitand  t v = withEmpty     $ \r    -> _bitand r t  v

bitor_, bitor :: Dimensions d => Tensor d -> HsReal -> IO (Tensor d)
bitor_ t v = withInplace t $ \r t' -> _bitor r t' v
bitor  t v = withEmpty     $ \r    -> _bitor r t  v

bitxor_, bitxor :: Dimensions d => Tensor d -> HsReal -> IO (Tensor d)
bitxor_ t v = withInplace t $ \r t' -> _bitxor r t' v
bitxor  t v = withEmpty     $ \r    -> _bitxor r t  v


