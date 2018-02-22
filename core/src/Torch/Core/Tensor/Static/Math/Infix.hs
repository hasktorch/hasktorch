{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Core.Tensor.Static.Math.Infix
  ( (<.>)
  , (+^), (^+)
  , (-^), (^-)
  , (*^), (^*)
  , (/^), (^/)
  , (^+^)
  , (^-^)
  , (^*^)
  , (^/^)
  , (!*!), (!*)
  ) where

import Torch.Core.Tensor.Dim
import Torch.Core.Tensor.Static.Math
import Torch.Class.C.Internal (HsReal, HsAccReal)
import Torch.Class.C.Tensor.Static (asStatic)

import System.IO.Unsafe (unsafePerformIO)
import Torch.Core.Tensor.Static
import qualified Torch.Core.Tensor.Static as Static
import qualified Torch.Core.Tensor.Dynamic as Dynamic
import qualified Torch.Core.Tensor.Static.Math.Unsafe as Unsafe (abs, constant, sign)

(<.>) :: MathConstraint2 t d d' => t d -> t d' -> HsAccReal (t d')
(<.>) a b = unsafePerformIO (dot a b)
{-# NOINLINE (<.>) #-}

(!*) :: (MathConstraint3 t '[r, c] '[c] '[r]) => t '[r, c] -> t '[c] -> t '[r]
(!*) a b = unsafePerformIO (mv a b)
{-# NOINLINE (!*) #-}

(!*!)
  :: forall t a b c . MathConstraint3 t '[a, b] '[b, c] '[a, c]
  => t '[a, b] -> t '[b, c] -> t '[a, c]
(!*!) a b = unsafePerformIO $ mmult a b
{-# NOINLINE (!*!) #-}

(^+^) :: MathConstraint t d => t d -> t d -> t d
(^+^) t1 t2 = unsafePerformIO $ cadd t1 1 t2
{-# NOINLINE (^+^)  #-}

(^-^) :: MathConstraint t d => t d -> t d -> t d
(^-^) t1 t2 = unsafePerformIO $ csub t1 1 t2
{-# NOINLINE (^-^)  #-}

(^*^) :: MathConstraint t d => t d -> t d -> t d
(^*^) a b = unsafePerformIO $ cmul a b
{-# NOINLINE (^*^)  #-}

(^/^) :: MathConstraint t d => t d -> t d -> t d
(^/^) a b = unsafePerformIO $ cdiv a b
{-# NOINLINE (^/^)  #-}

(^+) :: MathConstraint t d => t d -> HsReal (t d) -> t d
(^+) a b = unsafePerformIO $ add a b
{-# NOINLINE (^+)  #-}

(+^) :: MathConstraint t d => HsReal (t d) -> t d -> t d
(+^) a b = unsafePerformIO $ flip add a b
{-# NOINLINE (+^)  #-}

(^-) :: MathConstraint t d => t d -> HsReal (t d) -> t d
(^-) a b = unsafePerformIO $ sub a b
{-# NOINLINE (^-)  #-}

(-^) :: MathConstraint t d => HsReal (t d) -> t d -> t d
(-^) a b = unsafePerformIO $ flip sub a b -- addConst (neg t) val a b
{-# NOINLINE (-^)  #-}

(^*) :: MathConstraint t d => t d -> HsReal (t d) -> t d
(^*) a b = unsafePerformIO $ mul a b
{-# NOINLINE (^*)  #-}

(*^) :: MathConstraint t d => HsReal (t d) -> t d -> t d
(*^) a b = unsafePerformIO $ flip mul a b
{-# NOINLINE (*^)  #-}

(^/) :: MathConstraint t d => t d -> HsReal (t d) -> t d
(^/) a b = unsafePerformIO $ Torch.Core.Tensor.Static.Math.div a b
{-# NOINLINE (^/)  #-}

(/^) :: MathConstraint t d => HsReal (t d) -> t d -> t d
(/^) a b = unsafePerformIO $ flip Torch.Core.Tensor.Static.Math.div a b -- div a (cinv a)
{-# NOINLINE (/^)  #-}

instance Dimensions d => Num (ByteTensor d) where
  (+) = (^+^)
  {-# NOINLINE (+) #-}
  (-) = (^-^)
  {-# NOINLINE (-) #-}
  (*) = (^*^)
  {-# NOINLINE (*) #-}
  abs = id
  signum = Unsafe.sign
  {-# NOINLINE signum #-}
  fromInteger = Unsafe.constant . fromIntegral
  {-# NOINLINE fromInteger #-}

instance Dimensions d => Num (ShortTensor d) where
  (+) = (^+^)
  {-# NOINLINE (+) #-}
  (-) = (^-^)
  {-# NOINLINE (-) #-}
  (*) = (^*^)
  {-# NOINLINE (*) #-}
  abs = Unsafe.abs
  {-# NOINLINE abs #-}
  signum = Unsafe.sign
  {-# NOINLINE signum #-}
  fromInteger = Unsafe.constant . fromIntegral
  {-# NOINLINE fromInteger #-}

instance Dimensions d => Num (IntTensor d) where
  (+) = (^+^)
  {-# NOINLINE (+) #-}
  (-) = (^-^)
  {-# NOINLINE (-) #-}
  (*) = (^*^)
  {-# NOINLINE (*) #-}
  abs = Unsafe.abs
  {-# NOINLINE abs #-}
  signum = Unsafe.sign
  {-# NOINLINE signum #-}
  fromInteger = Unsafe.constant . fromIntegral
  {-# NOINLINE fromInteger #-}

instance Dimensions d => Num (LongTensor d) where
  (+) = (^+^)
  {-# NOINLINE (+) #-}
  (-) = (^-^)
  {-# NOINLINE (-) #-}
  (*) = (^*^)
  {-# NOINLINE (*) #-}
  abs = Unsafe.abs
  {-# NOINLINE abs #-}
  signum = Unsafe.sign
  {-# NOINLINE signum #-}
  fromInteger = Unsafe.constant . fromIntegral
  {-# NOINLINE fromInteger #-}

-- importing this module also unlocks the unsafe Num instance
instance Dimensions d => Num (DoubleTensor d) where
  (+) = (^+^)
  {-# NOINLINE (+) #-}
  (-) = (^-^)
  {-# NOINLINE (-) #-}
  (*) = (^*^)
  {-# NOINLINE (*) #-}
  abs = Unsafe.abs
  {-# NOINLINE abs #-}
  signum = Unsafe.sign
  {-# NOINLINE signum #-}
  fromInteger = Unsafe.constant . fromIntegral
  {-# NOINLINE fromInteger #-}

instance Dimensions d => Num (FloatTensor d) where
  (+) = (^+^)
  {-# NOINLINE (+) #-}
  (-) = (^-^)
  {-# NOINLINE (-) #-}
  (*) = (^*^)
  {-# NOINLINE (*) #-}
  abs = Unsafe.abs
  {-# NOINLINE abs #-}
  signum = Unsafe.sign
  {-# NOINLINE signum #-}
  fromInteger = Unsafe.constant . fromIntegral
  {-# NOINLINE fromInteger #-}


