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

import Torch.Dimensions
import Torch.Core.Tensor.Static.Math
import Torch.Class.Internal (HsReal, HsAccReal)
import Torch.Class.Tensor.Static (asStatic)

import System.IO.Unsafe (unsafePerformIO)
import Torch.Core.Tensor.Static
import qualified Torch.Core.Tensor.Static as Static
import qualified Torch.Core.Tensor.Dynamic as Dynamic
import qualified Torch.Core.Tensor.Static.Math.Unsafe as Unsafe (abs, constant, sign)

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


