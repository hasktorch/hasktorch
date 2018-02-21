{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Core.Tensor.Static.Math.Unsafe
  ( sumall
  , Torch.Core.Tensor.Static.Math.Unsafe.abs
  , square
  , constant
  , sign
  , zerosLike
  ) where

import Torch.Class.C.Internal (HsReal, HsAccReal, AsDynamic)
import Torch.Class.C.IsTensor (IsTensor)
import Torch.Core.Tensor.Dim
import Torch.Core.Tensor.Static.Math (MathConstraint, TensorMath, TensorMathSigned)
import System.IO.Unsafe (unsafePerformIO)

import qualified Torch.Core.Tensor.Static.Math as Safe

sumall :: MathConstraint t d => t d -> HsAccReal (t d)
sumall = unsafePerformIO . Safe.sumall
{-# NOINLINE sumall #-}

abs :: (Dimensions d, IsTensor (t d), TensorMathSigned (t d)) => t d -> t d
abs = unsafePerformIO . Safe.abs
{-# NOINLINE abs #-}

square :: (MathConstraint t d) => t d -> t d
square = unsafePerformIO . Safe.square
{-# NOINLINE square #-}

constant :: MathConstraint t d => HsReal (t d) -> t d
constant = unsafePerformIO . Safe.constant
{-# NOINLINE constant #-}

sign :: MathConstraint t d => t d -> t d
sign = unsafePerformIO . Safe.sign
{-# NOINLINE sign #-}

zerosLike :: MathConstraint t d => t d
zerosLike = unsafePerformIO Safe.zerosLike
{-# NOINLINE zerosLike #-}
