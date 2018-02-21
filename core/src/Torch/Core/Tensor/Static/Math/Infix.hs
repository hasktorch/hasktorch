{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
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
import qualified Torch.Core.Tensor.Dynamic as Dynamic

(<.>) :: MathConstraint2 t d d' => t d -> t d' -> IO (HsAccReal (t d'))
(<.>) = dot

(!*) :: (MathConstraint3 t '[r, c] '[c] '[r]) => t '[r, c] -> t '[c] -> t '[r]
(!*) a b = unsafePerformIO (mv a b)
{-# NOINLINE (!*) #-}

(!*!)
  :: forall t a b c . MathConstraint3 t '[a, b] '[b, c] '[a, c]
  => t '[a, b] -> t '[b, c] -> t '[a, c]
(!*!) a b = unsafePerformIO $ (asStatic <$> Dynamic.new (dim :: Dim '[a, c])) >>= \n -> addmm 1 n 1 a b
{-# NOINLINE (!*!) #-}

(^+^) :: MathConstraint t d => t d -> t d -> t d
(^+^) t1 t2 = unsafePerformIO $ cadd t1 1 {-scale-} t2
{-# NOINLINE (^+^)  #-}

(^-^) :: MathConstraint t d => t d -> t d -> t d
(^-^) t1 t2 = unsafePerformIO $ csub t1 1 {-scale-} t2
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
(/^) a b = unsafePerformIO $ flip Torch.Core.Tensor.Static.Math.div a b
{-# NOINLINE (/^)  #-}

