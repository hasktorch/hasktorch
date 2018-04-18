{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Core.Tensor.Static.Math
  ( MathConstraint
  , MathConstraint2
  , MathConstraint3
  , getRow, getColumn

  , constant, fill_
  , zero, zero_
  , maskedFill, maskedFill_
  , maskedCopy, dangerMaskedCopy_
  , maskedSelect, dangerMaskedSelect_
  , nonzero
  , indexSelect_, indexSelect
  , indexCopy_
  , indexAdd_
  , indexFill_
  , take_
  , put_
  , gather_
  , scatter_
  , scatterAdd_
  , scatterFill_
  , dot, mdot, vdot
  , minall
  , maxall
  , medianall
  , sumall
  , prodall

  -- helpers:
  --
  -- , constOp_
  -- , constOp2r_
  -- , constOp
  -- , constOp2r
  -- , cScaledOp
  -- , cScaledOp_
  -- , cOp
  -- , cOp_
  -- , ttOp
  -- , ttOp_

  , add_, add
  , sub_, sub
  , add_scaled_, add_scaled
  , sub_scaled_, sub_scaled
  , mul_, mul
  , div_, Torch.Core.Tensor.Static.Math.div
  , lshift_, lshift
  , rshift_, rshift
  , fmod_, fmod
  , remainder_, remainder
  , clamp_, clamp
  , bitand_, bitand
  , bitor_, bitor
  , bitxor_, bitxor
  , cadd_, cadd
  , csub_, csub
  , cmul_, cmul, square, square_
  , cpow_, cpow
  , cdiv_, cdiv
  , clshift_, clshift
  , crshift_, crshift
  , cfmod_, cfmod
  , cremainder_, cremainder
  , cbitand_, cbitand
  , cbitor_, cbitor
  , cbitxor_, cbitxor
  , addcmul_, addcmul
  , addcdiv_, addcdiv
  , addmv_, addmv, mv
  , addmm_, addmm, mmult
  , addr_, addr, outer
  , addbmm_, addbmm
  , baddbmm_, baddbmm
  , match_, match
  , numel
  -- More helpers:
  -- , keepDimOps_
  -- , keepDimOps
  , max_, Torch.Core.Tensor.Static.Math.max
  , min_, Torch.Core.Tensor.Static.Math.min
  , kthvalue_, kthvalue
  , mode_, mode
  , median_, median
  , sum_, Torch.Core.Tensor.Static.Math.sum, rowsum, colsum
  , prod_, prod
  , cumsum_, cumsum
  , cumprod_, cumprod
  , sign_, sign
  , trace
  , cross, cross_
  , cmax_, cmax
  , cmin_, cmin
  , cmaxValue_, cmaxValue
  , cminValue_, cminValue
  , zeros_, zeros
  , zerosLike, zerosLike_
  , ones_, ones
  , onesLike, onesLike_
  , diag_, diag, diag1d
  , eye_, eye, eye2
  , arange_, arange
  , range_, range
  , randperm_, randperm
  , reshape_, reshape
  -- , returnDimOps2
  , DescendingOrder(..)
  , sort_, sort
  , TopKOrder(..)
  , topk_, topk
  , tril_, tril
  , triu_, triu
  , cat_, cat, cat1d
  , catArray_, catArray
  , equal
  , ltValue, ltValue_, leValue, leValue_, gtValue, gtValue_, geValue, geValue_, neValue, neValue_, eqValue, eqValue_
  , ltValueT, ltValueT_, leValueT, leValueT_, gtValueT, gtValueT_, geValueT, geValueT_, neValueT, neValueT_, eqValueT, eqValueT_
  , ltTensor, ltTensor_, leTensor, leTensor_, gtTensor, gtTensor_, geTensor, geTensor_, neTensor, neTensor_, eqTensor, eqTensor_
  , ltTensorT, ltTensorT_, leTensorT, leTensorT_, gtTensorT, gtTensorT_, geTensorT, geTensorT_, neTensorT, neTensorT_, eqTensorT, eqTensorT_

  , module Classes
  , module FloatingMath
  , module SignedMath
  ) where

import Control.Exception.Safe
import Data.Singletons
import Data.Singletons.Prelude.List
import Data.Singletons.TypeLits
import Foreign (Ptr)
import GHC.Int
import GHC.Natural

import Torch.Class.Internal (HsReal, HsAccReal, AsDynamic)
import Torch.Dimensions
import qualified Torch.Core.Tensor.Dynamic as Dynamic
import qualified Torch.Core.Storage as Storage
import Torch.Core.Tensor.Static (IsStatic(..), StaticConstraint, StaticConstraint2, withInplace, ByteTensor, LongTensor)
import Torch.Types.TH
import Torch.Types.TH.Random
-- ========================================================================= --
-- Reexports
import Torch.Class.Tensor.Math as Classes (TensorMath, TensorMathFloating, TensorMathSigned)
import Torch.Core.Tensor.Static.Math.Floating as FloatingMath
import Torch.Core.Tensor.Static.Math.Signed as SignedMath
import Torch.Core.ShortTensor.Static.Math.Signed ()
import Torch.Core.IntTensor.Static.Math.Signed ()
import Torch.Core.LongTensor.Static.Math.Signed ()
import Torch.Core.FloatTensor.Static.Math.Signed ()
import Torch.Core.DoubleTensor.Static.Math.Signed ()

-- ========================================================================= --
-- All static variants of the Torch.Class.Tensor.Math class
-- ========================================================================= --

type MathConstraint t d =
  ( Dynamic.TensorMath (AsDynamic (t d))
  , Dimensions d
  , StaticConstraint t d
  , HsAccReal (AsDynamic (t d)) ~ HsAccReal (t d)
  , HsReal (AsDynamic (t d)) ~ HsReal (t d)
  )

type MathConstraint2 t d d' =
  ( Dimensions d, Dimensions d'
  , Dynamic.TensorMath (AsDynamic (t d))
  -- , AsDynamic (t d) ~ AsDynamic (t d')
  , HsReal (t d) ~ HsReal (t d')
  , HsAccReal (t d) ~ HsAccReal (t d')
  , HsAccReal (t d) ~ HsAccReal (AsDynamic (t d'))
  , HsReal (AsDynamic (t d)) ~ HsReal (AsDynamic (t d'))
  , HsReal (t d) ~ HsReal (AsDynamic (t d'))
  , StaticConstraint2 t d d'
  )

-- FIXME: this is going to explode
type MathConstraint3 t d d' d'' =
  ( MathConstraint2 t d d'
  , MathConstraint2 t d d''
  , MathConstraint2 t d' d''
  )

-- ========================================================================= --
-- TODO: put these helpers somewhere better (maybe "Math.Matrix"?) also remove the code duplication here
-- by writing a static indexSelect function
-- ========================================================================= --

dot :: MathConstraint2 t d d' => t d -> t d' -> IO (HsAccReal (t d'))
dot a b = Dynamic.dot (asDynamic a) (asDynamic b)

vdot :: MathConstraint t '[n] => t '[n] -> t '[n] -> IO (HsAccReal (t '[n]))
vdot = dot

mdot
  :: MathConstraint2 t '[x, y] '[y, z]
  => t '[x, y] -> t '[y, z] -> IO (HsAccReal (t '[x, y]))
mdot = dot

-- ========================================================================= --

-- | added simplified use of addmv: src1 #> src2
mv :: forall t r c . (MathConstraint3 t '[r, c] '[c] '[r]) => t '[r, c] -> t '[c] -> IO (t '[r])
mv m v = Dynamic.new (dim :: Dim '[r]) >>= \n -> addmv 0 (asStatic n) 1 m v

-- | beta * t + alpha * (src1 #> src2)
addmv_
  :: (MathConstraint3 t '[r] '[r, c] '[c], MathConstraint t '[r])
  => t '[r] -> HsReal (t '[r]) -> t '[r] -> HsReal (t '[r]) -> t '[r, c] -> t '[c] -> IO ()
addmv_ r a t b x y = Dynamic.addmv_ (asDynamic r) a (asDynamic t) b (asDynamic x) (asDynamic y)

addmv
  :: (MathConstraint3 t '[r] '[r, c] '[c], MathConstraint t '[r])
  => HsReal (t '[r]) -> t '[r] -> HsReal (t '[r]) -> t '[r, c] -> t '[c] -> IO (t '[r])
addmv a t b x y = withInplace $ \r -> Dynamic.addmv_ r a (asDynamic t) b (asDynamic x) (asDynamic y)

-- ========================================================================= --

-- only matrix-matrix multiplication:
-- https://github.com/torch/torch7/blob/aed31711c6b8846b8337a263a7f9f998697994e7/doc/maths.md#res-torchaddmmres-v1-m-v2-mat1-mat2
addmm_
  :: forall t a b c . MathConstraint3 t '[a, b] '[b, c] '[a, c]
  => t '[a, c] -> HsReal (t '[a, c]) -> t '[a, c] -> HsReal (t '[a, c]) -> t '[a, b] -> t '[b, c] -> IO ()
addmm_ r a t b x y = Dynamic.addmm_ (asDynamic r) a (asDynamic t) b (asDynamic x) (asDynamic y)

-- res = (a * m) + (b * mat1 * mat2)
addmm
  :: forall t a b c . MathConstraint3 t '[a, b] '[b, c] '[a, c]
  => HsReal (t '[a, c]) -> t '[a, c] -> HsReal (t '[a, c]) -> t '[a, b] -> t '[b, c] -> IO (t '[a, c])
addmm a m b x y = withInplace $ \r -> Dynamic.addmm_ r a (asDynamic m) b (asDynamic x) (asDynamic y)

mmult :: forall t a b c . (MathConstraint3 t '[a, b] '[b, c] '[a, c]) => t '[a, b] -> t '[b, c] -> IO (t '[a, c])
mmult x y = constant 0 >>= \n -> addmm 1 n 1 x y

-- ========================================================================= --
-- ========================================================================= --

-- ========================================================================= --
diag_ :: MathConstraint2 t d d' => t d' -> t d -> DimVal -> IO ()
diag_ r t d = Dynamic.diag_ (asDynamic r) (asDynamic t) (fromIntegral d)

diag :: MathConstraint2 t d d' => t d -> DimVal -> IO (t d')
diag t d = withInplace $ \r -> Dynamic.diag_ r (asDynamic t) (fromIntegral d)

-- | Create a diagonal matrix from a 1D vector
diag1d :: (KnownNat n, MathConstraint2 t '[n] '[n,n]) => t '[n] -> IO (t '[n, n])
diag1d v = diag v 0

eye_ :: MathConstraint t d => t d -> Int64 -> Int64 -> IO ()
eye_ r x y = Dynamic.eye_ (asDynamic r) x y

eye :: MathConstraint t d => Int64 -> Int64 -> IO (t d)
eye x y = withInplace $ \r -> Dynamic.eye_ r x y

-- square matrix identity
eye2 :: forall t n . (KnownNat n, MathConstraint t '[n, n]) => IO (t '[n, n])
eye2 = eye n n
  where
   n :: Int64
   n = fromIntegral (natVal (Proxy :: Proxy n))

tril_ :: MathConstraint t '[x, y] => t '[x, y] -> t '[x, y] -> Int64 -> IO ()
tril_ r t k = Dynamic.tril_ (asDynamic r) (asDynamic t) k

tril :: MathConstraint t '[x, y] => t '[x, y] -> Int64 -> IO (t '[x, y])
tril t k = withInplace $ \r -> Dynamic.tril_ r (asDynamic t) k

triu_ :: MathConstraint t '[x, y] => t '[x, y] -> t '[x, y] -> Int64 -> IO ()
triu_ r t k = Dynamic.triu_ (asDynamic r) (asDynamic t) k

triu :: MathConstraint t '[x, y] => t '[x, y] -> Int64 -> IO (t '[x, y])
triu t k = withInplace $ \r -> Dynamic.triu_ r (asDynamic t) k

cat_ :: forall t d d' d'' . (MathConstraint3 t d d' d'') => t d'' -> t d -> t d' -> DimVal -> IO ()
cat_ r a b d = Dynamic.cat_ (asDynamic r) (asDynamic a) (asDynamic b) (fromIntegral d)

cat :: forall t d d' d'' . (MathConstraint3 t d d' d'') => t d -> t d' -> DimVal -> IO (t d'')
cat a b d = withInplace $ \r -> Dynamic.cat_ r (asDynamic a) (asDynamic b) (fromIntegral d)

-- Specialized version of cat
cat1d
  :: forall t n1 n2 n . (SingI n1, SingI n2, SingI n, n ~ Sum [n1, n2])
  => (MathConstraint3 t '[n] '[n1] '[n2])
  => t '[n1] -> t '[n2] -> IO (t '[n])
cat1d a b = cat a b 0

-- FIXME: someone should do more advanced dependent typing to sort this one out. For now use the dynamic version:
-- catArray_
--   :: forall t d d' . (MathConstraint2 t d' d)
--   => t d' -> (forall ds . Dimensions ds => [t ds]) -> Int32 -> DimVal -> IO ()
-- catArray_ r xs n_inputs dimension = Dynamic.catArray_ (asDynamic r) (asDynamic <$> xs) n_inputs (fromIntegral dimension)
catArray_ :: MathConstraint t d => t d -> [AsDynamic (t d)] -> Int32 -> DimVal -> IO ()
catArray_ r xs n_inputs dimension = Dynamic.catArray_ (asDynamic r) xs n_inputs (fromIntegral dimension)

catArray :: MathConstraint t d => [AsDynamic (t d)] -> Int32 -> DimVal -> IO (t d)
catArray xs n_inputs dimension = withInplace $ \r -> Dynamic.catArray_ r xs n_inputs (fromIntegral dimension)


