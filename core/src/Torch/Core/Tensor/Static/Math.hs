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

-- retrieves a single row
getRow
  :: forall t n m . (KnownNat2 n m)
  => (MathConstraint2 t '[n, m] '[1, m])
  => t '[n, m] -> Natural -> IO (t '[1, m])
getRow t r
  | r > fromIntegral (natVal (Proxy :: Proxy n)) = throwString "Row out of bounds"
  | otherwise = do
      res <- Dynamic.new (dim :: Dim '[1, m])
      ixs :: Dynamic.LongTensor <- Dynamic.fromList1d [ fromIntegral r ]
      Dynamic.indexSelect_ res (asDynamic t) 0 ixs
      pure (asStatic res)

-- retrieves a single column
getColumn
  :: forall t n m . (KnownNat2 n m)
  => (MathConstraint2 t '[n, m] '[n, 1])
  => t '[n, m] -> Natural -> IO (t '[n, 1])
getColumn t r
  | r > fromIntegral (natVal (Proxy :: Proxy m)) = throwString "Column out of bounds"
  | otherwise = do
      res <- Dynamic.new (dim :: Dim '[n, 1])
      ixs :: Dynamic.LongTensor <- Dynamic.fromList1d [ fromIntegral r ]
      Dynamic.indexSelect_ res (asDynamic t) 1 ixs
      pure (asStatic res)

-- ========================================================================= --

constant :: MathConstraint t d => HsReal (t d) -> IO (t d)
constant v = withInplace (`Dynamic.fill_` v)

fill_ :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
fill_ t v = Dynamic.fill_ (asDynamic t) v >> pure t

zero :: MathConstraint t d => IO (t d)
zero = withInplace Dynamic.zero_

zero_ :: MathConstraint t d => t d -> IO ()
zero_ t = Dynamic.zero_ (asDynamic t)

maskedFill :: MathConstraint t d => ByteTensor d -> HsReal (t d) -> IO (t d)
maskedFill b v = withInplace (\res -> Dynamic.maskedFill_ res (asDynamic b) v)

maskedFill_ :: MathConstraint t d => t d -> ByteTensor d -> HsReal (t d) -> IO ()
maskedFill_ t b v = Dynamic.maskedFill_ (asDynamic t) (asDynamic b) v

maskedCopy :: MathConstraint2 t d d' => ByteTensor d -> t d -> IO (t d')
maskedCopy b t = withInplace (\res -> Dynamic.maskedCopy_ res (asDynamic b) (asDynamic t))

-- TODO: check out if we can use a linear-like type to invalidate the first parameter. Otherwise this impurely mutates the first variable
dangerMaskedCopy_ :: MathConstraint2 t d d' => t d -> ByteTensor d -> t d' -> IO (t d')
dangerMaskedCopy_ t b t' = do
  let res = asDynamic t
  Dynamic.maskedCopy_ res (asDynamic b) (asDynamic t')
  pure (asStatic res)

maskedSelect :: MathConstraint2 t d d' => t d -> ByteTensor d -> IO (t d')
maskedSelect t b = withInplace (\res -> Dynamic.maskedSelect_ res (asDynamic t) (asDynamic b))

-- TODO: check out if we can use a linear-like type to invalidate the first parameter. Otherwise this impurely mutates the first variable
dangerMaskedSelect_ :: MathConstraint2 t d d' => t d -> t d -> ByteTensor d -> IO (t d')
dangerMaskedSelect_ t t' b = do
  let res = asDynamic t
  Dynamic.maskedSelect_ res (asDynamic t') (asDynamic b)
  pure (asStatic res)

nonzero :: forall t d . (Dimensions d, IsStatic (t d), Dynamic.TensorMath (AsDynamic (t d))) => t d -> IO Dynamic.LongTensor
nonzero t = do
  l <- Dynamic.new (dim :: Dim d)
  Dynamic.nonzero_ l (asDynamic t)
  pure l

indexSelect_ :: MathConstraint t d => t d -> t d -> DimVal -> LongTensor '[n] -> IO ()
indexSelect_ r t d ix = Dynamic.indexSelect_ (asDynamic r) (asDynamic t) (fromIntegral d) (asDynamic ix)

indexSelect :: MathConstraint t d => t d -> DimVal -> LongTensor '[n] -> IO (t d)
indexSelect t d ix = withInplace $ \r -> Dynamic.indexSelect_ r (asDynamic t) (fromIntegral d) (asDynamic ix)

-- FIXME: Not sure if this is in-place or not
indexCopy_ :: MathConstraint t d => t d -> DimVal -> LongTensor '[n] -> t d -> IO ()
indexCopy_ r d ix t = Dynamic.indexCopy_ (asDynamic r) (fromIntegral d) (asDynamic ix) (asDynamic t)

indexAdd_ :: MathConstraint t d => t d -> DimVal -> LongTensor '[n] -> t d -> IO ()
indexAdd_ r d ix t = Dynamic.indexAdd_ (asDynamic r) (fromIntegral d) (asDynamic ix) (asDynamic t)

indexFill_ :: MathConstraint t d => t d -> DimVal -> LongTensor '[n] -> HsReal (t d) -> IO ()
indexFill_ r d ix v = Dynamic.indexFill_ (asDynamic r) (fromIntegral d) (asDynamic ix) v

take_ :: MathConstraint t d => t d -> t d -> LongTensor '[n] -> IO ()
take_ r t ix = Dynamic.take_ (asDynamic r) (asDynamic t) (asDynamic ix)

put_ :: MathConstraint t d => t d -> LongTensor '[n] -> t d -> DimVal -> IO ()
put_ r ix t d = Dynamic.put_ (asDynamic r) (asDynamic ix) (asDynamic t) (fromIntegral d)

gather_ :: MathConstraint t d => t d -> t d -> DimVal -> LongTensor '[n] -> IO ()
gather_ r t d ix = Dynamic.gather_ (asDynamic r) (asDynamic t) (fromIntegral d) (asDynamic ix)

scatter_ :: MathConstraint t d => t d -> DimVal -> LongTensor '[n] -> t d -> IO ()
scatter_ r d ix t = Dynamic.scatter_ (asDynamic r) (fromIntegral d) (asDynamic ix) (asDynamic t)

scatterAdd_ :: MathConstraint t d => t d -> DimVal -> LongTensor '[n] -> t d -> IO ()
scatterAdd_ r d ix t = Dynamic.scatterAdd_ (asDynamic r) (fromIntegral d) (asDynamic ix) (asDynamic t)

scatterFill_ :: MathConstraint t d => t d -> DimVal -> LongTensor '[n] -> HsReal (t d) -> IO ()
scatterFill_ r d ix v = Dynamic.scatterFill_ (asDynamic r) (fromIntegral d) (asDynamic ix) v

dot :: MathConstraint2 t d d' => t d -> t d' -> IO (HsAccReal (t d'))
dot a b = Dynamic.dot (asDynamic a) (asDynamic b)

vdot :: MathConstraint t '[n] => t '[n] -> t '[n] -> IO (HsAccReal (t '[n]))
vdot = dot

mdot
  :: MathConstraint2 t '[x, y] '[y, z]
  => t '[x, y] -> t '[y, z] -> IO (HsAccReal (t '[x, y]))
mdot = dot

minall :: MathConstraint t d => t d -> IO (HsReal (t d))
minall t = Dynamic.minall (asDynamic t)

maxall :: MathConstraint t d => t d -> IO (HsReal (t d))
maxall t = Dynamic.maxall (asDynamic t)

medianall :: MathConstraint t d => t d -> IO (HsReal (t d))
medianall t = Dynamic.medianall (asDynamic t)

sumall :: MathConstraint t d => t d -> IO (HsAccReal (t d))
sumall t = Dynamic.sumall (asDynamic t)

prodall :: MathConstraint t d => t d -> IO (HsAccReal (t d))
prodall t = Dynamic.prodall (asDynamic t)


-- helper functions operations

constOp_ :: MathConstraint t d => (AsDynamic (t d) -> AsDynamic (t d) -> HsReal (t d) -> IO ()) -> t d -> t d -> HsReal (t d) -> IO ()
constOp_ op r t v = op (asDynamic r) (asDynamic t) v

constOp2r_
  :: MathConstraint t d
  => (AsDynamic (t d) -> AsDynamic (t d) -> HsReal (t d) -> HsReal (t d) -> IO ())
  -> t d -> t d -> HsReal (t d) -> HsReal (t d) -> IO ()
constOp2r_ op r t v0 v1 = op (asDynamic r) (asDynamic t) v0 v1

constOp :: MathConstraint t d => (AsDynamic (t d) -> AsDynamic (t d) -> HsReal (t d) -> IO ()) -> t d -> HsReal (t d) -> IO (t d)
constOp op t v = withInplace $ \res -> op res (asDynamic t) v

constOp2r
  :: MathConstraint t d
  => (AsDynamic (t d) -> AsDynamic (t d) -> HsReal (t d) -> HsReal (t d) -> IO ())
  -> t d -> HsReal (t d) -> HsReal (t d) -> IO (t d)
constOp2r op t v0 v1 = withInplace $ \res -> op res (asDynamic t) v0 v1

cScaledOp :: MathConstraint t d => (AsDynamic (t d) -> AsDynamic (t d) -> HsReal (t d) -> AsDynamic (t d) -> IO ()) -> t d -> HsReal (t d) -> t d -> IO (t d)
cScaledOp op t v t' = withInplace $ \res -> op res (asDynamic t) v (asDynamic t')

cScaledOp_
  :: MathConstraint t d
  => (AsDynamic (t d) -> AsDynamic (t d) -> HsReal (t d) -> AsDynamic (t d) -> IO ())
  -> t d -> t d -> HsReal (t d) -> t d -> IO ()
cScaledOp_ op r t v t' = op (asDynamic r) (asDynamic t) v (asDynamic t')

cOp :: MathConstraint t d => (AsDynamic (t d) -> AsDynamic (t d) -> AsDynamic (t d) -> IO ()) -> t d -> t d -> IO (t d)
cOp op t t' = withInplace $ \res -> op res (asDynamic t) (asDynamic t')

cOp_
  :: MathConstraint t d
  => (AsDynamic (t d) -> AsDynamic (t d) -> AsDynamic (t d) -> IO ())
  -> t d -> t d -> t d -> IO ()
cOp_ op r t t' = op (asDynamic r) (asDynamic t) (asDynamic t')

-- Tensor-tensor op
ttOp
  :: MathConstraint t d
  => (AsDynamic (t d) -> HsReal (t d) -> AsDynamic (t d) -> HsReal (t d) -> AsDynamic (t d) -> AsDynamic (t d) -> IO ())
  -> HsReal (t d) -> t d -> HsReal (t d) -> t d -> t d -> IO (t d)
ttOp op v0 t v1 t' t'' = withInplace $ \res -> op res v0 (asDynamic t) v1 (asDynamic t') (asDynamic t'')

ttOp_
  :: MathConstraint t d
  => (AsDynamic (t d) -> HsReal (t d) -> AsDynamic (t d) -> HsReal (t d) -> AsDynamic (t d) -> AsDynamic (t d) -> IO ())
  -> t d -> HsReal (t d) -> t d -> HsReal (t d) -> t d -> t d -> IO ()
ttOp_ op res v0 t v1 t' t'' = op (asDynamic res) v0 (asDynamic t) v1 (asDynamic t') (asDynamic t'')


-- ========================================================================= --

add_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
add_ = constOp_ Dynamic.add_

add :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
add = constOp Dynamic.add_

sub_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
sub_ = constOp_ Dynamic.sub_

sub :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
sub = constOp Dynamic.sub_

add_scaled_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> HsReal (t d) -> IO ()
add_scaled_ = constOp2r_ Dynamic.add_scaled_

add_scaled :: MathConstraint t d => t d -> HsReal (t d) -> HsReal (t d) -> IO (t d)
add_scaled = constOp2r Dynamic.add_scaled_

sub_scaled_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> HsReal (t d) -> IO ()
sub_scaled_ = constOp2r_ Dynamic.sub_scaled_

sub_scaled :: MathConstraint t d => t d -> HsReal (t d) -> HsReal (t d) -> IO (t d)
sub_scaled = constOp2r Dynamic.sub_scaled_

mul_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
mul_ = constOp_ Dynamic.mul_

mul :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
mul = constOp Dynamic.mul_

div_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
div_ = constOp_ Dynamic.div_

div :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
div = constOp Dynamic.div_

lshift_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
lshift_ = constOp_ Dynamic.lshift_

lshift :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
lshift = constOp Dynamic.lshift_

rshift_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
rshift_ = constOp_ Dynamic.rshift_

rshift :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
rshift = constOp Dynamic.rshift_

fmod_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
fmod_ = constOp_ Dynamic.fmod_

fmod :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
fmod = constOp Dynamic.fmod_

remainder_    :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
remainder_ = constOp_ Dynamic.remainder_

remainder     :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
remainder = constOp Dynamic.remainder_

clamp_  :: MathConstraint t d => t d -> t d -> HsReal (t d) -> HsReal (t d) -> IO ()
clamp_  = constOp2r_ Dynamic.clamp_
clamp   :: MathConstraint t d => t d -> HsReal (t d) -> HsReal (t d) -> IO (t d)
clamp   = constOp2r Dynamic.clamp_

bitand_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
bitand_ = constOp_ Dynamic.bitand_
bitand  :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
bitand  = constOp Dynamic.bitand_
bitor_  :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
bitor_  = constOp_ Dynamic.bitor_
bitor   :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
bitor   = constOp Dynamic.bitor_
bitxor_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
bitxor_ = constOp_ Dynamic.bitxor_
bitxor  :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
bitxor  = constOp Dynamic.bitxor_

cadd_   :: MathConstraint t d => t d -> t d -> HsReal (t d) -> t d -> IO ()
cadd_   = cScaledOp_ Dynamic.cadd_
cadd    :: MathConstraint t d => t d -> HsReal (t d) -> t d -> IO (t d)
cadd    = cScaledOp Dynamic.cadd_
csub_   :: MathConstraint t d => t d -> t d -> HsReal (t d) -> t d -> IO ()
csub_   = cScaledOp_ Dynamic.csub_
csub    :: MathConstraint t d => t d -> HsReal (t d) -> t d -> IO (t d)
csub    = cScaledOp Dynamic.csub_


cmul_       :: MathConstraint t d => t d -> t d -> t d -> IO ()
cmul_       = cOp_ Dynamic.cmul_
cmul        :: MathConstraint t d => t d -> t d -> IO (t d)
cmul        = cOp Dynamic.cmul_

square_ :: MathConstraint t d => t d -> IO ()
square_ t = cmul_ t t t
square :: MathConstraint t d => t d -> IO (t d)
square t = cmul t t

cpow_       :: MathConstraint t d => t d -> t d -> t d -> IO ()
cpow_       = cOp_ Dynamic.cpow_
cpow        :: MathConstraint t d => t d -> t d -> IO (t d)
cpow        = cOp Dynamic.cpow_
cdiv_       :: MathConstraint t d => t d -> t d -> t d -> IO ()
cdiv_       = cOp_ Dynamic.cdiv_
cdiv        :: MathConstraint t d => t d -> t d -> IO (t d)
cdiv        = cOp Dynamic.cdiv_
clshift_    :: MathConstraint t d => t d -> t d -> t d -> IO ()
clshift_    = cOp_ Dynamic.clshift_
clshift     :: MathConstraint t d => t d -> t d -> IO (t d)
clshift     = cOp Dynamic.clshift_
crshift_    :: MathConstraint t d => t d -> t d -> t d -> IO ()
crshift_    = cOp_ Dynamic.crshift_
crshift     :: MathConstraint t d => t d -> t d -> IO (t d)
crshift     = cOp Dynamic.crshift_
cfmod_      :: MathConstraint t d => t d -> t d -> t d -> IO ()
cfmod_      = cOp_ Dynamic.cfmod_
cfmod       :: MathConstraint t d => t d -> t d -> IO (t d)
cfmod       = cOp Dynamic.cfmod_
cremainder_ :: MathConstraint t d => t d -> t d -> t d -> IO ()
cremainder_ = cOp_ Dynamic.cremainder_
cremainder  :: MathConstraint t d => t d -> t d -> IO (t d)
cremainder  = cOp Dynamic.cremainder_
cbitand_    :: MathConstraint t d => t d -> t d -> t d -> IO ()
cbitand_    = cOp_ Dynamic.cbitand_
cbitand     :: MathConstraint t d => t d -> t d -> IO (t d)
cbitand     = cOp Dynamic.cbitand_
cbitor_     :: MathConstraint t d => t d -> t d -> t d -> IO ()
cbitor_     = cOp_ Dynamic.cbitor_
cbitor      :: MathConstraint t d => t d -> t d -> IO (t d)
cbitor      = cOp Dynamic.cbitor_
cbitxor_    :: MathConstraint t d => t d -> t d -> t d -> IO ()
cbitxor_    = cOp_ Dynamic.cbitxor_
cbitxor     :: MathConstraint t d => t d -> t d -> IO (t d)
cbitxor     = cOp Dynamic.cbitxor_

addcmul_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> t d -> t d -> IO ()
addcmul_ r t v t' t'' = Dynamic.addcmul_ (asDynamic r) (asDynamic t) v (asDynamic t') (asDynamic t'')

addcmul :: MathConstraint t d => t d -> HsReal (t d) -> t d -> t d -> IO (t d)
addcmul t v t' t'' = withInplace $ \r -> Dynamic.addcmul_ r (asDynamic t) v (asDynamic t') (asDynamic t'')

addcdiv_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> t d -> t d -> IO ()
addcdiv_ r t v t' t'' = Dynamic.addcdiv_ (asDynamic r) (asDynamic t) v (asDynamic t') (asDynamic t'')

addcdiv :: MathConstraint t d => t d -> HsReal (t d) -> t d -> t d -> IO (t d)
addcdiv t v t' t'' = withInplace $ \r -> Dynamic.addcdiv_ r (asDynamic t) v (asDynamic t') (asDynamic t'')

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

-- outer product between a 1D tensor and a 1D tensor:
-- https://github.com/torch/torch7/blob/aed31711c6b8846b8337a263a7f9f998697994e7/doc/maths.md#res-torchaddrres-v1-mat-v2-vec1-vec2

-- res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
addr_
  :: (MathConstraint3 t '[r] '[c] '[r, c])
  => t '[r, c] -> HsReal (t '[r,c]) -> t '[r,c] -> HsReal (t '[r,c]) -> t '[r] -> t '[c] -> IO ()
addr_ r a t b x y = Dynamic.addr_ (asDynamic r) a (asDynamic t) b (asDynamic x) (asDynamic y)

addr :: (MathConstraint3 t '[r] '[c] '[r, c]) => HsReal (t '[r,c]) -> t '[r,c] -> HsReal (t '[r,c]) -> t '[r] -> t '[c] -> IO (t '[r, c])
addr  a t b x y = withInplace $ \r -> Dynamic.addr_ r a (asDynamic t) b (asDynamic x) (asDynamic y)

outer :: forall t r c . (MathConstraint3 t '[r] '[c] '[r, c]) => t '[r] -> t '[c] -> IO (t '[r, c])
outer v1 v2 = do
  t :: t '[r, c] <- zerosLike
  addr 0 t 1 v1 v2

addbmm_       :: MathConstraint t d => t d -> HsReal (t d) -> t d -> HsReal (t d) -> t d -> t d -> IO ()
addbmm_       = ttOp_ Dynamic.addbmm_
addbmm        :: MathConstraint t d => HsReal (t d) -> t d -> HsReal (t d) -> t d -> t d -> IO (t d)
addbmm        = ttOp  Dynamic.addbmm_
baddbmm_      :: MathConstraint t d => t d -> HsReal (t d) -> t d -> HsReal (t d) -> t d -> t d -> IO ()
baddbmm_      = ttOp_ Dynamic.baddbmm_
baddbmm       :: MathConstraint t d => HsReal (t d) -> t d -> HsReal (t d) -> t d -> t d -> IO (t d)
baddbmm       = ttOp  Dynamic.baddbmm_

match_ :: MathConstraint t d => t d -> t d -> t d -> HsReal (t d) -> IO ()
match_ r t t' v = Dynamic.match_ (asDynamic r) (asDynamic t) (asDynamic t') v

match  :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO (t d)
match t t' v = withInplace $ \r -> Dynamic.match_ r (asDynamic t) (asDynamic t') v

numel :: MathConstraint t d => t d -> IO Int64
numel t = Dynamic.numel (asDynamic t)

-- ========================================================================= --

keepDimOps_
  :: MathConstraint t d
  => ((AsDynamic (t d), Dynamic.LongTensor) -> AsDynamic (t d) -> Int32 -> Int32 -> IO ())
  -> (t d, Dynamic.LongTensor) -> t d -> DimVal -> Bool -> IO ()
keepDimOps_ op (r, ix) t d keep = op (asDynamic r, ix) (asDynamic t) (fromIntegral d) (fromIntegral $ fromEnum keep)

-- FIXME: find out how to _not_ pass in the index, I think that would add a small performance bump.
keepDimOps
  :: forall t d . MathConstraint t d
  => ((AsDynamic (t d), Dynamic.LongTensor) -> AsDynamic (t d) -> Int32 -> Int32 -> IO ())
  -> t d -> DimVal -> Bool -> IO (t d, Maybe Dynamic.LongTensor)
keepDimOps op t d keep = do
  ix :: Dynamic.LongTensor <- Dynamic.new (dim :: Dim d)
  res <- withInplace $ \r -> op (r, ix) (asDynamic t) (fromIntegral d) (fromIntegral $ fromEnum keep)
  pure (res, if keep then Just ix else Nothing)

-------------------------------------------------------------------------------

max_ :: MathConstraint t d => (t d, Dynamic.LongTensor) -> t d -> DimVal -> Bool -> IO ()
max_ = keepDimOps_ Dynamic.max_

max :: MathConstraint t d => t d -> DimVal -> Bool -> IO (t d, Maybe Dynamic.LongTensor)
max = keepDimOps Dynamic.max_

min_ :: MathConstraint t d => (t d, Dynamic.LongTensor) -> t d -> DimVal -> Bool -> IO ()
min_ = keepDimOps_ Dynamic.min_

min :: MathConstraint t d => t d -> DimVal -> Bool -> IO (t d, Maybe Dynamic.LongTensor)
min = keepDimOps Dynamic.min_

-- FIXME: unify with other 'keepDimOps_'
kthvalue_ :: MathConstraint t d => (t d, Dynamic.LongTensor) -> t d -> Int64 -> DimVal -> Bool -> IO ()
kthvalue_ (r, ix) t k d keep = Dynamic.kthvalue_ (asDynamic r, ix) (asDynamic t) k (fromIntegral d) (fromIntegral $ fromEnum keep)

-- FIXME: unify with other 'keepDimOps'
kthvalue :: forall t d . MathConstraint t d => t d -> Int64 -> DimVal -> Bool -> IO (t d, Maybe Dynamic.LongTensor)
kthvalue t k d keep = do
  ix :: Dynamic.LongTensor <- Dynamic.new (dim :: Dim d)
  res <- withInplace $ \r -> Dynamic.kthvalue_ (r, ix) (asDynamic t) k (fromIntegral d) (fromIntegral $ fromEnum keep)
  pure (res, if keep then Just ix else Nothing)

mode_ :: MathConstraint t d => (t d, Dynamic.LongTensor) -> t d -> DimVal -> Bool -> IO ()
mode_ = keepDimOps_ Dynamic.mode_

mode :: MathConstraint t d => t d -> DimVal -> Bool -> IO (t d, Maybe Dynamic.LongTensor)
mode = keepDimOps Dynamic.mode_

median_ :: MathConstraint t d => (t d, Dynamic.LongTensor) -> t d -> DimVal -> Bool -> IO ()
median_ = keepDimOps_ Dynamic.median_

median :: MathConstraint t d => t d -> DimVal -> Bool -> IO (t d, Maybe Dynamic.LongTensor)
median = keepDimOps Dynamic.median_

-- ========================================================================= --

-- TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
sum_ :: MathConstraint2 t d d' => t d' -> t d -> DimVal -> Bool -> IO ()
sum_ r t d k = Dynamic.sum_ (asDynamic r) (asDynamic t) (fromIntegral d) (fromIntegral $ fromEnum k)

sum :: MathConstraint2 t d d' => t d -> DimVal -> Bool -> IO (t d')
sum t d k = withInplace $ \r -> Dynamic.sum_ r (asDynamic t) (fromIntegral d) (fromIntegral $ fromEnum k)

rowsum :: MathConstraint2 t '[r, c] '[1, c] => t '[r, c] -> IO (t '[1, c])
rowsum t = Torch.Core.Tensor.Static.Math.sum t 0 True

colsum :: MathConstraint2 t '[r, c] '[r, 1] => t '[r, c] -> IO (t '[r, 1])
colsum t = Torch.Core.Tensor.Static.Math.sum t 1 True

prod_ :: MathConstraint2 t d d' => t d' -> t d -> Int32 -> Bool -> IO ()
prod_ r t d k = Dynamic.prod_ (asDynamic r) (asDynamic t) d (fromIntegral $ fromEnum k)

prod :: MathConstraint2 t d d' => t d -> Int32 -> Bool -> IO (t d')
prod t d k = withInplace $ \r -> Dynamic.prod_ r (asDynamic t) d (fromIntegral $ fromEnum k)

cumsum_ :: MathConstraint2 t d d' => t d' -> t d -> Int32 -> IO ()
cumsum_ r t d = Dynamic.cumsum_ (asDynamic r) (asDynamic t) d

cumsum :: MathConstraint2 t d d' => t d -> Int32 -> IO (t d')
cumsum t d = withInplace $ \r -> Dynamic.cumsum_ r (asDynamic t) d

cumprod_ :: MathConstraint2 t d d' => t d' -> t d -> Int32 -> IO ()
cumprod_ r t d = Dynamic.cumprod_ (asDynamic r) (asDynamic t) d

cumprod :: MathConstraint2 t d d' => t d -> Int32 -> IO (t d')
cumprod t d = withInplace $ \r -> Dynamic.cumprod_ r (asDynamic t) d

sign_ :: MathConstraint t d => t d -> t d -> IO ()
sign_ r t = Dynamic.sign_ (asDynamic r) (asDynamic t)

sign :: MathConstraint t d => t d -> IO (t d)
sign t = withInplace $ \r -> Dynamic.sign_ r (asDynamic t)

trace :: MathConstraint t d => t d -> IO (HsAccReal (t d))
trace = Dynamic.trace . asDynamic

cross :: MathConstraint3 t d d' d'' => t d -> t d' -> DimVal -> IO (t d'')
cross a b d = withInplace $ \res -> Dynamic.cross_ res (asDynamic a) (asDynamic b) (fromIntegral d)

cross_ :: MathConstraint t d => t d -> t d -> t d -> DimVal -> IO ()
cross_ r a b d = Dynamic.cross_ (asDynamic r) (asDynamic a) (asDynamic b) (fromIntegral d)

cmax_ :: MathConstraint t d => t d -> t d -> t d -> IO ()
cmax_ = cOp_ Dynamic.cmax_
cmax  :: MathConstraint t d => t d -> t d -> IO (t d)
cmax  = cOp Dynamic.cmax_
cmin_ :: MathConstraint t d => t d -> t d -> t d -> IO ()
cmin_ = cOp_ Dynamic.cmin_
cmin  :: MathConstraint t d => t d -> t d -> IO (t d)
cmin  = cOp Dynamic.cmin_

cmaxValue_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
cmaxValue_ r t v = Dynamic.cmaxValue_ (asDynamic r) (asDynamic t) v

cmaxValue :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
cmaxValue t v = withInplace $ \r -> Dynamic.cmaxValue_ r (asDynamic t) v

cminValue_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
cminValue_ r t v = Dynamic.cminValue_ (asDynamic r) (asDynamic t) v

cminValue :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
cminValue t v = withInplace $ \r -> Dynamic.cminValue_ r (asDynamic t) v

zeros_ :: MathConstraint t d => t d -> Storage.LongStorage -> IO ()
zeros_ r ls = Dynamic.zeros_ (asDynamic r) ls

zeros :: MathConstraint t d => Storage.LongStorage -> IO (t d)
zeros ls = withInplace $ \r -> Dynamic.zeros_ r ls

zerosLike :: forall t d . MathConstraint t d => IO (t d)
zerosLike = withInplace $ \r -> do
  shape <- Dynamic.new (dim :: Dim d)
  Dynamic.zerosLike_ r shape

zerosLike_ :: MathConstraint t d => t d -> t d -> IO ()
zerosLike_ r t = Dynamic.zerosLike_ (asDynamic r) (asDynamic t)

ones_ :: MathConstraint t d => t d -> Storage.LongStorage -> IO ()
ones_ r ls = Dynamic.ones_ (asDynamic r) ls

ones :: MathConstraint t d => Storage.LongStorage -> IO (t d)
ones ls = withInplace $ \r -> Dynamic.ones_ r ls

onesLike :: MathConstraint t d => t d -> IO (t d)
onesLike t = withInplace $ \r -> Dynamic.onesLike_ r (asDynamic t)

onesLike_ :: MathConstraint t d => t d -> t d -> IO ()
onesLike_ r t = Dynamic.onesLike_ (asDynamic r) (asDynamic t)

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

arange_ :: MathConstraint t d => t d -> HsAccReal (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
arange_ r s e step = Dynamic.arange_ (asDynamic r) s e step

arange :: MathConstraint t d => HsAccReal (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
arange s e step = withInplace $ \r -> Dynamic.arange_ r s e step

range_ :: MathConstraint t d => t d -> HsAccReal (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
range_ r s e step = Dynamic.range_ (asDynamic r) s e step

range :: MathConstraint t d => HsAccReal (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
range s e step = withInplace $ \r -> Dynamic.range_ r s e step

-- FIXME: find out if @n@ _must be_ in the dimension
randperm_ :: MathConstraint t d => t d -> Generator -> Int64 -> IO ()
randperm_ r g n = Dynamic.randperm_ (asDynamic r) g n

randperm :: MathConstraint t d => Generator -> Int64 -> IO (t d)
randperm g n = withInplace $ \r -> Dynamic.randperm_ r g n

reshape_ :: MathConstraint2 t d d' => t d' -> t d -> Storage.LongStorage -> IO ()
reshape_ r t s = Dynamic.reshape_ (asDynamic r) (asDynamic t) s

reshape :: MathConstraint2 t d d' => t d -> Storage.LongStorage -> IO (t d')
reshape t s = withInplace $ \r -> Dynamic.reshape_ r (asDynamic t) s

data DescendingOrder = Ascending | Descending
  deriving (Eq, Show, Ord, Enum, Bounded)

-- FIXME: unify with keepDimOps via fromIntegal on Bool
returnDimOps2
  :: forall t d a b . (MathConstraint t d, Integral a, Integral b)
  => (AsDynamic (t d) -> Dynamic.LongTensor -> AsDynamic (t d) -> Int32 -> Int32 -> IO ())
  -> t d -> a -> b -> IO (t d, Dynamic.LongTensor)
returnDimOps2 op t a b = do
  ix :: Dynamic.LongTensor <- Dynamic.new (dim :: Dim d)
  res <- withInplace $ \r -> op r ix (asDynamic t) (fromIntegral a) (fromIntegral b)
  pure (res, ix)

sort_ :: MathConstraint t d => (t d, Dynamic.LongTensor) -> t d -> DimVal -> DescendingOrder -> IO ()
sort_ (r, ix) t d o = Dynamic.sort_ (asDynamic r) ix (asDynamic t) (fromIntegral d) (fromIntegral $ fromEnum o)

sort :: MathConstraint t d => t d -> DimVal -> DescendingOrder -> IO (t d, Dynamic.LongTensor)
sort t d o = returnDimOps2 Dynamic.sort_ t d (fromEnum o)

-- https://github.com/torch/torch7/blob/75a86469aa9e2f5f04e11895b269ec22eb0e4687/lib/TH/generic/THTensorMath.c#L2545
data TopKOrder = KAscending | KNone | KDescending
  deriving (Eq, Show, Ord, Enum, Bounded)

topk_ :: MathConstraint2 t d d' => (t d', Dynamic.LongTensor) -> t d -> Int64 -> DimVal -> TopKOrder -> Bool -> IO ()
topk_ (r, ri) t k d o sorted = Dynamic.topk_ (asDynamic r) ri (asDynamic t) k (fromIntegral d) (fromIntegral $ fromEnum o) (fromIntegral $ fromEnum sorted)

topk :: forall t d d' . MathConstraint2 t d d' => t d -> Int64 -> DimVal -> TopKOrder -> Bool -> IO (t d', Dynamic.LongTensor)
topk t k d o sorted = do
  ix :: Dynamic.LongTensor <- Dynamic.new (dim :: Dim d')
  res <- withInplace $ \r -> Dynamic.topk_ r ix (asDynamic t) k (fromIntegral d) (fromIntegral $ fromEnum o) (fromIntegral $ fromEnum sorted)
  pure (res, ix)

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

equal :: MathConstraint t d => t d -> t d -> IO Bool
equal a b = (== 1) <$> Dynamic.equal (asDynamic a) (asDynamic b)

ltValue_ :: MathConstraint t d => ByteTensor d -> t d -> HsReal (t d) -> IO ()
ltValue_ b t v = Dynamic.ltValue_ (asDynamic b) (asDynamic t) v
ltValue :: MathConstraint t d => t d -> HsReal (t d) -> IO (ByteTensor d)
ltValue t v = withInplace $ \r -> Dynamic.ltValue_ r (asDynamic t) v
leValue_ :: MathConstraint t d => ByteTensor d -> t d -> HsReal (t d) -> IO ()
leValue_ b t v = Dynamic.leValue_ (asDynamic b) (asDynamic t) v
leValue :: MathConstraint t d => t d -> HsReal (t d) -> IO (ByteTensor d)
leValue t v = withInplace $ \r -> Dynamic.leValue_ r (asDynamic t) v
gtValue_ :: MathConstraint t d => ByteTensor d -> t d -> HsReal (t d) -> IO ()
gtValue_ b t v = Dynamic.gtValue_ (asDynamic b) (asDynamic t) v
gtValue :: MathConstraint t d => t d -> HsReal (t d) -> IO (ByteTensor d)
gtValue t v = withInplace $ \r -> Dynamic.gtValue_ r (asDynamic t) v
geValue_ :: MathConstraint t d => ByteTensor d -> t d -> HsReal (t d) -> IO ()
geValue_ b t v = Dynamic.geValue_ (asDynamic b) (asDynamic t) v
geValue :: MathConstraint t d => t d -> HsReal (t d) -> IO (ByteTensor d)
geValue t v = withInplace $ \r -> Dynamic.geValue_ r (asDynamic t) v
neValue_ :: MathConstraint t d => ByteTensor d -> t d -> HsReal (t d) -> IO ()
neValue_ b t v = Dynamic.neValue_ (asDynamic b) (asDynamic t) v
neValue :: MathConstraint t d => t d -> HsReal (t d) -> IO (ByteTensor d)
neValue t v = withInplace $ \r -> Dynamic.neValue_ r (asDynamic t) v
eqValue_ :: MathConstraint t d => ByteTensor d -> t d -> HsReal (t d) -> IO ()
eqValue_ b t v = Dynamic.eqValue_ (asDynamic b) (asDynamic t) v
eqValue :: MathConstraint t d => t d -> HsReal (t d) -> IO (ByteTensor d)
eqValue t v = withInplace $ \r -> Dynamic.eqValue_ r (asDynamic t) v

ltValueT_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
ltValueT_ r a v = Dynamic.ltValueT_ (asDynamic r) (asDynamic a) v
ltValueT :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
ltValueT a v = withInplace $ \r -> Dynamic.ltValueT_ r (asDynamic a) v
leValueT_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
leValueT_ r a v = Dynamic.leValueT_ (asDynamic r) (asDynamic a) v
leValueT :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
leValueT a v = withInplace $ \r -> Dynamic.leValueT_ r (asDynamic a) v
gtValueT_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
gtValueT_ r a v = Dynamic.gtValueT_ (asDynamic r) (asDynamic a) v
gtValueT :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
gtValueT a v = withInplace $ \r -> Dynamic.gtValueT_ r (asDynamic a) v
geValueT_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
geValueT_ r a v = Dynamic.geValueT_ (asDynamic r) (asDynamic a) v
geValueT :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
geValueT a v = withInplace $ \r -> Dynamic.geValueT_ r (asDynamic a) v
neValueT_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
neValueT_ r a v = Dynamic.neValueT_ (asDynamic r) (asDynamic a) v
neValueT :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
neValueT a v = withInplace $ \r -> Dynamic.neValueT_ r (asDynamic a) v
eqValueT_ :: MathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
eqValueT_ r a v = Dynamic.eqValueT_ (asDynamic r) (asDynamic a) v
eqValueT :: MathConstraint t d => t d -> HsReal (t d) -> IO (t d)
eqValueT a v = withInplace $ \r -> Dynamic.eqValueT_ r (asDynamic a) v

ltTensor_ :: MathConstraint t d => ByteTensor d -> t d -> t d -> IO ()
ltTensor_ r a b = Dynamic.ltTensor_ (asDynamic r) (asDynamic a) (asDynamic b)
ltTensor :: MathConstraint t d => t d -> t d -> IO (ByteTensor d)
ltTensor a b = withInplace $ \r -> Dynamic.ltTensor_ r (asDynamic a) (asDynamic b)
leTensor_ :: MathConstraint t d => ByteTensor d -> t d -> t d -> IO ()
leTensor_ r a b = Dynamic.leTensor_ (asDynamic r) (asDynamic a) (asDynamic b)
leTensor :: MathConstraint t d => t d -> t d -> IO (ByteTensor d)
leTensor a b = withInplace $ \r -> Dynamic.leTensor_ r (asDynamic a) (asDynamic b)
gtTensor_ :: MathConstraint t d => ByteTensor d -> t d -> t d -> IO ()
gtTensor_ r a b = Dynamic.gtTensor_ (asDynamic r) (asDynamic a) (asDynamic b)
gtTensor :: MathConstraint t d => t d -> t d -> IO (ByteTensor d)
gtTensor a b = withInplace $ \r -> Dynamic.gtTensor_ r (asDynamic a) (asDynamic b)
geTensor_ :: MathConstraint t d => ByteTensor d -> t d -> t d -> IO ()
geTensor_ r a b = Dynamic.geTensor_ (asDynamic r) (asDynamic a) (asDynamic b)
geTensor :: MathConstraint t d => t d -> t d -> IO (ByteTensor d)
geTensor a b = withInplace $ \r -> Dynamic.geTensor_ r (asDynamic a) (asDynamic b)
neTensor_ :: MathConstraint t d => ByteTensor d -> t d -> t d -> IO ()
neTensor_ r a b = Dynamic.neTensor_ (asDynamic r) (asDynamic a) (asDynamic b)
neTensor :: MathConstraint t d => t d -> t d -> IO (ByteTensor d)
neTensor a b = withInplace $ \r -> Dynamic.neTensor_ r (asDynamic a) (asDynamic b)
eqTensor_ :: MathConstraint t d => ByteTensor d -> t d -> t d -> IO ()
eqTensor_ r a b = Dynamic.eqTensor_ (asDynamic r) (asDynamic a) (asDynamic b)
eqTensor :: MathConstraint t d => t d -> t d -> IO (ByteTensor d)
eqTensor a b = withInplace $ \r -> Dynamic.eqTensor_ r (asDynamic a) (asDynamic b)

ltTensorT_ :: MathConstraint t d => t d -> t d -> t d -> IO ()
ltTensorT_ r a b = Dynamic.ltTensorT_ (asDynamic r) (asDynamic a) (asDynamic b)
ltTensorT :: MathConstraint t d => t d -> t d -> IO (t d)
ltTensorT a b = withInplace $ \r -> Dynamic.ltTensorT_ r (asDynamic a) (asDynamic b)
leTensorT_ :: MathConstraint t d => t d -> t d -> t d -> IO ()
leTensorT_ r a b = Dynamic.leTensorT_ (asDynamic r) (asDynamic a) (asDynamic b)
leTensorT :: MathConstraint t d => t d -> t d -> IO (t d)
leTensorT a b = withInplace $ \r -> Dynamic.leTensorT_ r (asDynamic a) (asDynamic b)
gtTensorT_ :: MathConstraint t d => t d -> t d -> t d -> IO ()
gtTensorT_ r a b = Dynamic.gtTensorT_ (asDynamic r) (asDynamic a) (asDynamic b)
gtTensorT :: MathConstraint t d => t d -> t d -> IO (t d)
gtTensorT a b = withInplace $ \r -> Dynamic.gtTensorT_ r (asDynamic a) (asDynamic b)
geTensorT_ :: MathConstraint t d => t d -> t d -> t d -> IO ()
geTensorT_ r a b = Dynamic.geTensorT_ (asDynamic r) (asDynamic a) (asDynamic b)
geTensorT :: MathConstraint t d => t d -> t d -> IO (t d)
geTensorT a b = withInplace $ \r -> Dynamic.geTensorT_ r (asDynamic a) (asDynamic b)
neTensorT_ :: MathConstraint t d => t d -> t d -> t d -> IO ()
neTensorT_ r a b = Dynamic.neTensorT_ (asDynamic r) (asDynamic a) (asDynamic b)
neTensorT :: MathConstraint t d => t d -> t d -> IO (t d)
neTensorT a b = withInplace $ \r -> Dynamic.neTensorT_ r (asDynamic a) (asDynamic b)
eqTensorT_ :: MathConstraint t d => t d -> t d -> t d -> IO ()
eqTensorT_ r a b = Dynamic.eqTensorT_ (asDynamic r) (asDynamic a) (asDynamic b)
eqTensorT :: MathConstraint t d => t d -> t d -> IO (t d)
eqTensorT a b = withInplace $ \r -> Dynamic.eqTensorT_ r (asDynamic a) (asDynamic b)

