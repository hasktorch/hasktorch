{-# LANGUAGE ConstraintKinds #-}
{- LANGUAGE DataKinds #-}
{- LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
module Torch.Core.Tensor.Static.Math.Floating where

import Data.Singletons
import Data.Singletons.Prelude.List
import Data.Singletons.TypeLits
import Foreign (Ptr)
import GHC.Int
import Data.Function (on)

import Torch.Class.C.Internal (HsReal, HsAccReal, AsDynamic)
import Torch.Core.Tensor.Dim
import Torch.Core.Tensor.Static (IsStatic(..), StaticConstraint, StaticConstraint2, withInplace, ByteTensor, LongTensor)
import THTypes
import THRandomTypes

import qualified Torch.Class.C.Tensor.Math as Class
import qualified Torch.Core.Tensor.Dynamic as Dynamic
import qualified Torch.Core.Storage as Storage
import qualified THLongTypes as Long

type FloatingMathConstraint t d =
  ( Class.TensorMathFloating (AsDynamic (t d))
  , HsReal (t d) ~ HsReal (AsDynamic (t d))
  , HsAccReal (t d) ~ HsAccReal (AsDynamic (t d))
  , IsStatic (t d)
  -- , Dimensions d
  )

cinv_         :: FloatingMathConstraint t d => t d -> t d -> IO ()
cinv_ = Class.cinv_ `on` asDynamic
sigmoid_      :: FloatingMathConstraint t d => t d -> t d -> IO ()
sigmoid_ = Class.sigmoid_ `on` asDynamic
log_          :: FloatingMathConstraint t d => t d -> t d -> IO ()
log_ = Class.log_ `on` asDynamic
lgamma_       :: FloatingMathConstraint t d => t d -> t d -> IO ()
lgamma_ = Class.lgamma_ `on` asDynamic
log1p_        :: FloatingMathConstraint t d => t d -> t d -> IO ()
log1p_ = Class.log1p_ `on` asDynamic
exp_          :: FloatingMathConstraint t d => t d -> t d -> IO ()
exp_ = Class.exp_ `on` asDynamic
cos_          :: FloatingMathConstraint t d => t d -> t d -> IO ()
cos_ = Class.cos_ `on` asDynamic
acos_         :: FloatingMathConstraint t d => t d -> t d -> IO ()
acos_ = Class.acos_ `on` asDynamic
cosh_         :: FloatingMathConstraint t d => t d -> t d -> IO ()
cosh_ = Class.cosh_ `on` asDynamic
sin_          :: FloatingMathConstraint t d => t d -> t d -> IO ()
sin_ = Class.sin_ `on` asDynamic
asin_         :: FloatingMathConstraint t d => t d -> t d -> IO ()
asin_ = Class.asin_ `on` asDynamic
sinh_         :: FloatingMathConstraint t d => t d -> t d -> IO ()
sinh_ = Class.sinh_ `on` asDynamic
tan_          :: FloatingMathConstraint t d => t d -> t d -> IO ()
tan_  = Class.tan_ `on` asDynamic
atan_         :: FloatingMathConstraint t d => t d -> t d -> IO ()
atan_ = Class.atan_ `on` asDynamic
atan2_        :: FloatingMathConstraint t d => t d -> t d -> t d -> IO ()
atan2_ r a b = Class.atan2_ (asDynamic r) (asDynamic a) (asDynamic b)
tanh_         :: FloatingMathConstraint t d => t d -> t d -> IO ()
tanh_ = Class.tanh_ `on` asDynamic
erf_          :: FloatingMathConstraint t d => t d -> t d -> IO ()
erf_  = Class.erf_ `on` asDynamic
erfinv_       :: FloatingMathConstraint t d => t d -> t d -> IO ()
erfinv_ = Class.erfinv_ `on` asDynamic
pow_          :: FloatingMathConstraint t d => t d -> t d -> HsReal (t d) -> IO ()
pow_ r a b = Class.pow_ (asDynamic r) (asDynamic a) b
tpow_         :: FloatingMathConstraint t d => t d -> HsReal (t d) -> t d -> IO ()
tpow_ r a b = Class.tpow_ (asDynamic r) a (asDynamic b)
sqrt_         :: FloatingMathConstraint t d => t d -> t d -> IO ()
sqrt_ = Class.sqrt_ `on` asDynamic
rsqrt_        :: FloatingMathConstraint t d => t d -> t d -> IO ()
rsqrt_ = Class.rsqrt_ `on` asDynamic
ceil_         :: FloatingMathConstraint t d => t d -> t d -> IO ()
ceil_ = Class.ceil_ `on` asDynamic
floor_        :: FloatingMathConstraint t d => t d -> t d -> IO ()
floor_ = Class.floor_ `on` asDynamic
round_        :: FloatingMathConstraint t d => t d -> t d -> IO ()
round_ = Class.round_ `on` asDynamic
trunc_        :: FloatingMathConstraint t d => t d -> t d -> IO ()
trunc_ = Class.trunc_ `on` asDynamic
frac_         :: FloatingMathConstraint t d => t d -> t d -> IO ()
frac_ = Class.frac_ `on` asDynamic
lerp_         :: FloatingMathConstraint t d => t d -> t d -> t d -> HsReal (t d) -> IO ()
lerp_ r a b = Class.lerp_ (asDynamic r) (asDynamic a) (asDynamic b)
mean_         :: FloatingMathConstraint t d => t d -> t d -> Int32 -> Int32 -> IO ()
mean_ r a b c = Class.mean_ (asDynamic r) (asDynamic a) b c
std_          :: FloatingMathConstraint t d => t d -> t d -> Int32 -> Int32 -> Int32 -> IO ()
std_ r a b c d = Class.std_ (asDynamic r) (asDynamic a) b c d
var_          :: FloatingMathConstraint t d => t d -> t d -> Int32 -> Int32 -> Int32 -> IO ()
var_ r a b c d = Class.var_ (asDynamic r) (asDynamic a) b c d
norm_         :: FloatingMathConstraint t d => t d -> t d -> HsReal (t d) -> Int32 -> Int32 -> IO ()
norm_ r a b c d = Class.norm_ (asDynamic r) (asDynamic a) b c d
renorm_       :: FloatingMathConstraint t d => t d -> t d -> HsReal (t d) -> Int32 -> HsReal (t d) -> IO ()
renorm_ r a b c d = Class.renorm_ (asDynamic r) (asDynamic a) b c d
dist          :: FloatingMathConstraint t d => t d -> t d -> HsReal (t d) -> IO (HsAccReal (t d))
dist r a b = Class.dist (asDynamic r) (asDynamic a) b
histc_        :: FloatingMathConstraint t d => t d -> t d -> Int64 -> HsReal (t d) -> HsReal (t d) -> IO ()
histc_ r a b c d = Class.histc_ (asDynamic r) (asDynamic a) b c d
bhistc_       :: FloatingMathConstraint t d => t d -> t d -> Int64 -> HsReal (t d) -> HsReal (t d) -> IO ()
bhistc_ r a b c d = Class.bhistc_ (asDynamic r) (asDynamic a) b c d
meanall       :: FloatingMathConstraint t d => t d -> IO (HsAccReal (t d))
meanall t     = Class.meanall (asDynamic t)
varall        :: FloatingMathConstraint t d => t d -> Int32 -> IO (HsAccReal (t d))
varall t      = Class.varall (asDynamic t)
stdall        :: FloatingMathConstraint t d => t d -> Int32 -> IO (HsAccReal (t d))
stdall t      = Class.stdall (asDynamic t)
normall       :: FloatingMathConstraint t d => t d -> HsReal (t d) -> IO (HsAccReal (t d))
normall t     = Class.normall (asDynamic t)
linspace_     :: FloatingMathConstraint t d => t d -> HsReal (t d) -> HsReal (t d) -> Int64 -> IO ()
linspace_ t   = Class.linspace_ (asDynamic t)
logspace_     :: FloatingMathConstraint t d => t d -> HsReal (t d) -> HsReal (t d) -> Int64 -> IO ()
logspace_ t   = Class.logspace_ (asDynamic t)
rand_         :: FloatingMathConstraint t d => t d -> Generator -> Long.Storage -> IO ()
rand_ t       = Class.rand_ (asDynamic t)
randn_        :: FloatingMathConstraint t d => t d -> Generator -> Long.Storage -> IO ()
randn_ t      = Class.randn_ (asDynamic t)

