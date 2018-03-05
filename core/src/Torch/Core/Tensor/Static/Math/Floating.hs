{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
module Torch.Core.Tensor.Static.Math.Floating where

import Data.Singletons
import Data.Singletons.Prelude.List
import Data.Singletons.TypeLits
import Foreign (Ptr)
import GHC.Int
import Data.Function (on)

import Torch.Class.Internal (HsReal, HsAccReal, AsDynamic)
import Torch.Dimensions
import Torch.Core.Tensor.Static (IsStatic(..), StaticConstraint, StaticConstraint2, withInplace, ByteTensor, LongTensor)
import Torch.Types.TH
import Torch.Types.TH.Random

import Torch.Class.Tensor.Math (TensorMathFloating)
import Torch.Core.FloatTensor.Static.Math.Floating ()
import Torch.Core.DoubleTensor.Static.Math.Floating ()

import qualified Torch.Class.Tensor.Math as Class
import qualified Torch.Core.Tensor.Dynamic as Dynamic
import qualified Torch.Core.Storage as Storage
import qualified Torch.Types.TH.Long as Long

type FloatingMathConstraint t d =
  ( Class.TensorMathFloating (AsDynamic (t d))
  , HsReal (t d) ~ HsReal (AsDynamic (t d))
  , HsAccReal (t d) ~ HsAccReal (AsDynamic (t d))
  , IsStatic (t d)
  , Dynamic.IsTensor (AsDynamic (t d))
  , Num (HsReal (t d))
  , Dimensions d
  )

cinv :: FloatingMathConstraint t d => t d -> IO (t d)
cinv t = withInplace $ \r -> Class.cinv_ r (asDynamic t)

sigmoid :: FloatingMathConstraint t d => t d -> IO (t d)
sigmoid t = withInplace $ \r -> Class.sigmoid_ r (asDynamic t)

log :: FloatingMathConstraint t d => t d -> IO (t d)
log t = withInplace $ \r -> Class.log_ r (asDynamic t)

lgamma :: FloatingMathConstraint t d => t d -> IO (t d)
lgamma t = withInplace $ \r -> Class.lgamma_ r (asDynamic t)

log1p :: FloatingMathConstraint t d => t d -> IO (t d)
log1p t = withInplace $ \r -> Class.log1p_ r (asDynamic t)

exp :: FloatingMathConstraint t d => t d -> IO (t d)
exp t = withInplace $ \r -> Class.exp_ r (asDynamic t)

cos :: FloatingMathConstraint t d => t d -> IO (t d)
cos t = withInplace $ \r -> Class.cos_ r (asDynamic t)

acos :: FloatingMathConstraint t d => t d -> IO (t d)
acos t = withInplace $ \r -> Class.acos_ r (asDynamic t)

cosh :: FloatingMathConstraint t d => t d -> IO (t d)
cosh t = withInplace $ \r -> Class.cosh_ r (asDynamic t)

sin :: FloatingMathConstraint t d => t d -> IO (t d)
sin t = withInplace $ \r -> Class.sin_ r (asDynamic t)

asin :: FloatingMathConstraint t d => t d -> IO (t d)
asin t = withInplace $ \r -> Class.asin_ r (asDynamic t)

sinh :: FloatingMathConstraint t d => t d -> IO (t d)
sinh t = withInplace $ \r -> Class.sinh_ r (asDynamic t)

tan :: FloatingMathConstraint t d => t d -> IO (t d)
tan t = withInplace $ \r -> Class.tan_ r (asDynamic t)

atan :: FloatingMathConstraint t d => t d -> IO (t d)
atan t = withInplace $ \r -> Class.atan_ r (asDynamic t)

atan2 :: FloatingMathConstraint t d => t d -> t d -> IO (t d)
atan2 a b = withInplace $ \r -> Class.atan2_ r (asDynamic a) (asDynamic b)

tanh :: FloatingMathConstraint t d => t d -> IO (t d)
tanh t = withInplace $ \r -> Class.tanh_ r (asDynamic t)

erf :: FloatingMathConstraint t d => t d -> IO (t d)
erf t = withInplace $ \r -> Class.erf_ r (asDynamic t)

erfinv :: FloatingMathConstraint t d => t d -> IO (t d)
erfinv t = withInplace $ \r -> Class.erfinv_ r (asDynamic t)

pow :: FloatingMathConstraint t d => t d -> HsReal (t d) -> IO (t d)
pow a b = withInplace $ \r -> Class.pow_ r (asDynamic a) b

round :: FloatingMathConstraint t d => t d -> IO (t d)
round t = withInplace $ \r -> Class.round_ r (asDynamic t)
