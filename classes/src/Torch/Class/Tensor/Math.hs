{-# LANGUAGE DataKinds #-}
module Torch.Class.Tensor.Math where

import Foreign
import Foreign.C.Types
import GHC.TypeLits (Nat)
import Torch.Dimensions
import Torch.Class.Types
import GHC.Int
import Torch.Class.Tensor (Tensor(empty)) -- , inplace, inplace1)
import qualified Torch.Types.TH as TH

class TensorMath t where
  fill_        :: t -> HsReal t -> IO ()
  zero_        :: t -> IO ()
  zeros_       :: t -> IndexStorage t -> IO ()
  zerosLike_   :: t -> t -> IO ()
  ones_        :: t -> TH.IndexStorage -> IO ()
  onesLike_    :: t -> t -> IO ()
  numel        :: t -> IO Integer
  reshape_     :: t -> t -> TH.IndexStorage -> IO ()
  cat_         :: t -> t -> t -> DimVal -> IO ()
  catArray_    :: t -> [t] -> Int -> DimVal -> IO ()
  nonzero_     :: IndexDynamic t -> t -> IO ()
  tril_        :: t -> t -> Integer -> IO ()
  triu_        :: t -> t -> Integer -> IO ()
  diag_        :: t -> t -> Int -> IO ()
  eye_         :: t -> Integer -> Integer -> IO ()
  trace        :: t -> IO (HsAccReal t)
  arange_      :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> IO ()
  range_       :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> IO ()

class CPUTensorMath t where
  match    :: t -> t -> t -> IO (HsReal t)
  kthvalue :: t -> IndexDynamic t -> t -> Integer -> Int -> IO Int
  randperm :: t -> Generator t -> Integer -> IO ()

class TensorMathFloating t where
  linspace_     :: t -> HsReal t -> HsReal t -> Int64 -> IO ()
  logspace_     :: t -> HsReal t -> HsReal t -> Int64 -> IO ()

{-
constant :: (IsTensor t, TensorMath t) => Dim (d::[Nat]) -> HsReal t -> IO t
constant d v = inplace (`fill_` v) d

zero :: (IsTensor t, TensorMath t) => Dim (d::[Nat]) -> IO t
zero d = inplace zero_ d

add :: (IsTensor t, TensorMath t) => t -> HsReal t -> IO t
add t x = flip inplace1 t $ \r -> add_ r t x

sub :: (IsTensor t, TensorMath t) => t -> HsReal t -> IO t
sub t x = flip inplace1 t $ \r -> sub_ r t x

add_scaled  :: (IsTensor t, TensorMath t) => t -> HsReal t -> HsReal t -> IO t
add_scaled t x y = flip inplace1 t $ \r -> add_scaled_ r t x y

sub_scaled  :: (IsTensor t, TensorMath t) => t -> HsReal t -> HsReal t -> IO t
sub_scaled t x y = flip inplace1 t $ \r -> sub_scaled_ r t x y

mul :: (IsTensor t, TensorMath t) => t -> HsReal t -> IO t
mul t x = flip inplace1 t $ \r -> mul_ r t x

div :: (IsTensor t, TensorMath t) => t -> HsReal t -> IO t
div t x = flip inplace1 t $ \r -> div_ r t x

lshift :: (IsTensor t, TensorMath t) => t -> HsReal t -> IO t
lshift t x = flip inplace1 t $ \r -> lshift_ r t x

rshift :: (IsTensor t, TensorMath t) => t -> HsReal t -> IO t
rshift t x = flip inplace1 t $ \r -> rshift_ r t x

fmod :: (IsTensor t, TensorMath t) => t -> HsReal t -> IO t
fmod t x = flip inplace1 t $ \r -> fmod_ r t x

remainder :: (IsTensor t, TensorMath t) => t -> HsReal t -> IO t
remainder t x = flip inplace1 t $ \r -> remainder_ r t x

clamp :: (IsTensor t, TensorMath t) => t -> HsReal t -> HsReal t -> IO t
clamp t x y = flip inplace1 t $ \r -> clamp_ r t x y

bitand :: (IsTensor t, TensorMath t) => t -> HsReal t -> IO t
bitand t x = flip inplace1 t $ \r -> bitand_ r t x

bitor :: (IsTensor t, TensorMath t) => t -> HsReal t -> IO t
bitor t x = flip inplace1 t $ \r -> bitor_ r t x

bitxor :: (IsTensor t, TensorMath t) => t -> HsReal t -> IO t
bitxor t x = flip inplace1 t $ \r -> bitxor_ r t x

cadd :: (IsTensor t, TensorMath t) => t -> HsReal t -> t -> IO t
cadd t v x = flip inplace1 t $ \r -> cadd_ r t v x

csub :: (IsTensor t, TensorMath t) => t -> HsReal t -> t -> IO t
csub t v x = flip inplace1 t $ \r -> csub_ r t v x

cmul :: (IsTensor t, TensorMath t) => t -> t -> IO t
cmul t x = flip inplace1 t $ \r -> cmul_ r t x

cpow :: (IsTensor t, TensorMath t) => t -> t -> IO t
cpow t x = flip inplace1 t $ \r -> cpow_ r t x

cdiv :: (IsTensor t, TensorMath t) => t -> t -> IO t
cdiv t x = flip inplace1 t $ \r -> cdiv_ r t x

clshift :: (IsTensor t, TensorMath t) => t -> t -> IO t
clshift t x = flip inplace1 t $ \r -> clshift_ r t x

crshift :: (IsTensor t, TensorMath t) => t -> t -> IO t
crshift t x = flip inplace1 t $ \r -> crshift_ r t x

cfmod :: (IsTensor t, TensorMath t) => t -> t -> IO t
cfmod t x = flip inplace1 t $ \r -> cfmod_ r t x

cremainder :: (IsTensor t, TensorMath t) => t -> t -> IO t
cremainder t x = flip inplace1 t $ \r -> cremainder_ r t x

cbitand :: (IsTensor t, TensorMath t) => t -> t -> IO t
cbitand t x = flip inplace1 t $ \r -> cbitand_ r t x

cbitor :: (IsTensor t, TensorMath t) => t -> t -> IO t
cbitor t x = flip inplace1 t $ \r -> cbitor_ r t x

cbitxor :: (IsTensor t, TensorMath t) => t -> t -> IO t
cbitxor t x = flip inplace1 t $ \r -> cbitxor_ r t x

-- addcmul_     :: t -> t -> HsReal t -> t -> t -> IO ()
-- addcdiv_     :: t -> t -> HsReal t -> t -> t -> IO ()

addmv :: (IsTensor t, TensorMath t) => HsReal t -> t -> HsReal t -> t -> t -> IO t
addmv m0 m v0 v x = flip inplace1 x $ \r -> addmv_ r m0 m v0 v x

addmm :: (IsTensor t, TensorMath t) => HsReal t -> t -> HsReal t -> t -> t -> IO t
addmm m0 m v0 v x = flip inplace1 x $ \r -> addmv_ r m0 m v0 v x

--  addr_        :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
--  addbmm_      :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
--  baddbmm_     :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
--  match_       :: t -> t -> t -> HsReal t -> IO ()
--  numel        :: t -> IO Int64
--  max_         :: (t, IndexTensor t) -> t -> Int32 -> Int32 -> IO ()
--  min_         :: (t, IndexTensor t) -> t -> Int32 -> Int32 -> IO ()
--  kthvalue_    :: (t, IndexTensor t) -> t -> Int64 -> Int32 -> Int32 -> IO ()
--  mode_        :: (t, IndexTensor t) -> t -> Int32 -> Int32 -> IO ()
--  median_      :: (t, IndexTensor t) -> t -> Int32 -> Int32 -> IO ()
--  sum_         :: t -> t -> Int32 -> Int32 -> IO ()
--  prod_        :: t -> t -> Int32 -> Int32 -> IO ()
--  cumsum_      :: t -> t -> Int32 -> IO ()
--  cumprod_     :: t -> t -> Int32 -> IO ()
--  sign_        :: t -> t -> IO ()
--  trace        :: t -> IO (HsAccReal t)
--  cross_       :: t -> t -> t -> Int32 -> IO ()
--  cmax_        :: t -> t -> t -> IO ()
--  cmin_        :: t -> t -> t -> IO ()
--  cmaxValue_   :: t -> t -> HsReal t -> IO ()
--  cminValue_   :: t -> t -> HsReal t -> IO ()
--  zeros_       :: t -> IndexStorage t -> IO ()
--  zerosLike_   :: t -> t -> IO ()
--  ones_        :: t -> IndexStorage t -> IO ()
--  onesLike_    :: t -> t -> IO ()
--  diag_        :: t -> t -> Int32 -> IO ()
--  eye_         :: t -> Int64 -> Int64 -> IO ()
--  arange_      :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> IO ()
--  range_       :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> IO ()
--  randperm_    :: t -> Generator -> Int64 -> IO ()
--  reshape_     :: t -> t -> IndexStorage t -> IO ()
--  sort_        :: t -> IndexTensor t -> t -> Int32 -> Int32 -> IO ()
--  topk_        :: t -> IndexTensor t -> t -> Int64 -> Int32 -> Int32 -> Int32 -> IO ()
--  tril_        :: t -> t -> Int64 -> IO ()
--  triu_        :: t -> t -> Int64 -> IO ()
--  cat_         :: t -> t -> t -> Int32 -> IO ()
--  catArray_    :: t -> [t] -> Int32 -> Int32 -> IO ()
--  equal        :: t -> t -> IO Int32
--  ltValue_     :: MaskTensor t -> t -> HsReal t -> IO ()
--  leValue_     :: MaskTensor t -> t -> HsReal t -> IO ()
--  gtValue_     :: MaskTensor t -> t -> HsReal t -> IO ()
--  geValue_     :: MaskTensor t -> t -> HsReal t -> IO ()
--  neValue_     :: MaskTensor t -> t -> HsReal t -> IO ()
--  eqValue_     :: MaskTensor t -> t -> HsReal t -> IO ()
--  ltValueT_    :: t -> t -> HsReal t -> IO ()
--  leValueT_    :: t -> t -> HsReal t -> IO ()
--  gtValueT_    :: t -> t -> HsReal t -> IO ()
--  geValueT_    :: t -> t -> HsReal t -> IO ()
--  neValueT_    :: t -> t -> HsReal t -> IO ()
--  eqValueT_    :: t -> t -> HsReal t -> IO ()
-- ltTensor_   :: t -> t -> IO MaskTensor t
-- leTensor_   :: t -> t -> IO MaskTensor t
-- gtTensor_   :: t -> t -> IO MaskTensor t
-- geTensor_   :: t -> t -> IO MaskTensor t
-- neTensor_   :: t -> t -> IO MaskTensor t
-- eqTensor_   :: t -> t -> IO MaskTensor t

ltTensorT :: (IsTensor t, TensorMath t) => t -> t -> IO t
ltTensorT t x = flip inplace1 t $ \r -> ltTensorT_ r t x
leTensorT :: (IsTensor t, TensorMath t) => t -> t -> IO t
leTensorT t x = flip inplace1 t $ \r -> leTensorT_ r t x
gtTensorT :: (IsTensor t, TensorMath t) => t -> t -> IO t
gtTensorT t x = flip inplace1 t $ \r -> gtTensorT_ r t x
geTensorT :: (IsTensor t, TensorMath t) => t -> t -> IO t
geTensorT t x = flip inplace1 t $ \r -> geTensorT_ r t x
neTensorT :: (IsTensor t, TensorMath t) => t -> t -> IO t
neTensorT t x = flip inplace1 t $ \r -> neTensorT_ r t x
eqTensorT :: (IsTensor t, TensorMath t) => t -> t -> IO t
eqTensorT t x = flip inplace1 t $ \r -> eqTensorT_ r t x

neg :: (IsTensor t, TensorMathSigned t) => t -> IO t
neg t = inplace1 (`neg_` t) t

abs :: (IsTensor t, TensorMathSigned t) => t -> IO t
abs t = inplace1 (`abs_` t) t

class TensorMathSigned t where
  neg_         :: t -> t -> IO ()
  abs_         :: t -> t -> IO ()

cinv :: (TensorMathFloating t, IsTensor t) => t -> IO t
cinv t = flip inplace1 t $ \r -> cinv_ r t

sigmoid :: (TensorMathFloating t, IsTensor t) => t -> IO t
sigmoid t = flip inplace1 t $ \r -> sigmoid_ r t

log :: (TensorMathFloating t, IsTensor t) => t -> IO t
log t = flip inplace1 t $ \r -> log_ r t

lgamma :: (TensorMathFloating t, IsTensor t) => t -> IO t
lgamma t = flip inplace1 t $ \r -> lgamma_ r t

log1p :: (TensorMathFloating t, IsTensor t) => t -> IO t
log1p t = flip inplace1 t $ \r -> log1p_ r t

exp :: (TensorMathFloating t, IsTensor t) => t -> IO t
exp t = flip inplace1 t $ \r -> exp_ r t

cos :: (TensorMathFloating t, IsTensor t) => t -> IO t
cos t = flip inplace1 t $ \r -> cos_ r t

acos :: (TensorMathFloating t, IsTensor t) => t -> IO t
acos t = flip inplace1 t $ \r -> acos_ r t

cosh :: (TensorMathFloating t, IsTensor t) => t -> IO t
cosh t = flip inplace1 t $ \r -> cosh_ r t

sin :: (TensorMathFloating t, IsTensor t) => t -> IO t
sin t = flip inplace1 t $ \r -> sin_ r t

asin :: (TensorMathFloating t, IsTensor t) => t -> IO t
asin t = flip inplace1 t $ \r -> asin_ r t

sinh :: (TensorMathFloating t, IsTensor t) => t -> IO t
sinh t = flip inplace1 t $ \r -> sinh_ r t

tan :: (TensorMathFloating t, IsTensor t) => t -> IO t
tan t = flip inplace1 t $ \r -> tan_ r t

atan :: (TensorMathFloating t, IsTensor t) => t -> IO t
atan t = flip inplace1 t $ \r -> atan_ r t

-- FIXME: not sure which dimensions to use as final dimensions
_atan2 :: (TensorMathFloating t, IsTensor t) => Dim (d::[Nat]) -> t -> t -> IO t
_atan2 d t0 t1 = flip inplace d $ \r -> atan2_ r t0 t1

tanh :: (TensorMathFloating t, IsTensor t) => t -> IO t
tanh t = flip inplace1 t $ \r -> tanh_ r t

erf :: (TensorMathFloating t, IsTensor t) => t -> IO t
erf t = flip inplace1 t $ \r -> erf_ r t

erfinv :: (TensorMathFloating t, IsTensor t) => t -> IO t
erfinv t = flip inplace1 t $ \r -> erfinv_ r t

pow :: (TensorMathFloating t, IsTensor t) => t -> HsReal t -> IO t
pow t v = flip inplace1 t $ \r -> pow_ r t v

_tpow :: (TensorMathFloating t, IsTensor t) => HsReal t -> t -> IO t
_tpow v t = flip inplace1 t $ \r -> tpow_ r v t

sqrt :: (TensorMathFloating t, IsTensor t) => t -> IO t
sqrt t = flip inplace1 t $ \r -> sqrt_ r t

rsqrt :: (TensorMathFloating t, IsTensor t) => t -> IO t
rsqrt t = flip inplace1 t $ \r -> rsqrt_ r t

ceil :: (TensorMathFloating t, IsTensor t) => t -> IO t
ceil t = flip inplace1 t $ \r -> ceil_ r t

floor :: (TensorMathFloating t, IsTensor t) => t -> IO t
floor t = flip inplace1 t $ \r -> floor_ r t

round :: (TensorMathFloating t, IsTensor t) => t -> IO t
round t = flip inplace1 t $ \r -> round_ r t

trunc :: (TensorMathFloating t, IsTensor t) => t -> IO t
trunc t = flip inplace1 t $ \r -> trunc_ r t

frac :: (TensorMathFloating t, IsTensor t) => t -> IO t
frac t = flip inplace1 t $ \r -> frac_ r t

_lerp :: (TensorMathFloating t, IsTensor t) => t -> t -> HsReal t -> IO t
_lerp t0 t1 v = flip inplace1 t0 $ \r -> lerp_ r t0 t1 v

mean :: (TensorMathFloating t, IsTensor t) => t -> Int32 -> Int32 -> IO t
mean t v0 v1 = flip inplace1 t $ \r -> mean_ r t v0 v1

std :: (TensorMathFloating t, IsTensor t) => t -> Int32 -> Int32 -> Int32 -> IO t
std t v0 v1 v2 = flip inplace1 t $ \r -> std_ r t v0 v1 v2

var :: (TensorMathFloating t, IsTensor t) => t -> Int32 -> Int32 -> Int32 -> IO t
var t v0 v1 v2 = flip inplace1 t $ \r -> var_ r t v0 v1 v2

norm :: (TensorMathFloating t, IsTensor t) => t -> HsReal t -> Int32 -> Int32 -> IO t
norm t v0 v1 v2 = flip inplace1 t $ \r -> norm_ r t v0 v1 v2

renorm :: (TensorMathFloating t, IsTensor t) => t -> HsReal t -> Int32 -> HsReal t -> IO t
renorm t v0 v1 v2 = flip inplace1 t $ \r -> renorm_ r t v0 v1 v2
-}

