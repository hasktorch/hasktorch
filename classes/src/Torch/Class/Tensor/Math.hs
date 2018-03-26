{-# LANGUAGE DataKinds #-}
module Torch.Class.Tensor.Math where

import Foreign
import Foreign.C.Types
import GHC.TypeLits (Nat)
import Torch.Dimensions
import Torch.Class.Types
import GHC.Int
import Torch.Class.Tensor (Tensor(empty), inplace, inplace1)

class TensorMath t where
  fill_        :: t -> HsReal t -> io ()
  zero_        :: t -> io ()
  zeros_       :: t -> IndexStorage t -> io ()
  zerosLike_   :: t -> t -> io ()
  ones_        :: t -> IndexStorage t -> io ()
  onesLike_    :: t -> t -> io ()
  numel        :: t -> io Int64
  reshape_     :: t -> t -> IndexStorage t -> io ()
  cat_         :: t -> t -> t -> Int32 -> io ()
  catArray_    :: t -> [t] -> Int32 -> Int32 -> io ()
  nonzero_     :: IndexTensor t -> t -> io ()
  tril_        :: t -> t -> Int64 -> io ()
  triu_        :: t -> t -> Int64 -> io ()
  diag_        :: t -> t -> Int32 -> io ()
  eye_         :: t -> Int64 -> Int64 -> io ()
  trace        :: t -> io (HsAccReal t)
  arange_      :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> io ()
  range_       :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> io ()

class CPUTensorMath t where
  match    :: t -> t -> t -> io (HsReal t)
  kthvalue :: t -> IndexTensor t -> t -> Integer -> Int -> io Int
  randperm :: t -> Generator t -> Integer -> io ()

class TensorMathFloating t where
  linspace_     :: t -> HsReal t -> HsReal t -> Int64 -> io ()
  logspace_     :: t -> HsReal t -> HsReal t -> Int64 -> io ()


class TensorMasked t where
  maskedFill_  :: t -> MaskTensor t -> HsReal t -> io ()
  maskedCopy_  :: t -> MaskTensor t -> t -> io ()
  maskedSelect_ :: t -> t -> MaskTensor t -> io ()

class TensorIndex t where
  indexSelect_ :: t -> t -> Int32 -> IndexTensor t -> io ()
  indexCopy_   :: t -> Int32 -> IndexTensor t -> t -> io ()
  indexAdd_    :: t -> Int32 -> IndexTensor t -> t -> io ()
  indexFill_   :: t -> Int32 -> IndexTensor t -> HsReal t -> io ()
  take_        :: t -> t -> IndexTensor t -> io ()
  put_         :: t -> IndexTensor t -> t -> Int32 -> io ()

{-
constant :: (IsTensor t, TensorMath t) => Dim (d::[Nat]) -> HsReal t -> io t
constant d v = inplace (`fill_` v) d

zero :: (IsTensor t, TensorMath t) => Dim (d::[Nat]) -> io t
zero d = inplace zero_ d

add :: (IsTensor t, TensorMath t) => t -> HsReal t -> io t
add t x = flip inplace1 t $ \r -> add_ r t x

sub :: (IsTensor t, TensorMath t) => t -> HsReal t -> io t
sub t x = flip inplace1 t $ \r -> sub_ r t x

add_scaled  :: (IsTensor t, TensorMath t) => t -> HsReal t -> HsReal t -> io t
add_scaled t x y = flip inplace1 t $ \r -> add_scaled_ r t x y

sub_scaled  :: (IsTensor t, TensorMath t) => t -> HsReal t -> HsReal t -> io t
sub_scaled t x y = flip inplace1 t $ \r -> sub_scaled_ r t x y

mul :: (IsTensor t, TensorMath t) => t -> HsReal t -> io t
mul t x = flip inplace1 t $ \r -> mul_ r t x

div :: (IsTensor t, TensorMath t) => t -> HsReal t -> io t
div t x = flip inplace1 t $ \r -> div_ r t x

lshift :: (IsTensor t, TensorMath t) => t -> HsReal t -> io t
lshift t x = flip inplace1 t $ \r -> lshift_ r t x

rshift :: (IsTensor t, TensorMath t) => t -> HsReal t -> io t
rshift t x = flip inplace1 t $ \r -> rshift_ r t x

fmod :: (IsTensor t, TensorMath t) => t -> HsReal t -> io t
fmod t x = flip inplace1 t $ \r -> fmod_ r t x

remainder :: (IsTensor t, TensorMath t) => t -> HsReal t -> io t
remainder t x = flip inplace1 t $ \r -> remainder_ r t x

clamp :: (IsTensor t, TensorMath t) => t -> HsReal t -> HsReal t -> io t
clamp t x y = flip inplace1 t $ \r -> clamp_ r t x y

bitand :: (IsTensor t, TensorMath t) => t -> HsReal t -> io t
bitand t x = flip inplace1 t $ \r -> bitand_ r t x

bitor :: (IsTensor t, TensorMath t) => t -> HsReal t -> io t
bitor t x = flip inplace1 t $ \r -> bitor_ r t x

bitxor :: (IsTensor t, TensorMath t) => t -> HsReal t -> io t
bitxor t x = flip inplace1 t $ \r -> bitxor_ r t x

cadd :: (IsTensor t, TensorMath t) => t -> HsReal t -> t -> io t
cadd t v x = flip inplace1 t $ \r -> cadd_ r t v x

csub :: (IsTensor t, TensorMath t) => t -> HsReal t -> t -> io t
csub t v x = flip inplace1 t $ \r -> csub_ r t v x

cmul :: (IsTensor t, TensorMath t) => t -> t -> io t
cmul t x = flip inplace1 t $ \r -> cmul_ r t x

cpow :: (IsTensor t, TensorMath t) => t -> t -> io t
cpow t x = flip inplace1 t $ \r -> cpow_ r t x

cdiv :: (IsTensor t, TensorMath t) => t -> t -> io t
cdiv t x = flip inplace1 t $ \r -> cdiv_ r t x

clshift :: (IsTensor t, TensorMath t) => t -> t -> io t
clshift t x = flip inplace1 t $ \r -> clshift_ r t x

crshift :: (IsTensor t, TensorMath t) => t -> t -> io t
crshift t x = flip inplace1 t $ \r -> crshift_ r t x

cfmod :: (IsTensor t, TensorMath t) => t -> t -> io t
cfmod t x = flip inplace1 t $ \r -> cfmod_ r t x

cremainder :: (IsTensor t, TensorMath t) => t -> t -> io t
cremainder t x = flip inplace1 t $ \r -> cremainder_ r t x

cbitand :: (IsTensor t, TensorMath t) => t -> t -> io t
cbitand t x = flip inplace1 t $ \r -> cbitand_ r t x

cbitor :: (IsTensor t, TensorMath t) => t -> t -> io t
cbitor t x = flip inplace1 t $ \r -> cbitor_ r t x

cbitxor :: (IsTensor t, TensorMath t) => t -> t -> io t
cbitxor t x = flip inplace1 t $ \r -> cbitxor_ r t x

-- addcmul_     :: t -> t -> HsReal t -> t -> t -> io ()
-- addcdiv_     :: t -> t -> HsReal t -> t -> t -> io ()

addmv :: (IsTensor t, TensorMath t) => HsReal t -> t -> HsReal t -> t -> t -> io t
addmv m0 m v0 v x = flip inplace1 x $ \r -> addmv_ r m0 m v0 v x

addmm :: (IsTensor t, TensorMath t) => HsReal t -> t -> HsReal t -> t -> t -> io t
addmm m0 m v0 v x = flip inplace1 x $ \r -> addmv_ r m0 m v0 v x

--  addr_        :: t -> HsReal t -> t -> HsReal t -> t -> t -> io ()
--  addbmm_      :: t -> HsReal t -> t -> HsReal t -> t -> t -> io ()
--  baddbmm_     :: t -> HsReal t -> t -> HsReal t -> t -> t -> io ()
--  match_       :: t -> t -> t -> HsReal t -> io ()
--  numel        :: t -> io Int64
--  max_         :: (t, IndexTensor t) -> t -> Int32 -> Int32 -> io ()
--  min_         :: (t, IndexTensor t) -> t -> Int32 -> Int32 -> io ()
--  kthvalue_    :: (t, IndexTensor t) -> t -> Int64 -> Int32 -> Int32 -> io ()
--  mode_        :: (t, IndexTensor t) -> t -> Int32 -> Int32 -> io ()
--  median_      :: (t, IndexTensor t) -> t -> Int32 -> Int32 -> io ()
--  sum_         :: t -> t -> Int32 -> Int32 -> io ()
--  prod_        :: t -> t -> Int32 -> Int32 -> io ()
--  cumsum_      :: t -> t -> Int32 -> io ()
--  cumprod_     :: t -> t -> Int32 -> io ()
--  sign_        :: t -> t -> io ()
--  trace        :: t -> io (HsAccReal t)
--  cross_       :: t -> t -> t -> Int32 -> io ()
--  cmax_        :: t -> t -> t -> io ()
--  cmin_        :: t -> t -> t -> io ()
--  cmaxValue_   :: t -> t -> HsReal t -> io ()
--  cminValue_   :: t -> t -> HsReal t -> io ()
--  zeros_       :: t -> IndexStorage t -> io ()
--  zerosLike_   :: t -> t -> io ()
--  ones_        :: t -> IndexStorage t -> io ()
--  onesLike_    :: t -> t -> io ()
--  diag_        :: t -> t -> Int32 -> io ()
--  eye_         :: t -> Int64 -> Int64 -> io ()
--  arange_      :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> io ()
--  range_       :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> io ()
--  randperm_    :: t -> Generator -> Int64 -> io ()
--  reshape_     :: t -> t -> IndexStorage t -> io ()
--  sort_        :: t -> IndexTensor t -> t -> Int32 -> Int32 -> io ()
--  topk_        :: t -> IndexTensor t -> t -> Int64 -> Int32 -> Int32 -> Int32 -> io ()
--  tril_        :: t -> t -> Int64 -> io ()
--  triu_        :: t -> t -> Int64 -> io ()
--  cat_         :: t -> t -> t -> Int32 -> io ()
--  catArray_    :: t -> [t] -> Int32 -> Int32 -> io ()
--  equal        :: t -> t -> io Int32
--  ltValue_     :: MaskTensor t -> t -> HsReal t -> io ()
--  leValue_     :: MaskTensor t -> t -> HsReal t -> io ()
--  gtValue_     :: MaskTensor t -> t -> HsReal t -> io ()
--  geValue_     :: MaskTensor t -> t -> HsReal t -> io ()
--  neValue_     :: MaskTensor t -> t -> HsReal t -> io ()
--  eqValue_     :: MaskTensor t -> t -> HsReal t -> io ()
--  ltValueT_    :: t -> t -> HsReal t -> io ()
--  leValueT_    :: t -> t -> HsReal t -> io ()
--  gtValueT_    :: t -> t -> HsReal t -> io ()
--  geValueT_    :: t -> t -> HsReal t -> io ()
--  neValueT_    :: t -> t -> HsReal t -> io ()
--  eqValueT_    :: t -> t -> HsReal t -> io ()
-- ltTensor_   :: t -> t -> io MaskTensor t
-- leTensor_   :: t -> t -> io MaskTensor t
-- gtTensor_   :: t -> t -> io MaskTensor t
-- geTensor_   :: t -> t -> io MaskTensor t
-- neTensor_   :: t -> t -> io MaskTensor t
-- eqTensor_   :: t -> t -> io MaskTensor t

ltTensorT :: (IsTensor t, TensorMath t) => t -> t -> io t
ltTensorT t x = flip inplace1 t $ \r -> ltTensorT_ r t x
leTensorT :: (IsTensor t, TensorMath t) => t -> t -> io t
leTensorT t x = flip inplace1 t $ \r -> leTensorT_ r t x
gtTensorT :: (IsTensor t, TensorMath t) => t -> t -> io t
gtTensorT t x = flip inplace1 t $ \r -> gtTensorT_ r t x
geTensorT :: (IsTensor t, TensorMath t) => t -> t -> io t
geTensorT t x = flip inplace1 t $ \r -> geTensorT_ r t x
neTensorT :: (IsTensor t, TensorMath t) => t -> t -> io t
neTensorT t x = flip inplace1 t $ \r -> neTensorT_ r t x
eqTensorT :: (IsTensor t, TensorMath t) => t -> t -> io t
eqTensorT t x = flip inplace1 t $ \r -> eqTensorT_ r t x

neg :: (IsTensor t, TensorMathSigned t) => t -> io t
neg t = inplace1 (`neg_` t) t

abs :: (IsTensor t, TensorMathSigned t) => t -> io t
abs t = inplace1 (`abs_` t) t

class TensorMathSigned t where
  neg_         :: t -> t -> io ()
  abs_         :: t -> t -> io ()

cinv :: (TensorMathFloating t, IsTensor t) => t -> io t
cinv t = flip inplace1 t $ \r -> cinv_ r t

sigmoid :: (TensorMathFloating t, IsTensor t) => t -> io t
sigmoid t = flip inplace1 t $ \r -> sigmoid_ r t

log :: (TensorMathFloating t, IsTensor t) => t -> io t
log t = flip inplace1 t $ \r -> log_ r t

lgamma :: (TensorMathFloating t, IsTensor t) => t -> io t
lgamma t = flip inplace1 t $ \r -> lgamma_ r t

log1p :: (TensorMathFloating t, IsTensor t) => t -> io t
log1p t = flip inplace1 t $ \r -> log1p_ r t

exp :: (TensorMathFloating t, IsTensor t) => t -> io t
exp t = flip inplace1 t $ \r -> exp_ r t

cos :: (TensorMathFloating t, IsTensor t) => t -> io t
cos t = flip inplace1 t $ \r -> cos_ r t

acos :: (TensorMathFloating t, IsTensor t) => t -> io t
acos t = flip inplace1 t $ \r -> acos_ r t

cosh :: (TensorMathFloating t, IsTensor t) => t -> io t
cosh t = flip inplace1 t $ \r -> cosh_ r t

sin :: (TensorMathFloating t, IsTensor t) => t -> io t
sin t = flip inplace1 t $ \r -> sin_ r t

asin :: (TensorMathFloating t, IsTensor t) => t -> io t
asin t = flip inplace1 t $ \r -> asin_ r t

sinh :: (TensorMathFloating t, IsTensor t) => t -> io t
sinh t = flip inplace1 t $ \r -> sinh_ r t

tan :: (TensorMathFloating t, IsTensor t) => t -> io t
tan t = flip inplace1 t $ \r -> tan_ r t

atan :: (TensorMathFloating t, IsTensor t) => t -> io t
atan t = flip inplace1 t $ \r -> atan_ r t

-- FIXME: not sure which dimensions to use as final dimensions
_atan2 :: (TensorMathFloating t, IsTensor t) => Dim (d::[Nat]) -> t -> t -> io t
_atan2 d t0 t1 = flip inplace d $ \r -> atan2_ r t0 t1

tanh :: (TensorMathFloating t, IsTensor t) => t -> io t
tanh t = flip inplace1 t $ \r -> tanh_ r t

erf :: (TensorMathFloating t, IsTensor t) => t -> io t
erf t = flip inplace1 t $ \r -> erf_ r t

erfinv :: (TensorMathFloating t, IsTensor t) => t -> io t
erfinv t = flip inplace1 t $ \r -> erfinv_ r t

pow :: (TensorMathFloating t, IsTensor t) => t -> HsReal t -> io t
pow t v = flip inplace1 t $ \r -> pow_ r t v

_tpow :: (TensorMathFloating t, IsTensor t) => HsReal t -> t -> io t
_tpow v t = flip inplace1 t $ \r -> tpow_ r v t

sqrt :: (TensorMathFloating t, IsTensor t) => t -> io t
sqrt t = flip inplace1 t $ \r -> sqrt_ r t

rsqrt :: (TensorMathFloating t, IsTensor t) => t -> io t
rsqrt t = flip inplace1 t $ \r -> rsqrt_ r t

ceil :: (TensorMathFloating t, IsTensor t) => t -> io t
ceil t = flip inplace1 t $ \r -> ceil_ r t

floor :: (TensorMathFloating t, IsTensor t) => t -> io t
floor t = flip inplace1 t $ \r -> floor_ r t

round :: (TensorMathFloating t, IsTensor t) => t -> io t
round t = flip inplace1 t $ \r -> round_ r t

trunc :: (TensorMathFloating t, IsTensor t) => t -> io t
trunc t = flip inplace1 t $ \r -> trunc_ r t

frac :: (TensorMathFloating t, IsTensor t) => t -> io t
frac t = flip inplace1 t $ \r -> frac_ r t

_lerp :: (TensorMathFloating t, IsTensor t) => t -> t -> HsReal t -> io t
_lerp t0 t1 v = flip inplace1 t0 $ \r -> lerp_ r t0 t1 v

mean :: (TensorMathFloating t, IsTensor t) => t -> Int32 -> Int32 -> io t
mean t v0 v1 = flip inplace1 t $ \r -> mean_ r t v0 v1

std :: (TensorMathFloating t, IsTensor t) => t -> Int32 -> Int32 -> Int32 -> io t
std t v0 v1 v2 = flip inplace1 t $ \r -> std_ r t v0 v1 v2

var :: (TensorMathFloating t, IsTensor t) => t -> Int32 -> Int32 -> Int32 -> io t
var t v0 v1 v2 = flip inplace1 t $ \r -> var_ r t v0 v1 v2

norm :: (TensorMathFloating t, IsTensor t) => t -> HsReal t -> Int32 -> Int32 -> io t
norm t v0 v1 v2 = flip inplace1 t $ \r -> norm_ r t v0 v1 v2

renorm :: (TensorMathFloating t, IsTensor t) => t -> HsReal t -> Int32 -> HsReal t -> io t
renorm t v0 v1 v2 = flip inplace1 t $ \r -> renorm_ r t v0 v1 v2
-}

