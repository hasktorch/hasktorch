{-# LANGUAGE DataKinds #-}
module Torch.Class.Tensor.Math where

import Foreign hiding (new)
import Foreign.C.Types
import GHC.TypeLits (Nat)
import Torch.Dimensions
import Torch.Class.Types
import GHC.Int
import Torch.Class.Tensor (Tensor(empty), withInplace1, new) -- , withInplace, inplace1)
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

constant :: (TensorMath t, Tensor t) => Dim (d :: [Nat]) -> HsReal t -> IO t
constant d v = new d >>= withInplace1 (`fill_` v)

_tenLike
  :: (Tensor t, TensorMath t)
  => (t -> t -> IO ())
  -> Dim (d::[Nat]) -> IO t
_tenLike fn_ d = do
  src <- new d
  shape <- new d
  fn_ src shape
  pure src

onesLike, zerosLike
  :: (Tensor t, TensorMath t)
  => Dim (d::[Nat]) -> IO t
onesLike = _tenLike onesLike_
zerosLike = _tenLike zerosLike_


{-
add :: (Tensor t, TensorMath t) => t -> HsReal t -> IO t
add t x = flip withInplace1 t $ \r -> add_ r t x

sub :: (Tensor t, TensorMath t) => t -> HsReal t -> IO t
sub t x = flip withInplace1 t $ \r -> sub_ r t x

add_scaled  :: (Tensor t, TensorMath t) => t -> HsReal t -> HsReal t -> IO t
add_scaled t x y = flip withInplace1 t $ \r -> add_scaled_ r t x y

sub_scaled  :: (Tensor t, TensorMath t) => t -> HsReal t -> HsReal t -> IO t
sub_scaled t x y = flip withInplace1 t $ \r -> sub_scaled_ r t x y

mul :: (Tensor t, TensorMath t) => t -> HsReal t -> IO t
mul t x = flip withInplace1 t $ \r -> mul_ r t x

div :: (Tensor t, TensorMath t) => t -> HsReal t -> IO t
div t x = flip withInplace1 t $ \r -> div_ r t x

lshift :: (Tensor t, TensorMath t) => t -> HsReal t -> IO t
lshift t x = flip withInplace1 t $ \r -> lshift_ r t x

rshift :: (Tensor t, TensorMath t) => t -> HsReal t -> IO t
rshift t x = flip withInplace1 t $ \r -> rshift_ r t x

fmod :: (Tensor t, TensorMath t) => t -> HsReal t -> IO t
fmod t x = flip withInplace1 t $ \r -> fmod_ r t x

remainder :: (Tensor t, TensorMath t) => t -> HsReal t -> IO t
remainder t x = flip withInplace1 t $ \r -> remainder_ r t x

clamp :: (Tensor t, TensorMath t) => t -> HsReal t -> HsReal t -> IO t
clamp t x y = flip withInplace1 t $ \r -> clamp_ r t x y

bitand :: (Tensor t, TensorMath t) => t -> HsReal t -> IO t
bitand t x = flip withInplace1 t $ \r -> bitand_ r t x

bitor :: (Tensor t, TensorMath t) => t -> HsReal t -> IO t
bitor t x = flip withInplace1 t $ \r -> bitor_ r t x

bitxor :: (Tensor t, TensorMath t) => t -> HsReal t -> IO t
bitxor t x = flip withInplace1 t $ \r -> bitxor_ r t x

cadd :: (Tensor t, TensorMath t) => t -> HsReal t -> t -> IO t
cadd t v x = flip withInplace1 t $ \r -> cadd_ r t v x

csub :: (Tensor t, TensorMath t) => t -> HsReal t -> t -> IO t
csub t v x = flip withInplace1 t $ \r -> csub_ r t v x

cmul :: (Tensor t, TensorMath t) => t -> t -> IO t
cmul t x = flip withInplace1 t $ \r -> cmul_ r t x

cpow :: (Tensor t, TensorMath t) => t -> t -> IO t
cpow t x = flip withInplace1 t $ \r -> cpow_ r t x

cdiv :: (Tensor t, TensorMath t) => t -> t -> IO t
cdiv t x = flip withInplace1 t $ \r -> cdiv_ r t x

clshift :: (Tensor t, TensorMath t) => t -> t -> IO t
clshift t x = flip withInplace1 t $ \r -> clshift_ r t x

crshift :: (Tensor t, TensorMath t) => t -> t -> IO t
crshift t x = flip withInplace1 t $ \r -> crshift_ r t x

cfmod :: (Tensor t, TensorMath t) => t -> t -> IO t
cfmod t x = flip withInplace1 t $ \r -> cfmod_ r t x

cremainder :: (Tensor t, TensorMath t) => t -> t -> IO t
cremainder t x = flip withInplace1 t $ \r -> cremainder_ r t x

cbitand :: (Tensor t, TensorMath t) => t -> t -> IO t
cbitand t x = flip withInplace1 t $ \r -> cbitand_ r t x

cbitor :: (Tensor t, TensorMath t) => t -> t -> IO t
cbitor t x = flip withInplace1 t $ \r -> cbitor_ r t x

cbitxor :: (Tensor t, TensorMath t) => t -> t -> IO t
cbitxor t x = flip withInplace1 t $ \r -> cbitxor_ r t x

-- addcmul_     :: t -> t -> HsReal t -> t -> t -> IO ()
-- addcdiv_     :: t -> t -> HsReal t -> t -> t -> IO ()

addmv :: (Tensor t, TensorMath t) => HsReal t -> t -> HsReal t -> t -> t -> IO t
addmv m0 m v0 v x = flip withInplace1 x $ \r -> addmv_ r m0 m v0 v x

addmm :: (Tensor t, TensorMath t) => HsReal t -> t -> HsReal t -> t -> t -> IO t
addmm m0 m v0 v x = flip withInplace1 x $ \r -> addmv_ r m0 m v0 v x

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

ltTensorT :: (Tensor t, TensorMath t) => t -> t -> IO t
ltTensorT t x = flip withInplace1 t $ \r -> ltTensorT_ r t x
leTensorT :: (Tensor t, TensorMath t) => t -> t -> IO t
leTensorT t x = flip withInplace1 t $ \r -> leTensorT_ r t x
gtTensorT :: (Tensor t, TensorMath t) => t -> t -> IO t
gtTensorT t x = flip withInplace1 t $ \r -> gtTensorT_ r t x
geTensorT :: (Tensor t, TensorMath t) => t -> t -> IO t
geTensorT t x = flip withInplace1 t $ \r -> geTensorT_ r t x
neTensorT :: (Tensor t, TensorMath t) => t -> t -> IO t
neTensorT t x = flip withInplace1 t $ \r -> neTensorT_ r t x
eqTensorT :: (Tensor t, TensorMath t) => t -> t -> IO t
eqTensorT t x = flip withInplace1 t $ \r -> eqTensorT_ r t x

neg :: (Tensor t, TensorMathSigned t) => t -> IO t
neg t = withInplace1 (`neg_` t) t

abs :: (Tensor t, TensorMathSigned t) => t -> IO t
abs t = withInplace1 (`abs_` t) t

class TensorMathSigned t where
  neg_         :: t -> t -> IO ()
  abs_         :: t -> t -> IO ()

cinv :: (TensorMathFloating t, Tensor t) => t -> IO t
cinv t = flip withInplace1 t $ \r -> cinv_ r t

sigmoid :: (TensorMathFloating t, Tensor t) => t -> IO t
sigmoid t = flip withInplace1 t $ \r -> sigmoid_ r t

log :: (TensorMathFloating t, Tensor t) => t -> IO t
log t = flip withInplace1 t $ \r -> log_ r t

lgamma :: (TensorMathFloating t, Tensor t) => t -> IO t
lgamma t = flip withInplace1 t $ \r -> lgamma_ r t

log1p :: (TensorMathFloating t, Tensor t) => t -> IO t
log1p t = flip withInplace1 t $ \r -> log1p_ r t

exp :: (TensorMathFloating t, Tensor t) => t -> IO t
exp t = flip withInplace1 t $ \r -> exp_ r t

cos :: (TensorMathFloating t, Tensor t) => t -> IO t
cos t = flip withInplace1 t $ \r -> cos_ r t

acos :: (TensorMathFloating t, Tensor t) => t -> IO t
acos t = flip withInplace1 t $ \r -> acos_ r t

cosh :: (TensorMathFloating t, Tensor t) => t -> IO t
cosh t = flip withInplace1 t $ \r -> cosh_ r t

sin :: (TensorMathFloating t, Tensor t) => t -> IO t
sin t = flip withInplace1 t $ \r -> sin_ r t

asin :: (TensorMathFloating t, Tensor t) => t -> IO t
asin t = flip withInplace1 t $ \r -> asin_ r t

sinh :: (TensorMathFloating t, Tensor t) => t -> IO t
sinh t = flip withInplace1 t $ \r -> sinh_ r t

tan :: (TensorMathFloating t, Tensor t) => t -> IO t
tan t = flip withInplace1 t $ \r -> tan_ r t

atan :: (TensorMathFloating t, Tensor t) => t -> IO t
atan t = flip withInplace1 t $ \r -> atan_ r t

-- FIXME: not sure which dimensions to use as final dimensions
_atan2 :: (TensorMathFloating t, Tensor t) => Dim (d::[Nat]) -> t -> t -> IO t
_atan2 d t0 t1 = flip withInplace d $ \r -> atan2_ r t0 t1

tanh :: (TensorMathFloating t, Tensor t) => t -> IO t
tanh t = flip withInplace1 t $ \r -> tanh_ r t

erf :: (TensorMathFloating t, Tensor t) => t -> IO t
erf t = flip withInplace1 t $ \r -> erf_ r t

erfinv :: (TensorMathFloating t, Tensor t) => t -> IO t
erfinv t = flip withInplace1 t $ \r -> erfinv_ r t

pow :: (TensorMathFloating t, Tensor t) => t -> HsReal t -> IO t
pow t v = flip withInplace1 t $ \r -> pow_ r t v

_tpow :: (TensorMathFloating t, Tensor t) => HsReal t -> t -> IO t
_tpow v t = flip withInplace1 t $ \r -> tpow_ r v t

sqrt :: (TensorMathFloating t, Tensor t) => t -> IO t
sqrt t = flip withInplace1 t $ \r -> sqrt_ r t

rsqrt :: (TensorMathFloating t, Tensor t) => t -> IO t
rsqrt t = flip withInplace1 t $ \r -> rsqrt_ r t

ceil :: (TensorMathFloating t, Tensor t) => t -> IO t
ceil t = flip withInplace1 t $ \r -> ceil_ r t

floor :: (TensorMathFloating t, Tensor t) => t -> IO t
floor t = flip withInplace1 t $ \r -> floor_ r t

round :: (TensorMathFloating t, Tensor t) => t -> IO t
round t = flip withInplace1 t $ \r -> round_ r t

trunc :: (TensorMathFloating t, Tensor t) => t -> IO t
trunc t = flip withInplace1 t $ \r -> trunc_ r t

frac :: (TensorMathFloating t, Tensor t) => t -> IO t
frac t = flip withInplace1 t $ \r -> frac_ r t

_lerp :: (TensorMathFloating t, Tensor t) => t -> t -> HsReal t -> IO t
_lerp t0 t1 v = flip withInplace1 t0 $ \r -> lerp_ r t0 t1 v

mean :: (TensorMathFloating t, Tensor t) => t -> Int32 -> Int32 -> IO t
mean t v0 v1 = flip withInplace1 t $ \r -> mean_ r t v0 v1

std :: (TensorMathFloating t, Tensor t) => t -> Int32 -> Int32 -> Int32 -> IO t
std t v0 v1 v2 = flip withInplace1 t $ \r -> std_ r t v0 v1 v2

var :: (TensorMathFloating t, Tensor t) => t -> Int32 -> Int32 -> Int32 -> IO t
var t v0 v1 v2 = flip withInplace1 t $ \r -> var_ r t v0 v1 v2

norm :: (TensorMathFloating t, Tensor t) => t -> HsReal t -> Int32 -> Int32 -> IO t
norm t v0 v1 v2 = flip withInplace1 t $ \r -> norm_ r t v0 v1 v2

renorm :: (TensorMathFloating t, Tensor t) => t -> HsReal t -> Int32 -> HsReal t -> IO t
renorm t v0 v1 v2 = flip withInplace1 t $ \r -> renorm_ r t v0 v1 v2
-}

