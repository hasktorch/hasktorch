{-# LANGUAGE TypeApplications #-}

module Torch.Functions (
    module Torch.Functions
  , module Torch.Functions.Native
) where

import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified ATen.Managed.Native as ATen
import qualified ATen.Managed.Type.Tensor as ATen
import qualified ATen.Managed.Type.Scalar as ATen
import qualified ATen.Managed.Type.Tuple as ATen
import qualified ATen.Const as ATen
import qualified ATen.Type as ATen
import qualified ATen.Managed.Cast
import ATen.Cast

import Torch.Scalar
import Torch.Tensor
import Torch.DType
import Torch.Functions.Native

kOne :: ForeignPtr ATen.Scalar
kOne = unsafePerformIO $ ATen.newScalar_i 1

instance Num Tensor where
  (+) = add
  (-) = sub
  (*) = mul
  negate t = unsafePerformIO $ (cast1 ATen.neg_t) t
  abs t = unsafePerformIO $ (cast1 ATen.abs_t) t
  signum t = unsafePerformIO $ (cast1 ATen.sign_t) t
  fromInteger i = asTensor @Int $ fromInteger @Int i

instance Fractional Tensor where
  a / b = unsafePerformIO $ (cast2 ATen.div_tt) a b
  recip t = unsafePerformIO $ (cast1 ATen.reciprocal_t) t
  fromRational i = asTensor @Float $ fromRational @Float i

data Tri = Upper | Lower

isUpper Upper = True
isUpper Lower = False

sumAll :: Tensor -> Tensor
sumAll t = unsafePerformIO $ (cast1 ATen.sum_t) t

abs :: Tensor -> Tensor
abs t = unsafePerformIO $ (cast1 ATen.abs_t) t

add :: Tensor -> Tensor -> Tensor
add a b = unsafePerformIO $ (cast3 ATen.add_tts) a b kOne

ceil :: Tensor -> Tensor
ceil t = unsafePerformIO $ (cast1 ATen.ceil_t) t

floor :: Tensor -> Tensor
floor t = unsafePerformIO $ (cast1 ATen.floor_t) t

min :: Tensor -> Tensor
min t = unsafePerformIO $ (cast1 ATen.min_t) t

max :: Tensor -> Tensor
max t = unsafePerformIO $ (cast1 ATen.max_t) t

median :: Tensor -> Tensor
median t = unsafePerformIO $ (cast1 ATen.median_t) t

sub :: Tensor -> Tensor -> Tensor
sub a b = unsafePerformIO $ (cast3 ATen.sub_tts) a b kOne

mul :: Tensor -> Tensor -> Tensor
mul a b = unsafePerformIO $ (cast2 ATen.mul_tt) a b

cmul :: Scalar a => Tensor -> a -> Tensor
cmul t a = unsafePerformIO $ (cast2 ATen.mul_ts) t a

matmul :: Tensor -> Tensor -> Tensor
matmul a b =
    unsafePerformIO $ case (dim a, dim b) of
      (2, 2) -> mm a b
      _ -> error "Unsupported case in matmul!"
  where
    mm = cast2 ATen.mm_tt

erf :: Tensor -> Tensor
erf t = unsafePerformIO $ (cast1 ATen.erf_t) t

exp :: Tensor -> Tensor
exp t = unsafePerformIO $ (cast1 ATen.exp_t) t

log1p :: Tensor -> Tensor
log1p t = unsafePerformIO $ (cast1 ATen.log1p_t) t

log2 :: Tensor -> Tensor
log2 t = unsafePerformIO $ (cast1 ATen.log2_t) t

log10 :: Tensor -> Tensor
log10 t = unsafePerformIO $ (cast1 ATen.log10_t) t

pow :: Scalar a => Tensor -> a -> Tensor
pow t s = unsafePerformIO $ (cast2 ATen.pow_ts) t s

relu :: Tensor -> Tensor
relu t = unsafePerformIO $ (cast1 ATen.relu_t) t

selu :: Tensor -> Tensor
selu t = unsafePerformIO $ (cast1 ATen.selu_t) t

sigmoid :: Tensor -> Tensor
sigmoid t = unsafePerformIO $ (cast1 ATen.sigmoid_t) t

sin :: Tensor -> Tensor
sin t = unsafePerformIO $ (cast1 ATen.sin_t) t

sinh :: Tensor -> Tensor
sinh t = unsafePerformIO $ (cast1 ATen.sinh_t) t

cos :: Tensor -> Tensor
cos t = unsafePerformIO $ (cast1 ATen.cos_t) t

sqrt :: Tensor -> Tensor
sqrt t = unsafePerformIO $ (cast1 ATen.sqrt_t) t

tanh :: Tensor -> Tensor
tanh t = unsafePerformIO $ (cast1 ATen.tanh_t) t

gt :: Tensor -> Tensor -> Tensor
gt a b = unsafePerformIO $ (cast2 ATen.gt_tt) a b

(>.) = gt

lt :: Tensor -> Tensor -> Tensor
lt a b = unsafePerformIO $ (cast2 ATen.lt_tt) a b

(<.) = lt

ge :: Tensor -> Tensor -> Tensor
ge a b = unsafePerformIO $ (cast2 ATen.ge_tt) a b

(>=.) = ge

le :: Tensor -> Tensor -> Tensor
le a b = unsafePerformIO $ (cast2 ATen.le_tt) a b

(<=.) = le

eq :: Tensor -> Tensor -> Tensor
eq a b = unsafePerformIO $ (cast2 ATen.eq_tt) a b

(==.) = eq

ne :: Tensor -> Tensor -> Tensor
ne a b = unsafePerformIO $ (cast2 ATen.ne_tt) a b

(/=.) = ne

toDType :: DType -> Tensor -> Tensor
toDType dtype t = unsafePerformIO $ (cast4 ATen.tensor_to_sbb) t dtype False False

squeezeAll :: Tensor -> Tensor
squeezeAll t = unsafePerformIO $ (cast1 ATen.squeeze_t) t

mse_loss :: Tensor -> Tensor -> Tensor
mse_loss a b = unsafePerformIO $ (cast3 ATen.mse_loss_ttl) a b ATen.kMean

conv2d :: Tensor -> Tensor -> Tensor -> (Int, Int) -> (Int, Int) -> Tensor
conv2d input weight bias (dh, dw) (ph, pw) = unsafePerformIO $
    (cast7 ATen.conv2d_tttllll) input weight bias
                                [dh, dw] [ph, pw] ([1, 1] :: [Int]) (0 :: Int)

maxPool2d :: Tensor -> (Int, Int) -> (Int, Int) -> (Int, Int) -> Tensor
maxPool2d input (kh, kw) (dh, dw) (ph, pw) = unsafePerformIO $
    (cast6 ATen.max_pool2d_tllllb) input [kh, kw] [dh, dw] [ph, pw] ([1, 1] :: [Int]) False

logSoftmax :: Tensor -> Int -> Tensor
logSoftmax input dim = unsafePerformIO $ (cast3 ATen.log_softmax_tls) input dim (dtype input)

inverse :: Tensor -> Tensor
inverse t = unsafePerformIO $ (cast1 ATen.inverse_t) t

gels :: Tensor -> Tensor -> (Tensor, Tensor)
gels _B _A = unsafePerformIO $ (cast2 ATen.gels_tt) _B _A

symeig :: Tensor -> Bool -> Tri -> (Tensor, Tensor)
symeig t eigenvectors upper = unsafePerformIO $ (cast3 ATen.symeig_tbb) t eigenvectors boolUpper
  where boolUpper = isUpper upper


eig :: Tensor -> Bool -> (Tensor, Tensor)
eig t eigenvectors = unsafePerformIO $ (cast2 ATen.eig_tb) t eigenvectors

svd :: Tensor -> Bool -> Bool -> (Tensor, Tensor, Tensor)
svd t some compute_uv = unsafePerformIO $ (cast3 ATen.svd_tbb) t some compute_uv

cholesky :: Tensor -> Tri -> Tensor
cholesky t upper = unsafePerformIO $ (cast2 ATen.cholesky_tb) t boolUpper
  where boolUpper = isUpper upper

cholesky_solve :: Tensor -> Tensor -> Tri -> Tensor
cholesky_solve t1 t2 upper = unsafePerformIO $ (cast3 ATen.cholesky_solve_ttb) t1 t2 boolUpper
  where boolUpper = isUpper upper

solve :: Tensor -> Tensor -> (Tensor,Tensor)
solve b a = unsafePerformIO $ (cast2 ATen.solve_tt) b a 

cholesky_inverse :: Tensor -> Tri -> Tensor
cholesky_inverse t upper = unsafePerformIO $ (cast2 ATen.cholesky_inverse_tb) t boolUpper
  where boolUpper = isUpper upper

-- pstrf :: Tensor -> Bool -> Double -> (Tensor, Tensor)
-- pstrf t upper tol = unsafePerformIO $ (cast3 ATen.pstrf_tbs) t upper tol

qr :: Tensor -> (Tensor, Tensor)
qr t = unsafePerformIO $ (cast1 ATen.qr_t) t

geqrf :: Tensor -> (Tensor, Tensor)
geqrf t = unsafePerformIO $ (cast1 ATen.geqrf_t) t 

orgqr :: Tensor -> Tensor -> Tensor
orgqr b a = unsafePerformIO $ (cast2 ATen.orgqr_tt) b a 

sign :: Tensor -> Tensor
sign t = unsafePerformIO $ (cast1 ATen.sign_t) t

transpose :: Tensor -> Int -> Int -> Tensor
transpose t a b = unsafePerformIO $ (cast3 ATen.transpose_tll) t a b

-- transpose special case for a 2D tensor
transpose2D :: Tensor -> Tensor
transpose2D t = transpose t 0 1

diag :: Tensor -> Int -> Tensor
diag t index = unsafePerformIO $ (cast2 ATen.tensor_diag_l) t index

all :: Tensor -> Bool
all t = toInt (unsafePerformIO $ (cast1 ATen.all_t) t) == 1

any :: Tensor -> Bool
any t = toInt (unsafePerformIO $ (cast1 ATen.any_t) t) == 1

all' :: Tensor -> Int -> Bool -> Tensor
all' t dim keepdim = unsafePerformIO $ (cast3 ATen.all_tlb) t dim keepdim

any' :: Tensor -> Int -> Bool -> Tensor
any' t dim keepdim = unsafePerformIO $ (cast3 ATen.any_tlb) t dim keepdim
