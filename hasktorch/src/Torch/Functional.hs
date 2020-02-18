{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Functional (
    module Torch.Functional,
    addmv, addr, allclose, argmin, baddbmm, bmm, acos, asin, atan, dot, lstsq, mv, slice
    , sumWithDimnames
) where

import          Prelude                 hiding ( all
                                                , any
                                                , sin
                                                , sinh
                                                , cos
                                                , cosh
                                                , tan
                                                , tanh
                                                , asin
                                                , asinh
                                                , acos
                                                , acosh
                                                , atan
                                                , atanh
                                                , max
                                                , min
                                                , exp
                                                , log
                                                , round
                                                , isNaN
                                                , floor
                                                , ceil
                                                )

import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Type.Scalar as ATen
import qualified Torch.Internal.Managed.Type.Tuple as ATen
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.Internal.Managed.Cast
import qualified Torch.Functional.Internal as Internal
import Torch.Internal.Cast
import Torch.Internal.Class
import Data.Int

import Torch.Scalar
import Torch.Tensor
import Torch.DType
import Torch.Functional.Internal hiding (argmax, clamp, cosh, conv1d, linear, softmax)
import Torch.TensorFactories (onesLike, ones')

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

-- Return upper or lower triangular matrices
data Tri = Upper | Lower deriving (Eq, Show)

-- Reductions, used by BCE loss, see -
-- https://github.com/pytorch/pytorch/blob/3762cf9cc63e2032410d50f218c1406668177c23/aten/src/ATen/core/Reduction.h
data Reduction = ReduceNone | ReduceMean | ReduceSum deriving (Eq, Show)

data Dim = Dim Int

data KeepDim = KeepDim | RemoveDim deriving (Eq, Show)

instance Castable Reduction Int64 where
  cast ReduceNone f = f 0
  cast ReduceMean f = f 1
  cast ReduceSum f = f 2
  uncast 0 f = f ReduceNone
  uncast 1 f = f ReduceMean
  uncast _ f = f ReduceSum

isUpper Upper = True
isUpper Lower = False

mean :: Tensor -> Tensor
mean t = unsafePerformIO $ (cast1 ATen.mean_t) t

std :: Tensor -> Tensor
std t = unsafePerformIO $ (cast1 ATen.std_t) t

var :: Tensor -> Tensor
var t = unsafePerformIO $ (cast1 ATen.var_t) t

sumAll :: Tensor -> Tensor
sumAll t = unsafePerformIO $ (cast1 ATen.sum_t) t

abs :: Tensor -> Tensor
abs t = unsafePerformIO $ (cast1 ATen.abs_t) t

keepdim KeepDim = True
keepdim RemoveDim = False

-- deprecates Native version
argmax :: Dim -> KeepDim -> Tensor -> Tensor
argmax (Dim d) k t = unsafePerformIO $ (cast3 ATen.argmax_tlb) t d (keepdim k)

add :: Tensor -> Tensor -> Tensor
add a b = unsafePerformIO $ (cast3 ATen.add_tts) a b kOne

mul :: Tensor -> Tensor -> Tensor
mul a b = unsafePerformIO $ (cast2 ATen.mul_tt) a b

sub :: Tensor -> Tensor -> Tensor
sub a b = unsafePerformIO $ (cast3 ATen.sub_tts) a b kOne

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

addScalar :: Scalar a => Tensor -> a -> Tensor
addScalar t a = unsafePerformIO $ (cast2 ATen.add_ts) t a

subScalar :: Scalar a => Tensor -> a -> Tensor
subScalar t a = unsafePerformIO $ (cast2 ATen.sub_ts) t a

mulScalar :: Scalar a => Tensor -> a -> Tensor
mulScalar t a = unsafePerformIO $ (cast2 ATen.mul_ts) t a

divScalar :: Scalar a => Tensor -> a -> Tensor
divScalar t a = unsafePerformIO $ (cast2 ATen.div_ts) t a

matmul :: Tensor -> Tensor -> Tensor
matmul a b = unsafePerformIO $ (cast2 ATen.matmul_tt) a b

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

cos :: Tensor -> Tensor
cos t = unsafePerformIO $ (cast1 ATen.cos_t) t

sinh :: Tensor -> Tensor
sinh t = unsafePerformIO $ (cast1 ATen.sinh_t) t

cosh :: Tensor -> Tensor
cosh t = unsafePerformIO $ (cast1 ATen.cosh_t) t

tanh :: Tensor -> Tensor
tanh t = unsafePerformIO $ (cast1 ATen.tanh_t) t

sqrt :: Tensor -> Tensor
sqrt t = unsafePerformIO $ (cast1 ATen.sqrt_t) t

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

binary_cross_entropy_loss :: Tensor -> Tensor -> Tensor -> Reduction-> Tensor
binary_cross_entropy_loss t target weight reduction = unsafePerformIO $ (cast4 ATen.binary_cross_entropy_tttl) t target weight reduction

-- | BCE with weights defaulted to 1.0 & reduction defaulted to ReduceMean
binary_cross_entropy_loss' :: Tensor -> Tensor -> Tensor
binary_cross_entropy_loss' t target = unsafePerformIO $ (cast4 ATen.binary_cross_entropy_tttl) t target (onesLike target) ReduceMean

mse_loss :: Tensor -> Tensor -> Tensor
mse_loss a b = unsafePerformIO $ (cast3 ATen.mse_loss_ttl) a b ATen.kMean

nllLoss' :: Tensor -> Tensor -> Tensor
nllLoss' t target = unsafePerformIO $ (cast5 ATen.nll_loss_tttll) t target weight ReduceMean (-100 :: Int)
    where
        nClass = (shape t) !! 1 -- TODO nicer runtime error if input dimensions don't conform
        weight = ones' [nClass]

maxPool2d :: Tensor -> (Int, Int) -> (Int, Int) -> (Int, Int) -> Tensor
maxPool2d input (kh, kw) (dh, dw) (ph, pw) = unsafePerformIO $
    (cast6 ATen.max_pool2d_tllllb) input ([kh, kw] :: [Int]) ([dh, dw] :: [Int]) ([ph, pw] :: [Int]) ([1, 1] :: [Int]) False

softmax :: Int -> Tensor -> Tensor
softmax dim input = unsafePerformIO $ (cast3 ATen.softmax_tls) input dim (dtype input)

logSoftmax :: Int -> Tensor -> Tensor
logSoftmax dim input = unsafePerformIO $ (cast3 ATen.log_softmax_tls) input dim (dtype input)

inverse :: Tensor -> Tensor
inverse t = unsafePerformIO $ (cast1 ATen.inverse_t) t

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

dropout
  :: Double -- ^ dropout probability
  -> Bool -- ^ whether or not to activate dropout
  -> Tensor -- ^ input
  -> IO Tensor -- ^ output
dropout p train input = cast3 ATen.dropout_tdb input p train

featureDropout
  :: Double -- ^ dropout probability
  -> Bool -- ^ whether or not to activate dropout
  -> Tensor -- ^ input
  -> IO Tensor -- ^ output
featureDropout p train input =
  cast3 ATen.feature_dropout_tdb input p train

alphaDropout
  :: Double -- ^ dropout probability
  -> Bool -- ^ whether or not to activate dropout
  -> Tensor -- ^ input
  -> IO Tensor -- ^ output
alphaDropout p train input =
  cast3 ATen.alpha_dropout_tdb input p train

featureAlphaDropout
  :: Double -- ^ dropout probability
  -> Bool -- ^ whether or not to activate dropout
  -> Tensor -- ^ input
  -> IO Tensor -- ^ output
featureAlphaDropout p train input =
  cast3 ATen.feature_alpha_dropout_tdb input p train

avgPool1d
  :: Int -- ^ kernel size
  -> Int -- ^ stride
  -> Int -- ^ padding
  -> Bool -- ^ ceil mode
  -> Bool -- ^ count include pad
  -> Tensor -- ^ input
  -> Tensor -- ^ output
avgPool1d kernelSize stride padding ceilMode countIncludePad input =
    unsafePerformIO $ cast6 ATen.avg_pool1d_tlllbb
        input
        kernelSize
        stride
        padding
        ceilMode
        countIncludePad

avgPool1d'
  :: Int -- ^ kernel size
  -> Int -- ^ stride
  -> Int -- ^ padding
  -> Tensor -- ^ input
  -> Tensor -- ^ output
avgPool1d' kernelSize stride padding input = avgPool1d kernelSize stride padding False True input

adaptiveAvgPool1d
  :: Int -- outputSize
  -> Tensor -- ^ input
  -> Tensor -- ^ output
adaptiveAvgPool1d outputSize input = unsafePerformIO
  $ cast2 ATen.adaptive_avg_pool1d_tl input outputSize

bitwiseNot :: Tensor -> Tensor
bitwiseNot input = unsafePerformIO $ cast1 ATen.bitwise_not_t input

cat
  :: Int -> [Tensor] -> Tensor
cat dim tensors = unsafePerformIO $ cast2 ATen.cat_ll tensors dim

chunk
  :: Int -- ^ chunks
  -> Int -- ^ dim
  -> Tensor -- ^ input tensor
  -> [Tensor] -- ^ output list of tensors
chunk chunks dim input = unsafePerformIO
  $ cast3 ATen.chunk_tll input chunks dim

clamp
  :: Float -- ^ minimum value
  -> Float -- ^ maximum value
  -> Tensor -- ^ input
  -> Tensor -- ^ output
clamp min max input = unsafePerformIO $ cast3 ATen.clamp_tss input min max

clampMax
  :: Float -- ^ maximum value
  -> Tensor -- ^ input
  -> Tensor -- ^ output
clampMax max input = unsafePerformIO $ cast2 ATen.clamp_max_ts input max

clampMin
  :: Float -- ^ minimum value
  -> Tensor -- ^ input
  -> Tensor -- ^ output
clampMin min input = unsafePerformIO $ cast2 ATen.clamp_min_ts input min

cudnnIsAcceptable
  :: Tensor -- ^ input 
  -> Bool -- ^ output
cudnnIsAcceptable input =
  unsafePerformIO $ cast1 ATen.cudnn_is_acceptable_t input

constantPadNd1d
  :: [Int] -- ^ list of padding per dimension
  -> Float -- ^ value
  -> Tensor -- ^ input
  -> Tensor -- ^ ouptut
constantPadNd1d padding value input = unsafePerformIO $ cast3
  ATen.constant_pad_nd_tls
  input
  padding
  value

conv1d
  :: Tensor -- ^ weight
  -> Tensor -- ^ bias
  -> Int -- ^ stride
  -> Int -- ^ padding
  -> Int -- ^ dilation
  -> Int -- ^ groups
  -> Tensor -- ^ input
  -> Tensor -- ^ output
conv1d weight bias stride padding dilation groups input =
    unsafePerformIO $ (cast7 ATen.conv1d_tttllll)
        input
        weight
        bias
        stride
        padding
        dilation
        groups

conv1d' weight bias stride padding input = conv1d weight bias stride padding 1 1 input


{-
conv2d :: Tensor -> Tensor -> Tensor -> (Int, Int) -> (Int, Int) -> Tensor
conv2d input weight bias (dh, dw) (ph, pw) = unsafePerformIO $
    (cast7 ATen.conv2d_tttllll) input weight bias
                                ([dh, dw] :: [Int]) ([ph, pw] :: [Int]) ([1, 1] :: [Int]) (1 :: Int)
-}

conv2d
  :: Tensor -- ^ weight
  -> Tensor -- ^ bias
  -> (Int, Int) -- ^ strides
  -> (Int, Int) -- ^ padding
  -> (Int, Int) -- ^ dilation
  -> Int -- ^ groups
  -> Tensor -- ^ input
  -> Tensor -- ^ output
conv2d weight bias (stride0, stride1) (padding0, padding1) (dilation0, dilation1) groups input =
    unsafePerformIO $ (cast7 ATen.conv2d_tttllll)
        input
        weight
        bias
        ([stride0, stride1] :: [Int])
        ([padding0, padding1] :: [Int])
        ([dilation0, dilation1] :: [Int])
        groups

conv2d'
  :: Tensor -- ^ weight
  -> Tensor -- ^ bias
  -> (Int, Int) -- ^ strides
  -> (Int, Int) -- ^ padding
  -> Tensor -- ^ input
  -> Tensor -- ^ output
conv2d' weight bias stride padding input = 
    conv2d weight bias stride padding
        (1, 1) -- dilation
        (1 :: Int) -- groups
        input

solve :: Tensor -> Tensor -> (Tensor,Tensor)
solve b a = unsafePerformIO $ (cast2 ATen.solve_tt) b a

cholesky_inverse :: Tensor -> Tri -> Tensor
cholesky_inverse t upper = unsafePerformIO $ (cast2 ATen.cholesky_inverse_tb) t boolUpper
  where boolUpper = isUpper upper

-- pstrf :: Tensor -> Bool -> Double -> (Tensor, Tensor)
-- pstrf t upper tol = unsafePerformIO $ (cast3 ATen.pstrf_tbs) t upper tol

--qr :: Tensor -> (Tensor, Tensor)
--qr t = unsafePerformIO $ (cast1 ATen.qr_t) t

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

permute :: Tensor -> [Int] -> Tensor
permute t dims = unsafePerformIO $ (cast2 ATen.tensor_permute_l) t dims
