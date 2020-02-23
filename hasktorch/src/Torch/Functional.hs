{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Functional (
    module Torch.Functional,
    addmv, addr, allclose, argmin, baddbmm, bmm, acos, asin, atan, dot, einsum, lstsq, mv, slice
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

-- | mean
-- Returns the mean value of all elements in the input tensor.
mean 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
mean t = unsafePerformIO $ (cast1 ATen.mean_t) t

-- | std
-- Returns the standard deviation of all elements in the input tensor.
std 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
std t = unsafePerformIO $ (cast1 ATen.std_t) t

-- | var
-- Returns the variance of all elements in the input tensor.
var 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
var t = unsafePerformIO $ (cast1 ATen.var_t) t

-- | sumAll
-- Returns the sum of all elements in the input tensor.
sumAll 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
sumAll t = unsafePerformIO $ (cast1 ATen.sum_t) t

-- | abs
-- Computes the element-wise absolute value of the given input tensor.
abs 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
abs t = unsafePerformIO $ (cast1 ATen.abs_t) t

-- | frac
-- Computes the fractional portion of each element in input.
-- out_i = input_i - (floor . abs) input_i * (sign input_i)
frac :: Tensor -> Tensor
frac _self = unsafePerformIO $ (cast1 ATen.frac_t) _self

keepdim KeepDim = True
keepdim RemoveDim = False

-- | argmax
-- Returns the indices of the maximum value of all elements in the input tensor.
argmax :: Dim -> KeepDim -> Tensor -> Tensor
argmax (Dim d) k t = unsafePerformIO $ (cast3 ATen.argmax_tlb) t d (keepdim k)

-- | add
-- Each element of the tensor other added to each element of the tensor input. The resulting tensor is returned.
add 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
add a b = unsafePerformIO $ (cast3 ATen.add_tts) a b kOne

-- | mul
-- Multiplies each element of the input input with the scalar other and returns a new resulting tensor.
mul
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
mul a b = unsafePerformIO $ (cast2 ATen.mul_tt) a b

-- | sub
sub 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
sub a b = unsafePerformIO $ (cast3 ATen.sub_tts) a b kOne

-- | ceil
ceil 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
ceil t = unsafePerformIO $ (cast1 ATen.ceil_t) t

-- | floor
floor 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
floor t = unsafePerformIO $ (cast1 ATen.floor_t) t

-- | min
min 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
min t = unsafePerformIO $ (cast1 ATen.min_t) t

-- | max
max 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
max t = unsafePerformIO $ (cast1 ATen.max_t) t

-- | median
median 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
median t = unsafePerformIO $ (cast1 ATen.median_t) t

-- | addScalar
addScalar :: Scalar a => Tensor -> a -> Tensor
addScalar t a = unsafePerformIO $ (cast2 ATen.add_ts) t a

-- | subScalar
subScalar :: Scalar a => Tensor -> a -> Tensor
subScalar t a = unsafePerformIO $ (cast2 ATen.sub_ts) t a

-- | mulScalar
mulScalar :: Scalar a => Tensor -> a -> Tensor
mulScalar t a = unsafePerformIO $ (cast2 ATen.mul_ts) t a

-- | divScalar
divScalar :: Scalar a => Tensor -> a -> Tensor
divScalar t a = unsafePerformIO $ (cast2 ATen.div_ts) t a

-- | matmul 
-- Matrix product of two tensors.
--
-- The behavior depends on the dimensionality of the tensors as follows:
-- 
-- If both tensors are 1-dimensional, the dot product (scalar) is returned.
-- If both arguments are 2-dimensional, the matrix-matrix product is returned.
-- If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
-- If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
-- If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a batched matrix multiply is returned. If the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after. If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched matrix multiple and removed after. The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable). For example, if input is a (j \times 1 \times n \times m)(j×1×n×m) tensor and other is a (k \times m \times p)(k×m×p) tensor, out will be an (j \times k \times n \times p)(j×k×n×p) tensor.
matmul :: Tensor -> Tensor -> Tensor
matmul a b = unsafePerformIO $ (cast2 ATen.matmul_tt) a b

-- | erf
-- Computes the error function of each element
erf 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
erf t = unsafePerformIO $ (cast1 ATen.erf_t) t

-- | exp
-- Returns a new tensor with the exponential of the elements of the input tensor input.
exp 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
exp t = unsafePerformIO $ (cast1 ATen.exp_t) t

-- | log1p
-- Returns a new tensor with the natural logarithm of (1 + input).
log1p 
    :: Tensor -> Tensor
log1p t = unsafePerformIO $ (cast1 ATen.log1p_t) t

-- | log2
-- Returns a new tensor with the logarithm to the base 2 of the elements of input.
log2 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
log2 t = unsafePerformIO $ (cast1 ATen.log2_t) t

-- | log10
-- Returns a new tensor with the logarithm to the base 10 of the elements of input.
log10 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
log10 t = unsafePerformIO $ (cast1 ATen.log10_t) t

-- | pow
-- Takes the power of each element in input with exponent and returns a tensor with the result.
pow :: Scalar a 
    => Tensor -- ^ input
    -> a -- ^ exponent
    -> Tensor -- ^ output
pow t s = unsafePerformIO $ (cast2 ATen.pow_ts) t s

-- | powt
-- Takes the power of each element in input with exponent and returns a tensor with the result.
-- Exponent is a tensor with the same number of elements as input.
powt
    :: Tensor -- ^ input
    -> Tensor -- ^ exponent
    -> Tensor -- ^ output
powt t t' = unsafePerformIO $ (cast2 ATen.pow_tt) t t'

-- | relu
-- Applies the rectified linear unit function element-wise.
relu 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
relu t = unsafePerformIO $ (cast1 ATen.relu_t) t

-- | selu
-- Applies element-wise, \text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))SELU(x)=scale∗(max(0,x)+min(0,α∗(exp(x)−1))) , with \alpha=1.6732632423543772848170429916717α=1.6732632423543772848170429916717 and scale=1.0507009873554804934193349852946scale=1.0507009873554804934193349852946 .
selu 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
selu t = unsafePerformIO $ (cast1 ATen.selu_t) t

-- | celu
-- Applies element-wise, \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))CELU(x)=max(0,x)+min(0,α∗(exp(x/α)−1)) .
celu 
   :: Float -- ^ alpha
   -> Tensor -- ^ input
   -> Tensor -- ^ output
celu _alpha _self = unsafePerformIO $ (cast2 ATen.celu_ts) _self _alpha

-- | sigmoid
-- Applies the element-wise function sigmoid.
sigmoid 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
sigmoid t = unsafePerformIO $ (cast1 ATen.sigmoid_t) t

-- | softmax
-- Applies a softmax function. 
-- It is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1.
softmax 
    :: Int -- ^ dimension
    -> Tensor -- ^ input
    -> Tensor -- ^ output
softmax dim input = unsafePerformIO $ (cast3 ATen.softmax_tls) 
    input dim (dtype input)

-- | logSoftmax
-- Applies a softmax followed by a logarithm.
-- While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower, and numerically unstable. This function uses an alternative formulation to compute the output and gradient correctly.
logSoftmax 
    :: Int -- ^ dimension
    -> Tensor -- ^ input
    -> Tensor -- ^ output
logSoftmax dim input = unsafePerformIO $ (cast3 ATen.log_softmax_tls) 
    input dim (dtype input)

-- | threshold
-- Thresholds each element of the input Tensor.
threshold 
    :: Float -- ^ threshold
    -> Float -- ^ value
    -> Tensor -- ^ input
    -> Tensor -- ^ output
threshold threshold value self = 
    unsafePerformIO $ (cast3 ATen.threshold_tss) self threshold value

-- | sin
-- Returns a new tensor with the sine of the elements of input.
sin 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
sin t = unsafePerformIO $ (cast1 ATen.sin_t) t

-- | cos
-- Returns a new tensor with the cos of the elements of input.
cos 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
cos t = unsafePerformIO $ (cast1 ATen.cos_t) t

-- | tan
-- Returns a new tensor with the tangent of the elements of input.
tan 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
tan t = unsafePerformIO $ (cast1 ATen.tan_t) t

-- | sinh
-- Returns a new tensor with the hyperbolic sine of the elements of input.
sinh 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
sinh t = unsafePerformIO $ (cast1 ATen.sinh_t) t

-- | cosh
-- Returns a new tensor with the hyperbolic cosine of the elements of input.
cosh 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
cosh t = unsafePerformIO $ (cast1 ATen.cosh_t) t

-- | tanh
-- Returns a new tensor with the hyperbolic tangent of the elements of input.
tanh 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
tanh t = unsafePerformIO $ (cast1 ATen.tanh_t) t

-- | sqrt
-- Returns a new tensor with the square-root of the elements of input.
sqrt 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
sqrt t = unsafePerformIO $ (cast1 ATen.sqrt_t) t

-- | gt
-- Computes input > other element-wise.
-- The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
gt
    :: Tensor -- ^ input
    -> Tensor -- ^ output
    -> Tensor -- ^ other
gt a b = unsafePerformIO $ (cast2 ATen.gt_tt) a b

(>.) = gt

-- | lt
-- Computes input < other element-wise.
-- The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
lt 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
lt a b = unsafePerformIO $ (cast2 ATen.lt_tt) a b

(<.) = lt

-- | ge
-- Computes input >= other element-wise.
-- The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
ge 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
ge a b = unsafePerformIO $ (cast2 ATen.ge_tt) a b

(>=.) = ge

-- | le
-- Computes input <= other element-wise.
-- The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
le 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
le a b = unsafePerformIO $ (cast2 ATen.le_tt) a b

(<=.) = le

-- | eq
-- Computes input == other element-wise.
-- The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
eq 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
eq a b = unsafePerformIO $ (cast2 ATen.eq_tt) a b

(==.) = eq

-- | ne
-- Computes input /= other element-wise.
-- The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
ne 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
ne a b = unsafePerformIO $ (cast2 ATen.ne_tt) a b

(/=.) = ne

-- | toDType
toDType :: DType -> Tensor -> Tensor
toDType dtype t = unsafePerformIO $ (cast4 ATen.tensor_to_sbb) t dtype False False

-- | squeezeAll
squeezeAll 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
squeezeAll t = unsafePerformIO $ (cast1 ATen.squeeze_t) t

-- | binary_cross_entropy_loss
binary_cross_entropy_loss :: Tensor -> Tensor -> Tensor -> Reduction-> Tensor
binary_cross_entropy_loss t target weight reduction = unsafePerformIO $ (cast4 ATen.binary_cross_entropy_tttl) t target weight reduction

-- | BCE with weights defaulted to 1.0 & reduction defaulted to ReduceMean
binary_cross_entropy_loss' :: Tensor -> Tensor -> Tensor
binary_cross_entropy_loss' t target = unsafePerformIO $ (cast4 ATen.binary_cross_entropy_tttl) t target (onesLike target) ReduceMean

-- | mse_loss
mseLoss :: Tensor -> Tensor -> Tensor
mseLoss a b = unsafePerformIO $ (cast3 ATen.mse_loss_ttl) a b ATen.kMean

-- | nllLoss
nllLoss' :: Tensor -> Tensor -> Tensor
nllLoss' t target = unsafePerformIO $ (cast5 ATen.nll_loss_tttll) t target weight ReduceMean (-100 :: Int)
    where
        nClass = (shape t) !! 1 -- TODO nicer runtime error if input dimensions don't conform
        weight = ones' [nClass]

-- | adaptiveMaxPool1d
-- Applies a 1D adaptive max pooling over an input signal composed of several input planes.
adaptiveMaxPool1d 
    :: Int -- ^ output size
    -> Tensor -- ^ input
    -> (Tensor,Tensor) -- ^ 
adaptiveMaxPool1d outputSize self =
    unsafePerformIO $ (cast2 ATen.adaptive_max_pool1d_tl)
        self outputSize

-- | adaptiveMaxPool2d
-- Applies a 2D adaptive max pooling over an input signal composed of several input planes.
adaptiveMaxPool2d 
    :: (Int,Int) 
    -> Tensor 
    -> (Tensor,Tensor)
adaptiveMaxPool2d _output_size _self =
    unsafePerformIO $ (cast2 ATen.adaptive_max_pool2d_tl)
        _self _output_size

-- | maxPool1dWithIndices
maxPool1dWithIndices 
    :: Int -- ^ kernel size
    -> Int -- ^ stride
    -> Int -- ^ padding
    -> Int -- ^ dilation
    -> Bool -- ^ ceil mode
    -> Tensor -- ^ input
    -> (Tensor,Tensor) -- ^ output, indices
maxPool1dWithIndices kernelSize stride padding dilation ceilMode self =
    unsafePerformIO $ (cast6 ATen.max_pool1d_with_indices_tllllb)
        self kernelSize stride padding dilation ceilMode

-- | maxPool1d
-- Applies a 1D max pooling over an input signal composed of several input planes.
maxPool1d 
    :: Int -- ^ kernel size
    -> Int -- ^ stride
    -> Int -- ^ padding
    -> Int -- ^ dilation
    -> Bool -- ^ ceil mode
    -> Tensor -- ^ input
    -> Tensor -- ^ output
maxPool1d kernelSize stride padding dilation ceilMode self =
    unsafePerformIO $ (cast6 ATen.max_pool1d_tllllb)
        self kernelSize stride padding dilation ceilMode

-- | maxPool2d
-- Applies a 2D max pooling over an input signal composed of several input planes.
maxPool2d 
    :: (Int,Int) -- ^ kernel size
    -> (Int,Int) -- ^ stride
    -> (Int,Int) -- ^ padding
    -> (Int,Int) -- ^ dilation
    -> Bool -- ^ ceil mode
    -> Tensor -- ^ input
    -> Tensor -- ^ output
maxPool2d kernelSize stride padding dilation ceilMode self =
    unsafePerformIO $ (cast6 ATen.max_pool2d_tllllb)
        self kernelSize stride padding dilation ceilMode

-- | maxPool3d 
-- Applies a 3D max pooling over an input signal composed of several input planes.
maxPool3d 
    :: (Int,Int,Int) -- ^ kernel size
    -> (Int,Int,Int) -- ^ stride
    -> (Int,Int,Int) -- ^ padding
    -> (Int,Int,Int) -- ^ dilation
    -> Bool -- ^ ceil mode
    -> Tensor -- ^ input
    -> Tensor -- ^ output
maxPool3d kernelSize stride padding dilation ceilMode self =
    unsafePerformIO $ (cast6 ATen.max_pool3d_tllllb)
        self kernelSize stride padding dilation ceilMode

inverse 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
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

choleskySolve :: Tensor -> Tensor -> Tri -> Tensor
choleskySolve t1 t2 upper = unsafePerformIO $ (cast3 ATen.cholesky_solve_ttb) t1 t2 boolUpper
  where boolUpper = isUpper upper

-- | dropout
-- During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
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

-- | alphaDropout
-- Applies alpha dropout to the input.
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

-- | avgPool1d
-- Applies a 1D average pooling over an input signal composed of several input planes.
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
avgPool1d' kernelSize stride padding input =
    avgPool1d kernelSize stride padding False True input

-- | adaptiveAvgPool1d
-- Applies a 1D adaptive average pooling over an input signal composed of several input planes.
adaptiveAvgPool1d
  :: Int -- outputSize
  -> Tensor -- ^ input
  -> Tensor -- ^ output
adaptiveAvgPool1d outputSize input = unsafePerformIO
  $ cast2 ATen.adaptive_avg_pool1d_tl input outputSize

-- | adaptiveAvgPool2d
-- Applies a 2D adaptive average pooling over an input signal composed of several input planes.
adaptive_avg_pool2d :: Tensor -> (Int,Int) -> Tensor
adaptive_avg_pool2d _self _output_size = unsafePerformIO $ (cast2 ATen.adaptive_avg_pool2d_tl) _self _output_size

-- | adaptiveAvgPool3d
-- Applies a 3D adaptive average pooling over an input signal composed of several input planes.
adaptive_avg_pool3d :: Tensor -> (Int,Int,Int) -> Tensor
adaptive_avg_pool3d _self _output_size = unsafePerformIO $ (cast2 ATen.adaptive_avg_pool3d_tl) _self _output_size

bitwiseNot 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
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

-- | conv1d
-- Applies a 1D convolution over an input signal composed of several input planes.
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

-- | conv2d
-- Applies a 2D convolution over an input signal composed of several input planes.
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

solve 
    :: Tensor 
    -> Tensor 
    -> (Tensor,Tensor)
solve b a = unsafePerformIO $ (cast2 ATen.solve_tt) b a

-- | Solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix uu .
cholesky_inverse
    :: Tensor -- ^ input
    -> Tri -- ^ upper or lower triangle
    -> Tensor -- ^ solution
cholesky_inverse t upper = unsafePerformIO $ (cast2 ATen.cholesky_inverse_tb) t boolUpper
  where boolUpper = isUpper upper

-- pstrf :: Tensor -> Bool -> Double -> (Tensor, Tensor)
-- pstrf t upper tol = unsafePerformIO $ (cast3 ATen.pstrf_tbs) t upper tol

--qr :: Tensor -> (Tensor, Tensor)
--qr t = unsafePerformIO $ (cast1 ATen.qr_t) t

-- | This is a low-level function for calling LAPACK directly. This function returns a namedtuple (a, tau) as defined in LAPACK documentation for geqrf.
geqrf 
    :: Tensor -- ^ input
    -> (Tensor, Tensor) -- ^ a, tau output matrices (see https://software.intel.com/en-us/node/521004)
geqrf t = unsafePerformIO $ (cast1 ATen.geqrf_t) t


-- | Computes the orthogonal matrix Q of a QR factorization, from the (input, input2) tuple returned by torch.geqrf().
-- This directly calls the underlying LAPACK function ?orgqr. See LAPACK documentation for orgqr for further details.
orgqr :: Tensor -> Tensor -> Tensor
orgqr b a = unsafePerformIO $ (cast2 ATen.orgqr_tt) b a

sign 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
sign t = unsafePerformIO $ (cast1 ATen.sign_t) t

transpose :: Tensor -> Int -> Int -> Tensor
transpose t a b = unsafePerformIO $ (cast3 ATen.transpose_tll) t a b

-- transpose special case for a 2D tensor
transpose2D 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
transpose2D t = transpose t 0 1

-- | diag
-- | Returns a tensor with the elements of input as the diagonal.
-- | The second argument controls which diagonal to consider:
--        If Int = 0, it is the main diagonal.
--        If Int > 0, it is above the main diagonal.
--        If Int < 0, it is below the main diagonal.

diag
    ::  Tensor -- ^ input
    ->  Int -- ^ diagonal
    ->  Tensor -- ^ output
diag t index = unsafePerformIO $ (cast2 ATen.tensor_diag_l) t index

-- | all
-- Returns True if all elements in the tensor are True, False otherwise.
all
 :: Tensor -- ^ input
 -> Bool -- ^ output
all t = toInt (unsafePerformIO $ (cast1 ATen.all_t) t) == 1

-- | any
-- Returns True if any elements in the tensor are True, False otherwise.
any
 :: Tensor -- ^ input
 -> Bool -- ^ output
any t = toInt (unsafePerformIO $ (cast1 ATen.any_t) t) == 1

-- | all'
-- Returns True if all elements in each row of the tensor in the given dimension dim are True, False otherwise.
-- If keepdim is True, the output tensor is of the same size as input except in the dimension dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 fewer dimension than input.  
all' 
 :: Tensor -- ^ input
 -> Int -- ^ dimension
 -> Bool -- ^ boolean corresponding to keepdim
 -> Tensor -- ^ output
all' t dim keepdim = unsafePerformIO $ (cast3 ATen.all_tlb) t dim keepdim

-- | any'
-- Returns True if any elements in each row of the tensor in the given dimension dim are True, False otherwise.
-- If keepdim is True, the output tensor is of the same size as input except in the dimension dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 fewer dimension than input.
any' 
 :: Tensor -- ^ input 
 -> Int -- ^ dimension 
 -> Bool -- ^ boolean corresponding to keepdim
 -> Tensor -- output
any' t dim keepdim = unsafePerformIO $ (cast3 ATen.any_tlb) t dim keepdim

-- | permute
-- Permute the dimensions of this tensor.
permute 
 :: Tensor -- ^ input
 -> [Int] -- ^ list corresponding to ordering of dimensions to permute with 
 -> Tensor -- output
permute t dims = unsafePerformIO $ (cast2 ATen.tensor_permute_l) t dims
