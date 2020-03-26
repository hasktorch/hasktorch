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

-- | Returns the mean value of all elements in the input tensor.
mean 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
mean t = unsafePerformIO $ (cast1 ATen.mean_t) t


-- | Returns the standard deviation of all elements in the input tensor.
std 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
std t = unsafePerformIO $ (cast1 ATen.std_t) t

-- | Returns the variance of all elements in the input tensor.
var 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
var t = unsafePerformIO $ (cast1 ATen.var_t) t

-- | Returns the sum of all elements in the input tensor.
sumAll 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
sumAll t = unsafePerformIO $ (cast1 ATen.sum_t) t

-- | Computes the element-wise absolute value of the given input tensor.
abs 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
abs t = unsafePerformIO $ (cast1 ATen.abs_t) t

-- | Computes the fractional portion of each element in input.
-- out_i = input_i - (floor . abs) input_i * (sign input_i)
frac 
 :: Tensor -- ^ input
 -> Tensor -- ^ output
frac _self = unsafePerformIO $ (cast1 ATen.frac_t) _self

keepdim KeepDim = True
keepdim RemoveDim = False

-- | Returns the indices of the maximum value of all elements in the input tensor.
argmax 
 :: Dim -- ^ the dimension to reduce
 -> KeepDim -- ^ whether the output tensor has dim retained or not
 -> Tensor -- ^ input
 -> Tensor -- ^ output
argmax (Dim d) k t = unsafePerformIO $ (cast3 ATen.argmax_tlb) t d (keepdim k)

-- | Each element of the tensor other added to each element of the tensor input. The resulting tensor is returned.
add 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
add a b = unsafePerformIO $ (cast3 ATen.add_tts) a b kOne

-- | Multiplies each element of the tensor other to each element of the input tensor and returns a new resulting tensor.
mul
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
mul a b = unsafePerformIO $ (cast2 ATen.mul_tt) a b

-- | Element wise subtraction of other tensor from input tensor and returns a new resulting tensor
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

-- | Adds each element of the input input with the scalar and returns a new resulting tensor.
addScalar 
 :: Scalar a 
 => Tensor -- ^ input
 -> a -- ^ summand
 -> Tensor -- ^ output
addScalar t a = unsafePerformIO $ (cast2 ATen.add_ts) t a

-- | Subtracts each element of the input input with the scalar and returns a new resulting tensor.
subScalar 
 :: Scalar a 
 => Tensor -- ^ input
 -> a -- ^ subtrahend
 -> Tensor -- ^ output
subScalar t a = unsafePerformIO $ (cast2 ATen.sub_ts) t a

-- | Multiplies each element of the input input with the scalar and returns a new resulting tensor.
mulScalar 
 :: Scalar a 
 => Tensor -- ^ input
 -> a -- ^ multiplier
 -> Tensor -- ^ output
mulScalar t a = unsafePerformIO $ (cast2 ATen.mul_ts) t a

-- | Divides each element of the input input with the scalar and returns a new resulting tensor.
divScalar 
 :: Scalar a 
 => Tensor -- ^ input
 -> a -- ^ divisor
 -> Tensor -- ^ output
divScalar t a = unsafePerformIO $ (cast2 ATen.div_ts) t a

-- |  Matrix product of two tensors.
--
-- The behavior depends on the dimensionality of the tensors as follows:
-- 
-- If both tensors are 1-dimensional, the dot product (scalar) is returned.
-- If both arguments are 2-dimensional, the matrix-matrix product is returned.
-- If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
-- If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
-- If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a batched matrix multiply is returned. If the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after. If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched matrix multiple and removed after. The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable). For example, if input is a (j \times 1 \times n \times m)(j×1×n×m) tensor and other is a (k \times m \times p)(k×m×p) tensor, out will be an (j \times k \times n \times p)(j×k×n×p) tensor.
matmul 
 :: Tensor -- ^ first tensor for matrix multiplication 
 -> Tensor -- ^ second tensor for matrix multiplication
 -> Tensor -- ^ output
matmul a b = unsafePerformIO $ (cast2 ATen.matmul_tt) a b

-- | Computes the error function of each element
erf 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
erf t = unsafePerformIO $ (cast1 ATen.erf_t) t

-- | Returns a new tensor with the exponential of the elements of the input tensor input.
exp 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
exp t = unsafePerformIO $ (cast1 ATen.exp_t) t

-- | Returns a new tensor with the natural logarithm of (1 + input).
log1p 
    :: Tensor -> Tensor
log1p t = unsafePerformIO $ (cast1 ATen.log1p_t) t

-- | Returns a new tensor with the logarithm to the base 2 of the elements of input.
log2 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
log2 t = unsafePerformIO $ (cast1 ATen.log2_t) t

-- | Returns a new tensor with the logarithm to the base 10 of the elements of input.
log10 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
log10 t = unsafePerformIO $ (cast1 ATen.log10_t) t

-- | Takes the power of each element in input with exponent and returns a tensor with the result.
pow 
 :: Scalar a 
 => Tensor -- ^ input
 -> a -- ^ exponent
 -> Tensor -- ^ output
pow t s = unsafePerformIO $ (cast2 ATen.pow_ts) t s

-- | Takes the power of each element in input with exponent and returns a tensor with the result.
-- Exponent is a tensor with the same number of elements as input.
powt
    :: Tensor -- ^ input
    -> Tensor -- ^ exponent
    -> Tensor -- ^ output
powt t t' = unsafePerformIO $ (cast2 ATen.pow_tt) t t'

-- | Applies the rectified linear unit function element-wise.
relu 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
relu t = unsafePerformIO $ (cast1 ATen.relu_t) t

-- | Applies element-wise, \(\text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1))\) , with α=1.6732632423543772848170429916717 and scale=1.0507009873554804934193349852946.
selu 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
selu t = unsafePerformIO $ (cast1 ATen.selu_t) t

-- | Applies element-wise, \(\text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))\).
celu 
   :: Float -- ^ alpha
   -> Tensor -- ^ input
   -> Tensor -- ^ output
celu _alpha _self = unsafePerformIO $ (cast2 ATen.celu_ts) _self _alpha

-- | Applies the element-wise function sigmoid.
sigmoid 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
sigmoid t = unsafePerformIO $ (cast1 ATen.sigmoid_t) t

-- | Applies a softmax function. 
-- It is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1.
softmax 
    :: Int -- ^ dimension
    -> Tensor -- ^ input
    -> Tensor -- ^ output
softmax dim input = unsafePerformIO $ (cast3 ATen.softmax_tls) 
    input dim (dtype input)

-- | Applies a softmax followed by a logarithm.
-- While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower, and numerically unstable. This function uses an alternative formulation to compute the output and gradient correctly.
logSoftmax 
    :: Int -- ^ dimension
    -> Tensor -- ^ input
    -> Tensor -- ^ output
logSoftmax dim input = unsafePerformIO $ (cast3 ATen.log_softmax_tls) 
    input dim (dtype input)

-- | Thresholds each element of the input Tensor.
threshold 
    :: Float -- ^ threshold
    -> Float -- ^ value
    -> Tensor -- ^ input
    -> Tensor -- ^ output
threshold threshold value self = 
    unsafePerformIO $ (cast3 ATen.threshold_tss) self threshold value

-- | Returns a new tensor with the sine of the elements of input.
sin 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
sin t = unsafePerformIO $ (cast1 ATen.sin_t) t


-- | Returns a new tensor with the cos of the elements of input.
cos 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
cos t = unsafePerformIO $ (cast1 ATen.cos_t) t

-- | Returns a new tensor with the tangent of the elements of input.
tan 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
tan t = unsafePerformIO $ (cast1 ATen.tan_t) t

-- | Returns a new tensor with the hyperbolic sine of the elements of input.
sinh 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
sinh t = unsafePerformIO $ (cast1 ATen.sinh_t) t

-- | Returns a new tensor with the hyperbolic cosine of the elements of input.
cosh 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
cosh t = unsafePerformIO $ (cast1 ATen.cosh_t) t

-- | Returns a new tensor with the hyperbolic tangent of the elements of input.
tanh 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
tanh t = unsafePerformIO $ (cast1 ATen.tanh_t) t

-- | Returns a new tensor with the square-root of the elements of input.
sqrt 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
sqrt t = unsafePerformIO $ (cast1 ATen.sqrt_t) t

-- | Computes input > other element-wise.
-- The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
gt
    :: Tensor -- ^ input
    -> Tensor -- ^ output
    -> Tensor -- ^ other
gt a b = unsafePerformIO $ (cast2 ATen.gt_tt) a b

(>.) = gt

-- | Computes input < other element-wise.
-- The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
lt 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
lt a b = unsafePerformIO $ (cast2 ATen.lt_tt) a b

(<.) = lt

-- | Computes input >= other element-wise.
-- The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
ge 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
ge a b = unsafePerformIO $ (cast2 ATen.ge_tt) a b

(>=.) = ge

-- | Computes input <= other element-wise.
-- The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
le 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
le a b = unsafePerformIO $ (cast2 ATen.le_tt) a b

(<=.) = le

-- | Computes input == other element-wise.
-- The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
eq 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
eq a b = unsafePerformIO $ (cast2 ATen.eq_tt) a b

(==.) = eq

-- | Computes input /= other element-wise.
-- The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
ne 
    :: Tensor -- ^ input
    -> Tensor -- ^ other
    -> Tensor -- ^ output
ne a b = unsafePerformIO $ (cast2 ATen.ne_tt) a b

(/=.) = ne

-- | Casting to given 'Dtype', where 'Dtype' is an object that represents the data type of a tensor in hasktorch.
toDType 
 :: DType -- ^ data type to cast to 
 -> Tensor -- ^ input
 -> Tensor -- ^ output
toDType dtype t = unsafePerformIO $ (cast4 ATen.tensor_to_sbb) t dtype False False

-- | squeezeAll
squeezeAll 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
squeezeAll t = unsafePerformIO $ (cast1 ATen.squeeze_t) t

-- | Function that measures the Binary Cross Entropy between the target and the output.
binaryCrossEntropyLoss 
 :: Tensor -- ^ input
 -> Tensor -- ^ target
 -> Tensor -- ^ weight
 -> Reduction -- ^ Specifies the reduction to apply to the output
 -> Tensor -- ^ output
binaryCrossEntropyLoss t target weight reduction = unsafePerformIO $ (cast4 ATen.binary_cross_entropy_tttl) t target weight reduction

-- | Binary Cross Entropy with weights defaulted to 1.0 & reduction defaulted to ReduceMean
binaryCrossEntropyLoss' 
 :: Tensor -- ^ input
 -> Tensor -- ^ target
 -> Tensor -- ^ output
binaryCrossEntropyLoss' t target = unsafePerformIO $ (cast4 ATen.binary_cross_entropy_tttl) t target (onesLike target) ReduceMean

-- | Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the @input@ and @target@.
mseLoss 
 :: Tensor -- ^ input
 -> Tensor -- ^ target
 -> Tensor -- ^ output
mseLoss input output = unsafePerformIO $ (cast3 ATen.mse_loss_ttl) input output ATen.kMean

-- | The negative log likelihood loss.
nllLoss' 
 :: Tensor -- ^ input
 -> Tensor -- ^ target tensor
 -> Tensor -- ^ output
nllLoss' t target = unsafePerformIO $ (cast5 ATen.nll_loss_tttll) t target weight ReduceMean (-100 :: Int)
    where
        nClass = (shape t) !! 1 -- TODO nicer runtime error if input dimensions don't conform
        weight = ones' [nClass]

-- | Applies a 1D adaptive max pooling over an input signal composed of several input planes.
adaptiveMaxPool1d 
    :: Int -- ^ output size
    -> Tensor -- ^ input
    -> (Tensor,Tensor) -- ^ output
adaptiveMaxPool1d outputSize self =
    unsafePerformIO $ (cast2 ATen.adaptive_max_pool1d_tl)
        self outputSize

-- | Applies a 2D adaptive max pooling over an input signal composed of several input planes.
adaptiveMaxPool2d 
    :: (Int,Int) -- ^ output size
    -> Tensor -- ^ input
    -> (Tensor,Tensor) -- ^ output 
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

-- | Applies a 1D max pooling over an input signal composed of several input planes.
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

-- | Applies a 2D max pooling over an input signal composed of several input planes.
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
 
-- | Applies a 3D max pooling over an input signal composed of several input planes.
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

-- | Takes the inverse of the square matrix input. @input@ can be batches of 2D square tensors, in which case this function would return a tensor composed of individual inverses.
inverse 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
inverse t = unsafePerformIO $ (cast1 ATen.inverse_t) t

-- | This function returns eigenvalues and eigenvectors of a real symmetric matrix input or a batch of real symmetric matrices, represented by a namedtuple (eigenvalues, eigenvectors).
symeig 
 :: Tensor -- ^ input tensor  
 -> Bool -- ^ bool which controls whether eigenvectors have to be computed
 -> Tri -- ^ controls whether to consider upper-triangular or lower-triangular region
 -> (Tensor, Tensor) -- ^ output tensors
symeig t eigenvectors upper = unsafePerformIO $ (cast3 ATen.symeig_tbb) t eigenvectors boolUpper
  where boolUpper = isUpper upper

-- | Computes the eigenvalues and eigenvectors of a real square matrix
eig 
 :: Tensor -- ^ input (square matrix) for which the eigen values and eigen vectors are to be computed
 -> Bool -- ^ bool to compute both eigenvalues and eigenvectors; otherwise, only eigenvalues will be computed
 -> (Tensor, Tensor) -- ^ output tensors
eig t eigenvectors = unsafePerformIO $ (cast2 ATen.eig_tb) t eigenvectors

-- | This function returns a namedtuple (U, S, V) which is the singular value decomposition of a input real matrix or batches of real matrices input such that input = U * diag(S) * V^T
svd 
 :: Tensor -- ^ input 
 -> Bool -- ^ controls the shape of returned U and V
 -> Bool -- ^ option whether to compute U and V or not
 -> (Tensor, Tensor, Tensor) -- ^ output tuple of tensors
svd t some compute_uv = unsafePerformIO $ (cast3 ATen.svd_tbb) t some compute_uv

-- | Computes the Cholesky decomposition of a symmetric positive-definite matrix AA or for batches of symmetric positive-definite matrices.
cholesky 
 :: Tensor -- ^ input
 -> Tri -- ^ flag that indicates whether to return a upper or lower triangular matrix.
 -> Tensor -- ^ output
cholesky t upper = unsafePerformIO $ (cast2 ATen.cholesky_tb) t boolUpper
  where boolUpper = isUpper upper


-- | Solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix uu .
choleskySolve 
 :: Tensor -- ^ input matrix b 
 -> Tensor -- ^ input matrix u
 -> Tri -- ^ bool whether to consider the Cholesky factor as a lower or upper triangular matrix
 -> Tensor -- ^ output
choleskySolve t1 t2 upper = unsafePerformIO $ (cast3 ATen.cholesky_solve_ttb) t1 t2 boolUpper
  where boolUpper = isUpper upper


-- | During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
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

-- | Applies alpha dropout to the input.
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

-- | Applies a 1D average pooling over an input signal composed of several input planes.
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

-- | Applies a 1D adaptive average pooling over an input signal composed of several input planes.
adaptiveAvgPool1d
  :: Int -- outputSize
  -> Tensor -- ^ input
  -> Tensor -- ^ output
adaptiveAvgPool1d outputSize input = unsafePerformIO
  $ cast2 ATen.adaptive_avg_pool1d_tl input outputSize

-- | Applies a 2D adaptive average pooling over an input signal composed of several input planes.
adaptiveAvgPool2d 
 :: Tensor -- ^ input 
 -> (Int,Int) -- ^ output size (Height * Width)
 -> Tensor -- ^ output
adaptiveAvgPool2d _self _output_size = unsafePerformIO $ (cast2 ATen.adaptive_avg_pool2d_tl) _self _output_size

-- | Applies a 3D adaptive average pooling over an input signal composed of several input planes.
adaptiveAvgPool3d 
 :: Tensor -- ^ input
 -> (Int,Int,Int) -- ^ output size (Depth * Height * Width)
 -> Tensor -- ^ output
adaptiveAvgPool3d _self _output_size = unsafePerformIO $ (cast2 ATen.adaptive_avg_pool3d_tl) _self _output_size

-- | Computes the bitwise NOT of the given input tensor. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical NOT.
bitwiseNot 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
bitwiseNot input = unsafePerformIO $ cast1 ATen.bitwise_not_t input

-- | Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
cat
  :: Int -- ^ dimension 
  -> [Tensor] -- ^ list of tensors to concatenate
  -> Tensor -- ^ output tensor
cat dim tensors = unsafePerformIO $ cast2 ATen.cat_ll tensors dim

-- | Splits a tensor into a specific number of chunks.
-- Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by chunks.
chunk
  :: Int -- ^ chunks
  -> Int -- ^ dim
  -> Tensor -- ^ input tensor
  -> [Tensor] -- ^ output list of tensors
chunk chunks dim input = unsafePerformIO
  $ cast3 ATen.chunk_tll input chunks dim

-- | Clamp all elements in input into the range [ min, max ] and return a resulting tensor.
clamp
  :: Float -- ^ minimum value
  -> Float -- ^ maximum value
  -> Tensor -- ^ input
  -> Tensor -- ^ output
clamp min max input = unsafePerformIO $ cast3 ATen.clamp_tss input min max

-- | Clamps all elements in input to be smaller or equal max.
clampMax
  :: Float -- ^ maximum value
  -> Tensor -- ^ input
  -> Tensor -- ^ output
clampMax max input = unsafePerformIO $ cast2 ATen.clamp_max_ts input max

-- | Clamps all elements in input to be larger or equal min.
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

-- | Pads the input tensor boundaries with a constant value.
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

-- | Applies a 1D convolution over an input signal composed of several input planes.
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

-- | Applies a 2D convolution over an input signal composed of several input planes.
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

-- | This function returns the solution to the system of linear equations represented by AX = BAX=B and the LU factorization of A, in order as a namedtuple solution, LU.
-- LU contains L and U factors for LU factorization of A
solve 
    :: Tensor -- ^ input matrix
    -> Tensor -- ^ input square matrix
    -> (Tensor,Tensor) -- ^ output tuple with solution and LU
solve b a = unsafePerformIO $ (cast2 ATen.solve_tt) b a

-- | Solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix uu .
choleskyInverse
    :: Tensor -- ^ input
    -> Tri -- ^ upper or lower triangle
    -> Tensor -- ^ solution
choleskyInverse t upper = unsafePerformIO $ (cast2 ATen.cholesky_inverse_tb) t boolUpper
  where boolUpper = isUpper upper

-- pstrf :: Tensor -> Bool -> Double -> (Tensor, Tensor)
-- pstrf t upper tol = unsafePerformIO $ (cast3 ATen.pstrf_tbs) t upper tol

-- qr :: Tensor -> (Tensor, Tensor)
-- qr t = unsafePerformIO $ (cast1 ATen.qr_t) t

-- | This is a low-level function for calling LAPACK directly. This function returns a namedtuple (a, tau) as defined in LAPACK documentation for geqrf.
geqrf 
    :: Tensor -- ^ input
    -> (Tensor, Tensor) -- ^ a, tau output matrices (see https://software.intel.com/en-us/node/521004)
geqrf t = unsafePerformIO $ (cast1 ATen.geqrf_t) t


-- | Computes the orthogonal matrix Q of a QR factorization, from the @(input, input2)@ tuple returned by 'geqrf' function.
-- This directly calls the underlying LAPACK function @?orgqr@. See LAPACK documentation for @orgqr@ for further details.
orgqr 
 :: Tensor -- ^ the @a@ from @geqrf@ function
 -> Tensor -- ^ the @tau@ from @geqrf@ function
 -> Tensor -- ^ output
orgqr b a = unsafePerformIO $ (cast2 ATen.orgqr_tt) b a

-- | Returns a new tensor with the signs of the elements of @input@
sign 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
sign t = unsafePerformIO $ (cast1 ATen.sign_t) t

-- | Returns a tensor that is a transposed version of @input@. The given dimensions @dim0@ and @dim1@ are swapped.
transpose 
 :: Tensor -- ^ input
 -> Int -- ^ dim1
 -> Int -- ^ dim2
 -> Tensor -- ^ 
transpose t a b = unsafePerformIO $ (cast3 ATen.transpose_tll) t a b

-- | transpose special case for a 2D tensor
transpose2D 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
transpose2D t = transpose t 0 1


-- | Returns a tensor with the elements of input as the diagonal.
-- The second argument controls which diagonal to consider:
--        If Int = 0, it is the main diagonal.
--        If Int > 0, it is above the main diagonal.
--        If Int < 0, it is below the main diagonal.

diag
    ::  Tensor -- ^ input
    ->  Int -- ^ diagonal
    ->  Tensor -- ^ output
diag t index = unsafePerformIO $ (cast2 ATen.tensor_diag_l) t index

-- | Returns True if all elements in the tensor are True, False otherwise.
all
 :: Tensor -- ^ input
 -> Bool -- ^ output
all t = toInt (unsafePerformIO $ (cast1 ATen.all_t) t) == 1

-- | Returns True if any elements in the tensor are True, False otherwise.
any
 :: Tensor -- ^ input
 -> Bool -- ^ output
any t = toInt (unsafePerformIO $ (cast1 ATen.any_t) t) == 1


-- | Returns True if all elements in each row of the tensor in the given dimension dim are True, False otherwise.
-- If keepdim is True, the output tensor is of the same size as input except in the dimension dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 fewer dimension than input.  
allDim 
 :: Tensor -- ^ input
 -> Int -- ^ dimension
 -> Bool -- ^ boolean corresponding to keepdim
 -> Tensor -- ^ output
allDim t dim keepdim = unsafePerformIO $ (cast3 ATen.all_tlb) t dim keepdim

-- | Returns True if any elements in each row of the tensor in the given dimension dim are True, False otherwise.
-- If keepdim is True, the output tensor is of the same size as input except in the dimension dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 fewer dimension than input.
anyDim 
 :: Tensor -- ^ input 
 -> Int -- ^ dimension 
 -> Bool -- ^ boolean corresponding to keepdim
 -> Tensor -- output
anyDim t dim keepdim = unsafePerformIO $ (cast3 ATen.any_tlb) t dim keepdim

-- | Permute the dimensions of this tensor.
permute 
 :: Tensor -- ^ input
 -> [Int] -- ^ list corresponding to ordering of dimensions to permute with 
 -> Tensor -- output
permute t dims = unsafePerformIO $ (cast2 ATen.tensor_permute_l) t dims
