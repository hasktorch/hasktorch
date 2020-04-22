{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Functional
    ( module Torch.Functional
    , Internal.addmv
    , Internal.addr
    , Internal.allclose
    , Internal.argmin
    , Internal.baddbmm
    , Internal.bmm
    , Internal.acos
    , Internal.asin
    , Internal.atan
    , Internal.dot
    , Internal.einsum
    , Internal.lstsq
    , Internal.mv
    , Internal.slice
    , Internal.sumWithDimnames
) where

import Prelude hiding ( all
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
-- import Torch.Functional.Internal hiding (argmax, clamp, cosh, conv1d, linear, softmax)
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
 => a -- ^ summand
 -> Tensor -- ^ input
 -> Tensor -- ^ output
addScalar a t = unsafePerformIO $ (cast2 ATen.add_ts) t a

-- | Subtracts each element of the input input with the scalar and returns a new resulting tensor.
subScalar 
 :: Scalar a 
 => a -- ^ subtrahend
 -> Tensor -- ^ input
 -> Tensor -- ^ output
subScalar a t = unsafePerformIO $ (cast2 ATen.sub_ts) t a

-- | Multiplies each element of the input input with the scalar and returns a new resulting tensor.
mulScalar 
 :: Scalar a 
 => a -- ^ multiplier
 -> Tensor -- ^ input
 -> Tensor -- ^ output
mulScalar a t = unsafePerformIO $ (cast2 ATen.mul_ts) t a

-- | Divides each element of the input input with the scalar and returns a new resulting tensor.
divScalar 
 :: Scalar a 
 => a -- ^ divisor
 -> Tensor -- ^ input
 -> Tensor -- ^ output
divScalar a t = unsafePerformIO $ (cast2 ATen.div_ts) t a

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

-- | A simple lookup table that looks up embeddings in a fixed dictionary and size.
-- This module is often used to retrieve word embeddings using indices. The input to the module is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.
embedding 
    :: Bool -- ^ whether or not to scale the gradient by the frequencies
    -> Bool -- ^ whether or not the embedding is sparse
    -> Tensor -- ^ weights
    -> Int -- ^ padding
    -> Tensor -- ^ indices
    -> Tensor -- ^ output
embedding scaleByGradFreq sparse weights paddingIdx indices =
    unsafePerformIO $ (cast5 ATen.embedding_ttlbb)
        weights indices paddingIdx scaleByGradFreq sparse

embedding'
    :: Tensor -- ^ weights
    -> Tensor -- ^ indices
    -> Tensor -- ^ output
embedding' weights indices =
    unsafePerformIO $ (cast5 ATen.embedding_ttlbb)
        weights indices (-1 :: Int) False False

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
 => a -- ^ exponent
 -> Tensor -- ^ input
 -> Tensor -- ^ output
pow s t = unsafePerformIO $ (cast2 ATen.pow_ts) t s

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

-- | Applies Exponential linear unit function element-wise, with alpha input, \(\text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))\)
elu
    :: Scalar s 
    => s -- ^ alpha value for ELU formulation
    -> Tensor -- ^ input
    -> Tensor -- ^ output
elu a t = unsafePerformIO $ (cast2 ATen.elu_ts) t a

-- | Applies exponential linear unit function element wise with default alpha value = 1
elu'
    :: Tensor -- ^ input
    -> Tensor -- ^ output
elu' t = unsafePerformIO $ (cast1 ATen.elu_t) t

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
    :: Dim -- ^ dimension
    -> Tensor -- ^ input
    -> Tensor -- ^ output
softmax (Dim d) input = unsafePerformIO $ (cast3 ATen.softmax_tls) 
    input d (dtype input)

-- | Applies a softmax followed by a logarithm.
-- While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower, and numerically unstable. This function uses an alternative formulation to compute the output and gradient correctly.
logSoftmax 
    :: Dim -- ^ dimension
    -> Tensor -- ^ input
    -> Tensor -- ^ output
logSoftmax (Dim d) input = unsafePerformIO $ (cast3 ATen.log_softmax_tls) 
    input d (dtype input)

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
 :: Reduction -- ^ Specifies the reduction to apply to the output
 -> Tensor -- ^ target
 -> Tensor -- ^ weight
 -> Tensor -- ^ input
 -> Tensor -- ^ output
binaryCrossEntropyLoss reduction target weight t = unsafePerformIO $ (cast4 ATen.binary_cross_entropy_tttl) t target weight reduction

-- | Binary Cross Entropy with weights defaulted to 1.0 & reduction defaulted to ReduceMean
binaryCrossEntropyLoss' 
 :: Tensor -- ^ target
 -> Tensor -- ^ input
 -> Tensor -- ^ output
binaryCrossEntropyLoss' target t = unsafePerformIO $ (cast4 ATen.binary_cross_entropy_tttl) t target (onesLike target) ReduceMean

-- | Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the @input@ and @target@.
mseLoss 
 :: Tensor -- ^ target
 -> Tensor -- ^ input
 -> Tensor -- ^ output
mseLoss output input = unsafePerformIO $ (cast3 ATen.mse_loss_ttl) input output ATen.kMean

-- | The negative log likelihood loss.
nllLoss' 
 :: Tensor -- ^ target tensor
 -> Tensor -- ^ input
 -> Tensor -- ^ output
nllLoss' target t = unsafePerformIO $ (cast5 ATen.nll_loss_tttll) t target weight ReduceMean (-100 :: Int)
    where
        nClass = (shape t) !! 1 -- TODO nicer runtime error if input dimensions don't conform
        weight = ones' [nClass]

-- | Returns cosine similarity between x1 and x2, computed along dim.
cosineSimilarity 
    :: Int -- ^ dimension of vectors (default=1)
    -> Double -- ^ small value to avoid division by 0 (default=1e-8)
    -> Tensor -- ^ x1
    -> Tensor -- ^ x2
    -> Tensor -- ^ output
cosineSimilarity dim eps x1 x2 = 
    unsafePerformIO $ (cast4 ATen.cosine_similarity_ttld) x1 x2 dim eps

-- | Returns cosine similarity with defaulted options.
cosineSimilarity'
    :: Tensor -- ^ x1
    -> Tensor -- ^ x2
    -> Tensor -- ^ output
cosineSimilarity' x1 x2 = 
    unsafePerformIO $ 
        (cast4 ATen.cosine_similarity_ttld) x1 x2 (1 :: Int) (1e-8 :: Double)

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
        self 
        (asList kernelSize)
        (asList stride)
        (asList padding)
        (asList dilation)
        ceilMode
        where
            asList :: (Int, Int) -> [Int]
            asList (a0, a1) = [a0, a1]
 
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
 :: Bool -- ^ bool which controls whether eigenvectors have to be computed
 -> Tri -- ^ controls whether to consider upper-triangular or lower-triangular region
 -> Tensor -- ^ input tensor  
 -> (Tensor, Tensor) -- ^ output tensors
symeig eigenvectors upper t = unsafePerformIO $ (cast3 ATen.symeig_tbb) t eigenvectors boolUpper
  where boolUpper = isUpper upper

-- | Computes the eigenvalues and eigenvectors of a real square matrix
eig 
 :: Bool -- ^ bool to compute both eigenvalues and eigenvectors; otherwise, only eigenvalues will be computed
 -> Tensor -- ^ input (square matrix) for which the eigen values and eigen vectors are to be computed
 -> (Tensor, Tensor) -- ^ output tensors
eig eigenvectors t = unsafePerformIO $ (cast2 ATen.eig_tb) t eigenvectors

-- | This function returns a namedtuple (U, S, V) which is the singular value decomposition of a input real matrix or batches of real matrices input such that input = U * diag(S) * V^T
svd 
 :: Bool -- ^ controls the shape of returned U and V
 -> Bool -- ^ option whether to compute U and V or not
 -> Tensor -- ^ input 
 -> (Tensor, Tensor, Tensor) -- ^ output tuple of tensors
svd some compute_uv t = unsafePerformIO $ (cast3 ATen.svd_tbb) t some compute_uv

-- | Computes the Cholesky decomposition of a symmetric positive-definite matrix AA or for batches of symmetric positive-definite matrices.
cholesky 
 :: Tri -- ^ flag that indicates whether to return a upper or lower triangular matrix.
 -> Tensor -- ^ input
 -> Tensor -- ^ output
cholesky upper t = unsafePerformIO $ (cast2 ATen.cholesky_tb) t boolUpper
  where boolUpper = isUpper upper


-- | Solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix uu .
choleskySolve 
 :: Tri -- ^ bool whether to consider the Cholesky factor as a lower or upper triangular matrix
 -> Tensor -- ^ input matrix b 
 -> Tensor -- ^ input matrix u
 -> Tensor -- ^ output
choleskySolve upper t1 t2 = unsafePerformIO $ (cast3 ATen.cholesky_solve_ttb) t1 t2 boolUpper
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
 :: (Int,Int) -- ^ output size (Height * Width)
 -> Tensor -- ^ input 
 -> Tensor -- ^ output
adaptiveAvgPool2d (outputHeight, outputWidth) input = 
    unsafePerformIO $ (cast2 ATen.adaptive_avg_pool2d_tl) 
        input 
        ([outputHeight, outputWidth] :: [Int])

-- | Applies a 3D adaptive average pooling over an input signal composed of several input planes.
adaptiveAvgPool3d 
 :: (Int,Int,Int) -- ^ output size (Depth * Height * Width)
 -> Tensor -- ^ input
 -> Tensor -- ^ output
adaptiveAvgPool3d _output_size _self = unsafePerformIO $ (cast2 ATen.adaptive_avg_pool3d_tl) _self _output_size

-- | Computes the bitwise NOT of the given input tensor. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical NOT.
bitwiseNot 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
bitwiseNot input = unsafePerformIO $ cast1 ATen.bitwise_not_t input

-- | Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
cat
  :: Dim -- ^ dimension 
  -> [Tensor] -- ^ list of tensors to concatenate
  -> Tensor -- ^ output tensor
cat (Dim d) tensors = unsafePerformIO $ cast2 ATen.cat_ll tensors d

-- | Splits a tensor into a specific number of chunks.
-- Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by chunks.
chunk
  :: Int -- ^ chunks
  -> Dim -- ^ dim
  -> Tensor -- ^ input tensor
  -> [Tensor] -- ^ output list of tensors
chunk chunks (Dim d) input = unsafePerformIO
  $ cast3 ATen.chunk_tll input chunks d

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
    :: Tri -- ^ upper or lower triangle
    -> Tensor -- ^ input
    -> Tensor -- ^ solution
choleskyInverse upper t = unsafePerformIO $ (cast2 ATen.cholesky_inverse_tb) t boolUpper
  where boolUpper = isUpper upper

-- pstrf :: Bool -> Double -> Tensor -> (Tensor, Tensor)
-- pstrf upper tol t = unsafePerformIO $ (cast3 ATen.pstrf_tbs) t upper tol

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
 :: Dim -- ^ dim1
 -> Dim -- ^ dim2
 -> Tensor -- ^ input
 -> Tensor -- ^ output
transpose (Dim d1) (Dim d2) t = unsafePerformIO $ (cast3 ATen.transpose_tll) t d1 d2

-- | transpose special case for a 2D tensor
transpose2D 
    :: Tensor -- ^ input
    -> Tensor -- ^ output
transpose2D = transpose (Dim 0) (Dim 1)


-- | Returns a tensor with the elements of input as the diagonal.
-- The second argument controls which diagonal to consider:
--        If Int = 0, it is the main diagonal.
--        If Int > 0, it is above the main diagonal.
--        If Int < 0, it is below the main diagonal.

diag
    ::  Int -- ^ diagonal
    ->  Tensor -- ^ input
    ->  Tensor -- ^ output
diag index t = unsafePerformIO $ (cast2 ATen.tensor_diag_l) t index

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
 :: Dim -- ^ dimension
 -> Bool -- ^ boolean corresponding to keepdim
 -> Tensor -- ^ input
 -> Tensor -- ^ output
allDim (Dim d) keepdim t = unsafePerformIO $ (cast3 ATen.all_tlb) t d keepdim

-- | Returns True if any elements in each row of the tensor in the given dimension dim are True, False otherwise.
-- If keepdim is True, the output tensor is of the same size as input except in the dimension dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 fewer dimension than input.
anyDim 
 :: Dim -- ^ dimension 
 -> Bool -- ^ boolean corresponding to keepdim
 -> Tensor -- ^ input 
 -> Tensor -- output
anyDim (Dim d) keepdim t = unsafePerformIO $ (cast3 ATen.any_tlb) t d keepdim

-- | Permute the dimensions of this tensor.
permute 
 :: [Int] -- ^ list corresponding to ordering of dimensions to permute with 
 -> Tensor -- ^ input
 -> Tensor -- output
permute dims t = unsafePerformIO $ (cast2 ATen.tensor_permute_l) t dims

-- | expand
-- TODO: figure out what the `implicit` boolean value does
expand
  :: Tensor -- ^ input
  -> Bool -- ^ some boolean value with unknown function
  -> [Int] -- ^ the desired expanded size
  -> Tensor -- ^ output
expand t someBool dims = unsafePerformIO $ (cast3 ATen.tensor_expand_lb) t dims someBool

-- | flatten
flatten
  :: Dim -- ^ startDim
  -> Dim -- ^ endDim
  -> Tensor -- ^ self
  -> Tensor -- ^ output
flatten (Dim startDim) (Dim endDim) t = unsafePerformIO $ (cast3 ATen.flatten_tll) t startDim endDim

-- | flattenAll
flattenAll
  :: Tensor -- ^ input
  -> Tensor -- ^ output
flattenAll t =
  unsafePerformIO $ (cast3 ATen.flatten_tll) t (0 :: Int) (-1 :: Int)

bernoulli'
  :: Tensor
  -> IO Tensor
bernoulli' t =
  (cast1 ATen.bernoulli_t) t

bernoulli
  :: Tensor
  -> Double
  -> IO Tensor
bernoulli t p =
  (cast2 ATen.bernoulli_td) t p

poisson
  :: Tensor
  -> IO Tensor
poisson t =
  (cast1 ATen.poisson_t) t

multinomial'
  :: Tensor
  -> Int
  -> IO Tensor
multinomial' t num_samples =
  (cast2 ATen.multinomial_tl) t num_samples

multinomial
  :: Tensor
  -> Int
  -> Bool
  -> IO Tensor
multinomial t num_samples replacement =
  (cast3 ATen.multinomial_tlb) t num_samples replacement

normal'
  :: Tensor
  -> IO Tensor
normal' _mean =
  (cast1 ATen.normal_t) _mean

normal
  :: Tensor
  -> Tensor
  -> IO Tensor
normal _mean _std =
  (cast2 ATen.normal_tt) _mean _std

normalScalar
  :: Tensor
  -> Double
  -> IO Tensor
normalScalar _mean _std =
  (cast2 ATen.normal_td) _mean _std

normalScalar'
  :: Double
  -> Tensor
  -> IO Tensor
normalScalar' _mean _std =
  (cast2 ATen.normal_dt) _mean _std

normalWithSize
  :: Double
  -> Double
  -> Int
  -> IO Tensor
normalWithSize _mean _std _size =
  (cast3 ATen.normal_ddl) _mean _std _size

rrelu'''
  :: Tensor
  -> IO Tensor
rrelu''' t =
  (cast1 ATen.rrelu_t) t

rrelu''
  :: Scalar a 
  => Tensor
  -> a
  -> IO Tensor
rrelu'' t _upper =
  (cast2 ATen.rrelu_ts) t _upper

rrelu'
  :: Scalar a 
  => Tensor
  -> a
  -> a
  -> IO Tensor
rrelu' t _lower _upper =
  (cast3 ATen.rrelu_tss) t _lower _upper

rrelu
  :: Scalar a 
  => Tensor
  -> a
  -> a
  -> Bool
  -> IO Tensor
rrelu t _lower _upper _training =
  (cast4 ATen.rrelu_tssb) t _lower _upper _training

rreluWithNoise'''
  :: Tensor
  -> Tensor
  -> IO Tensor
rreluWithNoise''' t _noise =
  (cast2 ATen.rrelu_with_noise_tt) t _noise

rreluWithNoise''
  :: Scalar a 
  => Tensor
  -> Tensor
  -> a
  -> IO Tensor
rreluWithNoise'' t _noise _upper =
  (cast3 ATen.rrelu_with_noise_tts) t _noise _upper

rreluWithNoise'
  :: Scalar a 
  => Tensor
  -> Tensor
  -> a
  -> a
  -> IO Tensor
rreluWithNoise' t _noise _lower _upper =
  (cast4 ATen.rrelu_with_noise_ttss) t _noise _lower _upper

rreluWithNoise
  :: Scalar a 
  => Tensor
  -> Tensor
  -> a
  -> a
  -> Bool
  -> IO Tensor
rreluWithNoise t _noise _lower _upper _training =
  (cast5 ATen.rrelu_with_noise_ttssb) t _noise _lower _upper _training

-- Not used yet
data RNNParams = RNNParams {
    weightIH :: Tensor,
    weightHH :: Tensor,
    biasIH :: Tensor,
    biasHH :: Tensor
} deriving (Show)

-- | A long short-term memory (LSTM) cell.
lstmCell 
    :: Tensor -- ^ input-hidden weights (4*hidden_size, input_size)
    -> Tensor -- ^ hidden-hidden weights (4*hidden_size, hidden_size)
    -> Tensor -- ^ input-hidden bias (4*hidden_size)
    -> Tensor -- ^ hidden-hidden bias, of shape (4*hidden_size)
    -> (Tensor, Tensor) -- ^ hidden state
    -> Tensor -- ^ input
    -> (Tensor, Tensor) -- next hidden state, next cell state
lstmCell _w_ih _w_hh _b_ih _b_hh (_hx, _cx) _input =
    unsafePerformIO $
        (cast6 ATen.lstm_cell_tltttt) 
        _input ([_hx, _cx] :: [Tensor]) _w_ih _w_hh _b_ih _b_hh -- TODO: make cast work with 2-tuples

-- | A gated recurrent unit (GRU) cell
gruCell 
    :: Tensor -- ^ input-hidden weights
    -> Tensor -- ^ hidden-hidden weights
    -> Tensor -- ^ input-hidden bias
    -> Tensor -- ^ hidden-hidden bias
    -> Tensor -- ^ hidden state
    -> Tensor -- ^ input
    -> Tensor -- ^ output
gruCell _w_ih _w_hh _b_ih _b_hh _hx _input =
  unsafePerformIO $
    (cast6 ATen.gru_cell_tttttt) 
    _input _hx _w_ih _w_hh _b_ih _b_hh

-- | An Elman RNN cell with tanh non-linearity
rnnTanhCell 
    :: Tensor -- ^ input-hidden weights
    -> Tensor -- ^ hidden-hidden weights
    -> Tensor -- ^ input-hidden bias
    -> Tensor -- ^ hidden-hidden bias
    -> Tensor -- ^ hidden state
    -> Tensor -- ^ input
    -> Tensor -- ^ output
rnnTanhCell _w_ih _w_hh _b_ih _b_hh _hx _input =
  unsafePerformIO $ (cast6 ATen.rnn_tanh_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

-- | An Elman RNN cell with ReLU non-linearity
rnnReluCell 
    :: Tensor -- ^ input-hidden weights
    -> Tensor -- ^ hidden-hidden weights
    -> Tensor -- ^ input-hidden bias
    -> Tensor -- ^ hidden-hidden bias
    -> Tensor -- ^ hidden state
    -> Tensor -- ^ input
    -> Tensor -- ^ output
rnnReluCell _w_ih _w_hh _b_ih _b_hh _hx _input =
  unsafePerformIO $ (cast6 ATen.rnn_relu_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

-- | A quantized long short-term memory (LSTM) cell.
quantizedLstmCell 
    :: Tensor -- ^ input-hidden weights
    -> Tensor -- ^ hidden-hidden weights
    -> Tensor -- ^ input-hidden bias
    -> Tensor -- ^ hidden-hidden bias
    -> Tensor -- ^ input-hidden packed
    -> Tensor -- ^ hidden-hidden packed
    -> Tensor -- ^ input-hidden column offsets
    -> Tensor -- ^ hidden-hidden column offsets
    -> Float -- ^ input-hidden scale
    -> Float -- ^ hidden-hidden scale
    -> Float -- ^ input-hidden zero point
    -> Float -- ^ hidden-hidden zero point
    -> (Tensor, Tensor) -- ^ hidden state
    -> Tensor -- ^ input
    -> (Tensor, Tensor) -- ^ output
quantizedLstmCell _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh (_hx, _cx) _input =
  unsafePerformIO $
    (cast14 ATen.quantized_lstm_cell_tlttttttttssss)
        _input ([_hx, _cx] :: [Tensor]) _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- | A quantized long gated recurrent unit (GRU) cell.
quantizedGruCell 
    :: Tensor -- ^ input-hidden weights
    -> Tensor -- ^ hidden-hidden weights
    -> Tensor -- ^ input-hidden bias
    -> Tensor -- ^ hidden-hidden bias
    -> Tensor -- ^ input-hidden packed
    -> Tensor -- ^ hidden-hidden packed
    -> Tensor -- ^ input-hidden column offsets
    -> Tensor -- ^ hidden-hidden column offsets
    -> Float -- ^ input-hidden scale
    -> Float -- ^ hidden-hidden scale
    -> Float -- ^ input-hidden zero point
    -> Float -- ^ hidden-hidden zero point
    -> Tensor -- ^ hidden state
    -> Tensor -- ^ input
    -> Tensor -- ^ output
quantizedGruCell _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh _hx _input =
  unsafePerformIO $ (cast14 ATen.quantized_gru_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- | A quantized Elman RNN cell with relu non-linearity
quantizedRnnReluCell 
    :: Tensor -- ^ input-hidden weights
    -> Tensor -- ^ hidden-hidden weights
    -> Tensor -- ^ input-hidden bias
    -> Tensor -- ^ hidden-hidden bias
    -> Tensor -- ^ input-hidden packed
    -> Tensor -- ^ hidden-hidden packed
    -> Tensor -- ^ input-hidden column offsets
    -> Tensor -- ^ hidden-hidden column offsets
    -> Float -- ^ input-hidden scale
    -> Float -- ^ hidden-hidden scale
    -> Float -- ^ input-hidden zero point
    -> Float -- ^ hidden-hidden zero point
    -> Tensor -- ^ hidden state
    -> Tensor -- ^ input
    -> Tensor -- ^ output
quantizedRnnReluCell _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh _hx _input =
  unsafePerformIO $ (cast14 ATen.quantized_rnn_relu_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- | A quantized Elman RNN cell with tanh non-linearity
quantizedRnnTanhCell
    :: Tensor -- ^ input-hidden weights
    -> Tensor -- ^ hidden-hidden weights
    -> Tensor -- ^ input-hidden bias
    -> Tensor -- ^ hidden-hidden bias
    -> Tensor -- ^ input-hidden packed
    -> Tensor -- ^ hidden-hidden packed
    -> Tensor -- ^ input-hidden column offsets
    -> Tensor -- ^ hidden-hidden column offsets
    -> Float -- ^ input-hidden scale
    -> Float -- ^ hidden-hidden scale
    -> Float -- ^ input-hidden zero point
    -> Float -- ^ hidden-hidden zero point
    -> Tensor -- ^ hidden state
    -> Tensor -- ^ input
    -> Tensor -- ^ output
quantizedRnnTanhCell _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh _hx _input =
  unsafePerformIO $ (cast14 ATen.quantized_rnn_tanh_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- | smoothL1Loss
smoothL1Loss
  :: Reduction -- ^ reduction
  -> Tensor -- ^ input
  -> Tensor -- ^ target
  -> Tensor -- ^ output
smoothL1Loss reduction input target = unsafePerformIO $ (cast3 ATen.smooth_l1_loss_ttl) input target reduction

-- | softMarginLoss
softMarginLoss 
  :: Reduction -- ^ reduction
  -> Tensor -- ^ input
  -> Tensor -- ^ target
  -> Tensor -- ^ output
softMarginLoss reduction input target = unsafePerformIO $ (cast3 ATen.soft_margin_loss_ttl) input target reduction

-- | softShrink
softShrink
  :: Float -- ^ lambda
  -> Tensor -- ^ input
  -> Tensor -- ^ output
softShrink lambda input = unsafePerformIO $ (cast2 ATen.softshrink_ts) input lambda

-- | Concatenates sequence of tensors along a new dimension.
-- All tensors need to be of the same size.
stack
  :: Dim -- ^ dim
  -> [Tensor] -- ^ input
  -> Tensor -- ^ output
stack (Dim d) tensors = unsafePerformIO $ (cast2 ATen.stack_ll) tensors d

-- | Returns the sum of each row of the input tensor in the given dimension dim.
-- If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. 
-- Otherwise, dim is squeezed, resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
sumDim
  :: Dim -- ^ dim to sum along
  -> KeepDim -- ^ whether the output tensor has dim retained or not
  -> DType -- ^ datatype
  -> Tensor -- ^ input
  -> Tensor -- ^ output
sumDim (Dim d) k dtype input = unsafePerformIO $ (cast4 ATen.sum_tlbs) input d (keepdim k) dtype

-- | Returns the k largest elements of the given input tensor along a given dimension.
-- If largest is False then the k smallest elements are returned.
-- The boolean option sorted if True, will make sure that the returned k elements are themselves sorted
-- A tuple of (values, indices) is returned, where the indices are the indices of the elements in the original input tensor.  
topK 
  :: Int -- ^ k
  -> Dim -- ^ dim to find topK along
  -> Bool -- ^ largest
  -> Bool -- ^ sorted
  -> Tensor -- ^ input
  -> (Tensor,Tensor) -- ^ output
topK k (Dim d) largest sorted input = unsafePerformIO $ (cast5 ATen.topk_tllbb) input k d largest sorted

-- | Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
-- The upper triangular part of the matrix is defined as the elements on and above the diagonal.
-- The argument diagonal controls which diagonal to consider. If diagonal = 0, all elements on and above the main diagonal are retained. 
-- A positive value excludes just as many diagonals above the main diagonal, and similarly a negative value includes just as many diagonals below the main diagonal. 
-- The main diagonal are the set of indices \((i,i)\) for \(i\) \(\in [0,\min(d_1,d_2)-1]\) where \(d_1\) and \(d_2 \) are the dimensions of the matrix.
triu
  :: Int -- ^ diagonal
  -> Tensor -- ^ input
  -> Tensor -- ^ output
triu diagonal input = unsafePerformIO $ (cast2 ATen.triu_tl) input diagonal

-- | Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
-- The lower triangular part of the matrix is defined as the elements on and below the diagonal.
-- The argument diagonal controls which diagonal to consider. If diagonal = 0, all elements on and below the main diagonal are retained. 
-- A positive value includes just as many diagonals above the main diagonal, and similarly a negative value excludes just as many diagonals below the main diagonal. 
-- The main diagonals are the set of indices \((i,i)\) for \(i\) \(\in [0,\min(d_1,d_2)-1]\) where \(d_1\) and \(d_2 \) are the dimensions of the matrix.
tril
  :: Int -- ^ diagonal
  -> Tensor -- ^ input
  -> Tensor -- ^ output
tril diagonal input = unsafePerformIO $ (cast2 ATen.tril_tl) input diagonal

-- | Returns a new tensor with a dimension of size one inserted at the specified position.
-- The returned tensor shares the same underlying data with this tensor.
-- A dim value within the range [(dim input) - 1, (dim input) + 1) can be used. Negative dim will correspond to unsqueeze applied at dim = dim + (dim input) + 1
unsqueeze
  :: Dim  -- ^ dim
  -> Tensor -- ^ input
  -> Tensor -- ^ output
unsqueeze (Dim d) input = unsafePerformIO $ (cast2 ATen.unsqueeze_tl) input d