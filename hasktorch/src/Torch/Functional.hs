{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Functional
    ( module Torch.Functional
    , Internal.acos
    , Internal.addmv
    , Internal.addr
    , Internal.allclose
    , Internal.argmin
    , Internal.asin
    , Internal.atan
    , Internal.baddbmm
    , Internal.bmm
    , Internal.conj
    , Internal.det
    , Internal.dot
    , Internal.einsum
    , Internal.expm1
    , Internal.ger
    , Internal.logdet
    , Internal.lstsq
    , Internal.mv
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
import qualified Prelude as P

import System.IO.Unsafe
import Foreign.ForeignPtr
import Foreign.C.Types (CBool(..))

import Torch.Dimname
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

instance Eq Tensor where
    (==) t t' = all (t `eq` t')

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

data CeilMode = Ceil | Floor deriving (Eq, Show)

instance Castable CeilMode CBool where -- Word8 == CBool
  cast Ceil f = f 1
  cast Floor f = f 0
  uncast 0 f = f Floor
  uncast 1 f = f Ceil

instance Castable Reduction Int64 where
  cast ReduceNone f = f 0
  cast ReduceMean f = f 1
  cast ReduceSum f = f 2
  uncast 0 f = f ReduceNone
  uncast 1 f = f ReduceMean
  uncast _ f = f ReduceSum

data Diag = Diag Int

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

-- | Element wise division of input tensor by other tensor and returns a new resulting tensor
div
  :: Tensor -- ^ input
  -> Tensor -- ^ other
  -> Tensor
div a b = unsafePerformIO $ (cast2 ATen.div_tt) a b

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

--
-- element-wise transformations / non-linearities
--

-- | Computes the error function of each element
erf
    :: Tensor -- ^ input
    -> Tensor -- ^ output
erf t = unsafePerformIO $ (cast1 ATen.erf_t) t

-- | Computes the complementary error function of each element of input
erfc
  :: Tensor -- ^ input
  -> Tensor -- ^ output
erfc t = unsafePerformIO $ (cast1 ATen.erfc_t) t

-- | Computes the inverse error function of each element of input. The inverse error function is defined in the range (-1, 1)(−1,1) as: erfinv(erf(x)) = x
erfinv
  :: Tensor -- ^ input
  -> Tensor -- ^ output
erfinv t = unsafePerformIO $ (cast1 ATen.erfinv_t) t

-- | Computes the logarithm of the gamma function on input.
lgamma
  :: Tensor -- ^ input
  -> Tensor -- ^ output
lgamma t = unsafePerformIO $ (cast1 ATen.lgamma_t) t

-- | Computes the logarithmic derivative of the gamma function on input.
digamma
  :: Tensor -- ^ input
  -> Tensor -- ^ output
digamma t = unsafePerformIO $ (cast1 ATen.digamma_t) t

-- | Computes the nth derivative of the digamma function on input. n \geq 0n≥0 is called the order of the polygamma function.
polygamma
  :: Int -- ^ n
  -> Tensor -- ^ input
  -> Tensor -- ^ output
polygamma n t = unsafePerformIO $ (cast2 ATen.polygamma_lt) n t

-- | Computes the multivariate log-gamma function with dimension pp element-wise. All elements must be greater than (p-1)/2, otherwise an error would be thrown.
mvlgamma
  :: Int -- ^ p
  -> Tensor -- ^ input
  -> Tensor -- ^ output
mvlgamma p t = unsafePerformIO $ (cast2 ATen.mvlgamma_tl) t p

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

-- | Returns a new tensor with the natural logarithm of the elements of input.
log
  :: Tensor -- ^ input
  -> Tensor -- ^ output
log _self = unsafePerformIO $ (cast1 ATen.log_t) _self

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

--
-- infix operators
--

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

isclose
  :: Double -- ^ rtol
  -> Double -- ^ atol
  -> Bool -- ^ equal_nan
  -> Tensor -- ^ self
  -> Tensor -- ^ other
  -> Tensor
isclose rtol atol equalNan self other = unsafePerformIO $ (cast5 ATen.isclose_ttddb) self other rtol atol equalNan

isnan
  :: Tensor -- ^ self
  -> Tensor -- a new tensor with boolean elements representing if each element is NaN or not.
isnan t = unsafePerformIO $ (cast1 ATen.isnan_t) t

isNonzero
  :: Tensor -- ^ self
  -> Bool
isNonzero _self = unsafePerformIO $ (cast1 ATen.is_nonzero_t) _self

isSameSize
  :: Tensor -- ^ self
  -> Tensor -- ^ other
  -> Bool
isSameSize self other = unsafePerformIO $ (cast2 ATen.is_same_size_tt) self other

isSigned
  :: Tensor -- ^ input
  -> Bool -- ^ True if the data type of input is a signed type
isSigned t = unsafePerformIO $ (cast1 ATen.is_signed_t) t

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

-- | squeezeDim
squeezeDim
  :: Int -- ^ dim
  -> Tensor -- ^ input
  -> Tensor -- ^ output
squeezeDim dim t = unsafePerformIO $ (cast2 ATen.squeeze_tl) t dim

--
-- Loss Functions
--

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

-- | This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
binaryCrossEntropyWithLogits
  :: Reduction -- ^ Specifies the reduction to apply to the output
  -> Tensor -- ^ target
  -> Tensor -- ^ weight
  -> Tensor -- ^ pos_weight
  -> Tensor -- ^ input
  -> Tensor -- ^ output
binaryCrossEntropyWithLogits reduction target weight pos_weight input = unsafePerformIO $ (cast5 ATen.binary_cross_entropy_with_logits_ttttl) input target weight pos_weight reduction

-- | Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the @input@ and @target@.
mseLoss
 :: Tensor -- ^ target tensor
 -> Tensor -- ^ input
 -> Tensor -- ^ output
mseLoss target t = unsafePerformIO $ (cast3 ATen.mse_loss_ttl) t target ATen.kMean

-- | The negative log likelihood loss.
nllLoss'
 :: Tensor -- ^ target tensor
 -> Tensor -- ^ input
 -> Tensor -- ^ output
nllLoss' target t = unsafePerformIO $ (cast5 ATen.nll_loss_tttll) t target weight ReduceMean (-100 :: Int)
    where
        nClass = (shape t) !! 1 -- TODO nicer runtime error if input dimensions don't conform
        weight = toDevice (device target) $ ones' [nClass]

-- | Returns cosine similarity between x1 and x2, computed along dim.
cosineSimilarity
    :: Dim -- ^ dimension of vectors (default=1)
    -> Double -- ^ small value to avoid division by 0 (default=1e-8)
    -> Tensor -- ^ x1
    -> Tensor -- ^ x2
    -> Tensor -- ^ output
cosineSimilarity (Dim dim) eps x1 x2 =
    unsafePerformIO $ (cast4 ATen.cosine_similarity_ttld) x1 x2 dim eps

-- | Returns cosine similarity with defaulted options.
cosineSimilarity'
    :: Tensor -- ^ x1
    -> Tensor -- ^ x2
    -> Tensor -- ^ output
cosineSimilarity' x1 x2 =
    unsafePerformIO $
        (cast4 ATen.cosine_similarity_ttld) x1 x2 (1 :: Int) (1e-8 :: Double)

-- | The Connectionist Temporal Classification loss.
-- Calculates loss between a continuous (unsegmented) time series and a target sequence.
-- CTCLoss sums over the probability of possible alignments of input to target,
-- producing a loss value which is differentiable with respect to each input node.
-- The alignment of input to target is assumed to be “many-to-one”, which limits
-- the length of the target sequence such that it must be \leq≤ the input length.
ctcLoss
    :: Bool -- ^ zero_infinity - Whether to zero infinite losses and the associated gradients (False by default). Infinite losses mainly occur when the inputs are too short to be aligned to the targets.
    -> Int -- ^ blank label
    -> Reduction -- ^ reduction
    -> [Int] -- ^ input_lengths
    -> [Int] -- ^ target_lengths
    -> Tensor -- ^ log_probs
    -> Tensor -- ^ targets
    -> Tensor -- ^ output
ctcLoss zeroInfinity blank reduction inputLengths targetLengths logProbs targets  = unsafePerformIO $ (cast7 ATen.ctc_loss_ttllllb) logProbs targets inputLengths targetLengths blank reduction zeroInfinity

-- | Returns CTC loss with defaulted options.
ctcLoss'
    :: Reduction -- ^ reduction
    -> [Int] -- ^ input lengths
    -> [Int] -- ^ target lengths
    -> Tensor -- ^ log probs
    -> Tensor -- ^ targets
    -> Tensor -- ^ output
ctcLoss' reduction inputLengths targetLengths logProbs targets  = unsafePerformIO $ (cast7 ATen.ctc_loss_ttllllb) logProbs targets inputLengths targetLengths blank reduction zeroInfinity
    where
        blank = 0 :: Int
        zeroInfinity = False

-- | Measures the loss given an input tensor xx and a labels tensor yy (containing 1 or -1).
-- This is usually used for measuring whether two inputs are similar or dissimilar,
-- e.g. using the L1 pairwise distance as xx,
-- and is typically used for learning nonlinear embeddings or semi-supervised learning.
hingeEmbeddingLoss
  :: Double -- ^ margin
  -> Reduction -- ^ reduction
  -> Tensor -- ^ target
  -> Tensor -- ^ self
  -> Tensor -- ^ output
hingeEmbeddingLoss margin reduction target t = unsafePerformIO $ (cast4 ATen.hinge_embedding_loss_ttdl) t target margin reduction

marginRankingLoss
  :: Tensor -- ^ input1
  -> Tensor -- ^ input2
  -> Tensor -- ^ target
  -> Double -- ^ margin
  -> Reduction -- ^ reduction
  -> Tensor -- ^ output
marginRankingLoss input1 input2 target margin reduction = unsafePerformIO $ (cast5 ATen.margin_ranking_loss_tttdl) input1 input2 target margin reduction

-- | The 2D negative log likelihood loss
nllLoss2D
  :: Reduction -- reduction
  -> Int -- ignore_index
  -> Tensor -- input
  -> Tensor -- target
  -> Tensor -- weight
  -> Tensor -- output
nllLoss2D reduction ignoreindex input target weight = unsafePerformIO $ (cast5 ATen.nll_loss2d_tttll) input target weight reduction ignoreindex

-- | Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between input \(x\) (a 2D mini-batch Tensor) and output \(y\) (which is a 1D tensor of target class indices)
multiMarginLoss
  :: Reduction -- ^ reduction
  -> Float -- ^ p
  -> Float -- ^ margin
  -> Tensor -- ^ input
  -> Tensor -- ^ target
  -> Tensor -- ^ weight
  -> Tensor -- ^ output
multiMarginLoss reduction p margin input target weight = unsafePerformIO $ (cast6 ATen.multi_margin_loss_ttsstl) input target p margin weight reduction

-- | Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy, between input \(x\) and target \(y\) of size \((N,C)\) .
multiLabelMarginLoss
  :: Reduction -- reduction
  -> Tensor -- input
  -> Tensor -- target
  -> Tensor -- output
multiLabelMarginLoss reduction input target = unsafePerformIO $ (cast3 ATen.multilabel_margin_loss_ttl) input target reduction

-- | The Kullback-Leibler divergence Loss
-- KL divergence is a useful distance measure for continuous distributions and is often useful when performing direct regression over the space of (discretely sampled) continuous output distributions.
-- As with NLLLoss, the input given is expected to contain log-probabilities and is not restricted to a 2D Tensor. The targets are interpreted as probabilities by default, but could be considered as log-probabilities with log_target set to True.
-- This criterion expects a target Tensor of the same size as the input Tensor.
klDiv
  :: Reduction
  -> Tensor -- ^ self
  -> Tensor -- ^ target
  -> Tensor -- ^ output
klDiv reduction self target = unsafePerformIO $ (cast3 ATen.kl_div_ttl) self target reduction

-- | Creates a criterion that uses a squared term if the absolute element-wise
--  error falls below 1 and an L1 term otherwise. It is less sensitive to
-- outliers than the MSELoss and in some cases prevents exploding gradients
-- (e.g. see Fast R-CNN paper by Ross Girshick). Also known as the Huber loss.
smoothL1Loss
  :: Reduction -- ^ reduction
  -> Tensor -- ^ self
  -> Tensor -- ^ target
  -> Tensor -- ^ output
smoothL1Loss reduction self target = unsafePerformIO $ (cast3 ATen.smooth_l1_loss_ttl) self target reduction

-- | Creates a criterion that optimizes a two-class classification logistic loss
--  between input tensor \(x\) and target tensor \(y\) (containing 1 or -1).
softMarginLoss
  :: Reduction -- ^ reduction
  -> Tensor -- ^ input
  -> Tensor -- ^ target
  -> Tensor -- ^ output
softMarginLoss reduction input target = unsafePerformIO $ (cast3 ATen.soft_margin_loss_ttl) input target reduction

--
-- Pooling
--

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
adaptiveMaxPool2d outputSize self =
    unsafePerformIO $ (cast2 ATen.adaptive_max_pool2d_tl)
        self outputSize

-- | Applies a 3D adaptive max pooling over an input signal composed of several input planes
adaptiveMaxPool3d
    :: (Int, Int) -- ^ output size
    -> Tensor -- ^ input
    -> (Tensor, Tensor)
adaptiveMaxPool3d outputSize input = unsafePerformIO $ (cast2 ATen.adaptive_max_pool3d_tl) input outputSize


-- | maxPool1dWithIndices
maxPool1dWithIndices
    :: Int -- ^ kernel size
    -> Int -- ^ stride
    -> Int -- ^ padding
    -> Int -- ^ dilation
    -> CeilMode -- ^ ceil mode
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
    -> CeilMode -- ^ ceil mode
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
    -> CeilMode -- ^ ceil mode
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
    -> CeilMode -- ^ ceil mode
    -> Tensor -- ^ input
    -> Tensor -- ^ output
maxPool3d kernelSize stride padding dilation ceilMode self =
    unsafePerformIO $ (cast6 ATen.max_pool3d_tllllb)
        self kernelSize stride padding dilation ceilMode

-- | Calculates resulting dimensions from a 2d maxpool operation
-- see https://pytorch.org/docs/master/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
maxPool2dDim
    :: (Int, Int) -- ^ kernel size
    -> (Int, Int) -- ^ stride
    -> (Int, Int) -- ^ padding
    -> (Int, Int) -- ^ dilation
    -> CeilMode   -- ^ Ceiling or Floor
    -> (Int, Int) -- ^ image dimensions
    -> (Int, Int) -- ^ height, width after maxPool
maxPool2dDim kernelSize stride padding dilation ceilMode imgDim
    = (calc fst, calc snd)
    where
        trunc Ceil = P.ceiling
        trunc Floor = P.floor
        calc f' =
            let f = (fromIntegral . f' :: (Int, Int) -> Float) in
            (trunc ceilMode) $ (f imgDim
            + 2 * f padding
            - (f dilation) * ((f kernelSize) - 1)
            - 1) / (f stride) + 1

-- | Applies a 1D average pooling over an input signal composed of several input planes.
avgPool1d
  :: Int -- ^ kernel size
  -> Int -- ^ stride
  -> Int -- ^ padding
  -> CeilMode -- ^ ceil mode
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
    avgPool1d kernelSize stride padding Floor True input

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

--
-- matrix solvers
--

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

--
-- dropout
--

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

--
-- Element-wise logical operators
--

-- | Computes the bitwise NOT of the given input tensor. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical NOT.
bitwiseNot
    :: Tensor -- ^ input
    -> Tensor -- ^ output
bitwiseNot input = unsafePerformIO $ cast1 ATen.bitwise_not_t input

-- | Computes the element-wise logical NOT of the given input tensor. If not specified, the output tensor will have the bool dtype. If the input tensor is not a bool tensor, zeros are treated as False and non-zeros are treated as True.
logicalNot
  :: Tensor -- ^ input
  -> Tensor -- ^ output
logicalNot t = unsafePerformIO $ (cast1 ATen.logical_not_t) t

logicalXor
  :: Tensor -- ^ self
  -> Tensor -- ^ other
  -> Tensor
logicalXor self other = unsafePerformIO $ (cast2 ATen.logical_xor_tt) self other

logicalAnd
  :: Tensor -- ^ self
  -> Tensor -- ^ other
  -> Tensor
logicalAnd self other = unsafePerformIO $ (cast2 ATen.logical_and_tt) self other

logicalOr
  :: Tensor -- ^ self
  -> Tensor -- ^ other
  -> Tensor
logicalOr self other = unsafePerformIO $ (cast2 ATen.logical_or_tt) self other

-- | Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
cat
  :: Dim -- ^ dimension
  -> [Tensor] -- ^ list of tensors to concatenate
  -> Tensor -- ^ output tensor
cat (Dim d) tensors = unsafePerformIO $ cast2 ATen.cat_ll tensors d

index
  :: [Tensor] -- ^ indices
  -> Tensor -- ^ input
  -> Tensor -- ^ output
index _indices _self = unsafePerformIO $ (cast2 ATen.index_tl) _self _indices

-- Copies the elements of tensor into the self tensor (out-of-place) by selecting the indices in the order given in index.
-- For example, if dim == 0 and index[i] == j, then the ith row of tensor is copied to the jth row of self.
-- The dimth dimension of tensor must have the same size as the length of index (which must be a vector), and all other dimensions must match self, or an error will be raised.
indexCopy
  :: Int -- ^ dim
  -> Tensor -- ^ index
  -> Tensor -- ^ source
  -> Tensor -- ^ input
  -> Tensor -- ^ output
indexCopy dim index source t = unsafePerformIO $ (cast4 ATen.index_copy_tltt) t dim index source

indexCopyWithDimname
  :: Dimname -- ^ dim
  -> Tensor -- ^ index
  -> Tensor -- ^ source
  -> Tensor -- ^ input
  -> Tensor -- ^ output
indexCopyWithDimname dim index source t = unsafePerformIO $ (cast4 ATen.index_copy_tntt) t dim index source

-- | Puts values from the tensor value into the input tensor (out-of-place)
-- using the indices specified in indices (which is a tuple of Tensors).
-- The expression tensor.index_put_(indices, value) is equivalent to tensor[indices] = value.
-- If accumulate is True, the elements in value are added to self. If accumulate is False, the behavior is undefined if indices contain duplicate elements.
indexPut
  :: Bool -- ^ accumulate
  -> [Tensor] -- ^ indices
  -> Tensor -- ^ values
  -> Tensor -- ^ input
  -> Tensor -- ^ output
indexPut accumulate indices values self = unsafePerformIO $ (cast4 ATen.index_put_tltb) self indices values accumulate

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

--
-- convolutions
--

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
    ::  Diag -- ^ diagonal
    ->  Tensor -- ^ input
    ->  Tensor -- ^ output
diag (Diag index) t = unsafePerformIO $ (cast2 ATen.tensor_diag_l) t index

--
diagEmbed
  :: Diag -- ^ offset
  -> Dim -- ^ dim1
  -> Dim -- ^ dim2
  -> Tensor -- ^ self
  -> Tensor
diagEmbed (Diag offset) (Dim dim1) (Dim dim2) t = unsafePerformIO $ (cast4 ATen.diag_embed_tlll) t offset dim1 dim2

-- | If input is a vector (1-D tensor), then returns a 2-D square tensor with the elements of input as the diagonal.
-- If input is a tensor with more than one dimension, then returns a 2-D tensor with diagonal elements equal to a flattened input.
-- The argument offset controls which diagonal to consider:
--  If offset = 0, it is the main diagonal.
--  If offset > 0, it is above the main diagonal.
--  If offset < 0, it is below the main diagonal.
diagflat
    :: Diag -- ^ offset
    -> Tensor -- ^ self
    -> Tensor -- ^ output
diagflat (Diag offset) t = unsafePerformIO $ (cast2 ATen.diagflat_tl) t offset

-- | Returns a partial view of input with the its diagonal elements with respect to dim1 and dim2 appended as a dimension at the end of the shape.
-- Applying diagEmbed to the output of this function with the same arguments yields a diagonal matrix with the diagonal entries of the input. However, diagEmbed has different default dimensions, so those need to be explicitly specified.
diagonal
  :: Diag -- ^ offset
  -> Dim -- ^ dim1
  -> Dim -- ^ dim2
  -> Tensor -- ^ input
  -> Tensor -- ^ output
diagonal (Diag offset) (Dim dim1) (Dim dim2) t = unsafePerformIO $ (cast4 ATen.diagonal_tlll) t offset dim1 dim2


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

-- | Applies the soft shrinkage function elementwise
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

-- | Returns the log of summed exponentials of each row of the input tensor in the given dimension dim. The computation is numerically stabilized.
logsumexp
  :: Bool -- ^ keepdim
  -> Int -- ^ dim
  -> Tensor -- ^ input
  -> Tensor -- ^ output
logsumexp keepdim dim t = unsafePerformIO $ (cast3 ATen.logsumexp_tlb) t dim keepdim

-- | Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
-- The upper triangular part of the matrix is defined as the elements on and above the diagonal.
-- The argument diagonal controls which diagonal to consider. If diagonal = 0, all elements on and above the main diagonal are retained.
-- A positive value excludes just as many diagonals above the main diagonal, and similarly a negative value includes just as many diagonals below the main diagonal.
-- The main diagonal are the set of indices \((i,i)\) for \(i\) \(\in [0,\min(d_1,d_2)-1]\) where \(d_1\) and \(d_2 \) are the dimensions of the matrix.
triu
  :: Diag -- ^ diagonal
  -> Tensor -- ^ input
  -> Tensor -- ^ output
triu (Diag diagonal) input = unsafePerformIO $ (cast2 ATen.triu_tl) input diagonal

-- | Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
-- The lower triangular part of the matrix is defined as the elements on and below the diagonal.
-- The argument diagonal controls which diagonal to consider. If diagonal = 0, all elements on and below the main diagonal are retained.
-- A positive value includes just as many diagonals above the main diagonal, and similarly a negative value excludes just as many diagonals below the main diagonal.
-- The main diagonals are the set of indices \((i,i)\) for \(i\) \(\in [0,\min(d_1,d_2)-1]\) where \(d_1\) and \(d_2 \) are the dimensions of the matrix.
tril
  :: Diag -- ^ diagonal
  -> Tensor -- ^ input
  -> Tensor -- ^ output
tril (Diag diagonal) input = unsafePerformIO $ (cast2 ATen.tril_tl) input diagonal

-- | Returns a new tensor with the truncated integer values of the elements of input.
trunc
  :: Tensor -- ^ input
  -> Tensor -- ^ output
trunc input = unsafePerformIO $ (cast1 ATen.trunc_t) input

-- | Returns the unique elements of the input tensor along a dimension.
uniqueDim
  :: Int -- ^ dim
  -> Bool -- ^ sorted
  -> Bool -- ^ return_inverse
  -> Bool -- ^ return_counts
  -> Tensor -- ^ input
  -> (Tensor,Tensor,Tensor) -- ^ output
uniqueDim dim sorted returnInverse returnCounts self = unsafePerformIO $ (cast5 ATen.unique_dim_tlbbb) self dim sorted returnInverse returnCounts

-- | Eliminates all but the first element from every consecutive group of equivalent elements.
-- This function is different from uniqueDim in the sense that this function only eliminates consecutive duplicate values. 
uniqueConsecutive
  :: Bool -- ^ return_inverse
  -> Bool -- ^ return_counts
  -> Int -- ^ dim
  -> Tensor -- ^ input
  -> (Tensor,Tensor,Tensor) -- ^ output
uniqueConsecutive returnInverse returnCounts dim self = unsafePerformIO $ (cast4 ATen.unique_consecutive_tbbl) self returnInverse returnCounts dim

-- | Eliminates all but the first element from every consecutive group of equivalent elements along a dimension.
-- This function is different from uniqueDim in the sense that this function only eliminates consecutive duplicate values. 
uniqueDimConsecutive
  :: Int -- ^ dim
  -> Bool -- ^ return_inverse
  -> Bool -- ^ return_counts
  -> Tensor -- ^ input
  -> (Tensor,Tensor,Tensor) -- ^ output
uniqueDimConsecutive dim returnInverse returnCounts self = unsafePerformIO $ (cast4 ATen.unique_dim_consecutive_tlbb) self dim returnInverse returnCounts

-- | Returns a new tensor with a dimension of size one inserted at the specified position.
-- The returned tensor shares the same underlying data with this tensor.
-- A dim value within the range [(dim input) - 1, (dim input) + 1)] can be used. Negative dim will correspond to unsqueeze applied at dim = dim + (dim input) + 1
unsqueeze
  :: Dim  -- ^ dim
  -> Tensor -- ^ input
  -> Tensor -- ^ output
unsqueeze (Dim d) input = unsafePerformIO $ (cast2 ATen.unsqueeze_tl) input d

-- | Upsamples the input, using bilinear upsampling. Expected inputs are spatial (4 dimensional).
upsampleBilinear2d
  :: (Int,Int) -- ^ output-size
  -> Bool -- ^ align corners
  -> Tensor -- ^ self
  -> Tensor
upsampleBilinear2d (outputHeight, outputWidth) alignCorners input = unsafePerformIO $ (cast3 ATen.upsample_bilinear2d_tlb) input [outputHeight, outputWidth] alignCorners

-- | Splits the tensor into chunks of given size if possible.
split
  :: Int -- ^ split-size
  -> Dim -- ^ dim
  -> Tensor -- ^ self
  -> [Tensor]
split splitSize (Dim d) input = unsafePerformIO $ (cast3 ATen.split_tll) input splitSize d

-- | Creates a criterion that measures the mean absolute error (MAE) between each element in the input \(x\) and target \(y\) .
l1Loss
  ::  Reduction -- ^ reduction
  -> Tensor -- ^ input
  -> Tensor -- ^ target
  -> Tensor -- ^ output
l1Loss reduction input target = unsafePerformIO $ (cast3 ATen.l1_loss_ttl) input target reduction

-- | Applies the element-wise function:
-- \(\text{LeakyReLU}(x) = \max(0,x) + \text{negative_slope} ∗ \min(0,x)\)
leakyRelu
  :: Float -- ^ negative slope
  -> Tensor -- ^ input
  -> Tensor -- ^ output
leakyRelu negSlope input = unsafePerformIO $ (cast2 ATen.leaky_relu_ts) input negSlope

-- | Applies the element-wise function:
-- \(\text{LogSigmoid}(x) = \log(\frac{ 1 }{ 1 + \exp(-x)})\)
logSigmoid
  :: Tensor -- ^ input
  -> Tensor -- ^ output
logSigmoid input = unsafePerformIO $ (cast1 ATen.log_sigmoid_t) input

-- | Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim.
-- And indices is the index location of each maximum value found (argmax).
-- If keepdim is True, the output tensors are of the same size as input except in the dimension dim where they are of size 1.
-- Otherwise, dim is squeezed , resulting in the output tensors having 1 fewer dimension than input.
maxDim
  :: Dim -- ^ dimension
  -> KeepDim -- ^ keepdim
  -> Tensor -- ^ input
  -> (Tensor, Tensor) -- ^ output
maxDim (Dim d) k input = unsafePerformIO $ (cast3 ATen.max_tlb) input d (keepdim k)

-- | Returns a namedtuple (values, indices) where values is the minimum value of each row of the input tensor in the given dimension dim.
-- And indices is the index location of each minimum value found (argmin).
-- If keepdim is True, the output tensors are of the same size as input except in the dimension dim where they are of size 1.
-- Otherwise, dim is squeezed, resulting in the output tensors having 1 fewer dimension than input.
minDim
  :: Dim -- ^ dimension
  -> KeepDim -- ^ keepdim
  -> Tensor -- ^ input
  -> (Tensor, Tensor)
minDim (Dim d) k input = unsafePerformIO $ (cast3 ATen.min_tlb) input d (keepdim k)

-- | Returns the mean value of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them.
-- If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1.
-- Otherwise, dim is squeezed (see torch.squeeze()), resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
meanDim
  :: Dim -- ^ dimension
  -> KeepDim -- ^ keepdim
  -> DType -- ^ dtype
  -> Tensor -- ^ input
  -> Tensor -- ^ output
meanDim (Dim d) k dtype input = unsafePerformIO $ (cast4 ATen.mean_tlbs) input d (keepdim k) dtype

-- | Returns a namedtuple (values, indices) where values is the median value of each row of the input tensor in the given dimension dim.
-- And indices is the index location of each median value found.
-- By default, dim is the last dimension of the input tensor.
-- If keepdim is True, the output tensors are of the same size as input except in the dimension dim where they are of size 1.
-- Otherwise, dim is squeezed (see torch.squeeze()), resulting in the outputs tensor having 1 fewer dimension than input.
medianDim
  :: Dim -- ^ dimension
  -> KeepDim -- ^ keepdim
  -> Tensor -- ^ input
  -> (Tensor, Tensor) -- ^ output
medianDim (Dim d) k input = unsafePerformIO $ (cast3 ATen.median_tlb) input d (keepdim k)

-- | Returns the matrix product of the NN 2-D tensors.
-- This product is efficiently computed using the matrix chain order algorithm which selects the order in which incurs the lowest cost in terms of arithmetic operations.
-- Note that since this is a function to compute the product, NN needs to be greater than or equal to 2; if equal to 2 then a trivial matrix-matrix product is returned.
-- If NN is 1, then this is a no-op - the original matrix is returned as is.
chainMatmul
  :: [Tensor] -- ^ list of tensors
  -> Tensor -- ^ output
chainMatmul tensors = unsafePerformIO $ (cast1 ATen.chain_matmul_l) tensors

-- | Applies element-wise the function \(\text{GELU}(x) = x * \Phi(x)\)
-- where \(\Phi(x)\) is the Cumulative Distribution Function for Gaussian Distribution.
gelu
  :: Tensor -- ^ input
  -> Tensor -- ^ output
gelu input = unsafePerformIO $ (cast1 ATen.gelu_t) input

-- | The gated linear unit. Computes:
-- \(\text{GLU}(a, b) = a \otimes \sigma(b)\)
-- where input is split in half along dim to form a and b, \(\sigma\) is the sigmoid function and \(\otimes\) is the element-wise product between matrices.
glu
  :: Dim -- ^ dimension
  -> Tensor -- ^ input
  -> Tensor -- ^ output
glu (Dim d) input = unsafePerformIO $ (cast2 ATen.glu_tl) input d

-- | Returns the standard-deviation and mean of all elements in the input tensor.
-- If unbiased is False, then the standard-deviation will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.
stdMean
  :: Bool -- ^ unbiased
  -> Tensor -- ^ input
  -> (Tensor,Tensor) -- ^ output
stdMean unbiased input = unsafePerformIO $ (cast2 ATen.std_mean_tb) input unbiased

-- | Returns the standard-deviation and mean of each row of the input tensor in the dimension dim. If dim is a list of dimensions, reduce over all of them.
-- If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1.
-- Otherwise, dim is squeezed, resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
-- If unbiased is False, then the standard-deviation will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.
stdMeanDim
  :: Dim -- ^ dimension
  -> Bool -- ^ unbiased
  -> KeepDim -- ^ whether the output tensor has dim retained or not
  -> Tensor -- ^ input
  -> (Tensor,Tensor) -- ^ output
stdMeanDim (Dim d) unbiased k input = unsafePerformIO $ (cast4 ATen.std_mean_tlbb) input d unbiased (keepdim k)
