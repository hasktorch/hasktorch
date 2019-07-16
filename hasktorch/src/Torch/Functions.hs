{-# LANGUAGE TypeApplications #-}

module Torch.Functions where

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

import Torch.Tensor
import Torch.DType

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

gels :: Tensor -> Tensor -> (Tensor,Tensor)
gels _B _A = unsafePerformIO $ (cast2 ATen.gels_tt) _B _A

transpose :: Tensor -> Int -> Int -> Tensor
transpose t a b = unsafePerformIO $ (cast3 ATen.transpose_tll) t a b

-- transpose special case for a 2D tensor
transpose2D :: Tensor -> Tensor
transpose2D t = transpose t 0 1

diag :: Tensor -> Int -> Tensor
diag t index = unsafePerformIO $ (cast2 ATen.tensor_diag_l) t index


---



dropout ::  Tensor -> Double -> Bool ->  Tensor
dropout _input _p _train = unsafePerformIO $ (cast3 Managed.dropout_tdb) _input _p _train

feature_dropout ::  Tensor -> Double -> Bool ->  Tensor
feature_dropout _input _p _train = unsafePerformIO $ (cast3 Managed.feature_dropout_tdb) _input _p _train

alpha_dropout ::  Tensor -> Double -> Bool ->  Tensor
alpha_dropout _input _p _train = unsafePerformIO $ (cast3 Managed.alpha_dropout_tdb) _input _p _train

feature_alpha_dropout ::  Tensor -> Double -> Bool ->  Tensor
feature_alpha_dropout _input _p _train = unsafePerformIO $ (cast3 Managed.feature_alpha_dropout_tdb) _input _p _train

abs ::  Tensor ->  Tensor
abs _self = unsafePerformIO $ (cast1 Managed.abs_t) _self

acos ::  Tensor ->  Tensor
acos _self = unsafePerformIO $ (cast1 Managed.acos_t) _self

avg_pool1d ::  Tensor ->  [Int] ->  [Int] ->  [Int] -> Bool -> Bool ->  Tensor
avg_pool1d _self _kernel_size _stride _padding _ceil_mode _count_include_pad = unsafePerformIO $ (cast6 Managed.avg_pool1d_tlllbb) _self _kernel_size _stride _padding _ceil_mode _count_include_pad

adaptive_avg_pool1d ::  Tensor ->  [Int] ->  Tensor
adaptive_avg_pool1d _self _output_size = unsafePerformIO $ (cast2 Managed.adaptive_avg_pool1d_tl) _self _output_size

adaptive_max_pool1d ::  Tensor ->  [Int] ->  (Tensor,Tensor)
adaptive_max_pool1d _self _output_size = unsafePerformIO $ (cast2 Managed.adaptive_max_pool1d_tl) _self _output_size

addmv ::  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Tensor
addmv _self _mat _vec _beta _alpha = unsafePerformIO $ (cast5 Managed.addmv_tttss) _self _mat _vec _beta _alpha

addr ::  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Tensor
addr _self _vec1 _vec2 _beta _alpha = unsafePerformIO $ (cast5 Managed.addr_tttss) _self _vec1 _vec2 _beta _alpha

affine_grid_generator ::  Tensor ->  [Int] ->  Tensor
affine_grid_generator _theta _size = unsafePerformIO $ (cast2 Managed.affine_grid_generator_tl) _theta _size

affine_grid_generator_backward ::  Tensor ->  [Int] ->  Tensor
affine_grid_generator_backward _grad _size = unsafePerformIO $ (cast2 Managed.affine_grid_generator_backward_tl) _grad _size

allclose ::  Tensor ->  Tensor -> Double -> Double -> Bool -> Bool
allclose _self _other _rtol _atol _equal_nan = unsafePerformIO $ (cast5 Managed.allclose_ttddb) _self _other _rtol _atol _equal_nan

argmax ::  Tensor -> Int64 -> Bool ->  Tensor
argmax _self _dim _keepdim = unsafePerformIO $ (cast3 Managed.argmax_tlb) _self _dim _keepdim

argmin ::  Tensor -> Int64 -> Bool ->  Tensor
argmin _self _dim _keepdim = unsafePerformIO $ (cast3 Managed.argmin_tlb) _self _dim _keepdim

as_strided ::  Tensor ->  [Int] ->  [Int] -> Int64 ->  Tensor
as_strided _self _size _stride _storage_offset = unsafePerformIO $ (cast4 Managed.as_strided_tlll) _self _size _stride _storage_offset

asin ::  Tensor ->  Tensor
asin _self = unsafePerformIO $ (cast1 Managed.asin_t) _self

atan ::  Tensor ->  Tensor
atan _self = unsafePerformIO $ (cast1 Managed.atan_t) _self

baddbmm ::  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Tensor
baddbmm _self _batch1 _batch2 _beta _alpha = unsafePerformIO $ (cast5 Managed.baddbmm_tttss) _self _batch1 _batch2 _beta _alpha

batch_norm ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor -> Bool -> Double -> Double -> Bool ->  Tensor
batch_norm _input _weight _bias _running_mean _running_var _training _momentum _eps _cudnn_enabled = unsafePerformIO $ (cast9 Managed.batch_norm_tttttbddb) _input _weight _bias _running_mean _running_var _training _momentum _eps _cudnn_enabled

bilinear ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor
bilinear _input1 _input2 _weight _bias = unsafePerformIO $ (cast4 Managed.bilinear_tttt) _input1 _input2 _weight _bias

binary_cross_entropy_with_logits ::  Tensor ->  Tensor ->  Tensor ->  Tensor -> Int64 ->  Tensor
binary_cross_entropy_with_logits _self _target _weight _pos_weight _reduction = unsafePerformIO $ (cast5 Managed.binary_cross_entropy_with_logits_ttttl) _self _target _weight _pos_weight _reduction

binary_cross_entropy_with_logits_backward ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor -> Int64 ->  Tensor
binary_cross_entropy_with_logits_backward _grad_output _self _target _weight _pos_weight _reduction = unsafePerformIO $ (cast6 Managed.binary_cross_entropy_with_logits_backward_tttttl) _grad_output _self _target _weight _pos_weight _reduction

bincount ::  Tensor ->  Tensor -> Int64 ->  Tensor
bincount _self _weights _minlength = unsafePerformIO $ (cast3 Managed.bincount_ttl) _self _weights _minlength

bmm ::  Tensor ->  Tensor ->  Tensor
bmm _self _mat2 = unsafePerformIO $ (cast2 Managed.bmm_tt) _self _mat2

broadcast_tensors ::  [Tensor] ->  [Tensor]
broadcast_tensors _tensors = unsafePerformIO $ (cast1 Managed.broadcast_tensors_l) _tensors

cat ::  [Tensor] -> Int64 ->  Tensor
cat _tensors _dim = unsafePerformIO $ (cast2 Managed.cat_ll) _tensors _dim

ceil ::  Tensor ->  Tensor
ceil _self = unsafePerformIO $ (cast1 Managed.ceil_t) _self

chain_matmul ::  [Tensor] ->  Tensor
chain_matmul _matrices = unsafePerformIO $ (cast1 Managed.chain_matmul_l) _matrices

chunk ::  Tensor -> Int64 -> Int64 ->  [Tensor]
chunk _self _chunks _dim = unsafePerformIO $ (cast3 Managed.chunk_tll) _self _chunks _dim

clamp ::  Tensor ->  Double ->  Double ->  Tensor
clamp _self _min _max = unsafePerformIO $ (cast3 Managed.clamp_tss) _self _min _max

clamp_max ::  Tensor ->  Double ->  Tensor
clamp_max _self _max = unsafePerformIO $ (cast2 Managed.clamp_max_ts) _self _max

clamp_min ::  Tensor ->  Double ->  Tensor
clamp_min _self _min = unsafePerformIO $ (cast2 Managed.clamp_min_ts) _self _min

cudnn_is_acceptable ::  Tensor -> Bool
cudnn_is_acceptable _self = unsafePerformIO $ (cast1 Managed.cudnn_is_acceptable_t) _self

constant_pad_nd ::  Tensor ->  [Int] ->  Double ->  Tensor
constant_pad_nd _self _pad _value = unsafePerformIO $ (cast3 Managed.constant_pad_nd_tls) _self _pad _value

convolution ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Bool ->  [Int] -> Int64 ->  Tensor
convolution _input _weight _bias _stride _padding _dilation _transposed _output_padding _groups = unsafePerformIO $ (cast9 Managed.convolution_tttlllbll) _input _weight _bias _stride _padding _dilation _transposed _output_padding _groups

conv1d ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 ->  Tensor
conv1d _input _weight _bias _stride _padding _dilation _groups = unsafePerformIO $ (cast7 Managed.conv1d_tttllll) _input _weight _bias _stride _padding _dilation _groups

conv2d ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 ->  Tensor
conv2d _input _weight _bias _stride _padding _dilation _groups = unsafePerformIO $ (cast7 Managed.conv2d_tttllll) _input _weight _bias _stride _padding _dilation _groups

conv3d ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 ->  Tensor
conv3d _input _weight _bias _stride _padding _dilation _groups = unsafePerformIO $ (cast7 Managed.conv3d_tttllll) _input _weight _bias _stride _padding _dilation _groups

conv_tbc ::  Tensor ->  Tensor ->  Tensor -> Int64 ->  Tensor
conv_tbc _self _weight _bias _pad = unsafePerformIO $ (cast4 Managed.conv_tbc_tttl) _self _weight _bias _pad

conv_tbc_backward ::  Tensor ->  Tensor ->  Tensor ->  Tensor -> Int64 ->  (Tensor,Tensor,Tensor)
conv_tbc_backward _self _input _weight _bias _pad = unsafePerformIO $ (cast5 Managed.conv_tbc_backward_ttttl) _self _input _weight _bias _pad

conv_transpose1d ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 ->  [Int] ->  Tensor
conv_transpose1d _input _weight _bias _stride _padding _output_padding _groups _dilation = unsafePerformIO $ (cast8 Managed.conv_transpose1d_tttlllll) _input _weight _bias _stride _padding _output_padding _groups _dilation

conv_transpose2d ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 ->  [Int] ->  Tensor
conv_transpose2d _input _weight _bias _stride _padding _output_padding _groups _dilation = unsafePerformIO $ (cast8 Managed.conv_transpose2d_tttlllll) _input _weight _bias _stride _padding _output_padding _groups _dilation

conv_transpose3d ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 ->  [Int] ->  Tensor
conv_transpose3d _input _weight _bias _stride _padding _output_padding _groups _dilation = unsafePerformIO $ (cast8 Managed.conv_transpose3d_tttlllll) _input _weight _bias _stride _padding _output_padding _groups _dilation

cos ::  Tensor ->  Tensor
cos _self = unsafePerformIO $ (cast1 Managed.cos_t) _self

cosh ::  Tensor ->  Tensor
cosh _self = unsafePerformIO $ (cast1 Managed.cosh_t) _self

cosine_embedding_loss ::  Tensor ->  Tensor ->  Tensor -> Double -> Int64 ->  Tensor
cosine_embedding_loss _input1 _input2 _target _margin _reduction = unsafePerformIO $ (cast5 Managed.cosine_embedding_loss_tttdl) _input1 _input2 _target _margin _reduction

cudnn_affine_grid_generator ::  Tensor -> Int64 -> Int64 -> Int64 -> Int64 ->  Tensor
cudnn_affine_grid_generator _theta _N _C _H _W = unsafePerformIO $ (cast5 Managed.cudnn_affine_grid_generator_tllll) _theta _N _C _H _W

cudnn_affine_grid_generator_backward ::  Tensor -> Int64 -> Int64 -> Int64 -> Int64 ->  Tensor
cudnn_affine_grid_generator_backward _grad _N _C _H _W = unsafePerformIO $ (cast5 Managed.cudnn_affine_grid_generator_backward_tllll) _grad _N _C _H _W

cudnn_batch_norm ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor -> Bool -> Double -> Double ->  (Tensor,Tensor,Tensor)
cudnn_batch_norm _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon = unsafePerformIO $ (cast8 Managed.cudnn_batch_norm_tttttbdd) _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon

cudnn_batch_norm_backward ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor -> Double ->  (Tensor,Tensor,Tensor)
cudnn_batch_norm_backward _input _grad_output _weight _running_mean _running_var _save_mean _save_var _epsilon = unsafePerformIO $ (cast8 Managed.cudnn_batch_norm_backward_tttttttd) _input _grad_output _weight _running_mean _running_var _save_mean _save_var _epsilon

cudnn_convolution ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
cudnn_convolution _self _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 Managed.cudnn_convolution_tttllllbb) _self _weight _bias _padding _stride _dilation _groups _benchmark _deterministic

cudnn_convolution_backward_input ::  [Int] ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
cudnn_convolution_backward_input _self_size _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 Managed.cudnn_convolution_backward_input_lttllllbb) _self_size _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic

cudnn_convolution_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
cudnn_convolution_backward _self _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic _output_mask = unsafePerformIO $ (cast10 Managed.cudnn_convolution_backward_tttllllbba) _self _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic _output_mask

cudnn_convolution_backward_bias ::  Tensor ->  Tensor
cudnn_convolution_backward_bias _grad_output = unsafePerformIO $ (cast1 Managed.cudnn_convolution_backward_bias_t) _grad_output

cudnn_convolution_backward_weight ::  [Int] ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
cudnn_convolution_backward_weight _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 Managed.cudnn_convolution_backward_weight_lttllllbb) _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic

cudnn_convolution_transpose ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
cudnn_convolution_transpose _self _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast10 Managed.cudnn_convolution_transpose_tttlllllbb) _self _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic

cudnn_convolution_transpose_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
cudnn_convolution_transpose_backward _self _grad_output _weight _padding _output_padding _stride _dilation _groups _benchmark _deterministic _output_mask = unsafePerformIO $ (cast11 Managed.cudnn_convolution_transpose_backward_tttlllllbba) _self _grad_output _weight _padding _output_padding _stride _dilation _groups _benchmark _deterministic _output_mask

cudnn_convolution_transpose_backward_bias ::  Tensor ->  Tensor
cudnn_convolution_transpose_backward_bias _grad_output = unsafePerformIO $ (cast1 Managed.cudnn_convolution_transpose_backward_bias_t) _grad_output

cudnn_convolution_transpose_backward_input ::  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
cudnn_convolution_transpose_backward_input _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast8 Managed.cudnn_convolution_transpose_backward_input_ttllllbb) _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic

cudnn_convolution_transpose_backward_weight ::  [Int] ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
cudnn_convolution_transpose_backward_weight _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 Managed.cudnn_convolution_transpose_backward_weight_lttllllbb) _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic

cudnn_grid_sampler ::  Tensor ->  Tensor ->  Tensor
cudnn_grid_sampler _self _grid = unsafePerformIO $ (cast2 Managed.cudnn_grid_sampler_tt) _self _grid

cudnn_grid_sampler_backward ::  Tensor ->  Tensor ->  Tensor ->  (Tensor,Tensor)
cudnn_grid_sampler_backward _self _grid _grad_output = unsafePerformIO $ (cast3 Managed.cudnn_grid_sampler_backward_ttt) _self _grid _grad_output

det ::  Tensor ->  Tensor
det _self = unsafePerformIO $ (cast1 Managed.det_t) _self

diag_embed ::  Tensor -> Int64 -> Int64 -> Int64 ->  Tensor
diag_embed _self _offset _dim1 _dim2 = unsafePerformIO $ (cast4 Managed.diag_embed_tlll) _self _offset _dim1 _dim2

diagflat ::  Tensor -> Int64 ->  Tensor
diagflat _self _offset = unsafePerformIO $ (cast2 Managed.diagflat_tl) _self _offset

diagonal ::  Tensor -> Int64 -> Int64 -> Int64 ->  Tensor
diagonal _self _offset _dim1 _dim2 = unsafePerformIO $ (cast4 Managed.diagonal_tlll) _self _offset _dim1 _dim2

dot ::  Tensor ->  Tensor ->  Tensor
dot _self _tensor = unsafePerformIO $ (cast2 Managed.dot_tt) _self _tensor

einsum ::  StdString ->  [Tensor] ->  Tensor
einsum _equation _tensors = unsafePerformIO $ (cast2 Managed.einsum_sl) _equation _tensors

embedding ::  Tensor ->  Tensor -> Int64 -> Bool -> Bool ->  Tensor
embedding _weight _indices _padding_idx _scale_grad_by_freq _sparse = unsafePerformIO $ (cast5 Managed.embedding_ttlbb) _weight _indices _padding_idx _scale_grad_by_freq _sparse

embedding_backward ::  Tensor ->  Tensor -> Int64 -> Int64 -> Bool -> Bool ->  Tensor
embedding_backward _grad _indices _num_weights _padding_idx _scale_grad_by_freq _sparse = unsafePerformIO $ (cast6 Managed.embedding_backward_ttllbb) _grad _indices _num_weights _padding_idx _scale_grad_by_freq _sparse

embedding_dense_backward ::  Tensor ->  Tensor -> Int64 -> Int64 -> Bool ->  Tensor
embedding_dense_backward _grad_output _indices _num_weights _padding_idx _scale_grad_by_freq = unsafePerformIO $ (cast5 Managed.embedding_dense_backward_ttllb) _grad_output _indices _num_weights _padding_idx _scale_grad_by_freq

embedding_sparse_backward ::  Tensor ->  Tensor -> Int64 -> Int64 -> Bool ->  Tensor
embedding_sparse_backward _grad _indices _num_weights _padding_idx _scale_grad_by_freq = unsafePerformIO $ (cast5 Managed.embedding_sparse_backward_ttllb) _grad _indices _num_weights _padding_idx _scale_grad_by_freq

embedding_bag ::  Tensor ->  Tensor ->  Tensor -> Bool -> Int64 -> Bool ->  Tensor ->  (Tensor,Tensor,Tensor,Tensor)
embedding_bag _weight _indices _offsets _scale_grad_by_freq _mode _sparse _per_sample_weights = unsafePerformIO $ (cast7 Managed.embedding_bag_tttblbt) _weight _indices _offsets _scale_grad_by_freq _mode _sparse _per_sample_weights

erf ::  Tensor ->  Tensor
erf _self = unsafePerformIO $ (cast1 Managed.erf_t) _self

erfc ::  Tensor ->  Tensor
erfc _self = unsafePerformIO $ (cast1 Managed.erfc_t) _self

exp ::  Tensor ->  Tensor
exp _self = unsafePerformIO $ (cast1 Managed.exp_t) _self

expm1 ::  Tensor ->  Tensor
expm1 _self = unsafePerformIO $ (cast1 Managed.expm1_t) _self

flatten ::  Tensor -> Int64 -> Int64 ->  Tensor
flatten _self _start_dim _end_dim = unsafePerformIO $ (cast3 Managed.flatten_tll) _self _start_dim _end_dim

floor ::  Tensor ->  Tensor
floor _self = unsafePerformIO $ (cast1 Managed.floor_t) _self

frac ::  Tensor ->  Tensor
frac _self = unsafePerformIO $ (cast1 Managed.frac_t) _self

grid_sampler ::  Tensor ->  Tensor -> Int64 -> Int64 ->  Tensor
grid_sampler _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (cast4 Managed.grid_sampler_ttll) _input _grid _interpolation_mode _padding_mode

grid_sampler_2d ::  Tensor ->  Tensor -> Int64 -> Int64 ->  Tensor
grid_sampler_2d _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (cast4 Managed.grid_sampler_2d_ttll) _input _grid _interpolation_mode _padding_mode

grid_sampler_2d_backward ::  Tensor ->  Tensor ->  Tensor -> Int64 -> Int64 ->  (Tensor,Tensor)
grid_sampler_2d_backward _grad_output _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (cast5 Managed.grid_sampler_2d_backward_tttll) _grad_output _input _grid _interpolation_mode _padding_mode

grid_sampler_3d ::  Tensor ->  Tensor -> Int64 -> Int64 ->  Tensor
grid_sampler_3d _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (cast4 Managed.grid_sampler_3d_ttll) _input _grid _interpolation_mode _padding_mode

grid_sampler_3d_backward ::  Tensor ->  Tensor ->  Tensor -> Int64 -> Int64 ->  (Tensor,Tensor)
grid_sampler_3d_backward _grad_output _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (cast5 Managed.grid_sampler_3d_backward_tttll) _grad_output _input _grid _interpolation_mode _padding_mode

hinge_embedding_loss ::  Tensor ->  Tensor -> Double -> Int64 ->  Tensor
hinge_embedding_loss _self _target _margin _reduction = unsafePerformIO $ (cast4 Managed.hinge_embedding_loss_ttdl) _self _target _margin _reduction

ger ::  Tensor ->  Tensor ->  Tensor
ger _self _vec2 = unsafePerformIO $ (cast2 Managed.ger_tt) _self _vec2

group_norm ::  Tensor -> Int64 ->  Tensor ->  Tensor -> Double -> Bool ->  Tensor
group_norm _input _num_groups _weight _bias _eps _cudnn_enabled = unsafePerformIO $ (cast6 Managed.group_norm_tlttdb) _input _num_groups _weight _bias _eps _cudnn_enabled

fft ::  Tensor -> Int64 -> Bool ->  Tensor
fft _self _signal_ndim _normalized = unsafePerformIO $ (cast3 Managed.fft_tlb) _self _signal_ndim _normalized

ifft ::  Tensor -> Int64 -> Bool ->  Tensor
ifft _self _signal_ndim _normalized = unsafePerformIO $ (cast3 Managed.ifft_tlb) _self _signal_ndim _normalized

rfft ::  Tensor -> Int64 -> Bool -> Bool ->  Tensor
rfft _self _signal_ndim _normalized _onesided = unsafePerformIO $ (cast4 Managed.rfft_tlbb) _self _signal_ndim _normalized _onesided

irfft ::  Tensor -> Int64 -> Bool -> Bool ->  [Int] ->  Tensor
irfft _self _signal_ndim _normalized _onesided _signal_sizes = unsafePerformIO $ (cast5 Managed.irfft_tlbbl) _self _signal_ndim _normalized _onesided _signal_sizes

index ::  Tensor ->  [Tensor] ->  Tensor
index _self _indices = unsafePerformIO $ (cast2 Managed.index_tl) _self _indices

index_copy ::  Tensor -> Int64 ->  Tensor ->  Tensor ->  Tensor
index_copy _self _dim _index _source = unsafePerformIO $ (cast4 Managed.index_copy_tltt) _self _dim _index _source

index_put ::  Tensor ->  [Tensor] ->  Tensor -> Bool ->  Tensor
index_put _self _indices _values _accumulate = unsafePerformIO $ (cast4 Managed.index_put_tltb) _self _indices _values _accumulate

instance_norm ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor -> Bool -> Double -> Double -> Bool ->  Tensor
instance_norm _input _weight _bias _running_mean _running_var _use_input_stats _momentum _eps _cudnn_enabled = unsafePerformIO $ (cast9 Managed.instance_norm_tttttbddb) _input _weight _bias _running_mean _running_var _use_input_stats _momentum _eps _cudnn_enabled

inverse ::  Tensor ->  Tensor
inverse _self = unsafePerformIO $ (cast1 Managed.inverse_t) _self

isclose ::  Tensor ->  Tensor -> Double -> Double -> Bool ->  Tensor
isclose _self _other _rtol _atol _equal_nan = unsafePerformIO $ (cast5 Managed.isclose_ttddb) _self _other _rtol _atol _equal_nan

isnan ::  Tensor ->  Tensor
isnan _self = unsafePerformIO $ (cast1 Managed.isnan_t) _self

is_distributed ::  Tensor -> Bool
is_distributed _self = unsafePerformIO $ (cast1 Managed.is_distributed_t) _self

is_floating_point ::  Tensor -> Bool
is_floating_point _self = unsafePerformIO $ (cast1 Managed.is_floating_point_t) _self

is_complex ::  Tensor -> Bool
is_complex _self = unsafePerformIO $ (cast1 Managed.is_complex_t) _self

is_nonzero ::  Tensor -> Bool
is_nonzero _self = unsafePerformIO $ (cast1 Managed.is_nonzero_t) _self

is_same_size ::  Tensor ->  Tensor -> Bool
is_same_size _self _other = unsafePerformIO $ (cast2 Managed.is_same_size_tt) _self _other

is_signed ::  Tensor -> Bool
is_signed _self = unsafePerformIO $ (cast1 Managed.is_signed_t) _self

kl_div ::  Tensor ->  Tensor -> Int64 ->  Tensor
kl_div _self _target _reduction = unsafePerformIO $ (cast3 Managed.kl_div_ttl) _self _target _reduction

kl_div_backward ::  Tensor ->  Tensor ->  Tensor -> Int64 ->  Tensor
kl_div_backward _grad_output _self _target _reduction = unsafePerformIO $ (cast4 Managed.kl_div_backward_tttl) _grad_output _self _target _reduction

kthvalue ::  Tensor -> Int64 -> Int64 -> Bool ->  (Tensor,Tensor)
kthvalue _self _k _dim _keepdim = unsafePerformIO $ (cast4 Managed.kthvalue_tllb) _self _k _dim _keepdim

layer_norm ::  Tensor ->  [Int] ->  Tensor ->  Tensor -> Double -> Bool ->  Tensor
layer_norm _input _normalized_shape _weight _bias _eps _cudnn_enable = unsafePerformIO $ (cast6 Managed.layer_norm_tlttdb) _input _normalized_shape _weight _bias _eps _cudnn_enable

linear ::  Tensor ->  Tensor ->  Tensor ->  Tensor
linear _input _weight _bias = unsafePerformIO $ (cast3 Managed.linear_ttt) _input _weight _bias

mkldnn_linear ::  Tensor ->  Tensor ->  Tensor ->  Tensor
mkldnn_linear _input _weight _bias = unsafePerformIO $ (cast3 Managed.mkldnn_linear_ttt) _input _weight _bias

fbgemm_linear_int8_weight ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Tensor ->  Tensor
fbgemm_linear_int8_weight _input _weight _packed _col_offsets _weight_scale _weight_zero_point _bias = unsafePerformIO $ (cast7 Managed.fbgemm_linear_int8_weight_ttttsst) _input _weight _packed _col_offsets _weight_scale _weight_zero_point _bias

fbgemm_linear_quantize_weight ::  Tensor ->  (Tensor,Tensor,CDouble,Int64)
fbgemm_linear_quantize_weight _input = unsafePerformIO $ (cast1 Managed.fbgemm_linear_quantize_weight_t) _input

fbgemm_pack_quantized_matrix ::  Tensor -> Int64 -> Int64 ->  Tensor
fbgemm_pack_quantized_matrix _input _K _N = unsafePerformIO $ (cast3 Managed.fbgemm_pack_quantized_matrix_tll) _input _K _N

fbgemm_is_cpu_supported :: Bool
fbgemm_is_cpu_supported  = unsafePerformIO $ (cast0 Managed.fbgemm_is_cpu_supported) 

log ::  Tensor ->  Tensor
log _self = unsafePerformIO $ (cast1 Managed.log_t) _self

log10 ::  Tensor ->  Tensor
log10 _self = unsafePerformIO $ (cast1 Managed.log10_t) _self

log1p ::  Tensor ->  Tensor
log1p _self = unsafePerformIO $ (cast1 Managed.log1p_t) _self

log2 ::  Tensor ->  Tensor
log2 _self = unsafePerformIO $ (cast1 Managed.log2_t) _self

logdet ::  Tensor ->  Tensor
logdet _self = unsafePerformIO $ (cast1 Managed.logdet_t) _self

logsumexp ::  Tensor ->  [Int] -> Bool ->  Tensor
logsumexp _self _dim _keepdim = unsafePerformIO $ (cast3 Managed.logsumexp_tlb) _self _dim _keepdim

margin_ranking_loss ::  Tensor ->  Tensor ->  Tensor -> Double -> Int64 ->  Tensor
margin_ranking_loss _input1 _input2 _target _margin _reduction = unsafePerformIO $ (cast5 Managed.margin_ranking_loss_tttdl) _input1 _input2 _target _margin _reduction

matmul ::  Tensor ->  Tensor ->  Tensor
matmul _self _other = unsafePerformIO $ (cast2 Managed.matmul_tt) _self _other

matrix_power ::  Tensor -> Int64 ->  Tensor
matrix_power _self _n = unsafePerformIO $ (cast2 Managed.matrix_power_tl) _self _n

max_values ::  Tensor ->  [Int] -> Bool ->  Tensor
max_values _self _dim _keepdim = unsafePerformIO $ (cast3 Managed.max_values_tlb) _self _dim _keepdim

max_pool1d_with_indices ::  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Bool ->  (Tensor,Tensor)
max_pool1d_with_indices _self _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 Managed.max_pool1d_with_indices_tllllb) _self _kernel_size _stride _padding _dilation _ceil_mode

max_pool1d ::  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Bool ->  Tensor
max_pool1d _self _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 Managed.max_pool1d_tllllb) _self _kernel_size _stride _padding _dilation _ceil_mode

max_pool2d ::  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Bool ->  Tensor
max_pool2d _self _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 Managed.max_pool2d_tllllb) _self _kernel_size _stride _padding _dilation _ceil_mode

mkldnn_max_pool2d ::  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Bool ->  Tensor
mkldnn_max_pool2d _self _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 Managed.mkldnn_max_pool2d_tllllb) _self _kernel_size _stride _padding _dilation _ceil_mode

max_pool3d ::  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Bool ->  Tensor
max_pool3d _self _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 Managed.max_pool3d_tllllb) _self _kernel_size _stride _padding _dilation _ceil_mode

min_values ::  Tensor ->  [Int] -> Bool ->  Tensor
min_values _self _dim _keepdim = unsafePerformIO $ (cast3 Managed.min_values_tlb) _self _dim _keepdim

mkldnn_convolution ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 ->  Tensor
mkldnn_convolution _self _weight _bias _padding _stride _dilation _groups = unsafePerformIO $ (cast7 Managed.mkldnn_convolution_tttllll) _self _weight _bias _padding _stride _dilation _groups

mkldnn_convolution_backward_input ::  [Int] ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool ->  Tensor
mkldnn_convolution_backward_input _self_size _grad_output _weight _padding _stride _dilation _groups _bias_defined = unsafePerformIO $ (cast8 Managed.mkldnn_convolution_backward_input_lttllllb) _self_size _grad_output _weight _padding _stride _dilation _groups _bias_defined

mkldnn_convolution_backward_weights ::  [Int] ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool ->  (Tensor,Tensor)
mkldnn_convolution_backward_weights _weight_size _grad_output _self _padding _stride _dilation _groups _bias_defined = unsafePerformIO $ (cast8 Managed.mkldnn_convolution_backward_weights_lttllllb) _weight_size _grad_output _self _padding _stride _dilation _groups _bias_defined

mkldnn_convolution_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
mkldnn_convolution_backward _self _grad_output _weight _padding _stride _dilation _groups _output_mask = unsafePerformIO $ (cast8 Managed.mkldnn_convolution_backward_tttlllla) _self _grad_output _weight _padding _stride _dilation _groups _output_mask

miopen_batch_norm ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor -> Bool -> Double -> Double ->  (Tensor,Tensor,Tensor)
miopen_batch_norm _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon = unsafePerformIO $ (cast8 Managed.miopen_batch_norm_tttttbdd) _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon

miopen_batch_norm_backward ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor -> Double ->  (Tensor,Tensor,Tensor)
miopen_batch_norm_backward _input _grad_output _weight _running_mean _running_var _save_mean _save_var _epsilon = unsafePerformIO $ (cast8 Managed.miopen_batch_norm_backward_tttttttd) _input _grad_output _weight _running_mean _running_var _save_mean _save_var _epsilon

miopen_convolution ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
miopen_convolution _self _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 Managed.miopen_convolution_tttllllbb) _self _weight _bias _padding _stride _dilation _groups _benchmark _deterministic

miopen_convolution_backward_input ::  [Int] ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
miopen_convolution_backward_input _self_size _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 Managed.miopen_convolution_backward_input_lttllllbb) _self_size _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic

miopen_convolution_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
miopen_convolution_backward _self _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic _output_mask = unsafePerformIO $ (cast10 Managed.miopen_convolution_backward_tttllllbba) _self _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic _output_mask

miopen_convolution_backward_bias ::  Tensor ->  Tensor
miopen_convolution_backward_bias _grad_output = unsafePerformIO $ (cast1 Managed.miopen_convolution_backward_bias_t) _grad_output

miopen_convolution_backward_weight ::  [Int] ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
miopen_convolution_backward_weight _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 Managed.miopen_convolution_backward_weight_lttllllbb) _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic

miopen_convolution_transpose ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
miopen_convolution_transpose _self _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast10 Managed.miopen_convolution_transpose_tttlllllbb) _self _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic

miopen_convolution_transpose_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
miopen_convolution_transpose_backward _self _grad_output _weight _padding _output_padding _stride _dilation _groups _benchmark _deterministic _output_mask = unsafePerformIO $ (cast11 Managed.miopen_convolution_transpose_backward_tttlllllbba) _self _grad_output _weight _padding _output_padding _stride _dilation _groups _benchmark _deterministic _output_mask

miopen_convolution_transpose_backward_input ::  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
miopen_convolution_transpose_backward_input _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast8 Managed.miopen_convolution_transpose_backward_input_ttllllbb) _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic

miopen_convolution_transpose_backward_weight ::  [Int] ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
miopen_convolution_transpose_backward_weight _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 Managed.miopen_convolution_transpose_backward_weight_lttllllbb) _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic

miopen_depthwise_convolution ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
miopen_depthwise_convolution _self _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 Managed.miopen_depthwise_convolution_tttllllbb) _self _weight _bias _padding _stride _dilation _groups _benchmark _deterministic

miopen_depthwise_convolution_backward_input ::  [Int] ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
miopen_depthwise_convolution_backward_input _self_size _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 Managed.miopen_depthwise_convolution_backward_input_lttllllbb) _self_size _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic

miopen_depthwise_convolution_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
miopen_depthwise_convolution_backward _self _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic _output_mask = unsafePerformIO $ (cast10 Managed.miopen_depthwise_convolution_backward_tttllllbba) _self _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic _output_mask

miopen_depthwise_convolution_backward_weight ::  [Int] ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 -> Bool -> Bool ->  Tensor
miopen_depthwise_convolution_backward_weight _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 Managed.miopen_depthwise_convolution_backward_weight_lttllllbb) _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic

mm ::  Tensor ->  Tensor ->  Tensor
mm _self _mat2 = unsafePerformIO $ (cast2 Managed.mm_tt) _self _mat2

mode ::  Tensor -> Int64 -> Bool ->  (Tensor,Tensor)
mode _self _dim _keepdim = unsafePerformIO $ (cast3 Managed.mode_tlb) _self _dim _keepdim

mv ::  Tensor ->  Tensor ->  Tensor
mv _self _vec = unsafePerformIO $ (cast2 Managed.mv_tt) _self _vec

mvlgamma ::  Tensor -> Int64 ->  Tensor
mvlgamma _self _p = unsafePerformIO $ (cast2 Managed.mvlgamma_tl) _self _p

narrow ::  Tensor -> Int64 -> Int64 -> Int64 ->  Tensor
narrow _self _dim _start _length = unsafePerformIO $ (cast4 Managed.narrow_tlll) _self _dim _start _length

native_batch_norm ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor -> Bool -> Double -> Double ->  (Tensor,Tensor,Tensor)
native_batch_norm _input _weight _bias _running_mean _running_var _training _momentum _eps = unsafePerformIO $ (cast8 Managed.native_batch_norm_tttttbdd) _input _weight _bias _running_mean _running_var _training _momentum _eps

batch_norm_stats ::  Tensor -> Double ->  (Tensor,Tensor)
batch_norm_stats _input _eps = unsafePerformIO $ (cast2 Managed.batch_norm_stats_td) _input _eps

batch_norm_elemt ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor -> Double ->  Tensor
batch_norm_elemt _input _weight _bias _mean _invstd _eps = unsafePerformIO $ (cast6 Managed.batch_norm_elemt_tttttd) _input _weight _bias _mean _invstd _eps

batch_norm_gather_stats ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor -> Double -> Double -> Int64 ->  (Tensor,Tensor)
batch_norm_gather_stats _input _mean _invstd _running_mean _running_var _momentum _eps _count = unsafePerformIO $ (cast8 Managed.batch_norm_gather_stats_tttttddl) _input _mean _invstd _running_mean _running_var _momentum _eps _count

native_batch_norm_backward ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor -> Bool -> Double ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
native_batch_norm_backward _grad_out _input _weight _running_mean _running_var _save_mean _save_invstd _train _eps _output_mask = unsafePerformIO $ (cast10 Managed.native_batch_norm_backward_tttttttbda) _grad_out _input _weight _running_mean _running_var _save_mean _save_invstd _train _eps _output_mask

batch_norm_backward_reduce ::  Tensor ->  Tensor ->  Tensor ->  Tensor -> Bool -> Bool -> Bool ->  (Tensor,Tensor,Tensor,Tensor)
batch_norm_backward_reduce _grad_out _input _mean _invstd _input_g _weight_g _bias_g = unsafePerformIO $ (cast7 Managed.batch_norm_backward_reduce_ttttbbb) _grad_out _input _mean _invstd _input_g _weight_g _bias_g

batch_norm_backward_elemt ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor
batch_norm_backward_elemt _grad_out _input _mean _invstd _weight _mean_dy _mean_dy_xmu = unsafePerformIO $ (cast7 Managed.batch_norm_backward_elemt_ttttttt) _grad_out _input _mean _invstd _weight _mean_dy _mean_dy_xmu

batch_norm_update_stats ::  Tensor ->  Tensor ->  Tensor -> Double ->  (Tensor,Tensor)
batch_norm_update_stats _input _running_mean _running_var _momentum = unsafePerformIO $ (cast4 Managed.batch_norm_update_stats_tttd) _input _running_mean _running_var _momentum

pairwise_distance ::  Tensor ->  Tensor -> Double -> Double -> Bool ->  Tensor
pairwise_distance _x1 _x2 _p _eps _keepdim = unsafePerformIO $ (cast5 Managed.pairwise_distance_ttddb) _x1 _x2 _p _eps _keepdim

cdist ::  Tensor ->  Tensor -> Double ->  Tensor
cdist _x1 _x2 _p = unsafePerformIO $ (cast3 Managed.cdist_ttd) _x1 _x2 _p

pdist ::  Tensor -> Double ->  Tensor
pdist _self _p = unsafePerformIO $ (cast2 Managed.pdist_td) _self _p

cosine_similarity ::  Tensor ->  Tensor -> Int64 -> Double ->  Tensor
cosine_similarity _x1 _x2 _dim _eps = unsafePerformIO $ (cast4 Managed.cosine_similarity_ttld) _x1 _x2 _dim _eps

pixel_shuffle ::  Tensor -> Int64 ->  Tensor
pixel_shuffle _self _upscale_factor = unsafePerformIO $ (cast2 Managed.pixel_shuffle_tl) _self _upscale_factor

pin_memory ::  Tensor ->  Tensor
pin_memory _self = unsafePerformIO $ (cast1 Managed.pin_memory_t) _self

pinverse ::  Tensor -> Double ->  Tensor
pinverse _self _rcond = unsafePerformIO $ (cast2 Managed.pinverse_td) _self _rcond

reciprocal ::  Tensor ->  Tensor
reciprocal _self = unsafePerformIO $ (cast1 Managed.reciprocal_t) _self

neg ::  Tensor ->  Tensor
neg _self = unsafePerformIO $ (cast1 Managed.neg_t) _self

reshape ::  Tensor ->  [Int] ->  Tensor
reshape _self _shape = unsafePerformIO $ (cast2 Managed.reshape_tl) _self _shape

mkldnn_reshape ::  Tensor ->  [Int] ->  Tensor
mkldnn_reshape _self _shape = unsafePerformIO $ (cast2 Managed.mkldnn_reshape_tl) _self _shape

round ::  Tensor ->  Tensor
round _self = unsafePerformIO $ (cast1 Managed.round_t) _self

relu ::  Tensor ->  Tensor
relu _self = unsafePerformIO $ (cast1 Managed.relu_t) _self

prelu ::  Tensor ->  Tensor ->  Tensor
prelu _self _weight = unsafePerformIO $ (cast2 Managed.prelu_tt) _self _weight

prelu_backward ::  Tensor ->  Tensor ->  Tensor ->  (Tensor,Tensor)
prelu_backward _grad_output _self _weight = unsafePerformIO $ (cast3 Managed.prelu_backward_ttt) _grad_output _self _weight

hardshrink ::  Tensor ->  Double ->  Tensor
hardshrink _self _lambd = unsafePerformIO $ (cast2 Managed.hardshrink_ts) _self _lambd

hardshrink_backward ::  Tensor ->  Tensor ->  Double ->  Tensor
hardshrink_backward _grad_out _self _lambd = unsafePerformIO $ (cast3 Managed.hardshrink_backward_tts) _grad_out _self _lambd

rsqrt ::  Tensor ->  Tensor
rsqrt _self = unsafePerformIO $ (cast1 Managed.rsqrt_t) _self

select ::  Tensor -> Int64 -> Int64 ->  Tensor
select _self _dim _index = unsafePerformIO $ (cast3 Managed.select_tll) _self _dim _index

selu ::  Tensor ->  Tensor
selu _self = unsafePerformIO $ (cast1 Managed.selu_t) _self

celu ::  Tensor ->  Double ->  Tensor
celu _self _alpha = unsafePerformIO $ (cast2 Managed.celu_ts) _self _alpha

sigmoid ::  Tensor ->  Tensor
sigmoid _self = unsafePerformIO $ (cast1 Managed.sigmoid_t) _self

sin ::  Tensor ->  Tensor
sin _self = unsafePerformIO $ (cast1 Managed.sin_t) _self

sinh ::  Tensor ->  Tensor
sinh _self = unsafePerformIO $ (cast1 Managed.sinh_t) _self

detach ::  Tensor ->  Tensor
detach _self = unsafePerformIO $ (cast1 Managed.detach_t) _self

size ::  Tensor -> Int64 -> Int64
size _self _dim = unsafePerformIO $ (cast2 Managed.size_tl) _self _dim

slice ::  Tensor -> Int64 -> Int64 -> Int64 -> Int64 ->  Tensor
slice _self _dim _start _end _step = unsafePerformIO $ (cast5 Managed.slice_tllll) _self _dim _start _end _step

slogdet ::  Tensor ->  (Tensor,Tensor)
slogdet _self = unsafePerformIO $ (cast1 Managed.slogdet_t) _self

smm ::  Tensor ->  Tensor ->  Tensor
smm _self _mat2 = unsafePerformIO $ (cast2 Managed.smm_tt) _self _mat2

split ::  Tensor -> Int64 -> Int64 ->  [Tensor]
split _self _split_size _dim = unsafePerformIO $ (cast3 Managed.split_tll) _self _split_size _dim

split_with_sizes ::  Tensor ->  [Int] -> Int64 ->  [Tensor]
split_with_sizes _self _split_sizes _dim = unsafePerformIO $ (cast3 Managed.split_with_sizes_tll) _self _split_sizes _dim

sspaddmm ::  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Tensor
sspaddmm _self _mat1 _mat2 _beta _alpha = unsafePerformIO $ (cast5 Managed.sspaddmm_tttss) _self _mat1 _mat2 _beta _alpha

stack ::  [Tensor] -> Int64 ->  Tensor
stack _tensors _dim = unsafePerformIO $ (cast2 Managed.stack_ll) _tensors _dim

stft ::  Tensor -> Int64 -> Int64 -> Int64 ->  Tensor -> Bool -> Bool ->  Tensor
stft _self _n_fft _hop_length _win_length _window _normalized _onesided = unsafePerformIO $ (cast7 Managed.stft_tllltbb) _self _n_fft _hop_length _win_length _window _normalized _onesided

stride ::  Tensor -> Int64 -> Int64
stride _self _dim = unsafePerformIO $ (cast2 Managed.stride_tl) _self _dim

sqrt ::  Tensor ->  Tensor
sqrt _self = unsafePerformIO $ (cast1 Managed.sqrt_t) _self

t ::  Tensor ->  Tensor
t _self = unsafePerformIO $ (cast1 Managed.t_t) _self

tan ::  Tensor ->  Tensor
tan _self = unsafePerformIO $ (cast1 Managed.tan_t) _self

tanh ::  Tensor ->  Tensor
tanh _self = unsafePerformIO $ (cast1 Managed.tanh_t) _self

tensordot ::  Tensor ->  Tensor ->  [Int] ->  [Int] ->  Tensor
tensordot _self _other _dims_self _dims_other = unsafePerformIO $ (cast4 Managed.tensordot_ttll) _self _other _dims_self _dims_other

threshold ::  Tensor ->  Double ->  Double ->  Tensor
threshold _self _threshold _value = unsafePerformIO $ (cast3 Managed.threshold_tss) _self _threshold _value

threshold_backward ::  Tensor ->  Tensor ->  Double ->  Tensor
threshold_backward _grad_output _self _threshold = unsafePerformIO $ (cast3 Managed.threshold_backward_tts) _grad_output _self _threshold

transpose ::  Tensor -> Int64 -> Int64 ->  Tensor
transpose _self _dim0 _dim1 = unsafePerformIO $ (cast3 Managed.transpose_tll) _self _dim0 _dim1

one_hot ::  Tensor -> Int64 ->  Tensor
one_hot _self _num_classes = unsafePerformIO $ (cast2 Managed.one_hot_tl) _self _num_classes

flip ::  Tensor ->  [Int] ->  Tensor
flip _self _dims = unsafePerformIO $ (cast2 Managed.flip_tl) _self _dims

roll ::  Tensor ->  [Int] ->  [Int] ->  Tensor
roll _self _shifts _dims = unsafePerformIO $ (cast3 Managed.roll_tll) _self _shifts _dims

rot90 ::  Tensor -> Int64 ->  [Int] ->  Tensor
rot90 _self _k _dims = unsafePerformIO $ (cast3 Managed.rot90_tll) _self _k _dims

triplet_margin_loss ::  Tensor ->  Tensor ->  Tensor -> Double -> Double -> Double -> Bool -> Int64 ->  Tensor
triplet_margin_loss _anchor _positive _negative _margin _p _eps _swap _reduction = unsafePerformIO $ (cast8 Managed.triplet_margin_loss_tttdddbl) _anchor _positive _negative _margin _p _eps _swap _reduction

trunc ::  Tensor ->  Tensor
trunc _self = unsafePerformIO $ (cast1 Managed.trunc_t) _self

unique_dim ::  Tensor -> Int64 -> Bool -> Bool -> Bool ->  (Tensor,Tensor,Tensor)
unique_dim _self _dim _sorted _return_inverse _return_counts = unsafePerformIO $ (cast5 Managed.unique_dim_tlbbb) _self _dim _sorted _return_inverse _return_counts

unique_consecutive ::  Tensor -> Bool -> Bool -> Int64 ->  (Tensor,Tensor,Tensor)
unique_consecutive _self _return_inverse _return_counts _dim = unsafePerformIO $ (cast4 Managed.unique_consecutive_tbbl) _self _return_inverse _return_counts _dim

unique_dim_consecutive ::  Tensor -> Int64 -> Bool -> Bool ->  (Tensor,Tensor,Tensor)
unique_dim_consecutive _self _dim _return_inverse _return_counts = unsafePerformIO $ (cast4 Managed.unique_dim_consecutive_tlbb) _self _dim _return_inverse _return_counts

unsqueeze ::  Tensor -> Int64 ->  Tensor
unsqueeze _self _dim = unsafePerformIO $ (cast2 Managed.unsqueeze_tl) _self _dim

where ::  Tensor ->  Tensor ->  Tensor ->  Tensor
where _condition _self _other = unsafePerformIO $ (cast3 Managed.where_ttt) _condition _self _other

norm_except_dim ::  Tensor -> Int64 -> Int64 ->  Tensor
norm_except_dim _v _pow _dim = unsafePerformIO $ (cast3 Managed.norm_except_dim_tll) _v _pow _dim

native_norm ::  Tensor ->  Double ->  Tensor
native_norm _self _p = unsafePerformIO $ (cast2 Managed.native_norm_ts) _self _p

nuclear_norm ::  Tensor -> Bool ->  Tensor
nuclear_norm _self _keepdim = unsafePerformIO $ (cast2 Managed.nuclear_norm_tb) _self _keepdim

clone ::  Tensor ->  Tensor
clone _self = unsafePerformIO $ (cast1 Managed.clone_t) _self

s_native_addmm ::  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Tensor
s_native_addmm _self _mat1 _mat2 _beta _alpha = unsafePerformIO $ (cast5 Managed.s_native_addmm_tttss) _self _mat1 _mat2 _beta _alpha

addmm ::  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Tensor
addmm _self _mat1 _mat2 _beta _alpha = unsafePerformIO $ (cast5 Managed.addmm_tttss) _self _mat1 _mat2 _beta _alpha

to_dense_backward ::  Tensor ->  Tensor ->  Tensor
to_dense_backward _grad _input = unsafePerformIO $ (cast2 Managed.to_dense_backward_tt) _grad _input

hspmm ::  Tensor ->  Tensor ->  Tensor
hspmm _mat1 _mat2 = unsafePerformIO $ (cast2 Managed.hspmm_tt) _mat1 _mat2

numel ::  Tensor -> Int64
numel _self = unsafePerformIO $ (cast1 Managed.numel_t) _self

unbind ::  Tensor -> Int64 ->  [Tensor]
unbind _self _dim = unsafePerformIO $ (cast2 Managed.unbind_tl) _self _dim

mkldnn_reorder_conv2d_weight ::  Tensor ->  [Int] ->  [Int] ->  [Int] -> Int64 ->  Tensor
mkldnn_reorder_conv2d_weight _self _padding _stride _dilation _groups = unsafePerformIO $ (cast5 Managed.mkldnn_reorder_conv2d_weight_tllll) _self _padding _stride _dilation _groups

to_mkldnn_backward ::  Tensor ->  Tensor ->  Tensor
to_mkldnn_backward _grad _input = unsafePerformIO $ (cast2 Managed.to_mkldnn_backward_tt) _grad _input

quantize_linear ::  Tensor -> Double -> Int64 ->  Tensor
quantize_linear _self _scale _zero_point = unsafePerformIO $ (cast3 Managed.quantize_linear_tdl) _self _scale _zero_point

dequantize ::  Tensor ->  Tensor
dequantize _self = unsafePerformIO $ (cast1 Managed.dequantize_t) _self

q_scale ::  Tensor ->  Double
q_scale _self = unsafePerformIO $ (cast1 Managed.q_scale_t) _self

q_zero_point ::  Tensor ->  Double
q_zero_point _self = unsafePerformIO $ (cast1 Managed.q_zero_point_t) _self

int_repr ::  Tensor ->  Tensor
int_repr _self = unsafePerformIO $ (cast1 Managed.int_repr_t) _self

meshgrid ::  [Tensor] ->  [Tensor]
meshgrid _tensors = unsafePerformIO $ (cast1 Managed.meshgrid_l) _tensors

cartesian_prod ::  [Tensor] ->  Tensor
cartesian_prod _tensors = unsafePerformIO $ (cast1 Managed.cartesian_prod_l) _tensors

combinations ::  Tensor -> Int64 -> Bool ->  Tensor
combinations _self _r _with_replacement = unsafePerformIO $ (cast3 Managed.combinations_tlb) _self _r _with_replacement

lstm_cell ::  Tensor ->  [Tensor] ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  (Tensor,Tensor)
lstm_cell _input _hx _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (cast6 Managed.lstm_cell_tltttt) _input _hx _w_ih _w_hh _b_ih _b_hh

gru_cell ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor
gru_cell _input _hx _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (cast6 Managed.gru_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

rnn_tanh_cell ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor
rnn_tanh_cell _input _hx _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (cast6 Managed.rnn_tanh_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

rnn_relu_cell ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor
rnn_relu_cell _input _hx _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (cast6 Managed.rnn_relu_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

quantized_lstm ::  Tensor ->  [Tensor] ->  [Tensor] -> Bool -> Int64 -> Double -> Bool -> Bool -> Bool ->  (Tensor,Tensor,Tensor)
quantized_lstm _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first = unsafePerformIO $ (cast9 Managed.quantized_lstm_tllbldbbb) _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first

quantized_lstm_cell ::  Tensor ->  [Tensor] ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Double ->  Double ->  (Tensor,Tensor)
quantized_lstm_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (cast14 Managed.quantized_lstm_cell_tlttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

quantized_gru_cell ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Double ->  Double ->  Tensor
quantized_gru_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (cast14 Managed.quantized_gru_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

quantized_rnn_relu_cell ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Double ->  Double ->  Tensor
quantized_rnn_relu_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (cast14 Managed.quantized_rnn_relu_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

quantized_rnn_tanh_cell ::  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Double ->  Double ->  Tensor
quantized_rnn_tanh_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (cast14 Managed.quantized_rnn_tanh_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

masked_scatter ::  Tensor ->  Tensor ->  Tensor ->  Tensor
masked_scatter _self _mask _source = unsafePerformIO $ (cast3 Managed.masked_scatter_ttt) _self _mask _source

index_add ::  Tensor -> Int64 ->  Tensor ->  Tensor ->  Tensor
index_add _self _dim _index _source = unsafePerformIO $ (cast4 Managed.index_add_tltt) _self _dim _index _source

scatter_add ::  Tensor -> Int64 ->  Tensor ->  Tensor ->  Tensor
scatter_add _self _dim _index _src = unsafePerformIO $ (cast4 Managed.scatter_add_tltt) _self _dim _index _src

addbmm ::  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Tensor
addbmm _self _batch1 _batch2 _beta _alpha = unsafePerformIO $ (cast5 Managed.addbmm_tttss) _self _batch1 _batch2 _beta _alpha

diag ::  Tensor -> Int64 ->  Tensor
diag _self _diagonal = unsafePerformIO $ (cast2 Managed.diag_tl) _self _diagonal

cross ::  Tensor ->  Tensor -> Int64 ->  Tensor
cross _self _other _dim = unsafePerformIO $ (cast3 Managed.cross_ttl) _self _other _dim

triu ::  Tensor -> Int64 ->  Tensor
triu _self _diagonal = unsafePerformIO $ (cast2 Managed.triu_tl) _self _diagonal

tril ::  Tensor -> Int64 ->  Tensor
tril _self _diagonal = unsafePerformIO $ (cast2 Managed.tril_tl) _self _diagonal

trace ::  Tensor ->  Tensor
trace _self = unsafePerformIO $ (cast1 Managed.trace_t) _self

take ::  Tensor ->  Tensor ->  Tensor
take _self _index = unsafePerformIO $ (cast2 Managed.take_tt) _self _index

index_select ::  Tensor -> Int64 ->  Tensor ->  Tensor
index_select _self _dim _index = unsafePerformIO $ (cast3 Managed.index_select_tlt) _self _dim _index

masked_select ::  Tensor ->  Tensor ->  Tensor
masked_select _self _mask = unsafePerformIO $ (cast2 Managed.masked_select_tt) _self _mask

nonzero ::  Tensor ->  Tensor
nonzero _self = unsafePerformIO $ (cast1 Managed.nonzero_t) _self

gather ::  Tensor -> Int64 ->  Tensor -> Bool ->  Tensor
gather _self _dim _index _sparse_grad = unsafePerformIO $ (cast4 Managed.gather_tltb) _self _dim _index _sparse_grad

addcmul ::  Tensor ->  Tensor ->  Tensor ->  Double ->  Tensor
addcmul _self _tensor1 _tensor2 _value = unsafePerformIO $ (cast4 Managed.addcmul_ttts) _self _tensor1 _tensor2 _value

addcdiv ::  Tensor ->  Tensor ->  Tensor ->  Double ->  Tensor
addcdiv _self _tensor1 _tensor2 _value = unsafePerformIO $ (cast4 Managed.addcdiv_ttts) _self _tensor1 _tensor2 _value

gels ::  Tensor ->  Tensor ->  (Tensor,Tensor)
gels _self _A = unsafePerformIO $ (cast2 Managed.gels_tt) _self _A

triangular_solve ::  Tensor ->  Tensor -> Bool -> Bool -> Bool ->  (Tensor,Tensor)
triangular_solve _self _A _upper _transpose _unitriangular = unsafePerformIO $ (cast5 Managed.triangular_solve_ttbbb) _self _A _upper _transpose _unitriangular

symeig ::  Tensor -> Bool -> Bool ->  (Tensor,Tensor)
symeig _self _eigenvectors _upper = unsafePerformIO $ (cast3 Managed.symeig_tbb) _self _eigenvectors _upper

eig ::  Tensor -> Bool ->  (Tensor,Tensor)
eig _self _eigenvectors = unsafePerformIO $ (cast2 Managed.eig_tb) _self _eigenvectors

svd ::  Tensor -> Bool -> Bool ->  (Tensor,Tensor,Tensor)
svd _self _some _compute_uv = unsafePerformIO $ (cast3 Managed.svd_tbb) _self _some _compute_uv

cholesky ::  Tensor -> Bool ->  Tensor
cholesky _self _upper = unsafePerformIO $ (cast2 Managed.cholesky_tb) _self _upper

cholesky_solve ::  Tensor ->  Tensor -> Bool ->  Tensor
cholesky_solve _self _input2 _upper = unsafePerformIO $ (cast3 Managed.cholesky_solve_ttb) _self _input2 _upper

solve ::  Tensor ->  Tensor ->  (Tensor,Tensor)
solve _self _A = unsafePerformIO $ (cast2 Managed.solve_tt) _self _A

cholesky_inverse ::  Tensor -> Bool ->  Tensor
cholesky_inverse _self _upper = unsafePerformIO $ (cast2 Managed.cholesky_inverse_tb) _self _upper

pstrf ::  Tensor -> Bool ->  Double ->  (Tensor,Tensor)
pstrf _self _upper _tol = unsafePerformIO $ (cast3 Managed.pstrf_tbs) _self _upper _tol

qr ::  Tensor ->  (Tensor,Tensor)
qr _self = unsafePerformIO $ (cast1 Managed.qr_t) _self

geqrf ::  Tensor ->  (Tensor,Tensor)
geqrf _self = unsafePerformIO $ (cast1 Managed.geqrf_t) _self

orgqr ::  Tensor ->  Tensor ->  Tensor
orgqr _self _input2 = unsafePerformIO $ (cast2 Managed.orgqr_tt) _self _input2

ormqr ::  Tensor ->  Tensor ->  Tensor -> Bool -> Bool ->  Tensor
ormqr _self _input2 _input3 _left _transpose = unsafePerformIO $ (cast5 Managed.ormqr_tttbb) _self _input2 _input3 _left _transpose

lu_solve ::  Tensor ->  Tensor ->  Tensor ->  Tensor
lu_solve _self _LU_data _LU_pivots = unsafePerformIO $ (cast3 Managed.lu_solve_ttt) _self _LU_data _LU_pivots

lgamma ::  Tensor ->  Tensor
lgamma _self = unsafePerformIO $ (cast1 Managed.lgamma_t) _self

digamma ::  Tensor ->  Tensor
digamma _self = unsafePerformIO $ (cast1 Managed.digamma_t) _self

polygamma :: Int64 ->  Tensor ->  Tensor
polygamma _n _self = unsafePerformIO $ (cast2 Managed.polygamma_lt) _n _self

erfinv ::  Tensor ->  Tensor
erfinv _self = unsafePerformIO $ (cast1 Managed.erfinv_t) _self

dist ::  Tensor ->  Tensor ->  Double ->  Tensor
dist _self _other _p = unsafePerformIO $ (cast3 Managed.dist_tts) _self _other _p

atan2 ::  Tensor ->  Tensor ->  Tensor
atan2 _self _other = unsafePerformIO $ (cast2 Managed.atan2_tt) _self _other

histc ::  Tensor -> Int64 ->  Double ->  Double ->  Tensor
histc _self _bins _min _max = unsafePerformIO $ (cast4 Managed.histc_tlss) _self _bins _min _max

sign ::  Tensor ->  Tensor
sign _self = unsafePerformIO $ (cast1 Managed.sign_t) _self

sort ::  Tensor -> Int64 -> Bool ->  (Tensor,Tensor)
sort _self _dim _descending = unsafePerformIO $ (cast3 Managed.sort_tlb) _self _dim _descending

argsort ::  Tensor -> Int64 -> Bool ->  Tensor
argsort _self _dim _descending = unsafePerformIO $ (cast3 Managed.argsort_tlb) _self _dim _descending

topk ::  Tensor -> Int64 -> Int64 -> Bool -> Bool ->  (Tensor,Tensor)
topk _self _k _dim _largest _sorted = unsafePerformIO $ (cast5 Managed.topk_tllbb) _self _k _dim _largest _sorted

renorm ::  Tensor ->  Double -> Int64 ->  Double ->  Tensor
renorm _self _p _dim _maxnorm = unsafePerformIO $ (cast4 Managed.renorm_tsls) _self _p _dim _maxnorm

equal ::  Tensor ->  Tensor -> Bool
equal _self _other = unsafePerformIO $ (cast2 Managed.equal_tt) _self _other

alias ::  Tensor ->  Tensor
alias _self = unsafePerformIO $ (cast1 Managed.alias_t) _self

binary_cross_entropy ::  Tensor ->  Tensor ->  Tensor -> Int64 ->  Tensor
binary_cross_entropy _self _target _weight _reduction = unsafePerformIO $ (cast4 Managed.binary_cross_entropy_tttl) _self _target _weight _reduction

binary_cross_entropy_backward ::  Tensor ->  Tensor ->  Tensor ->  Tensor -> Int64 ->  Tensor
binary_cross_entropy_backward _grad_output _self _target _weight _reduction = unsafePerformIO $ (cast5 Managed.binary_cross_entropy_backward_ttttl) _grad_output _self _target _weight _reduction

mse_loss ::  Tensor ->  Tensor -> Int64 ->  Tensor
mse_loss _self _target _reduction = unsafePerformIO $ (cast3 Managed.mse_loss_ttl) _self _target _reduction

mse_loss_backward ::  Tensor ->  Tensor ->  Tensor -> Int64 ->  Tensor
mse_loss_backward _grad_output _self _target _reduction = unsafePerformIO $ (cast4 Managed.mse_loss_backward_tttl) _grad_output _self _target _reduction

l1_loss ::  Tensor ->  Tensor -> Int64 ->  Tensor
l1_loss _self _target _reduction = unsafePerformIO $ (cast3 Managed.l1_loss_ttl) _self _target _reduction

l1_loss_backward ::  Tensor ->  Tensor ->  Tensor -> Int64 ->  Tensor
l1_loss_backward _grad_output _self _target _reduction = unsafePerformIO $ (cast4 Managed.l1_loss_backward_tttl) _grad_output _self _target _reduction

multi_margin_loss ::  Tensor ->  Tensor ->  Double ->  Double ->  Tensor -> Int64 ->  Tensor
multi_margin_loss _self _target _p _margin _weight _reduction = unsafePerformIO $ (cast6 Managed.multi_margin_loss_ttsstl) _self _target _p _margin _weight _reduction

multi_margin_loss_backward ::  Tensor ->  Tensor ->  Tensor ->  Double ->  Double ->  Tensor -> Int64 ->  Tensor
multi_margin_loss_backward _grad_output _self _target _p _margin _weight _reduction = unsafePerformIO $ (cast7 Managed.multi_margin_loss_backward_tttsstl) _grad_output _self _target _p _margin _weight _reduction

multilabel_margin_loss ::  Tensor ->  Tensor -> Int64 ->  Tensor
multilabel_margin_loss _self _target _reduction = unsafePerformIO $ (cast3 Managed.multilabel_margin_loss_ttl) _self _target _reduction

multilabel_margin_loss_forward ::  Tensor ->  Tensor -> Int64 ->  (Tensor,Tensor)
multilabel_margin_loss_forward _self _target _reduction = unsafePerformIO $ (cast3 Managed.multilabel_margin_loss_forward_ttl) _self _target _reduction

multilabel_margin_loss_backward ::  Tensor ->  Tensor ->  Tensor -> Int64 ->  Tensor ->  Tensor
multilabel_margin_loss_backward _grad_output _self _target _reduction _is_target = unsafePerformIO $ (cast5 Managed.multilabel_margin_loss_backward_tttlt) _grad_output _self _target _reduction _is_target

nll_loss ::  Tensor ->  Tensor ->  Tensor -> Int64 -> Int64 ->  Tensor
nll_loss _self _target _weight _reduction _ignore_index = unsafePerformIO $ (cast5 Managed.nll_loss_tttll) _self _target _weight _reduction _ignore_index

nll_loss_forward ::  Tensor ->  Tensor ->  Tensor -> Int64 -> Int64 ->  (Tensor,Tensor)
nll_loss_forward _self _target _weight _reduction _ignore_index = unsafePerformIO $ (cast5 Managed.nll_loss_forward_tttll) _self _target _weight _reduction _ignore_index

nll_loss_backward ::  Tensor ->  Tensor ->  Tensor ->  Tensor -> Int64 -> Int64 ->  Tensor ->  Tensor
nll_loss_backward _grad_output _self _target _weight _reduction _ignore_index _total_weight = unsafePerformIO $ (cast7 Managed.nll_loss_backward_ttttllt) _grad_output _self _target _weight _reduction _ignore_index _total_weight

nll_loss2d ::  Tensor ->  Tensor ->  Tensor -> Int64 -> Int64 ->  Tensor
nll_loss2d _self _target _weight _reduction _ignore_index = unsafePerformIO $ (cast5 Managed.nll_loss2d_tttll) _self _target _weight _reduction _ignore_index

nll_loss2d_forward ::  Tensor ->  Tensor ->  Tensor -> Int64 -> Int64 ->  (Tensor,Tensor)
nll_loss2d_forward _self _target _weight _reduction _ignore_index = unsafePerformIO $ (cast5 Managed.nll_loss2d_forward_tttll) _self _target _weight _reduction _ignore_index

nll_loss2d_backward ::  Tensor ->  Tensor ->  Tensor ->  Tensor -> Int64 -> Int64 ->  Tensor ->  Tensor
nll_loss2d_backward _grad_output _self _target _weight _reduction _ignore_index _total_weight = unsafePerformIO $ (cast7 Managed.nll_loss2d_backward_ttttllt) _grad_output _self _target _weight _reduction _ignore_index _total_weight

smooth_l1_loss ::  Tensor ->  Tensor -> Int64 ->  Tensor
smooth_l1_loss _self _target _reduction = unsafePerformIO $ (cast3 Managed.smooth_l1_loss_ttl) _self _target _reduction

smooth_l1_loss_backward ::  Tensor ->  Tensor ->  Tensor -> Int64 ->  Tensor
smooth_l1_loss_backward _grad_output _self _target _reduction = unsafePerformIO $ (cast4 Managed.smooth_l1_loss_backward_tttl) _grad_output _self _target _reduction

soft_margin_loss ::  Tensor ->  Tensor -> Int64 ->  Tensor
soft_margin_loss _self _target _reduction = unsafePerformIO $ (cast3 Managed.soft_margin_loss_ttl) _self _target _reduction

soft_margin_loss_backward ::  Tensor ->  Tensor ->  Tensor -> Int64 ->  Tensor
soft_margin_loss_backward _grad_output _self _target _reduction = unsafePerformIO $ (cast4 Managed.soft_margin_loss_backward_tttl) _grad_output _self _target _reduction

elu ::  Tensor ->  Double ->  Double ->  Double ->  Tensor
elu _self _alpha _scale _input_scale = unsafePerformIO $ (cast4 Managed.elu_tsss) _self _alpha _scale _input_scale

elu_backward ::  Tensor ->  Double ->  Double ->  Double ->  Tensor ->  Tensor
elu_backward _grad_output _alpha _scale _input_scale _output = unsafePerformIO $ (cast5 Managed.elu_backward_tssst) _grad_output _alpha _scale _input_scale _output

glu ::  Tensor -> Int64 ->  Tensor
glu _self _dim = unsafePerformIO $ (cast2 Managed.glu_tl) _self _dim

glu_backward ::  Tensor ->  Tensor -> Int64 ->  Tensor
glu_backward _grad_output _self _dim = unsafePerformIO $ (cast3 Managed.glu_backward_ttl) _grad_output _self _dim

hardtanh ::  Tensor ->  Double ->  Double ->  Tensor
hardtanh _self _min_val _max_val = unsafePerformIO $ (cast3 Managed.hardtanh_tss) _self _min_val _max_val

hardtanh_backward ::  Tensor ->  Tensor ->  Double ->  Double ->  Tensor
hardtanh_backward _grad_output _self _min_val _max_val = unsafePerformIO $ (cast4 Managed.hardtanh_backward_ttss) _grad_output _self _min_val _max_val

leaky_relu ::  Tensor ->  Double ->  Tensor
leaky_relu _self _negative_slope = unsafePerformIO $ (cast2 Managed.leaky_relu_ts) _self _negative_slope

leaky_relu_backward ::  Tensor ->  Tensor ->  Double ->  Tensor
leaky_relu_backward _grad_output _self _negative_slope = unsafePerformIO $ (cast3 Managed.leaky_relu_backward_tts) _grad_output _self _negative_slope

log_sigmoid ::  Tensor ->  Tensor
log_sigmoid _self = unsafePerformIO $ (cast1 Managed.log_sigmoid_t) _self

log_sigmoid_forward ::  Tensor ->  (Tensor,Tensor)
log_sigmoid_forward _self = unsafePerformIO $ (cast1 Managed.log_sigmoid_forward_t) _self

log_sigmoid_backward ::  Tensor ->  Tensor ->  Tensor ->  Tensor
log_sigmoid_backward _grad_output _self _buffer = unsafePerformIO $ (cast3 Managed.log_sigmoid_backward_ttt) _grad_output _self _buffer

rrelu_with_noise_backward ::  Tensor ->  Tensor ->  Tensor ->  Double ->  Double -> Bool ->  Tensor
rrelu_with_noise_backward _grad_output _self _noise _lower _upper _training = unsafePerformIO $ (cast6 Managed.rrelu_with_noise_backward_tttssb) _grad_output _self _noise _lower _upper _training

softplus ::  Tensor ->  Double ->  Double ->  Tensor
softplus _self _beta _threshold = unsafePerformIO $ (cast3 Managed.softplus_tss) _self _beta _threshold

softplus_backward ::  Tensor ->  Tensor ->  Double ->  Double ->  Tensor ->  Tensor
softplus_backward _grad_output _self _beta _threshold _output = unsafePerformIO $ (cast5 Managed.softplus_backward_ttsst) _grad_output _self _beta _threshold _output

softshrink ::  Tensor ->  Double ->  Tensor
softshrink _self _lambd = unsafePerformIO $ (cast2 Managed.softshrink_ts) _self _lambd

softshrink_backward ::  Tensor ->  Tensor ->  Double ->  Tensor
softshrink_backward _grad_output _self _lambd = unsafePerformIO $ (cast3 Managed.softshrink_backward_tts) _grad_output _self _lambd

adaptive_avg_pool2d ::  Tensor ->  [Int] ->  Tensor
adaptive_avg_pool2d _self _output_size = unsafePerformIO $ (cast2 Managed.adaptive_avg_pool2d_tl) _self _output_size

adaptive_avg_pool3d ::  Tensor ->  [Int] ->  Tensor
adaptive_avg_pool3d _self _output_size = unsafePerformIO $ (cast2 Managed.adaptive_avg_pool3d_tl) _self _output_size

adaptive_avg_pool3d_backward ::  Tensor ->  Tensor ->  Tensor
adaptive_avg_pool3d_backward _grad_output _self = unsafePerformIO $ (cast2 Managed.adaptive_avg_pool3d_backward_tt) _grad_output _self

adaptive_max_pool2d ::  Tensor ->  [Int] ->  (Tensor,Tensor)
adaptive_max_pool2d _self _output_size = unsafePerformIO $ (cast2 Managed.adaptive_max_pool2d_tl) _self _output_size

adaptive_max_pool2d_backward ::  Tensor ->  Tensor ->  Tensor ->  Tensor
adaptive_max_pool2d_backward _grad_output _self _indices = unsafePerformIO $ (cast3 Managed.adaptive_max_pool2d_backward_ttt) _grad_output _self _indices

adaptive_max_pool3d ::  Tensor ->  [Int] ->  (Tensor,Tensor)
adaptive_max_pool3d _self _output_size = unsafePerformIO $ (cast2 Managed.adaptive_max_pool3d_tl) _self _output_size

adaptive_max_pool3d_backward ::  Tensor ->  Tensor ->  Tensor ->  Tensor
adaptive_max_pool3d_backward _grad_output _self _indices = unsafePerformIO $ (cast3 Managed.adaptive_max_pool3d_backward_ttt) _grad_output _self _indices

avg_pool2d ::  Tensor ->  [Int] ->  [Int] ->  [Int] -> Bool -> Bool ->  Tensor
avg_pool2d _self _kernel_size _stride _padding _ceil_mode _count_include_pad = unsafePerformIO $ (cast6 Managed.avg_pool2d_tlllbb) _self _kernel_size _stride _padding _ceil_mode _count_include_pad

avg_pool2d_backward ::  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Bool -> Bool ->  Tensor
avg_pool2d_backward _grad_output _self _kernel_size _stride _padding _ceil_mode _count_include_pad = unsafePerformIO $ (cast7 Managed.avg_pool2d_backward_ttlllbb) _grad_output _self _kernel_size _stride _padding _ceil_mode _count_include_pad

avg_pool3d ::  Tensor ->  [Int] ->  [Int] ->  [Int] -> Bool -> Bool ->  Tensor
avg_pool3d _self _kernel_size _stride _padding _ceil_mode _count_include_pad = unsafePerformIO $ (cast6 Managed.avg_pool3d_tlllbb) _self _kernel_size _stride _padding _ceil_mode _count_include_pad

avg_pool3d_backward ::  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] -> Bool -> Bool ->  Tensor
avg_pool3d_backward _grad_output _self _kernel_size _stride _padding _ceil_mode _count_include_pad = unsafePerformIO $ (cast7 Managed.avg_pool3d_backward_ttlllbb) _grad_output _self _kernel_size _stride _padding _ceil_mode _count_include_pad

fractional_max_pool2d ::  Tensor ->  [Int] ->  [Int] ->  Tensor ->  (Tensor,Tensor)
fractional_max_pool2d _self _kernel_size _output_size _random_samples = unsafePerformIO $ (cast4 Managed.fractional_max_pool2d_tllt) _self _kernel_size _output_size _random_samples

fractional_max_pool2d_backward ::  Tensor ->  Tensor ->  [Int] ->  [Int] ->  Tensor ->  Tensor
fractional_max_pool2d_backward _grad_output _self _kernel_size _output_size _indices = unsafePerformIO $ (cast5 Managed.fractional_max_pool2d_backward_ttllt) _grad_output _self _kernel_size _output_size _indices

fractional_max_pool3d ::  Tensor ->  [Int] ->  [Int] ->  Tensor ->  (Tensor,Tensor)
fractional_max_pool3d _self _kernel_size _output_size _random_samples = unsafePerformIO $ (cast4 Managed.fractional_max_pool3d_tllt) _self _kernel_size _output_size _random_samples

fractional_max_pool3d_backward ::  Tensor ->  Tensor ->  [Int] ->  [Int] ->  Tensor ->  Tensor
fractional_max_pool3d_backward _grad_output _self _kernel_size _output_size _indices = unsafePerformIO $ (cast5 Managed.fractional_max_pool3d_backward_ttllt) _grad_output _self _kernel_size _output_size _indices

max_pool2d_with_indices ::  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Bool ->  (Tensor,Tensor)
max_pool2d_with_indices _self _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 Managed.max_pool2d_with_indices_tllllb) _self _kernel_size _stride _padding _dilation _ceil_mode

max_pool2d_with_indices_backward ::  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Bool ->  Tensor ->  Tensor
max_pool2d_with_indices_backward _grad_output _self _kernel_size _stride _padding _dilation _ceil_mode _indices = unsafePerformIO $ (cast8 Managed.max_pool2d_with_indices_backward_ttllllbt) _grad_output _self _kernel_size _stride _padding _dilation _ceil_mode _indices

max_pool3d_with_indices ::  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Bool ->  (Tensor,Tensor)
max_pool3d_with_indices _self _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 Managed.max_pool3d_with_indices_tllllb) _self _kernel_size _stride _padding _dilation _ceil_mode

max_pool3d_with_indices_backward ::  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] -> Bool ->  Tensor ->  Tensor
max_pool3d_with_indices_backward _grad_output _self _kernel_size _stride _padding _dilation _ceil_mode _indices = unsafePerformIO $ (cast8 Managed.max_pool3d_with_indices_backward_ttllllbt) _grad_output _self _kernel_size _stride _padding _dilation _ceil_mode _indices

max_unpool2d ::  Tensor ->  Tensor ->  [Int] ->  Tensor
max_unpool2d _self _indices _output_size = unsafePerformIO $ (cast3 Managed.max_unpool2d_ttl) _self _indices _output_size

max_unpool2d_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  Tensor
max_unpool2d_backward _grad_output _self _indices _output_size = unsafePerformIO $ (cast4 Managed.max_unpool2d_backward_tttl) _grad_output _self _indices _output_size

max_unpool3d ::  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  Tensor
max_unpool3d _self _indices _output_size _stride _padding = unsafePerformIO $ (cast5 Managed.max_unpool3d_ttlll) _self _indices _output_size _stride _padding

max_unpool3d_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  Tensor
max_unpool3d_backward _grad_output _self _indices _output_size _stride _padding = unsafePerformIO $ (cast6 Managed.max_unpool3d_backward_tttlll) _grad_output _self _indices _output_size _stride _padding

reflection_pad1d ::  Tensor ->  [Int] ->  Tensor
reflection_pad1d _self _padding = unsafePerformIO $ (cast2 Managed.reflection_pad1d_tl) _self _padding

reflection_pad1d_backward ::  Tensor ->  Tensor ->  [Int] ->  Tensor
reflection_pad1d_backward _grad_output _self _padding = unsafePerformIO $ (cast3 Managed.reflection_pad1d_backward_ttl) _grad_output _self _padding

reflection_pad2d ::  Tensor ->  [Int] ->  Tensor
reflection_pad2d _self _padding = unsafePerformIO $ (cast2 Managed.reflection_pad2d_tl) _self _padding

reflection_pad2d_backward ::  Tensor ->  Tensor ->  [Int] ->  Tensor
reflection_pad2d_backward _grad_output _self _padding = unsafePerformIO $ (cast3 Managed.reflection_pad2d_backward_ttl) _grad_output _self _padding

replication_pad1d ::  Tensor ->  [Int] ->  Tensor
replication_pad1d _self _padding = unsafePerformIO $ (cast2 Managed.replication_pad1d_tl) _self _padding

replication_pad1d_backward ::  Tensor ->  Tensor ->  [Int] ->  Tensor
replication_pad1d_backward _grad_output _self _padding = unsafePerformIO $ (cast3 Managed.replication_pad1d_backward_ttl) _grad_output _self _padding

replication_pad2d ::  Tensor ->  [Int] ->  Tensor
replication_pad2d _self _padding = unsafePerformIO $ (cast2 Managed.replication_pad2d_tl) _self _padding

replication_pad2d_backward ::  Tensor ->  Tensor ->  [Int] ->  Tensor
replication_pad2d_backward _grad_output _self _padding = unsafePerformIO $ (cast3 Managed.replication_pad2d_backward_ttl) _grad_output _self _padding

replication_pad3d ::  Tensor ->  [Int] ->  Tensor
replication_pad3d _self _padding = unsafePerformIO $ (cast2 Managed.replication_pad3d_tl) _self _padding

replication_pad3d_backward ::  Tensor ->  Tensor ->  [Int] ->  Tensor
replication_pad3d_backward _grad_output _self _padding = unsafePerformIO $ (cast3 Managed.replication_pad3d_backward_ttl) _grad_output _self _padding

upsample_linear1d ::  Tensor ->  [Int] -> Bool ->  Tensor
upsample_linear1d _self _output_size _align_corners = unsafePerformIO $ (cast3 Managed.upsample_linear1d_tlb) _self _output_size _align_corners

upsample_linear1d_backward ::  Tensor ->  [Int] ->  [Int] -> Bool ->  Tensor
upsample_linear1d_backward _grad_output _output_size _input_size _align_corners = unsafePerformIO $ (cast4 Managed.upsample_linear1d_backward_tllb) _grad_output _output_size _input_size _align_corners

upsample_bilinear2d ::  Tensor ->  [Int] -> Bool ->  Tensor
upsample_bilinear2d _self _output_size _align_corners = unsafePerformIO $ (cast3 Managed.upsample_bilinear2d_tlb) _self _output_size _align_corners

upsample_bilinear2d_backward ::  Tensor ->  [Int] ->  [Int] -> Bool ->  Tensor
upsample_bilinear2d_backward _grad_output _output_size _input_size _align_corners = unsafePerformIO $ (cast4 Managed.upsample_bilinear2d_backward_tllb) _grad_output _output_size _input_size _align_corners

upsample_bicubic2d ::  Tensor ->  [Int] -> Bool ->  Tensor
upsample_bicubic2d _self _output_size _align_corners = unsafePerformIO $ (cast3 Managed.upsample_bicubic2d_tlb) _self _output_size _align_corners

upsample_bicubic2d_backward ::  Tensor ->  [Int] ->  [Int] -> Bool ->  Tensor
upsample_bicubic2d_backward _grad_output _output_size _input_size _align_corners = unsafePerformIO $ (cast4 Managed.upsample_bicubic2d_backward_tllb) _grad_output _output_size _input_size _align_corners

upsample_trilinear3d ::  Tensor ->  [Int] -> Bool ->  Tensor
upsample_trilinear3d _self _output_size _align_corners = unsafePerformIO $ (cast3 Managed.upsample_trilinear3d_tlb) _self _output_size _align_corners

upsample_trilinear3d_backward ::  Tensor ->  [Int] ->  [Int] -> Bool ->  Tensor
upsample_trilinear3d_backward _grad_output _output_size _input_size _align_corners = unsafePerformIO $ (cast4 Managed.upsample_trilinear3d_backward_tllb) _grad_output _output_size _input_size _align_corners

upsample_nearest1d ::  Tensor ->  [Int] ->  Tensor
upsample_nearest1d _self _output_size = unsafePerformIO $ (cast2 Managed.upsample_nearest1d_tl) _self _output_size

upsample_nearest1d_backward ::  Tensor ->  [Int] ->  [Int] ->  Tensor
upsample_nearest1d_backward _grad_output _output_size _input_size = unsafePerformIO $ (cast3 Managed.upsample_nearest1d_backward_tll) _grad_output _output_size _input_size

upsample_nearest2d ::  Tensor ->  [Int] ->  Tensor
upsample_nearest2d _self _output_size = unsafePerformIO $ (cast2 Managed.upsample_nearest2d_tl) _self _output_size

upsample_nearest2d_backward ::  Tensor ->  [Int] ->  [Int] ->  Tensor
upsample_nearest2d_backward _grad_output _output_size _input_size = unsafePerformIO $ (cast3 Managed.upsample_nearest2d_backward_tll) _grad_output _output_size _input_size

upsample_nearest3d ::  Tensor ->  [Int] ->  Tensor
upsample_nearest3d _self _output_size = unsafePerformIO $ (cast2 Managed.upsample_nearest3d_tl) _self _output_size

upsample_nearest3d_backward ::  Tensor ->  [Int] ->  [Int] ->  Tensor
upsample_nearest3d_backward _grad_output _output_size _input_size = unsafePerformIO $ (cast3 Managed.upsample_nearest3d_backward_tll) _grad_output _output_size _input_size

sigmoid_backward ::  Tensor ->  Tensor ->  Tensor
sigmoid_backward _grad_output _output = unsafePerformIO $ (cast2 Managed.sigmoid_backward_tt) _grad_output _output

tanh_backward ::  Tensor ->  Tensor ->  Tensor
tanh_backward _grad_output _output = unsafePerformIO $ (cast2 Managed.tanh_backward_tt) _grad_output _output

thnn_conv_transpose2d ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  Tensor
thnn_conv_transpose2d _self _weight _kernel_size _bias _stride _padding _output_padding _dilation = unsafePerformIO $ (cast8 Managed.thnn_conv_transpose2d_ttltllll) _self _weight _kernel_size _bias _stride _padding _output_padding _dilation

thnn_conv_transpose2d_forward ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  (Tensor,Tensor,Tensor)
thnn_conv_transpose2d_forward _self _weight _kernel_size _bias _stride _padding _output_padding _dilation = unsafePerformIO $ (cast8 Managed.thnn_conv_transpose2d_forward_ttltllll) _self _weight _kernel_size _bias _stride _padding _output_padding _dilation

thnn_conv_transpose2d_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  Tensor ->  Tensor ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
thnn_conv_transpose2d_backward _grad_output _self _weight _kernel_size _stride _padding _output_padding _dilation _columns _ones _output_mask = unsafePerformIO $ (cast11 Managed.thnn_conv_transpose2d_backward_tttllllltta) _grad_output _self _weight _kernel_size _stride _padding _output_padding _dilation _columns _ones _output_mask

thnn_conv_transpose3d ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  Tensor
thnn_conv_transpose3d _self _weight _kernel_size _bias _stride _padding _output_padding _dilation = unsafePerformIO $ (cast8 Managed.thnn_conv_transpose3d_ttltllll) _self _weight _kernel_size _bias _stride _padding _output_padding _dilation

thnn_conv_transpose3d_forward ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  (Tensor,Tensor,Tensor)
thnn_conv_transpose3d_forward _self _weight _kernel_size _bias _stride _padding _output_padding _dilation = unsafePerformIO $ (cast8 Managed.thnn_conv_transpose3d_forward_ttltllll) _self _weight _kernel_size _bias _stride _padding _output_padding _dilation

thnn_conv_transpose3d_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  Tensor ->  Tensor ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
thnn_conv_transpose3d_backward _grad_output _self _weight _kernel_size _stride _padding _output_padding _dilation _finput _fgrad_input _output_mask = unsafePerformIO $ (cast11 Managed.thnn_conv_transpose3d_backward_tttllllltta) _grad_output _self _weight _kernel_size _stride _padding _output_padding _dilation _finput _fgrad_input _output_mask

thnn_conv2d ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  Tensor
thnn_conv2d _self _weight _kernel_size _bias _stride _padding = unsafePerformIO $ (cast6 Managed.thnn_conv2d_ttltll) _self _weight _kernel_size _bias _stride _padding

thnn_conv2d_forward ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  (Tensor,Tensor,Tensor)
thnn_conv2d_forward _self _weight _kernel_size _bias _stride _padding = unsafePerformIO $ (cast6 Managed.thnn_conv2d_forward_ttltll) _self _weight _kernel_size _bias _stride _padding

thnn_conv2d_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  Tensor ->  Tensor ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
thnn_conv2d_backward _grad_output _self _weight _kernel_size _stride _padding _finput _fgrad_input _output_mask = unsafePerformIO $ (cast9 Managed.thnn_conv2d_backward_tttllltta) _grad_output _self _weight _kernel_size _stride _padding _finput _fgrad_input _output_mask

thnn_conv_depthwise2d ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  Tensor
thnn_conv_depthwise2d _self _weight _kernel_size _bias _stride _padding _dilation = unsafePerformIO $ (cast7 Managed.thnn_conv_depthwise2d_ttltlll) _self _weight _kernel_size _bias _stride _padding _dilation

thnn_conv_depthwise2d_forward ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  Tensor
thnn_conv_depthwise2d_forward _self _weight _kernel_size _bias _stride _padding _dilation = unsafePerformIO $ (cast7 Managed.thnn_conv_depthwise2d_forward_ttltlll) _self _weight _kernel_size _bias _stride _padding _dilation

thnn_conv_depthwise2d_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  (StdArray CBool 2) ->  (Tensor,Tensor)
thnn_conv_depthwise2d_backward _grad_output _self _weight _kernel_size _stride _padding _dilation _output_mask = unsafePerformIO $ (cast8 Managed.thnn_conv_depthwise2d_backward_tttlllla) _grad_output _self _weight _kernel_size _stride _padding _dilation _output_mask

thnn_conv3d ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  Tensor
thnn_conv3d _self _weight _kernel_size _bias _stride _padding = unsafePerformIO $ (cast6 Managed.thnn_conv3d_ttltll) _self _weight _kernel_size _bias _stride _padding

thnn_conv3d_forward ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  (Tensor,Tensor,Tensor)
thnn_conv3d_forward _self _weight _kernel_size _bias _stride _padding = unsafePerformIO $ (cast6 Managed.thnn_conv3d_forward_ttltll) _self _weight _kernel_size _bias _stride _padding

thnn_conv3d_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  Tensor ->  Tensor ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
thnn_conv3d_backward _grad_output _self _weight _kernel_size _stride _padding _finput _fgrad_input _output_mask = unsafePerformIO $ (cast9 Managed.thnn_conv3d_backward_tttllltta) _grad_output _self _weight _kernel_size _stride _padding _finput _fgrad_input _output_mask

thnn_conv_dilated2d ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  Tensor
thnn_conv_dilated2d _self _weight _kernel_size _bias _stride _padding _dilation = unsafePerformIO $ (cast7 Managed.thnn_conv_dilated2d_ttltlll) _self _weight _kernel_size _bias _stride _padding _dilation

thnn_conv_dilated2d_forward ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  (Tensor,Tensor,Tensor)
thnn_conv_dilated2d_forward _self _weight _kernel_size _bias _stride _padding _dilation = unsafePerformIO $ (cast7 Managed.thnn_conv_dilated2d_forward_ttltlll) _self _weight _kernel_size _bias _stride _padding _dilation

thnn_conv_dilated2d_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  Tensor ->  Tensor ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
thnn_conv_dilated2d_backward _grad_output _self _weight _kernel_size _stride _padding _dilation _columns _ones _output_mask = unsafePerformIO $ (cast10 Managed.thnn_conv_dilated2d_backward_tttlllltta) _grad_output _self _weight _kernel_size _stride _padding _dilation _columns _ones _output_mask

thnn_conv_dilated3d ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  Tensor
thnn_conv_dilated3d _self _weight _kernel_size _bias _stride _padding _dilation = unsafePerformIO $ (cast7 Managed.thnn_conv_dilated3d_ttltlll) _self _weight _kernel_size _bias _stride _padding _dilation

thnn_conv_dilated3d_forward ::  Tensor ->  Tensor ->  [Int] ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  (Tensor,Tensor,Tensor)
thnn_conv_dilated3d_forward _self _weight _kernel_size _bias _stride _padding _dilation = unsafePerformIO $ (cast7 Managed.thnn_conv_dilated3d_forward_ttltlll) _self _weight _kernel_size _bias _stride _padding _dilation

thnn_conv_dilated3d_backward ::  Tensor ->  Tensor ->  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  Tensor ->  Tensor ->  (StdArray CBool 3) ->  (Tensor,Tensor,Tensor)
thnn_conv_dilated3d_backward _grad_output _self _weight _kernel_size _stride _padding _dilation _columns _ones _output_mask = unsafePerformIO $ (cast10 Managed.thnn_conv_dilated3d_backward_tttlllltta) _grad_output _self _weight _kernel_size _stride _padding _dilation _columns _ones _output_mask

thnn_col2im ::  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  Tensor
thnn_col2im _self _output_size _kernel_size _dilation _padding _stride = unsafePerformIO $ (cast6 Managed.thnn_col2im_tlllll) _self _output_size _kernel_size _dilation _padding _stride

thnn_col2im_backward ::  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  Tensor
thnn_col2im_backward _grad_output _kernel_size _dilation _padding _stride = unsafePerformIO $ (cast5 Managed.thnn_col2im_backward_tllll) _grad_output _kernel_size _dilation _padding _stride

thnn_im2col ::  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  Tensor
thnn_im2col _self _kernel_size _dilation _padding _stride = unsafePerformIO $ (cast5 Managed.thnn_im2col_tllll) _self _kernel_size _dilation _padding _stride

thnn_im2col_backward ::  Tensor ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  [Int] ->  Tensor
thnn_im2col_backward _grad_output _input_size _kernel_size _dilation _padding _stride = unsafePerformIO $ (cast6 Managed.thnn_im2col_backward_tlllll) _grad_output _input_size _kernel_size _dilation _padding _stride

