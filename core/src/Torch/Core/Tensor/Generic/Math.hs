{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Generic.Math where

import qualified THByteTensorMath as T
import qualified THDoubleTensorMath as T
import qualified THFloatTensorMath as T
import qualified THIntTensorMath as T
import qualified THLongTensorMath as T
import qualified THShortTensorMath as T

import Torch.Core.Tensor.Generic.Internal

class GenericMath t where
  fill         :: Ptr t -> HaskReal t -> IO ()
  zero         :: Ptr t -> IO ()
  maskedFill   :: Ptr t -> Ptr CTHByteTensor -> HaskReal t -> IO ()
  maskedCopy   :: Ptr t -> Ptr CTHByteTensor -> Ptr t -> IO ()
  maskedSelect :: Ptr t -> Ptr t -> Ptr CTHByteTensor -> IO ()
  nonzero      :: Ptr CTHLongTensor -> Ptr t -> IO ()
  indexSelect  :: Ptr t -> Ptr t -> CInt -> Ptr CTHLongTensor -> IO ()
  indexCopy    :: Ptr t -> CInt -> Ptr CTHLongTensor -> Ptr t -> IO ()
  indexAdd     :: Ptr t -> CInt -> Ptr CTHLongTensor -> Ptr t -> IO ()
  indexFill    :: Ptr t -> CInt -> Ptr CTHLongTensor -> HaskReal t -> IO ()
  take         :: Ptr t -> Ptr t -> Ptr CTHLongTensor -> IO ()
  put          :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> IO ()
  gather       :: Ptr t -> Ptr t -> CInt -> Ptr CTHLongTensor -> IO ()
  scatter      :: Ptr t -> CInt -> Ptr CTHLongTensor -> Ptr t -> IO ()
  scatterAdd   :: Ptr t -> CInt -> Ptr CTHLongTensor -> Ptr t -> IO ()
  scatterFill  :: Ptr t -> CInt -> Ptr CTHLongTensor -> HaskReal t -> IO ()
  dot          :: Ptr t -> Ptr t -> HaskAccReal t
  minall       :: Ptr t -> HaskReal t
  maxall       :: Ptr t -> HaskReal t
  medianall    :: Ptr t -> HaskReal t
  sumall       :: Ptr t -> HaskAccReal t
  prodall      :: Ptr t -> HaskAccReal t
  add          :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  sub          :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  add_scaled   :: Ptr t -> Ptr t -> HaskReal t -> HaskReal t -> IO ()
  sub_scaled   :: Ptr t -> Ptr t -> HaskReal t -> HaskReal t -> IO ()
  mul          :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  div          :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  lshift       :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  rshift       :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  fmod         :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  remainder    :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  clamp        :: Ptr t -> Ptr t -> HaskReal t -> HaskReal t -> IO ()
  bitand       :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  bitor        :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  bitxor       :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  cadd         :: Ptr t -> Ptr t -> HaskReal t -> Ptr t -> IO ()
  csub         :: Ptr t -> Ptr t -> HaskReal t -> Ptr t -> IO ()
  cmul         :: Ptr t -> Ptr t -> Ptr t -> IO ()
  cpow         :: Ptr t -> Ptr t -> Ptr t -> IO ()
  cdiv         :: Ptr t -> Ptr t -> Ptr t -> IO ()
  clshift      :: Ptr t -> Ptr t -> Ptr t -> IO ()
  crshift      :: Ptr t -> Ptr t -> Ptr t -> IO ()
  cfmod        :: Ptr t -> Ptr t -> Ptr t -> IO ()
  cremainder   :: Ptr t -> Ptr t -> Ptr t -> IO ()
  cbitand      :: Ptr t -> Ptr t -> Ptr t -> IO ()
  cbitor       :: Ptr t -> Ptr t -> Ptr t -> IO ()
  cbitxor      :: Ptr t -> Ptr t -> Ptr t -> IO ()
  addcmul      :: Ptr t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  addcdiv      :: Ptr t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  addmv        :: Ptr t -> HaskReal t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  addmm        :: Ptr t -> HaskReal t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  addr         :: Ptr t -> HaskReal t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  addbmm       :: Ptr t -> HaskReal t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  baddbmm      :: Ptr t -> HaskReal t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  match        :: Ptr t -> Ptr t -> Ptr t -> HaskReal t -> IO ()
  numel        :: Ptr t -> CPtrdiff
  max          :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> CInt -> IO ()
  min          :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> CInt -> IO ()
  kthvalue     :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CLLong -> CInt -> CInt -> IO ()
  mode         :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> CInt -> IO ()
  median       :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> CInt -> IO ()
  sum          :: Ptr t -> Ptr t -> CInt -> CInt -> IO ()
  prod         :: Ptr t -> Ptr t -> CInt -> CInt -> IO ()
  cumsum       :: Ptr t -> Ptr t -> CInt -> IO ()
  cumprod      :: Ptr t -> Ptr t -> CInt -> IO ()
  sign         :: Ptr t -> Ptr t -> IO ()
  trace        :: Ptr t -> HaskAccReal t
  cross        :: Ptr t -> Ptr t -> Ptr t -> CInt -> IO ()
  cmax         :: Ptr t -> Ptr t -> Ptr t -> IO ()
  cmin         :: Ptr t -> Ptr t -> Ptr t -> IO ()
  cmaxValue    :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  cminValue    :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  zeros        :: Ptr t -> Ptr CTHLongStorage -> IO ()
  zerosLike    :: Ptr t -> Ptr t -> IO ()
  ones         :: Ptr t -> Ptr CTHLongStorage -> IO ()
  onesLike     :: Ptr t -> Ptr t -> IO ()
  diag         :: Ptr t -> Ptr t -> CInt -> IO ()
  eye          :: Ptr t -> CLLong -> CLLong -> IO ()
  arange       :: Ptr t -> HaskAccReal t -> HaskAccReal t -> HaskAccReal t -> IO ()
  range        :: Ptr t -> HaskAccReal t -> HaskAccReal t -> HaskAccReal t -> IO ()
  randperm     :: Ptr t -> Ptr CTHGenerator -> CLLong -> IO ()
  reshape      :: Ptr t -> Ptr t -> Ptr CTHLongStorage -> IO ()
  sort         :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> CInt -> IO ()
  topk         :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CLLong -> CInt -> CInt -> CInt -> IO ()
  tril         :: Ptr t -> Ptr t -> CLLong -> IO ()
  triu         :: Ptr t -> Ptr t -> CLLong -> IO ()
  cat          :: Ptr t -> Ptr t -> Ptr t -> CInt -> IO ()
  catArray     :: Ptr t -> Ptr (Ptr t) -> CInt -> CInt -> IO ()
  equal        :: Ptr t -> Ptr t -> CInt
  ltValue      :: Ptr CTHByteTensor -> Ptr t -> HaskReal t -> IO ()
  leValue      :: Ptr CTHByteTensor -> Ptr t -> HaskReal t -> IO ()
  gtValue      :: Ptr CTHByteTensor -> Ptr t -> HaskReal t -> IO ()
  geValue      :: Ptr CTHByteTensor -> Ptr t -> HaskReal t -> IO ()
  neValue      :: Ptr CTHByteTensor -> Ptr t -> HaskReal t -> IO ()
  eqValue      :: Ptr CTHByteTensor -> Ptr t -> HaskReal t -> IO ()
  ltValueT     :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  leValueT     :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  gtValueT     :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  geValueT     :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  neValueT     :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  eqValueT     :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  ltTensor     :: Ptr CTHByteTensor -> Ptr t -> Ptr t -> IO ()
  leTensor     :: Ptr CTHByteTensor -> Ptr t -> Ptr t -> IO ()
  gtTensor     :: Ptr CTHByteTensor -> Ptr t -> Ptr t -> IO ()
  geTensor     :: Ptr CTHByteTensor -> Ptr t -> Ptr t -> IO ()
  neTensor     :: Ptr CTHByteTensor -> Ptr t -> Ptr t -> IO ()
  eqTensor     :: Ptr CTHByteTensor -> Ptr t -> Ptr t -> IO ()
  ltTensorT    :: Ptr t -> Ptr t -> Ptr t -> IO ()
  leTensorT    :: Ptr t -> Ptr t -> Ptr t -> IO ()
  gtTensorT    :: Ptr t -> Ptr t -> Ptr t -> IO ()
  geTensorT    :: Ptr t -> Ptr t -> Ptr t -> IO ()
  neTensorT    :: Ptr t -> Ptr t -> Ptr t -> IO ()
  eqTensorT    :: Ptr t -> Ptr t -> Ptr t -> IO ()

class GenericMath t => GenericNegativeOps t where
  neg          :: Ptr t -> Ptr t -> IO ()
  abs          :: Ptr t -> Ptr t -> IO ()

class GenericMath t => GenericFloatingMath t where
  cinv         :: Ptr t -> Ptr t -> IO ()
  sigmoid      :: Ptr t -> Ptr t -> IO ()
  log          :: Ptr t -> Ptr t -> IO ()
  lgamma       :: Ptr t -> Ptr t -> IO ()
  log1p        :: Ptr t -> Ptr t -> IO ()
  exp          :: Ptr t -> Ptr t -> IO ()
  cos          :: Ptr t -> Ptr t -> IO ()
  acos         :: Ptr t -> Ptr t -> IO ()
  cosh         :: Ptr t -> Ptr t -> IO ()
  sin          :: Ptr t -> Ptr t -> IO ()
  asin         :: Ptr t -> Ptr t -> IO ()
  sinh         :: Ptr t -> Ptr t -> IO ()
  tan          :: Ptr t -> Ptr t -> IO ()
  atan         :: Ptr t -> Ptr t -> IO ()
  atan2        :: Ptr t -> Ptr t -> Ptr t -> IO ()
  tanh         :: Ptr t -> Ptr t -> IO ()
  erf          :: Ptr t -> Ptr t -> IO ()
  erfinv       :: Ptr t -> Ptr t -> IO ()
  pow          :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  tpow         :: Ptr t -> HaskReal t -> Ptr t -> IO ()
  sqrt         :: Ptr t -> Ptr t -> IO ()
  rsqrt        :: Ptr t -> Ptr t -> IO ()
  ceil         :: Ptr t -> Ptr t -> IO ()
  floor        :: Ptr t -> Ptr t -> IO ()
  round        :: Ptr t -> Ptr t -> IO ()
  trunc        :: Ptr t -> Ptr t -> IO ()
  frac         :: Ptr t -> Ptr t -> IO ()
  lerp         :: Ptr t -> Ptr t -> Ptr t -> HaskReal t -> IO ()
  mean         :: Ptr t -> Ptr t -> CInt -> CInt -> IO ()
  std          :: Ptr t -> Ptr t -> CInt -> CInt -> CInt -> IO ()
  var          :: Ptr t -> Ptr t -> CInt -> CInt -> CInt -> IO ()
  norm         :: Ptr t -> Ptr t -> HaskReal t -> CInt -> CInt -> IO ()
  renorm       :: Ptr t -> Ptr t -> HaskReal t -> CInt -> HaskReal t -> IO ()
  dist         :: Ptr t -> Ptr t -> HaskReal t -> HaskAccReal t
  histc        :: Ptr t -> Ptr t -> CLLong -> HaskReal t -> HaskReal t -> IO ()
  bhistc       :: Ptr t -> Ptr t -> CLLong -> HaskReal t -> HaskReal t -> IO ()
  meanall      :: Ptr t -> HaskAccReal t
  varall       :: Ptr t -> CInt -> HaskAccReal t
  stdall       :: Ptr t -> CInt -> HaskAccReal t
  normall      :: Ptr t -> HaskReal t -> HaskAccReal t
  linspace     :: Ptr t -> HaskReal t -> HaskReal t -> CLLong -> IO ()
  logspace     :: Ptr t -> HaskReal t -> HaskReal t -> CLLong -> IO ()
  rand         :: Ptr t -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()
  randn        :: Ptr t -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()


instance GenericMath CTHByteTensor where
  fill = T.c_THByteTensor_fill
  zero = T.c_THByteTensor_zero
  maskedFill = T.c_THByteTensor_maskedFill
  maskedCopy = T.c_THByteTensor_maskedCopy
  maskedSelect = T.c_THByteTensor_maskedSelect
  nonzero = T.c_THByteTensor_nonzero
  indexSelect = T.c_THByteTensor_indexSelect
  indexCopy = T.c_THByteTensor_indexCopy
  indexAdd = T.c_THByteTensor_indexAdd
  indexFill = T.c_THByteTensor_indexFill
  take = T.c_THByteTensor_take
  put = T.c_THByteTensor_put
  gather = T.c_THByteTensor_gather
  scatter = T.c_THByteTensor_scatter
  scatterAdd = T.c_THByteTensor_scatterAdd
  scatterFill = T.c_THByteTensor_scatterFill
  dot = T.c_THByteTensor_dot
  minall = T.c_THByteTensor_minall
  maxall = T.c_THByteTensor_maxall
  medianall = T.c_THByteTensor_medianall
  sumall = T.c_THByteTensor_sumall
  prodall = T.c_THByteTensor_prodall
  add = T.c_THByteTensor_add
  sub = T.c_THByteTensor_sub
  add_scaled = T.c_THByteTensor_add_scaled
  sub_scaled = T.c_THByteTensor_sub_scaled
  mul = T.c_THByteTensor_mul
  div = T.c_THByteTensor_div
  lshift = T.c_THByteTensor_lshift
  rshift = T.c_THByteTensor_rshift
  fmod = T.c_THByteTensor_fmod
  remainder = T.c_THByteTensor_remainder
  clamp = T.c_THByteTensor_clamp
  bitand = T.c_THByteTensor_bitand
  bitor = T.c_THByteTensor_bitor
  bitxor = T.c_THByteTensor_bitxor
  cadd = T.c_THByteTensor_cadd
  csub = T.c_THByteTensor_csub
  cmul = T.c_THByteTensor_cmul
  cpow = T.c_THByteTensor_cpow
  cdiv = T.c_THByteTensor_cdiv
  clshift = T.c_THByteTensor_clshift
  crshift = T.c_THByteTensor_crshift
  cfmod = T.c_THByteTensor_cfmod
  cremainder = T.c_THByteTensor_cremainder
  cbitand = T.c_THByteTensor_cbitand
  cbitor = T.c_THByteTensor_cbitor
  cbitxor = T.c_THByteTensor_cbitxor
  addcmul = T.c_THByteTensor_addcmul
  addcdiv = T.c_THByteTensor_addcdiv
  addmv = T.c_THByteTensor_addmv
  addmm = T.c_THByteTensor_addmm
  addr = T.c_THByteTensor_addr
  addbmm = T.c_THByteTensor_addbmm
  baddbmm = T.c_THByteTensor_baddbmm
  match = T.c_THByteTensor_match
  numel = T.c_THByteTensor_numel
  max = T.c_THByteTensor_max
  min = T.c_THByteTensor_min
  kthvalue = T.c_THByteTensor_kthvalue
  mode = T.c_THByteTensor_mode
  median = T.c_THByteTensor_median
  sum = T.c_THByteTensor_sum
  prod = T.c_THByteTensor_prod
  cumsum = T.c_THByteTensor_cumsum
  cumprod = T.c_THByteTensor_cumprod
  sign = T.c_THByteTensor_sign
  trace = T.c_THByteTensor_trace
  cross = T.c_THByteTensor_cross
  cmax = T.c_THByteTensor_cmax
  cmin = T.c_THByteTensor_cmin
  cmaxValue = T.c_THByteTensor_cmaxValue
  cminValue = T.c_THByteTensor_cminValue
  zeros = T.c_THByteTensor_zeros
  zerosLike = T.c_THByteTensor_zerosLike
  ones = T.c_THByteTensor_ones
  onesLike = T.c_THByteTensor_onesLike
  diag = T.c_THByteTensor_diag
  eye = T.c_THByteTensor_eye
  arange = T.c_THByteTensor_arange
  range = T.c_THByteTensor_range
  randperm = T.c_THByteTensor_randperm
  reshape = T.c_THByteTensor_reshape
  sort = T.c_THByteTensor_sort
  topk = T.c_THByteTensor_topk
  tril = T.c_THByteTensor_tril
  triu = T.c_THByteTensor_triu
  cat = T.c_THByteTensor_cat
  catArray = T.c_THByteTensor_catArray
  equal = T.c_THByteTensor_equal
  ltValue = T.c_THByteTensor_ltValue
  leValue = T.c_THByteTensor_leValue
  gtValue = T.c_THByteTensor_gtValue
  geValue = T.c_THByteTensor_geValue
  neValue = T.c_THByteTensor_neValue
  eqValue = T.c_THByteTensor_eqValue
  ltValueT = T.c_THByteTensor_ltValueT
  leValueT = T.c_THByteTensor_leValueT
  gtValueT = T.c_THByteTensor_gtValueT
  geValueT = T.c_THByteTensor_geValueT
  neValueT = T.c_THByteTensor_neValueT
  eqValueT = T.c_THByteTensor_eqValueT
  ltTensor = T.c_THByteTensor_ltTensor
  leTensor = T.c_THByteTensor_leTensor
  gtTensor = T.c_THByteTensor_gtTensor
  geTensor = T.c_THByteTensor_geTensor
  neTensor = T.c_THByteTensor_neTensor
  eqTensor = T.c_THByteTensor_eqTensor
  ltTensorT = T.c_THByteTensor_ltTensorT
  leTensorT = T.c_THByteTensor_leTensorT
  gtTensorT = T.c_THByteTensor_gtTensorT
  geTensorT = T.c_THByteTensor_geTensorT
  neTensorT = T.c_THByteTensor_neTensorT
  eqTensorT = T.c_THByteTensor_eqTensorT

instance GenericMath CTHDoubleTensor where
  fill = T.c_THDoubleTensor_fill
  zero = T.c_THDoubleTensor_zero
  maskedFill = T.c_THDoubleTensor_maskedFill
  maskedCopy = T.c_THDoubleTensor_maskedCopy
  maskedSelect = T.c_THDoubleTensor_maskedSelect
  nonzero = T.c_THDoubleTensor_nonzero
  indexSelect = T.c_THDoubleTensor_indexSelect
  indexCopy = T.c_THDoubleTensor_indexCopy
  indexAdd = T.c_THDoubleTensor_indexAdd
  indexFill = T.c_THDoubleTensor_indexFill
  take = T.c_THDoubleTensor_take
  put = T.c_THDoubleTensor_put
  gather = T.c_THDoubleTensor_gather
  scatter = T.c_THDoubleTensor_scatter
  scatterAdd = T.c_THDoubleTensor_scatterAdd
  scatterFill = T.c_THDoubleTensor_scatterFill
  dot = T.c_THDoubleTensor_dot
  minall = T.c_THDoubleTensor_minall
  maxall = T.c_THDoubleTensor_maxall
  medianall = T.c_THDoubleTensor_medianall
  sumall = T.c_THDoubleTensor_sumall
  prodall = T.c_THDoubleTensor_prodall
  add = T.c_THDoubleTensor_add
  sub = T.c_THDoubleTensor_sub
  add_scaled = T.c_THDoubleTensor_add_scaled
  sub_scaled = T.c_THDoubleTensor_sub_scaled
  mul = T.c_THDoubleTensor_mul
  div = T.c_THDoubleTensor_div
  lshift = T.c_THDoubleTensor_lshift
  rshift = T.c_THDoubleTensor_rshift
  fmod = T.c_THDoubleTensor_fmod
  remainder = T.c_THDoubleTensor_remainder
  clamp = T.c_THDoubleTensor_clamp
  bitand = T.c_THDoubleTensor_bitand
  bitor = T.c_THDoubleTensor_bitor
  bitxor = T.c_THDoubleTensor_bitxor
  cadd = T.c_THDoubleTensor_cadd
  csub = T.c_THDoubleTensor_csub
  cmul = T.c_THDoubleTensor_cmul
  cpow = T.c_THDoubleTensor_cpow
  cdiv = T.c_THDoubleTensor_cdiv
  clshift = T.c_THDoubleTensor_clshift
  crshift = T.c_THDoubleTensor_crshift
  cfmod = T.c_THDoubleTensor_cfmod
  cremainder = T.c_THDoubleTensor_cremainder
  cbitand = T.c_THDoubleTensor_cbitand
  cbitor = T.c_THDoubleTensor_cbitor
  cbitxor = T.c_THDoubleTensor_cbitxor
  addcmul = T.c_THDoubleTensor_addcmul
  addcdiv = T.c_THDoubleTensor_addcdiv
  addmv = T.c_THDoubleTensor_addmv
  addmm = T.c_THDoubleTensor_addmm
  addr = T.c_THDoubleTensor_addr
  addbmm = T.c_THDoubleTensor_addbmm
  baddbmm = T.c_THDoubleTensor_baddbmm
  match = T.c_THDoubleTensor_match
  numel = T.c_THDoubleTensor_numel
  max = T.c_THDoubleTensor_max
  min = T.c_THDoubleTensor_min
  kthvalue = T.c_THDoubleTensor_kthvalue
  mode = T.c_THDoubleTensor_mode
  median = T.c_THDoubleTensor_median
  sum = T.c_THDoubleTensor_sum
  prod = T.c_THDoubleTensor_prod
  cumsum = T.c_THDoubleTensor_cumsum
  cumprod = T.c_THDoubleTensor_cumprod
  sign = T.c_THDoubleTensor_sign
  trace = T.c_THDoubleTensor_trace
  cross = T.c_THDoubleTensor_cross
  cmax = T.c_THDoubleTensor_cmax
  cmin = T.c_THDoubleTensor_cmin
  cmaxValue = T.c_THDoubleTensor_cmaxValue
  cminValue = T.c_THDoubleTensor_cminValue
  zeros = T.c_THDoubleTensor_zeros
  zerosLike = T.c_THDoubleTensor_zerosLike
  ones = T.c_THDoubleTensor_ones
  onesLike = T.c_THDoubleTensor_onesLike
  diag = T.c_THDoubleTensor_diag
  eye = T.c_THDoubleTensor_eye
  arange = T.c_THDoubleTensor_arange
  range = T.c_THDoubleTensor_range
  randperm = T.c_THDoubleTensor_randperm
  reshape = T.c_THDoubleTensor_reshape
  sort = T.c_THDoubleTensor_sort
  topk = T.c_THDoubleTensor_topk
  tril = T.c_THDoubleTensor_tril
  triu = T.c_THDoubleTensor_triu
  cat = T.c_THDoubleTensor_cat
  catArray = T.c_THDoubleTensor_catArray
  equal = T.c_THDoubleTensor_equal
  ltValue = T.c_THDoubleTensor_ltValue
  leValue = T.c_THDoubleTensor_leValue
  gtValue = T.c_THDoubleTensor_gtValue
  geValue = T.c_THDoubleTensor_geValue
  neValue = T.c_THDoubleTensor_neValue
  eqValue = T.c_THDoubleTensor_eqValue
  ltValueT = T.c_THDoubleTensor_ltValueT
  leValueT = T.c_THDoubleTensor_leValueT
  gtValueT = T.c_THDoubleTensor_gtValueT
  geValueT = T.c_THDoubleTensor_geValueT
  neValueT = T.c_THDoubleTensor_neValueT
  eqValueT = T.c_THDoubleTensor_eqValueT
  ltTensor = T.c_THDoubleTensor_ltTensor
  leTensor = T.c_THDoubleTensor_leTensor
  gtTensor = T.c_THDoubleTensor_gtTensor
  geTensor = T.c_THDoubleTensor_geTensor
  neTensor = T.c_THDoubleTensor_neTensor
  eqTensor = T.c_THDoubleTensor_eqTensor
  ltTensorT = T.c_THDoubleTensor_ltTensorT
  leTensorT = T.c_THDoubleTensor_leTensorT
  gtTensorT = T.c_THDoubleTensor_gtTensorT
  geTensorT = T.c_THDoubleTensor_geTensorT
  neTensorT = T.c_THDoubleTensor_neTensorT
  eqTensorT = T.c_THDoubleTensor_eqTensorT

instance GenericNegativeOps CTHDoubleTensor where
  neg = T.c_THDoubleTensor_neg
  abs = T.c_THDoubleTensor_abs

instance GenericFloatingMath CTHDoubleTensor where
  cinv = T.c_THDoubleTensor_cinv
  sigmoid = T.c_THDoubleTensor_sigmoid
  log = T.c_THDoubleTensor_log
  lgamma = T.c_THDoubleTensor_lgamma
  log1p = T.c_THDoubleTensor_log1p
  exp = T.c_THDoubleTensor_exp
  cos = T.c_THDoubleTensor_cos
  acos = T.c_THDoubleTensor_acos
  cosh = T.c_THDoubleTensor_cosh
  sin = T.c_THDoubleTensor_sin
  asin = T.c_THDoubleTensor_asin
  sinh = T.c_THDoubleTensor_sinh
  tan = T.c_THDoubleTensor_tan
  atan = T.c_THDoubleTensor_atan
  atan2 = T.c_THDoubleTensor_atan2
  tanh = T.c_THDoubleTensor_tanh
  erf = T.c_THDoubleTensor_erf
  erfinv = T.c_THDoubleTensor_erfinv
  pow = T.c_THDoubleTensor_pow
  tpow = T.c_THDoubleTensor_tpow
  sqrt = T.c_THDoubleTensor_sqrt
  rsqrt = T.c_THDoubleTensor_rsqrt
  ceil = T.c_THDoubleTensor_ceil
  floor = T.c_THDoubleTensor_floor
  round = T.c_THDoubleTensor_round
  trunc = T.c_THDoubleTensor_trunc
  frac = T.c_THDoubleTensor_frac
  lerp = T.c_THDoubleTensor_lerp
  mean = T.c_THDoubleTensor_mean
  std = T.c_THDoubleTensor_std
  var = T.c_THDoubleTensor_var
  norm = T.c_THDoubleTensor_norm
  renorm = T.c_THDoubleTensor_renorm
  dist = T.c_THDoubleTensor_dist
  histc = T.c_THDoubleTensor_histc
  bhistc = T.c_THDoubleTensor_bhistc
  meanall = T.c_THDoubleTensor_meanall
  varall = T.c_THDoubleTensor_varall
  stdall = T.c_THDoubleTensor_stdall
  normall = T.c_THDoubleTensor_normall
  linspace = T.c_THDoubleTensor_linspace
  logspace = T.c_THDoubleTensor_logspace
  rand = T.c_THDoubleTensor_rand
  randn = T.c_THDoubleTensor_randn

instance GenericMath CTHFloatTensor where
  fill = T.c_THFloatTensor_fill
  zero = T.c_THFloatTensor_zero
  maskedFill = T.c_THFloatTensor_maskedFill
  maskedCopy = T.c_THFloatTensor_maskedCopy
  maskedSelect = T.c_THFloatTensor_maskedSelect
  nonzero = T.c_THFloatTensor_nonzero
  indexSelect = T.c_THFloatTensor_indexSelect
  indexCopy = T.c_THFloatTensor_indexCopy
  indexAdd = T.c_THFloatTensor_indexAdd
  indexFill = T.c_THFloatTensor_indexFill
  take = T.c_THFloatTensor_take
  put = T.c_THFloatTensor_put
  gather = T.c_THFloatTensor_gather
  scatter = T.c_THFloatTensor_scatter
  scatterAdd = T.c_THFloatTensor_scatterAdd
  scatterFill = T.c_THFloatTensor_scatterFill
  dot = T.c_THFloatTensor_dot
  minall = T.c_THFloatTensor_minall
  maxall = T.c_THFloatTensor_maxall
  medianall = T.c_THFloatTensor_medianall
  sumall = T.c_THFloatTensor_sumall
  prodall = T.c_THFloatTensor_prodall
  add = T.c_THFloatTensor_add
  sub = T.c_THFloatTensor_sub
  add_scaled = T.c_THFloatTensor_add_scaled
  sub_scaled = T.c_THFloatTensor_sub_scaled
  mul = T.c_THFloatTensor_mul
  div = T.c_THFloatTensor_div
  lshift = T.c_THFloatTensor_lshift
  rshift = T.c_THFloatTensor_rshift
  fmod = T.c_THFloatTensor_fmod
  remainder = T.c_THFloatTensor_remainder
  clamp = T.c_THFloatTensor_clamp
  bitand = T.c_THFloatTensor_bitand
  bitor = T.c_THFloatTensor_bitor
  bitxor = T.c_THFloatTensor_bitxor
  cadd = T.c_THFloatTensor_cadd
  csub = T.c_THFloatTensor_csub
  cmul = T.c_THFloatTensor_cmul
  cpow = T.c_THFloatTensor_cpow
  cdiv = T.c_THFloatTensor_cdiv
  clshift = T.c_THFloatTensor_clshift
  crshift = T.c_THFloatTensor_crshift
  cfmod = T.c_THFloatTensor_cfmod
  cremainder = T.c_THFloatTensor_cremainder
  cbitand = T.c_THFloatTensor_cbitand
  cbitor = T.c_THFloatTensor_cbitor
  cbitxor = T.c_THFloatTensor_cbitxor
  addcmul = T.c_THFloatTensor_addcmul
  addcdiv = T.c_THFloatTensor_addcdiv
  addmv = T.c_THFloatTensor_addmv
  addmm = T.c_THFloatTensor_addmm
  addr = T.c_THFloatTensor_addr
  addbmm = T.c_THFloatTensor_addbmm
  baddbmm = T.c_THFloatTensor_baddbmm
  match = T.c_THFloatTensor_match
  numel = T.c_THFloatTensor_numel
  max = T.c_THFloatTensor_max
  min = T.c_THFloatTensor_min
  kthvalue = T.c_THFloatTensor_kthvalue
  mode = T.c_THFloatTensor_mode
  median = T.c_THFloatTensor_median
  sum = T.c_THFloatTensor_sum
  prod = T.c_THFloatTensor_prod
  cumsum = T.c_THFloatTensor_cumsum
  cumprod = T.c_THFloatTensor_cumprod
  sign = T.c_THFloatTensor_sign
  trace = T.c_THFloatTensor_trace
  cross = T.c_THFloatTensor_cross
  cmax = T.c_THFloatTensor_cmax
  cmin = T.c_THFloatTensor_cmin
  cmaxValue = T.c_THFloatTensor_cmaxValue
  cminValue = T.c_THFloatTensor_cminValue
  zeros = T.c_THFloatTensor_zeros
  zerosLike = T.c_THFloatTensor_zerosLike
  ones = T.c_THFloatTensor_ones
  onesLike = T.c_THFloatTensor_onesLike
  diag = T.c_THFloatTensor_diag
  eye = T.c_THFloatTensor_eye
  arange = T.c_THFloatTensor_arange
  range = T.c_THFloatTensor_range
  randperm = T.c_THFloatTensor_randperm
  reshape = T.c_THFloatTensor_reshape
  sort = T.c_THFloatTensor_sort
  topk = T.c_THFloatTensor_topk
  tril = T.c_THFloatTensor_tril
  triu = T.c_THFloatTensor_triu
  cat = T.c_THFloatTensor_cat
  catArray = T.c_THFloatTensor_catArray
  equal = T.c_THFloatTensor_equal
  ltValue = T.c_THFloatTensor_ltValue
  leValue = T.c_THFloatTensor_leValue
  gtValue = T.c_THFloatTensor_gtValue
  geValue = T.c_THFloatTensor_geValue
  neValue = T.c_THFloatTensor_neValue
  eqValue = T.c_THFloatTensor_eqValue
  ltValueT = T.c_THFloatTensor_ltValueT
  leValueT = T.c_THFloatTensor_leValueT
  gtValueT = T.c_THFloatTensor_gtValueT
  geValueT = T.c_THFloatTensor_geValueT
  neValueT = T.c_THFloatTensor_neValueT
  eqValueT = T.c_THFloatTensor_eqValueT
  ltTensor = T.c_THFloatTensor_ltTensor
  leTensor = T.c_THFloatTensor_leTensor
  gtTensor = T.c_THFloatTensor_gtTensor
  geTensor = T.c_THFloatTensor_geTensor
  neTensor = T.c_THFloatTensor_neTensor
  eqTensor = T.c_THFloatTensor_eqTensor
  ltTensorT = T.c_THFloatTensor_ltTensorT
  leTensorT = T.c_THFloatTensor_leTensorT
  gtTensorT = T.c_THFloatTensor_gtTensorT
  geTensorT = T.c_THFloatTensor_geTensorT
  neTensorT = T.c_THFloatTensor_neTensorT
  eqTensorT = T.c_THFloatTensor_eqTensorT

instance GenericNegativeOps CTHFloatTensor where
  neg = T.c_THFloatTensor_neg
  abs = T.c_THFloatTensor_abs

instance GenericFloatingMath CTHFloatTensor where
  cinv = T.c_THFloatTensor_cinv
  sigmoid = T.c_THFloatTensor_sigmoid
  log = T.c_THFloatTensor_log
  lgamma = T.c_THFloatTensor_lgamma
  log1p = T.c_THFloatTensor_log1p
  exp = T.c_THFloatTensor_exp
  cos = T.c_THFloatTensor_cos
  acos = T.c_THFloatTensor_acos
  cosh = T.c_THFloatTensor_cosh
  sin = T.c_THFloatTensor_sin
  asin = T.c_THFloatTensor_asin
  sinh = T.c_THFloatTensor_sinh
  tan = T.c_THFloatTensor_tan
  atan = T.c_THFloatTensor_atan
  atan2 = T.c_THFloatTensor_atan2
  tanh = T.c_THFloatTensor_tanh
  erf = T.c_THFloatTensor_erf
  erfinv = T.c_THFloatTensor_erfinv
  pow = T.c_THFloatTensor_pow
  tpow = T.c_THFloatTensor_tpow
  sqrt = T.c_THFloatTensor_sqrt
  rsqrt = T.c_THFloatTensor_rsqrt
  ceil = T.c_THFloatTensor_ceil
  floor = T.c_THFloatTensor_floor
  round = T.c_THFloatTensor_round
  trunc = T.c_THFloatTensor_trunc
  frac = T.c_THFloatTensor_frac
  lerp = T.c_THFloatTensor_lerp
  mean = T.c_THFloatTensor_mean
  std = T.c_THFloatTensor_std
  var = T.c_THFloatTensor_var
  norm = T.c_THFloatTensor_norm
  renorm = T.c_THFloatTensor_renorm
  dist = T.c_THFloatTensor_dist
  histc = T.c_THFloatTensor_histc
  bhistc = T.c_THFloatTensor_bhistc
  meanall = T.c_THFloatTensor_meanall
  varall = T.c_THFloatTensor_varall
  stdall = T.c_THFloatTensor_stdall
  normall = T.c_THFloatTensor_normall
  linspace = T.c_THFloatTensor_linspace
  logspace = T.c_THFloatTensor_logspace
  rand = T.c_THFloatTensor_rand
  randn = T.c_THFloatTensor_randn


instance GenericMath CTHIntTensor where
  fill = T.c_THIntTensor_fill
  zero = T.c_THIntTensor_zero
  maskedFill = T.c_THIntTensor_maskedFill
  maskedCopy = T.c_THIntTensor_maskedCopy
  maskedSelect = T.c_THIntTensor_maskedSelect
  nonzero = T.c_THIntTensor_nonzero
  indexSelect = T.c_THIntTensor_indexSelect
  indexCopy = T.c_THIntTensor_indexCopy
  indexAdd = T.c_THIntTensor_indexAdd
  indexFill = T.c_THIntTensor_indexFill
  take = T.c_THIntTensor_take
  put = T.c_THIntTensor_put
  gather = T.c_THIntTensor_gather
  scatter = T.c_THIntTensor_scatter
  scatterAdd = T.c_THIntTensor_scatterAdd
  scatterFill = T.c_THIntTensor_scatterFill
  dot = T.c_THIntTensor_dot
  minall = T.c_THIntTensor_minall
  maxall = T.c_THIntTensor_maxall
  medianall = T.c_THIntTensor_medianall
  sumall = T.c_THIntTensor_sumall
  prodall = T.c_THIntTensor_prodall
  add = T.c_THIntTensor_add
  sub = T.c_THIntTensor_sub
  add_scaled = T.c_THIntTensor_add_scaled
  sub_scaled = T.c_THIntTensor_sub_scaled
  mul = T.c_THIntTensor_mul
  div = T.c_THIntTensor_div
  lshift = T.c_THIntTensor_lshift
  rshift = T.c_THIntTensor_rshift
  fmod = T.c_THIntTensor_fmod
  remainder = T.c_THIntTensor_remainder
  clamp = T.c_THIntTensor_clamp
  bitand = T.c_THIntTensor_bitand
  bitor = T.c_THIntTensor_bitor
  bitxor = T.c_THIntTensor_bitxor
  cadd = T.c_THIntTensor_cadd
  csub = T.c_THIntTensor_csub
  cmul = T.c_THIntTensor_cmul
  cpow = T.c_THIntTensor_cpow
  cdiv = T.c_THIntTensor_cdiv
  clshift = T.c_THIntTensor_clshift
  crshift = T.c_THIntTensor_crshift
  cfmod = T.c_THIntTensor_cfmod
  cremainder = T.c_THIntTensor_cremainder
  cbitand = T.c_THIntTensor_cbitand
  cbitor = T.c_THIntTensor_cbitor
  cbitxor = T.c_THIntTensor_cbitxor
  addcmul = T.c_THIntTensor_addcmul
  addcdiv = T.c_THIntTensor_addcdiv
  addmv = T.c_THIntTensor_addmv
  addmm = T.c_THIntTensor_addmm
  addr = T.c_THIntTensor_addr
  addbmm = T.c_THIntTensor_addbmm
  baddbmm = T.c_THIntTensor_baddbmm
  match = T.c_THIntTensor_match
  numel = T.c_THIntTensor_numel
  max = T.c_THIntTensor_max
  min = T.c_THIntTensor_min
  kthvalue = T.c_THIntTensor_kthvalue
  mode = T.c_THIntTensor_mode
  median = T.c_THIntTensor_median
  sum = T.c_THIntTensor_sum
  prod = T.c_THIntTensor_prod
  cumsum = T.c_THIntTensor_cumsum
  cumprod = T.c_THIntTensor_cumprod
  sign = T.c_THIntTensor_sign
  trace = T.c_THIntTensor_trace
  cross = T.c_THIntTensor_cross
  cmax = T.c_THIntTensor_cmax
  cmin = T.c_THIntTensor_cmin
  cmaxValue = T.c_THIntTensor_cmaxValue
  cminValue = T.c_THIntTensor_cminValue
  zeros = T.c_THIntTensor_zeros
  zerosLike = T.c_THIntTensor_zerosLike
  ones = T.c_THIntTensor_ones
  onesLike = T.c_THIntTensor_onesLike
  diag = T.c_THIntTensor_diag
  eye = T.c_THIntTensor_eye
  arange = T.c_THIntTensor_arange
  range = T.c_THIntTensor_range
  randperm = T.c_THIntTensor_randperm
  reshape = T.c_THIntTensor_reshape
  sort = T.c_THIntTensor_sort
  topk = T.c_THIntTensor_topk
  tril = T.c_THIntTensor_tril
  triu = T.c_THIntTensor_triu
  cat = T.c_THIntTensor_cat
  catArray = T.c_THIntTensor_catArray
  equal = T.c_THIntTensor_equal
  ltValue = T.c_THIntTensor_ltValue
  leValue = T.c_THIntTensor_leValue
  gtValue = T.c_THIntTensor_gtValue
  geValue = T.c_THIntTensor_geValue
  neValue = T.c_THIntTensor_neValue
  eqValue = T.c_THIntTensor_eqValue
  ltValueT = T.c_THIntTensor_ltValueT
  leValueT = T.c_THIntTensor_leValueT
  gtValueT = T.c_THIntTensor_gtValueT
  geValueT = T.c_THIntTensor_geValueT
  neValueT = T.c_THIntTensor_neValueT
  eqValueT = T.c_THIntTensor_eqValueT
  ltTensor = T.c_THIntTensor_ltTensor
  leTensor = T.c_THIntTensor_leTensor
  gtTensor = T.c_THIntTensor_gtTensor
  geTensor = T.c_THIntTensor_geTensor
  neTensor = T.c_THIntTensor_neTensor
  eqTensor = T.c_THIntTensor_eqTensor
  ltTensorT = T.c_THIntTensor_ltTensorT
  leTensorT = T.c_THIntTensor_leTensorT
  gtTensorT = T.c_THIntTensor_gtTensorT
  geTensorT = T.c_THIntTensor_geTensorT
  neTensorT = T.c_THIntTensor_neTensorT
  eqTensorT = T.c_THIntTensor_eqTensorT

instance GenericNegativeOps CTHIntTensor where
  neg = T.c_THIntTensor_neg
  abs = T.c_THIntTensor_abs

instance GenericMath CTHLongTensor where
  fill = T.c_THLongTensor_fill
  zero = T.c_THLongTensor_zero
  maskedFill = T.c_THLongTensor_maskedFill
  maskedCopy = T.c_THLongTensor_maskedCopy
  maskedSelect = T.c_THLongTensor_maskedSelect
  nonzero = T.c_THLongTensor_nonzero
  indexSelect = T.c_THLongTensor_indexSelect
  indexCopy = T.c_THLongTensor_indexCopy
  indexAdd = T.c_THLongTensor_indexAdd
  indexFill = T.c_THLongTensor_indexFill
  take = T.c_THLongTensor_take
  put = T.c_THLongTensor_put
  gather = T.c_THLongTensor_gather
  scatter = T.c_THLongTensor_scatter
  scatterAdd = T.c_THLongTensor_scatterAdd
  scatterFill = T.c_THLongTensor_scatterFill
  dot = T.c_THLongTensor_dot
  minall = T.c_THLongTensor_minall
  maxall = T.c_THLongTensor_maxall
  medianall = T.c_THLongTensor_medianall
  sumall = T.c_THLongTensor_sumall
  prodall = T.c_THLongTensor_prodall
  add = T.c_THLongTensor_add
  sub = T.c_THLongTensor_sub
  add_scaled = T.c_THLongTensor_add_scaled
  sub_scaled = T.c_THLongTensor_sub_scaled
  mul = T.c_THLongTensor_mul
  div = T.c_THLongTensor_div
  lshift = T.c_THLongTensor_lshift
  rshift = T.c_THLongTensor_rshift
  fmod = T.c_THLongTensor_fmod
  remainder = T.c_THLongTensor_remainder
  clamp = T.c_THLongTensor_clamp
  bitand = T.c_THLongTensor_bitand
  bitor = T.c_THLongTensor_bitor
  bitxor = T.c_THLongTensor_bitxor
  cadd = T.c_THLongTensor_cadd
  csub = T.c_THLongTensor_csub
  cmul = T.c_THLongTensor_cmul
  cpow = T.c_THLongTensor_cpow
  cdiv = T.c_THLongTensor_cdiv
  clshift = T.c_THLongTensor_clshift
  crshift = T.c_THLongTensor_crshift
  cfmod = T.c_THLongTensor_cfmod
  cremainder = T.c_THLongTensor_cremainder
  cbitand = T.c_THLongTensor_cbitand
  cbitor = T.c_THLongTensor_cbitor
  cbitxor = T.c_THLongTensor_cbitxor
  addcmul = T.c_THLongTensor_addcmul
  addcdiv = T.c_THLongTensor_addcdiv
  addmv = T.c_THLongTensor_addmv
  addmm = T.c_THLongTensor_addmm
  addr = T.c_THLongTensor_addr
  addbmm = T.c_THLongTensor_addbmm
  baddbmm = T.c_THLongTensor_baddbmm
  match = T.c_THLongTensor_match
  numel = T.c_THLongTensor_numel
  max = T.c_THLongTensor_max
  min = T.c_THLongTensor_min
  kthvalue = T.c_THLongTensor_kthvalue
  mode = T.c_THLongTensor_mode
  median = T.c_THLongTensor_median
  sum = T.c_THLongTensor_sum
  prod = T.c_THLongTensor_prod
  cumsum = T.c_THLongTensor_cumsum
  cumprod = T.c_THLongTensor_cumprod
  sign = T.c_THLongTensor_sign
  trace = T.c_THLongTensor_trace
  cross = T.c_THLongTensor_cross
  cmax = T.c_THLongTensor_cmax
  cmin = T.c_THLongTensor_cmin
  cmaxValue = T.c_THLongTensor_cmaxValue
  cminValue = T.c_THLongTensor_cminValue
  zeros = T.c_THLongTensor_zeros
  zerosLike = T.c_THLongTensor_zerosLike
  ones = T.c_THLongTensor_ones
  onesLike = T.c_THLongTensor_onesLike
  diag = T.c_THLongTensor_diag
  eye = T.c_THLongTensor_eye
  arange = T.c_THLongTensor_arange
  range = T.c_THLongTensor_range
  randperm = T.c_THLongTensor_randperm
  reshape = T.c_THLongTensor_reshape
  sort = T.c_THLongTensor_sort
  topk = T.c_THLongTensor_topk
  tril = T.c_THLongTensor_tril
  triu = T.c_THLongTensor_triu
  cat = T.c_THLongTensor_cat
  catArray = T.c_THLongTensor_catArray
  equal = T.c_THLongTensor_equal
  ltValue = T.c_THLongTensor_ltValue
  leValue = T.c_THLongTensor_leValue
  gtValue = T.c_THLongTensor_gtValue
  geValue = T.c_THLongTensor_geValue
  neValue = T.c_THLongTensor_neValue
  eqValue = T.c_THLongTensor_eqValue
  ltValueT = T.c_THLongTensor_ltValueT
  leValueT = T.c_THLongTensor_leValueT
  gtValueT = T.c_THLongTensor_gtValueT
  geValueT = T.c_THLongTensor_geValueT
  neValueT = T.c_THLongTensor_neValueT
  eqValueT = T.c_THLongTensor_eqValueT
  ltTensor = T.c_THLongTensor_ltTensor
  leTensor = T.c_THLongTensor_leTensor
  gtTensor = T.c_THLongTensor_gtTensor
  geTensor = T.c_THLongTensor_geTensor
  neTensor = T.c_THLongTensor_neTensor
  eqTensor = T.c_THLongTensor_eqTensor
  ltTensorT = T.c_THLongTensor_ltTensorT
  leTensorT = T.c_THLongTensor_leTensorT
  gtTensorT = T.c_THLongTensor_gtTensorT
  geTensorT = T.c_THLongTensor_geTensorT
  neTensorT = T.c_THLongTensor_neTensorT
  eqTensorT = T.c_THLongTensor_eqTensorT

instance GenericNegativeOps CTHLongTensor where
  neg = T.c_THLongTensor_neg
  abs = T.c_THLongTensor_abs

instance GenericMath CTHShortTensor where
  fill = T.c_THShortTensor_fill
  zero = T.c_THShortTensor_zero
  maskedFill = T.c_THShortTensor_maskedFill
  maskedCopy = T.c_THShortTensor_maskedCopy
  maskedSelect = T.c_THShortTensor_maskedSelect
  nonzero = T.c_THShortTensor_nonzero
  indexSelect = T.c_THShortTensor_indexSelect
  indexCopy = T.c_THShortTensor_indexCopy
  indexAdd = T.c_THShortTensor_indexAdd
  indexFill = T.c_THShortTensor_indexFill
  take = T.c_THShortTensor_take
  put = T.c_THShortTensor_put
  gather = T.c_THShortTensor_gather
  scatter = T.c_THShortTensor_scatter
  scatterAdd = T.c_THShortTensor_scatterAdd
  scatterFill = T.c_THShortTensor_scatterFill
  dot = T.c_THShortTensor_dot
  minall = T.c_THShortTensor_minall
  maxall = T.c_THShortTensor_maxall
  medianall = T.c_THShortTensor_medianall
  sumall = T.c_THShortTensor_sumall
  prodall = T.c_THShortTensor_prodall
  add = T.c_THShortTensor_add
  sub = T.c_THShortTensor_sub
  add_scaled = T.c_THShortTensor_add_scaled
  sub_scaled = T.c_THShortTensor_sub_scaled
  mul = T.c_THShortTensor_mul
  div = T.c_THShortTensor_div
  lshift = T.c_THShortTensor_lshift
  rshift = T.c_THShortTensor_rshift
  fmod = T.c_THShortTensor_fmod
  remainder = T.c_THShortTensor_remainder
  clamp = T.c_THShortTensor_clamp
  bitand = T.c_THShortTensor_bitand
  bitor = T.c_THShortTensor_bitor
  bitxor = T.c_THShortTensor_bitxor
  cadd = T.c_THShortTensor_cadd
  csub = T.c_THShortTensor_csub
  cmul = T.c_THShortTensor_cmul
  cpow = T.c_THShortTensor_cpow
  cdiv = T.c_THShortTensor_cdiv
  clshift = T.c_THShortTensor_clshift
  crshift = T.c_THShortTensor_crshift
  cfmod = T.c_THShortTensor_cfmod
  cremainder = T.c_THShortTensor_cremainder
  cbitand = T.c_THShortTensor_cbitand
  cbitor = T.c_THShortTensor_cbitor
  cbitxor = T.c_THShortTensor_cbitxor
  addcmul = T.c_THShortTensor_addcmul
  addcdiv = T.c_THShortTensor_addcdiv
  addmv = T.c_THShortTensor_addmv
  addmm = T.c_THShortTensor_addmm
  addr = T.c_THShortTensor_addr
  addbmm = T.c_THShortTensor_addbmm
  baddbmm = T.c_THShortTensor_baddbmm
  match = T.c_THShortTensor_match
  numel = T.c_THShortTensor_numel
  max = T.c_THShortTensor_max
  min = T.c_THShortTensor_min
  kthvalue = T.c_THShortTensor_kthvalue
  mode = T.c_THShortTensor_mode
  median = T.c_THShortTensor_median
  sum = T.c_THShortTensor_sum
  prod = T.c_THShortTensor_prod
  cumsum = T.c_THShortTensor_cumsum
  cumprod = T.c_THShortTensor_cumprod
  sign = T.c_THShortTensor_sign
  trace = T.c_THShortTensor_trace
  cross = T.c_THShortTensor_cross
  cmax = T.c_THShortTensor_cmax
  cmin = T.c_THShortTensor_cmin
  cmaxValue = T.c_THShortTensor_cmaxValue
  cminValue = T.c_THShortTensor_cminValue
  zeros = T.c_THShortTensor_zeros
  zerosLike = T.c_THShortTensor_zerosLike
  ones = T.c_THShortTensor_ones
  onesLike = T.c_THShortTensor_onesLike
  diag = T.c_THShortTensor_diag
  eye = T.c_THShortTensor_eye
  arange = T.c_THShortTensor_arange
  range = T.c_THShortTensor_range
  randperm = T.c_THShortTensor_randperm
  reshape = T.c_THShortTensor_reshape
  sort = T.c_THShortTensor_sort
  topk = T.c_THShortTensor_topk
  tril = T.c_THShortTensor_tril
  triu = T.c_THShortTensor_triu
  cat = T.c_THShortTensor_cat
  catArray = T.c_THShortTensor_catArray
  equal = T.c_THShortTensor_equal
  ltValue = T.c_THShortTensor_ltValue
  leValue = T.c_THShortTensor_leValue
  gtValue = T.c_THShortTensor_gtValue
  geValue = T.c_THShortTensor_geValue
  neValue = T.c_THShortTensor_neValue
  eqValue = T.c_THShortTensor_eqValue
  ltValueT = T.c_THShortTensor_ltValueT
  leValueT = T.c_THShortTensor_leValueT
  gtValueT = T.c_THShortTensor_gtValueT
  geValueT = T.c_THShortTensor_geValueT
  neValueT = T.c_THShortTensor_neValueT
  eqValueT = T.c_THShortTensor_eqValueT
  ltTensor = T.c_THShortTensor_ltTensor
  leTensor = T.c_THShortTensor_leTensor
  gtTensor = T.c_THShortTensor_gtTensor
  geTensor = T.c_THShortTensor_geTensor
  neTensor = T.c_THShortTensor_neTensor
  eqTensor = T.c_THShortTensor_eqTensor
  ltTensorT = T.c_THShortTensor_ltTensorT
  leTensorT = T.c_THShortTensor_leTensorT
  gtTensorT = T.c_THShortTensor_gtTensorT
  geTensorT = T.c_THShortTensor_geTensorT
  neTensorT = T.c_THShortTensor_neTensorT
  eqTensorT = T.c_THShortTensor_eqTensorT

instance GenericNegativeOps CTHShortTensor where
  neg = T.c_THShortTensor_neg
  abs = T.c_THShortTensor_abs

