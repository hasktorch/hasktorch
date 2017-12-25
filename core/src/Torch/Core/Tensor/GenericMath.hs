{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.GenericMath where

import THTypes
import Foreign (Ptr)
import Foreign.C.Types
import qualified THByteTensorMath as T
import qualified THDoubleTensorMath as T
import qualified THFloatTensorMath as T
import qualified THIntTensorMath as T
import qualified THLongTensorMath as T
import qualified THShortTensorMath as T

import Torch.Core.Internal (HaskType)

type SHOULD_BE_HASK_TYPE = CDouble

class GenericMath t where
  fill         :: Ptr t -> HaskType t -> IO ()
  zero         :: Ptr t -> IO ()
  maskedFill   :: Ptr t -> Ptr CTHByteTensor -> HaskType t -> IO ()
  maskedCopy   :: Ptr t -> Ptr CTHByteTensor -> Ptr t -> IO ()
  maskedSelect :: Ptr t -> Ptr t -> Ptr CTHByteTensor -> IO ()
  nonzero      :: Ptr CTHLongTensor -> Ptr t -> IO ()
  indexSelect  :: Ptr t -> Ptr t -> CInt -> Ptr CTHLongTensor -> IO ()
  indexCopy    :: Ptr t -> CInt -> Ptr CTHLongTensor -> Ptr t -> IO ()
  indexAdd     :: Ptr t -> CInt -> Ptr CTHLongTensor -> Ptr t -> IO ()
  indexFill    :: Ptr t -> CInt -> Ptr CTHLongTensor -> HaskType t -> IO ()
  take         :: Ptr t -> Ptr t -> Ptr CTHLongTensor -> IO ()
  put          :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> IO ()
  gather       :: Ptr t -> Ptr t -> CInt -> Ptr CTHLongTensor -> IO ()
  scatter      :: Ptr t -> CInt -> Ptr CTHLongTensor -> Ptr t -> IO ()
  scatterAdd   :: Ptr t -> CInt -> Ptr CTHLongTensor -> Ptr t -> IO ()
  scatterFill  :: Ptr t -> CInt -> Ptr CTHLongTensor -> HaskType t -> IO ()
  dot          :: Ptr t -> Ptr t -> SHOULD_BE_HASK_TYPE
  minall       :: Ptr t -> HaskType t
  maxall       :: Ptr t -> HaskType t
  medianall    :: Ptr t -> HaskType t
  sumall       :: Ptr t -> SHOULD_BE_HASK_TYPE
  prodall      :: Ptr t -> SHOULD_BE_HASK_TYPE
  neg          :: Ptr t -> Ptr t -> IO ()
  cinv         :: Ptr t -> Ptr t -> IO ()
  add          :: Ptr t -> Ptr t -> HaskType t -> IO ()
  sub          :: Ptr t -> Ptr t -> HaskType t -> IO ()
  add_scaled   :: Ptr t -> Ptr t -> HaskType t -> HaskType t -> IO ()
  sub_scaled   :: Ptr t -> Ptr t -> HaskType t -> HaskType t -> IO ()
  mul          :: Ptr t -> Ptr t -> HaskType t -> IO ()
  div          :: Ptr t -> Ptr t -> HaskType t -> IO ()
  lshift       :: Ptr t -> Ptr t -> HaskType t -> IO ()
  rshift       :: Ptr t -> Ptr t -> HaskType t -> IO ()
  fmod         :: Ptr t -> Ptr t -> HaskType t -> IO ()
  remainder    :: Ptr t -> Ptr t -> HaskType t -> IO ()
  clamp        :: Ptr t -> Ptr t -> HaskType t -> HaskType t -> IO ()
  bitand       :: Ptr t -> Ptr t -> HaskType t -> IO ()
  bitor        :: Ptr t -> Ptr t -> HaskType t -> IO ()
  bitxor       :: Ptr t -> Ptr t -> HaskType t -> IO ()
  cadd         :: Ptr t -> Ptr t -> HaskType t -> Ptr t -> IO ()
  csub         :: Ptr t -> Ptr t -> HaskType t -> Ptr t -> IO ()
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
  addcmul      :: Ptr t -> Ptr t -> HaskType t -> Ptr t -> Ptr t -> IO ()
  addcdiv      :: Ptr t -> Ptr t -> HaskType t -> Ptr t -> Ptr t -> IO ()
  addmv        :: Ptr t -> HaskType t -> Ptr t -> HaskType t -> Ptr t -> Ptr t -> IO ()
  addmm        :: Ptr t -> HaskType t -> Ptr t -> HaskType t -> Ptr t -> Ptr t -> IO ()
  addr         :: Ptr t -> HaskType t -> Ptr t -> HaskType t -> Ptr t -> Ptr t -> IO ()
  addbmm       :: Ptr t -> HaskType t -> Ptr t -> HaskType t -> Ptr t -> Ptr t -> IO ()
  baddbmm      :: Ptr t -> HaskType t -> Ptr t -> HaskType t -> Ptr t -> Ptr t -> IO ()
  match        :: Ptr t -> Ptr t -> Ptr t -> HaskType t -> IO ()
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
  trace        :: Ptr t -> SHOULD_BE_HASK_TYPE
  cross        :: Ptr t -> Ptr t -> Ptr t -> CInt -> IO ()
  cmax         :: Ptr t -> Ptr t -> Ptr t -> IO ()
  cmin         :: Ptr t -> Ptr t -> Ptr t -> IO ()
  cmaxValue    :: Ptr t -> Ptr t -> HaskType t -> IO ()
  cminValue    :: Ptr t -> Ptr t -> HaskType t -> IO ()
  zeros        :: Ptr t -> Ptr CTHLongStorage -> IO ()
  zerosLike    :: Ptr t -> Ptr t -> IO ()
  ones         :: Ptr t -> Ptr CTHLongStorage -> IO ()
  onesLike     :: Ptr t -> Ptr t -> IO ()
  diag         :: Ptr t -> Ptr t -> CInt -> IO ()
  eye          :: Ptr t -> CLLong -> CLLong -> IO ()
  arange       :: Ptr t -> SHOULD_BE_HASK_TYPE -> SHOULD_BE_HASK_TYPE -> SHOULD_BE_HASK_TYPE -> IO ()
  range        :: Ptr t -> SHOULD_BE_HASK_TYPE -> SHOULD_BE_HASK_TYPE -> SHOULD_BE_HASK_TYPE -> IO ()
  randperm     :: Ptr t -> Ptr CTHGenerator -> CLLong -> IO ()
  reshape      :: Ptr t -> Ptr t -> Ptr CTHLongStorage -> IO ()
  sort         :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> CInt -> IO ()
  topk         :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CLLong -> CInt -> CInt -> CInt -> IO ()
  tril         :: Ptr t -> Ptr t -> CLLong -> IO ()
  triu         :: Ptr t -> Ptr t -> CLLong -> IO ()
  cat          :: Ptr t -> Ptr t -> Ptr t -> CInt -> IO ()
  catArray     :: Ptr t -> Ptr (Ptr t) -> CInt -> CInt -> IO ()
  equal        :: Ptr t -> Ptr t -> CInt
  ltValue      :: Ptr CTHByteTensor -> Ptr t -> HaskType t -> IO ()
  leValue      :: Ptr CTHByteTensor -> Ptr t -> HaskType t -> IO ()
  gtValue      :: Ptr CTHByteTensor -> Ptr t -> HaskType t -> IO ()
  geValue      :: Ptr CTHByteTensor -> Ptr t -> HaskType t -> IO ()
  neValue      :: Ptr CTHByteTensor -> Ptr t -> HaskType t -> IO ()
  eqValue      :: Ptr CTHByteTensor -> Ptr t -> HaskType t -> IO ()
  ltValueT     :: Ptr t -> Ptr t -> HaskType t -> IO ()
  leValueT     :: Ptr t -> Ptr t -> HaskType t -> IO ()
  gtValueT     :: Ptr t -> Ptr t -> HaskType t -> IO ()
  geValueT     :: Ptr t -> Ptr t -> HaskType t -> IO ()
  neValueT     :: Ptr t -> Ptr t -> HaskType t -> IO ()
  eqValueT     :: Ptr t -> Ptr t -> HaskType t -> IO ()
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
  abs          :: Ptr t -> Ptr t -> IO ()
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
  pow          :: Ptr t -> Ptr t -> HaskType t -> IO ()
  tpow         :: Ptr t -> HaskType t -> Ptr t -> IO ()
  sqrt         :: Ptr t -> Ptr t -> IO ()
  rsqrt        :: Ptr t -> Ptr t -> IO ()
  ceil         :: Ptr t -> Ptr t -> IO ()
  floor        :: Ptr t -> Ptr t -> IO ()
  round        :: Ptr t -> Ptr t -> IO ()
  trunc        :: Ptr t -> Ptr t -> IO ()
  frac         :: Ptr t -> Ptr t -> IO ()
  lerp         :: Ptr t -> Ptr t -> Ptr t -> HaskType t -> IO ()
  mean         :: Ptr t -> Ptr t -> CInt -> CInt -> IO ()
  std          :: Ptr t -> Ptr t -> CInt -> CInt -> CInt -> IO ()
  var          :: Ptr t -> Ptr t -> CInt -> CInt -> CInt -> IO ()
  norm         :: Ptr t -> Ptr t -> HaskType t -> CInt -> CInt -> IO ()
  renorm       :: Ptr t -> Ptr t -> HaskType t -> CInt -> HaskType t -> IO ()
  dist         :: Ptr t -> Ptr t -> HaskType t -> SHOULD_BE_HASK_TYPE
  histc        :: Ptr t -> Ptr t -> CLLong -> HaskType t -> HaskType t -> IO ()
  bhistc       :: Ptr t -> Ptr t -> CLLong -> HaskType t -> HaskType t -> IO ()
  meanall      :: Ptr t -> SHOULD_BE_HASK_TYPE
  varall       :: Ptr t -> CInt -> SHOULD_BE_HASK_TYPE
  stdall       :: Ptr t -> CInt -> SHOULD_BE_HASK_TYPE
  normall      :: Ptr t -> HaskType t -> SHOULD_BE_HASK_TYPE
  linspace     :: Ptr t -> HaskType t -> HaskType t -> CLLong -> IO ()
  logspace     :: Ptr t -> HaskType t -> HaskType t -> CLLong -> IO ()
  rand         :: Ptr t -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()
  randn        :: Ptr t -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

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
  neg = T.c_THDoubleTensor_neg
  cinv = T.c_THDoubleTensor_cinv
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
  abs = T.c_THDoubleTensor_abs
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
  neg = T.c_THFloatTensor_neg
  cinv = T.c_THFloatTensor_cinv
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
  abs = T.c_THFloatTensor_abs
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


