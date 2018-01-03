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
  c_fill         :: Ptr t -> HaskReal t -> IO ()
  c_zero         :: Ptr t -> IO ()
  c_maskedFill   :: Ptr t -> Ptr CTHByteTensor -> HaskReal t -> IO ()
  c_maskedCopy   :: Ptr t -> Ptr CTHByteTensor -> Ptr t -> IO ()
  c_maskedSelect :: Ptr t -> Ptr t -> Ptr CTHByteTensor -> IO ()
  c_nonzero      :: Ptr CTHLongTensor -> Ptr t -> IO ()
  c_indexSelect  :: Ptr t -> Ptr t -> CInt -> Ptr CTHLongTensor -> IO ()
  c_indexCopy    :: Ptr t -> CInt -> Ptr CTHLongTensor -> Ptr t -> IO ()
  c_indexAdd     :: Ptr t -> CInt -> Ptr CTHLongTensor -> Ptr t -> IO ()
  c_indexFill    :: Ptr t -> CInt -> Ptr CTHLongTensor -> HaskReal t -> IO ()
  c_take         :: Ptr t -> Ptr t -> Ptr CTHLongTensor -> IO ()
  c_put          :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> IO ()
  c_gather       :: Ptr t -> Ptr t -> CInt -> Ptr CTHLongTensor -> IO ()
  c_scatter      :: Ptr t -> CInt -> Ptr CTHLongTensor -> Ptr t -> IO ()
  c_scatterAdd   :: Ptr t -> CInt -> Ptr CTHLongTensor -> Ptr t -> IO ()
  c_scatterFill  :: Ptr t -> CInt -> Ptr CTHLongTensor -> HaskReal t -> IO ()
  c_dot          :: Ptr t -> Ptr t -> HaskAccReal t
  c_minall       :: Ptr t -> HaskReal t
  c_maxall       :: Ptr t -> HaskReal t
  c_medianall    :: Ptr t -> HaskReal t
  c_sumall       :: Ptr t -> HaskAccReal t
  c_prodall      :: Ptr t -> HaskAccReal t
  c_add          :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_sub          :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_add_scaled   :: Ptr t -> Ptr t -> HaskReal t -> HaskReal t -> IO ()
  c_sub_scaled   :: Ptr t -> Ptr t -> HaskReal t -> HaskReal t -> IO ()
  c_mul          :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_div          :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_lshift       :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_rshift       :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_fmod         :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_remainder    :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_clamp        :: Ptr t -> Ptr t -> HaskReal t -> HaskReal t -> IO ()
  c_bitand       :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_bitor        :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_bitxor       :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_cadd         :: Ptr t -> Ptr t -> HaskReal t -> Ptr t -> IO ()
  c_csub         :: Ptr t -> Ptr t -> HaskReal t -> Ptr t -> IO ()
  c_cmul         :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_cpow         :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_cdiv         :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_clshift      :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_crshift      :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_cfmod        :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_cremainder   :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_cbitand      :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_cbitor       :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_cbitxor      :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_addcmul      :: Ptr t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  c_addcdiv      :: Ptr t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  c_addmv        :: Ptr t -> HaskReal t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  c_addmm        :: Ptr t -> HaskReal t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  c_addr         :: Ptr t -> HaskReal t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  c_addbmm       :: Ptr t -> HaskReal t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  c_baddbmm      :: Ptr t -> HaskReal t -> Ptr t -> HaskReal t -> Ptr t -> Ptr t -> IO ()
  c_match        :: Ptr t -> Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_numel        :: Ptr t -> CPtrdiff
  c_max          :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> CInt -> IO ()
  c_min          :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> CInt -> IO ()
  c_kthvalue     :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CLLong -> CInt -> CInt -> IO ()
  c_mode         :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> CInt -> IO ()
  c_median       :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> CInt -> IO ()
  c_sum          :: Ptr t -> Ptr t -> CInt -> CInt -> IO ()
  c_prod         :: Ptr t -> Ptr t -> CInt -> CInt -> IO ()
  c_cumsum       :: Ptr t -> Ptr t -> CInt -> IO ()
  c_cumprod      :: Ptr t -> Ptr t -> CInt -> IO ()
  c_sign         :: Ptr t -> Ptr t -> IO ()
  c_trace        :: Ptr t -> HaskAccReal t
  c_cross        :: Ptr t -> Ptr t -> Ptr t -> CInt -> IO ()
  c_cmax         :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_cmin         :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_cmaxValue    :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_cminValue    :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_zeros        :: Ptr t -> Ptr CTHLongStorage -> IO ()
  c_zerosLike    :: Ptr t -> Ptr t -> IO ()
  c_ones         :: Ptr t -> Ptr CTHLongStorage -> IO ()
  c_onesLike     :: Ptr t -> Ptr t -> IO ()
  c_diag         :: Ptr t -> Ptr t -> CInt -> IO ()
  c_eye          :: Ptr t -> CLLong -> CLLong -> IO ()
  c_arange       :: Ptr t -> HaskAccReal t -> HaskAccReal t -> HaskAccReal t -> IO ()
  c_range        :: Ptr t -> HaskAccReal t -> HaskAccReal t -> HaskAccReal t -> IO ()
  c_randperm     :: Ptr t -> Ptr CTHGenerator -> CLLong -> IO ()
  c_reshape      :: Ptr t -> Ptr t -> Ptr CTHLongStorage -> IO ()
  c_sort         :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CInt -> CInt -> IO ()
  c_topk         :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> CLLong -> CInt -> CInt -> CInt -> IO ()
  c_tril         :: Ptr t -> Ptr t -> CLLong -> IO ()
  c_triu         :: Ptr t -> Ptr t -> CLLong -> IO ()
  c_cat          :: Ptr t -> Ptr t -> Ptr t -> CInt -> IO ()
  c_catArray     :: Ptr t -> Ptr (Ptr t) -> CInt -> CInt -> IO ()
  c_equal        :: Ptr t -> Ptr t -> CInt
  c_ltValue      :: Ptr CTHByteTensor -> Ptr t -> HaskReal t -> IO ()
  c_leValue      :: Ptr CTHByteTensor -> Ptr t -> HaskReal t -> IO ()
  c_gtValue      :: Ptr CTHByteTensor -> Ptr t -> HaskReal t -> IO ()
  c_geValue      :: Ptr CTHByteTensor -> Ptr t -> HaskReal t -> IO ()
  c_neValue      :: Ptr CTHByteTensor -> Ptr t -> HaskReal t -> IO ()
  c_eqValue      :: Ptr CTHByteTensor -> Ptr t -> HaskReal t -> IO ()
  c_ltValueT     :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_leValueT     :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_gtValueT     :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_geValueT     :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_neValueT     :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_eqValueT     :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_ltTensor     :: Ptr CTHByteTensor -> Ptr t -> Ptr t -> IO ()
  c_leTensor     :: Ptr CTHByteTensor -> Ptr t -> Ptr t -> IO ()
  c_gtTensor     :: Ptr CTHByteTensor -> Ptr t -> Ptr t -> IO ()
  c_geTensor     :: Ptr CTHByteTensor -> Ptr t -> Ptr t -> IO ()
  c_neTensor     :: Ptr CTHByteTensor -> Ptr t -> Ptr t -> IO ()
  c_eqTensor     :: Ptr CTHByteTensor -> Ptr t -> Ptr t -> IO ()
  c_ltTensorT    :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_leTensorT    :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_gtTensorT    :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_geTensorT    :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_neTensorT    :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_eqTensorT    :: Ptr t -> Ptr t -> Ptr t -> IO ()

class GenericMath t => GenericNegativeOps t where
  c_neg          :: Ptr t -> Ptr t -> IO ()
  c_abs          :: Ptr t -> Ptr t -> IO ()

class GenericMath t => GenericFloatingMath t where
  c_cinv         :: Ptr t -> Ptr t -> IO ()
  c_sigmoid      :: Ptr t -> Ptr t -> IO ()
  c_log          :: Ptr t -> Ptr t -> IO ()
  c_lgamma       :: Ptr t -> Ptr t -> IO ()
  c_log1p        :: Ptr t -> Ptr t -> IO ()
  c_exp          :: Ptr t -> Ptr t -> IO ()
  c_cos          :: Ptr t -> Ptr t -> IO ()
  c_acos         :: Ptr t -> Ptr t -> IO ()
  c_cosh         :: Ptr t -> Ptr t -> IO ()
  c_sin          :: Ptr t -> Ptr t -> IO ()
  c_asin         :: Ptr t -> Ptr t -> IO ()
  c_sinh         :: Ptr t -> Ptr t -> IO ()
  c_tan          :: Ptr t -> Ptr t -> IO ()
  c_atan         :: Ptr t -> Ptr t -> IO ()
  c_atan2        :: Ptr t -> Ptr t -> Ptr t -> IO ()
  c_tanh         :: Ptr t -> Ptr t -> IO ()
  c_erf          :: Ptr t -> Ptr t -> IO ()
  c_erfinv       :: Ptr t -> Ptr t -> IO ()
  c_pow          :: Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_tpow         :: Ptr t -> HaskReal t -> Ptr t -> IO ()
  c_sqrt         :: Ptr t -> Ptr t -> IO ()
  c_rsqrt        :: Ptr t -> Ptr t -> IO ()
  c_ceil         :: Ptr t -> Ptr t -> IO ()
  c_floor        :: Ptr t -> Ptr t -> IO ()
  c_round        :: Ptr t -> Ptr t -> IO ()
  c_trunc        :: Ptr t -> Ptr t -> IO ()
  c_frac         :: Ptr t -> Ptr t -> IO ()
  c_lerp         :: Ptr t -> Ptr t -> Ptr t -> HaskReal t -> IO ()
  c_mean         :: Ptr t -> Ptr t -> CInt -> CInt -> IO ()
  c_std          :: Ptr t -> Ptr t -> CInt -> CInt -> CInt -> IO ()
  c_var          :: Ptr t -> Ptr t -> CInt -> CInt -> CInt -> IO ()
  c_norm         :: Ptr t -> Ptr t -> HaskReal t -> CInt -> CInt -> IO ()
  c_renorm       :: Ptr t -> Ptr t -> HaskReal t -> CInt -> HaskReal t -> IO ()
  c_dist         :: Ptr t -> Ptr t -> HaskReal t -> HaskAccReal t
  c_histc        :: Ptr t -> Ptr t -> CLLong -> HaskReal t -> HaskReal t -> IO ()
  c_bhistc       :: Ptr t -> Ptr t -> CLLong -> HaskReal t -> HaskReal t -> IO ()
  c_meanall      :: Ptr t -> HaskAccReal t
  c_varall       :: Ptr t -> CInt -> HaskAccReal t
  c_stdall       :: Ptr t -> CInt -> HaskAccReal t
  c_normall      :: Ptr t -> HaskReal t -> HaskAccReal t
  c_linspace     :: Ptr t -> HaskReal t -> HaskReal t -> CLLong -> IO ()
  c_logspace     :: Ptr t -> HaskReal t -> HaskReal t -> CLLong -> IO ()
  c_rand         :: Ptr t -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()
  c_randn        :: Ptr t -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()


instance GenericMath CTHByteTensor where
  c_fill = T.c_THByteTensor_fill
  c_zero = T.c_THByteTensor_zero
  c_maskedFill = T.c_THByteTensor_maskedFill
  c_maskedCopy = T.c_THByteTensor_maskedCopy
  c_maskedSelect = T.c_THByteTensor_maskedSelect
  c_nonzero = T.c_THByteTensor_nonzero
  c_indexSelect = T.c_THByteTensor_indexSelect
  c_indexCopy = T.c_THByteTensor_indexCopy
  c_indexAdd = T.c_THByteTensor_indexAdd
  c_indexFill = T.c_THByteTensor_indexFill
  c_take = T.c_THByteTensor_take
  c_put = T.c_THByteTensor_put
  c_gather = T.c_THByteTensor_gather
  c_scatter = T.c_THByteTensor_scatter
  c_scatterAdd = T.c_THByteTensor_scatterAdd
  c_scatterFill = T.c_THByteTensor_scatterFill
  c_dot = T.c_THByteTensor_dot
  c_minall = T.c_THByteTensor_minall
  c_maxall = T.c_THByteTensor_maxall
  c_medianall = T.c_THByteTensor_medianall
  c_sumall = T.c_THByteTensor_sumall
  c_prodall = T.c_THByteTensor_prodall
  c_add = T.c_THByteTensor_add
  c_sub = T.c_THByteTensor_sub
  c_add_scaled = T.c_THByteTensor_add_scaled
  c_sub_scaled = T.c_THByteTensor_sub_scaled
  c_mul = T.c_THByteTensor_mul
  c_div = T.c_THByteTensor_div
  c_lshift = T.c_THByteTensor_lshift
  c_rshift = T.c_THByteTensor_rshift
  c_fmod = T.c_THByteTensor_fmod
  c_remainder = T.c_THByteTensor_remainder
  c_clamp = T.c_THByteTensor_clamp
  c_bitand = T.c_THByteTensor_bitand
  c_bitor = T.c_THByteTensor_bitor
  c_bitxor = T.c_THByteTensor_bitxor
  c_cadd = T.c_THByteTensor_cadd
  c_csub = T.c_THByteTensor_csub
  c_cmul = T.c_THByteTensor_cmul
  c_cpow = T.c_THByteTensor_cpow
  c_cdiv = T.c_THByteTensor_cdiv
  c_clshift = T.c_THByteTensor_clshift
  c_crshift = T.c_THByteTensor_crshift
  c_cfmod = T.c_THByteTensor_cfmod
  c_cremainder = T.c_THByteTensor_cremainder
  c_cbitand = T.c_THByteTensor_cbitand
  c_cbitor = T.c_THByteTensor_cbitor
  c_cbitxor = T.c_THByteTensor_cbitxor
  c_addcmul = T.c_THByteTensor_addcmul
  c_addcdiv = T.c_THByteTensor_addcdiv
  c_addmv = T.c_THByteTensor_addmv
  c_addmm = T.c_THByteTensor_addmm
  c_addr = T.c_THByteTensor_addr
  c_addbmm = T.c_THByteTensor_addbmm
  c_baddbmm = T.c_THByteTensor_baddbmm
  c_match = T.c_THByteTensor_match
  c_numel = T.c_THByteTensor_numel
  c_max = T.c_THByteTensor_max
  c_min = T.c_THByteTensor_min
  c_kthvalue = T.c_THByteTensor_kthvalue
  c_mode = T.c_THByteTensor_mode
  c_median = T.c_THByteTensor_median
  c_sum = T.c_THByteTensor_sum
  c_prod = T.c_THByteTensor_prod
  c_cumsum = T.c_THByteTensor_cumsum
  c_cumprod = T.c_THByteTensor_cumprod
  c_sign = T.c_THByteTensor_sign
  c_trace = T.c_THByteTensor_trace
  c_cross = T.c_THByteTensor_cross
  c_cmax = T.c_THByteTensor_cmax
  c_cmin = T.c_THByteTensor_cmin
  c_cmaxValue = T.c_THByteTensor_cmaxValue
  c_cminValue = T.c_THByteTensor_cminValue
  c_zeros = T.c_THByteTensor_zeros
  c_zerosLike = T.c_THByteTensor_zerosLike
  c_ones = T.c_THByteTensor_ones
  c_onesLike = T.c_THByteTensor_onesLike
  c_diag = T.c_THByteTensor_diag
  c_eye = T.c_THByteTensor_eye
  c_arange = T.c_THByteTensor_arange
  c_range = T.c_THByteTensor_range
  c_randperm = T.c_THByteTensor_randperm
  c_reshape = T.c_THByteTensor_reshape
  c_sort = T.c_THByteTensor_sort
  c_topk = T.c_THByteTensor_topk
  c_tril = T.c_THByteTensor_tril
  c_triu = T.c_THByteTensor_triu
  c_cat = T.c_THByteTensor_cat
  c_catArray = T.c_THByteTensor_catArray
  c_equal = T.c_THByteTensor_equal
  c_ltValue = T.c_THByteTensor_ltValue
  c_leValue = T.c_THByteTensor_leValue
  c_gtValue = T.c_THByteTensor_gtValue
  c_geValue = T.c_THByteTensor_geValue
  c_neValue = T.c_THByteTensor_neValue
  c_eqValue = T.c_THByteTensor_eqValue
  c_ltValueT = T.c_THByteTensor_ltValueT
  c_leValueT = T.c_THByteTensor_leValueT
  c_gtValueT = T.c_THByteTensor_gtValueT
  c_geValueT = T.c_THByteTensor_geValueT
  c_neValueT = T.c_THByteTensor_neValueT
  c_eqValueT = T.c_THByteTensor_eqValueT
  c_ltTensor = T.c_THByteTensor_ltTensor
  c_leTensor = T.c_THByteTensor_leTensor
  c_gtTensor = T.c_THByteTensor_gtTensor
  c_geTensor = T.c_THByteTensor_geTensor
  c_neTensor = T.c_THByteTensor_neTensor
  c_eqTensor = T.c_THByteTensor_eqTensor
  c_ltTensorT = T.c_THByteTensor_ltTensorT
  c_leTensorT = T.c_THByteTensor_leTensorT
  c_gtTensorT = T.c_THByteTensor_gtTensorT
  c_geTensorT = T.c_THByteTensor_geTensorT
  c_neTensorT = T.c_THByteTensor_neTensorT
  c_eqTensorT = T.c_THByteTensor_eqTensorT

instance GenericMath CTHDoubleTensor where
  c_fill = T.c_THDoubleTensor_fill
  c_zero = T.c_THDoubleTensor_zero
  c_maskedFill = T.c_THDoubleTensor_maskedFill
  c_maskedCopy = T.c_THDoubleTensor_maskedCopy
  c_maskedSelect = T.c_THDoubleTensor_maskedSelect
  c_nonzero = T.c_THDoubleTensor_nonzero
  c_indexSelect = T.c_THDoubleTensor_indexSelect
  c_indexCopy = T.c_THDoubleTensor_indexCopy
  c_indexAdd = T.c_THDoubleTensor_indexAdd
  c_indexFill = T.c_THDoubleTensor_indexFill
  c_take = T.c_THDoubleTensor_take
  c_put = T.c_THDoubleTensor_put
  c_gather = T.c_THDoubleTensor_gather
  c_scatter = T.c_THDoubleTensor_scatter
  c_scatterAdd = T.c_THDoubleTensor_scatterAdd
  c_scatterFill = T.c_THDoubleTensor_scatterFill
  c_dot = T.c_THDoubleTensor_dot
  c_minall = T.c_THDoubleTensor_minall
  c_maxall = T.c_THDoubleTensor_maxall
  c_medianall = T.c_THDoubleTensor_medianall
  c_sumall = T.c_THDoubleTensor_sumall
  c_prodall = T.c_THDoubleTensor_prodall
  c_add = T.c_THDoubleTensor_add
  c_sub = T.c_THDoubleTensor_sub
  c_add_scaled = T.c_THDoubleTensor_add_scaled
  c_sub_scaled = T.c_THDoubleTensor_sub_scaled
  c_mul = T.c_THDoubleTensor_mul
  c_div = T.c_THDoubleTensor_div
  c_lshift = T.c_THDoubleTensor_lshift
  c_rshift = T.c_THDoubleTensor_rshift
  c_fmod = T.c_THDoubleTensor_fmod
  c_remainder = T.c_THDoubleTensor_remainder
  c_clamp = T.c_THDoubleTensor_clamp
  c_bitand = T.c_THDoubleTensor_bitand
  c_bitor = T.c_THDoubleTensor_bitor
  c_bitxor = T.c_THDoubleTensor_bitxor
  c_cadd = T.c_THDoubleTensor_cadd
  c_csub = T.c_THDoubleTensor_csub
  c_cmul = T.c_THDoubleTensor_cmul
  c_cpow = T.c_THDoubleTensor_cpow
  c_cdiv = T.c_THDoubleTensor_cdiv
  c_clshift = T.c_THDoubleTensor_clshift
  c_crshift = T.c_THDoubleTensor_crshift
  c_cfmod = T.c_THDoubleTensor_cfmod
  c_cremainder = T.c_THDoubleTensor_cremainder
  c_cbitand = T.c_THDoubleTensor_cbitand
  c_cbitor = T.c_THDoubleTensor_cbitor
  c_cbitxor = T.c_THDoubleTensor_cbitxor
  c_addcmul = T.c_THDoubleTensor_addcmul
  c_addcdiv = T.c_THDoubleTensor_addcdiv
  c_addmv = T.c_THDoubleTensor_addmv
  c_addmm = T.c_THDoubleTensor_addmm
  c_addr = T.c_THDoubleTensor_addr
  c_addbmm = T.c_THDoubleTensor_addbmm
  c_baddbmm = T.c_THDoubleTensor_baddbmm
  c_match = T.c_THDoubleTensor_match
  c_numel = T.c_THDoubleTensor_numel
  c_max = T.c_THDoubleTensor_max
  c_min = T.c_THDoubleTensor_min
  c_kthvalue = T.c_THDoubleTensor_kthvalue
  c_mode = T.c_THDoubleTensor_mode
  c_median = T.c_THDoubleTensor_median
  c_sum = T.c_THDoubleTensor_sum
  c_prod = T.c_THDoubleTensor_prod
  c_cumsum = T.c_THDoubleTensor_cumsum
  c_cumprod = T.c_THDoubleTensor_cumprod
  c_sign = T.c_THDoubleTensor_sign
  c_trace = T.c_THDoubleTensor_trace
  c_cross = T.c_THDoubleTensor_cross
  c_cmax = T.c_THDoubleTensor_cmax
  c_cmin = T.c_THDoubleTensor_cmin
  c_cmaxValue = T.c_THDoubleTensor_cmaxValue
  c_cminValue = T.c_THDoubleTensor_cminValue
  c_zeros = T.c_THDoubleTensor_zeros
  c_zerosLike = T.c_THDoubleTensor_zerosLike
  c_ones = T.c_THDoubleTensor_ones
  c_onesLike = T.c_THDoubleTensor_onesLike
  c_diag = T.c_THDoubleTensor_diag
  c_eye = T.c_THDoubleTensor_eye
  c_arange = T.c_THDoubleTensor_arange
  c_range = T.c_THDoubleTensor_range
  c_randperm = T.c_THDoubleTensor_randperm
  c_reshape = T.c_THDoubleTensor_reshape
  c_sort = T.c_THDoubleTensor_sort
  c_topk = T.c_THDoubleTensor_topk
  c_tril = T.c_THDoubleTensor_tril
  c_triu = T.c_THDoubleTensor_triu
  c_cat = T.c_THDoubleTensor_cat
  c_catArray = T.c_THDoubleTensor_catArray
  c_equal = T.c_THDoubleTensor_equal
  c_ltValue = T.c_THDoubleTensor_ltValue
  c_leValue = T.c_THDoubleTensor_leValue
  c_gtValue = T.c_THDoubleTensor_gtValue
  c_geValue = T.c_THDoubleTensor_geValue
  c_neValue = T.c_THDoubleTensor_neValue
  c_eqValue = T.c_THDoubleTensor_eqValue
  c_ltValueT = T.c_THDoubleTensor_ltValueT
  c_leValueT = T.c_THDoubleTensor_leValueT
  c_gtValueT = T.c_THDoubleTensor_gtValueT
  c_geValueT = T.c_THDoubleTensor_geValueT
  c_neValueT = T.c_THDoubleTensor_neValueT
  c_eqValueT = T.c_THDoubleTensor_eqValueT
  c_ltTensor = T.c_THDoubleTensor_ltTensor
  c_leTensor = T.c_THDoubleTensor_leTensor
  c_gtTensor = T.c_THDoubleTensor_gtTensor
  c_geTensor = T.c_THDoubleTensor_geTensor
  c_neTensor = T.c_THDoubleTensor_neTensor
  c_eqTensor = T.c_THDoubleTensor_eqTensor
  c_ltTensorT = T.c_THDoubleTensor_ltTensorT
  c_leTensorT = T.c_THDoubleTensor_leTensorT
  c_gtTensorT = T.c_THDoubleTensor_gtTensorT
  c_geTensorT = T.c_THDoubleTensor_geTensorT
  c_neTensorT = T.c_THDoubleTensor_neTensorT
  c_eqTensorT = T.c_THDoubleTensor_eqTensorT

instance GenericNegativeOps CTHDoubleTensor where
  c_neg = T.c_THDoubleTensor_neg
  c_abs = T.c_THDoubleTensor_abs

instance GenericFloatingMath CTHDoubleTensor where
  c_cinv = T.c_THDoubleTensor_cinv
  c_sigmoid = T.c_THDoubleTensor_sigmoid
  c_log = T.c_THDoubleTensor_log
  c_lgamma = T.c_THDoubleTensor_lgamma
  c_log1p = T.c_THDoubleTensor_log1p
  c_exp = T.c_THDoubleTensor_exp
  c_cos = T.c_THDoubleTensor_cos
  c_acos = T.c_THDoubleTensor_acos
  c_cosh = T.c_THDoubleTensor_cosh
  c_sin = T.c_THDoubleTensor_sin
  c_asin = T.c_THDoubleTensor_asin
  c_sinh = T.c_THDoubleTensor_sinh
  c_tan = T.c_THDoubleTensor_tan
  c_atan = T.c_THDoubleTensor_atan
  c_atan2 = T.c_THDoubleTensor_atan2
  c_tanh = T.c_THDoubleTensor_tanh
  c_erf = T.c_THDoubleTensor_erf
  c_erfinv = T.c_THDoubleTensor_erfinv
  c_pow = T.c_THDoubleTensor_pow
  c_tpow = T.c_THDoubleTensor_tpow
  c_sqrt = T.c_THDoubleTensor_sqrt
  c_rsqrt = T.c_THDoubleTensor_rsqrt
  c_ceil = T.c_THDoubleTensor_ceil
  c_floor = T.c_THDoubleTensor_floor
  c_round = T.c_THDoubleTensor_round
  c_trunc = T.c_THDoubleTensor_trunc
  c_frac = T.c_THDoubleTensor_frac
  c_lerp = T.c_THDoubleTensor_lerp
  c_mean = T.c_THDoubleTensor_mean
  c_std = T.c_THDoubleTensor_std
  c_var = T.c_THDoubleTensor_var
  c_norm = T.c_THDoubleTensor_norm
  c_renorm = T.c_THDoubleTensor_renorm
  c_dist = T.c_THDoubleTensor_dist
  c_histc = T.c_THDoubleTensor_histc
  c_bhistc = T.c_THDoubleTensor_bhistc
  c_meanall = T.c_THDoubleTensor_meanall
  c_varall = T.c_THDoubleTensor_varall
  c_stdall = T.c_THDoubleTensor_stdall
  c_normall = T.c_THDoubleTensor_normall
  c_linspace = T.c_THDoubleTensor_linspace
  c_logspace = T.c_THDoubleTensor_logspace
  c_rand = T.c_THDoubleTensor_rand
  c_randn = T.c_THDoubleTensor_randn

instance GenericMath CTHFloatTensor where
  c_fill = T.c_THFloatTensor_fill
  c_zero = T.c_THFloatTensor_zero
  c_maskedFill = T.c_THFloatTensor_maskedFill
  c_maskedCopy = T.c_THFloatTensor_maskedCopy
  c_maskedSelect = T.c_THFloatTensor_maskedSelect
  c_nonzero = T.c_THFloatTensor_nonzero
  c_indexSelect = T.c_THFloatTensor_indexSelect
  c_indexCopy = T.c_THFloatTensor_indexCopy
  c_indexAdd = T.c_THFloatTensor_indexAdd
  c_indexFill = T.c_THFloatTensor_indexFill
  c_take = T.c_THFloatTensor_take
  c_put = T.c_THFloatTensor_put
  c_gather = T.c_THFloatTensor_gather
  c_scatter = T.c_THFloatTensor_scatter
  c_scatterAdd = T.c_THFloatTensor_scatterAdd
  c_scatterFill = T.c_THFloatTensor_scatterFill
  c_dot = T.c_THFloatTensor_dot
  c_minall = T.c_THFloatTensor_minall
  c_maxall = T.c_THFloatTensor_maxall
  c_medianall = T.c_THFloatTensor_medianall
  c_sumall = T.c_THFloatTensor_sumall
  c_prodall = T.c_THFloatTensor_prodall
  c_add = T.c_THFloatTensor_add
  c_sub = T.c_THFloatTensor_sub
  c_add_scaled = T.c_THFloatTensor_add_scaled
  c_sub_scaled = T.c_THFloatTensor_sub_scaled
  c_mul = T.c_THFloatTensor_mul
  c_div = T.c_THFloatTensor_div
  c_lshift = T.c_THFloatTensor_lshift
  c_rshift = T.c_THFloatTensor_rshift
  c_fmod = T.c_THFloatTensor_fmod
  c_remainder = T.c_THFloatTensor_remainder
  c_clamp = T.c_THFloatTensor_clamp
  c_bitand = T.c_THFloatTensor_bitand
  c_bitor = T.c_THFloatTensor_bitor
  c_bitxor = T.c_THFloatTensor_bitxor
  c_cadd = T.c_THFloatTensor_cadd
  c_csub = T.c_THFloatTensor_csub
  c_cmul = T.c_THFloatTensor_cmul
  c_cpow = T.c_THFloatTensor_cpow
  c_cdiv = T.c_THFloatTensor_cdiv
  c_clshift = T.c_THFloatTensor_clshift
  c_crshift = T.c_THFloatTensor_crshift
  c_cfmod = T.c_THFloatTensor_cfmod
  c_cremainder = T.c_THFloatTensor_cremainder
  c_cbitand = T.c_THFloatTensor_cbitand
  c_cbitor = T.c_THFloatTensor_cbitor
  c_cbitxor = T.c_THFloatTensor_cbitxor
  c_addcmul = T.c_THFloatTensor_addcmul
  c_addcdiv = T.c_THFloatTensor_addcdiv
  c_addmv = T.c_THFloatTensor_addmv
  c_addmm = T.c_THFloatTensor_addmm
  c_addr = T.c_THFloatTensor_addr
  c_addbmm = T.c_THFloatTensor_addbmm
  c_baddbmm = T.c_THFloatTensor_baddbmm
  c_match = T.c_THFloatTensor_match
  c_numel = T.c_THFloatTensor_numel
  c_max = T.c_THFloatTensor_max
  c_min = T.c_THFloatTensor_min
  c_kthvalue = T.c_THFloatTensor_kthvalue
  c_mode = T.c_THFloatTensor_mode
  c_median = T.c_THFloatTensor_median
  c_sum = T.c_THFloatTensor_sum
  c_prod = T.c_THFloatTensor_prod
  c_cumsum = T.c_THFloatTensor_cumsum
  c_cumprod = T.c_THFloatTensor_cumprod
  c_sign = T.c_THFloatTensor_sign
  c_trace = T.c_THFloatTensor_trace
  c_cross = T.c_THFloatTensor_cross
  c_cmax = T.c_THFloatTensor_cmax
  c_cmin = T.c_THFloatTensor_cmin
  c_cmaxValue = T.c_THFloatTensor_cmaxValue
  c_cminValue = T.c_THFloatTensor_cminValue
  c_zeros = T.c_THFloatTensor_zeros
  c_zerosLike = T.c_THFloatTensor_zerosLike
  c_ones = T.c_THFloatTensor_ones
  c_onesLike = T.c_THFloatTensor_onesLike
  c_diag = T.c_THFloatTensor_diag
  c_eye = T.c_THFloatTensor_eye
  c_arange = T.c_THFloatTensor_arange
  c_range = T.c_THFloatTensor_range
  c_randperm = T.c_THFloatTensor_randperm
  c_reshape = T.c_THFloatTensor_reshape
  c_sort = T.c_THFloatTensor_sort
  c_topk = T.c_THFloatTensor_topk
  c_tril = T.c_THFloatTensor_tril
  c_triu = T.c_THFloatTensor_triu
  c_cat = T.c_THFloatTensor_cat
  c_catArray = T.c_THFloatTensor_catArray
  c_equal = T.c_THFloatTensor_equal
  c_ltValue = T.c_THFloatTensor_ltValue
  c_leValue = T.c_THFloatTensor_leValue
  c_gtValue = T.c_THFloatTensor_gtValue
  c_geValue = T.c_THFloatTensor_geValue
  c_neValue = T.c_THFloatTensor_neValue
  c_eqValue = T.c_THFloatTensor_eqValue
  c_ltValueT = T.c_THFloatTensor_ltValueT
  c_leValueT = T.c_THFloatTensor_leValueT
  c_gtValueT = T.c_THFloatTensor_gtValueT
  c_geValueT = T.c_THFloatTensor_geValueT
  c_neValueT = T.c_THFloatTensor_neValueT
  c_eqValueT = T.c_THFloatTensor_eqValueT
  c_ltTensor = T.c_THFloatTensor_ltTensor
  c_leTensor = T.c_THFloatTensor_leTensor
  c_gtTensor = T.c_THFloatTensor_gtTensor
  c_geTensor = T.c_THFloatTensor_geTensor
  c_neTensor = T.c_THFloatTensor_neTensor
  c_eqTensor = T.c_THFloatTensor_eqTensor
  c_ltTensorT = T.c_THFloatTensor_ltTensorT
  c_leTensorT = T.c_THFloatTensor_leTensorT
  c_gtTensorT = T.c_THFloatTensor_gtTensorT
  c_geTensorT = T.c_THFloatTensor_geTensorT
  c_neTensorT = T.c_THFloatTensor_neTensorT
  c_eqTensorT = T.c_THFloatTensor_eqTensorT

instance GenericNegativeOps CTHFloatTensor where
  c_neg = T.c_THFloatTensor_neg
  c_abs = T.c_THFloatTensor_abs

instance GenericFloatingMath CTHFloatTensor where
  c_cinv = T.c_THFloatTensor_cinv
  c_sigmoid = T.c_THFloatTensor_sigmoid
  c_log = T.c_THFloatTensor_log
  c_lgamma = T.c_THFloatTensor_lgamma
  c_log1p = T.c_THFloatTensor_log1p
  c_exp = T.c_THFloatTensor_exp
  c_cos = T.c_THFloatTensor_cos
  c_acos = T.c_THFloatTensor_acos
  c_cosh = T.c_THFloatTensor_cosh
  c_sin = T.c_THFloatTensor_sin
  c_asin = T.c_THFloatTensor_asin
  c_sinh = T.c_THFloatTensor_sinh
  c_tan = T.c_THFloatTensor_tan
  c_atan = T.c_THFloatTensor_atan
  c_atan2 = T.c_THFloatTensor_atan2
  c_tanh = T.c_THFloatTensor_tanh
  c_erf = T.c_THFloatTensor_erf
  c_erfinv = T.c_THFloatTensor_erfinv
  c_pow = T.c_THFloatTensor_pow
  c_tpow = T.c_THFloatTensor_tpow
  c_sqrt = T.c_THFloatTensor_sqrt
  c_rsqrt = T.c_THFloatTensor_rsqrt
  c_ceil = T.c_THFloatTensor_ceil
  c_floor = T.c_THFloatTensor_floor
  c_round = T.c_THFloatTensor_round
  c_trunc = T.c_THFloatTensor_trunc
  c_frac = T.c_THFloatTensor_frac
  c_lerp = T.c_THFloatTensor_lerp
  c_mean = T.c_THFloatTensor_mean
  c_std = T.c_THFloatTensor_std
  c_var = T.c_THFloatTensor_var
  c_norm = T.c_THFloatTensor_norm
  c_renorm = T.c_THFloatTensor_renorm
  c_dist = T.c_THFloatTensor_dist
  c_histc = T.c_THFloatTensor_histc
  c_bhistc = T.c_THFloatTensor_bhistc
  c_meanall = T.c_THFloatTensor_meanall
  c_varall = T.c_THFloatTensor_varall
  c_stdall = T.c_THFloatTensor_stdall
  c_normall = T.c_THFloatTensor_normall
  c_linspace = T.c_THFloatTensor_linspace
  c_logspace = T.c_THFloatTensor_logspace
  c_rand = T.c_THFloatTensor_rand
  c_randn = T.c_THFloatTensor_randn


instance GenericMath CTHIntTensor where
  c_fill = T.c_THIntTensor_fill
  c_zero = T.c_THIntTensor_zero
  c_maskedFill = T.c_THIntTensor_maskedFill
  c_maskedCopy = T.c_THIntTensor_maskedCopy
  c_maskedSelect = T.c_THIntTensor_maskedSelect
  c_nonzero = T.c_THIntTensor_nonzero
  c_indexSelect = T.c_THIntTensor_indexSelect
  c_indexCopy = T.c_THIntTensor_indexCopy
  c_indexAdd = T.c_THIntTensor_indexAdd
  c_indexFill = T.c_THIntTensor_indexFill
  c_take = T.c_THIntTensor_take
  c_put = T.c_THIntTensor_put
  c_gather = T.c_THIntTensor_gather
  c_scatter = T.c_THIntTensor_scatter
  c_scatterAdd = T.c_THIntTensor_scatterAdd
  c_scatterFill = T.c_THIntTensor_scatterFill
  c_dot = T.c_THIntTensor_dot
  c_minall = T.c_THIntTensor_minall
  c_maxall = T.c_THIntTensor_maxall
  c_medianall = T.c_THIntTensor_medianall
  c_sumall = T.c_THIntTensor_sumall
  c_prodall = T.c_THIntTensor_prodall
  c_add = T.c_THIntTensor_add
  c_sub = T.c_THIntTensor_sub
  c_add_scaled = T.c_THIntTensor_add_scaled
  c_sub_scaled = T.c_THIntTensor_sub_scaled
  c_mul = T.c_THIntTensor_mul
  c_div = T.c_THIntTensor_div
  c_lshift = T.c_THIntTensor_lshift
  c_rshift = T.c_THIntTensor_rshift
  c_fmod = T.c_THIntTensor_fmod
  c_remainder = T.c_THIntTensor_remainder
  c_clamp = T.c_THIntTensor_clamp
  c_bitand = T.c_THIntTensor_bitand
  c_bitor = T.c_THIntTensor_bitor
  c_bitxor = T.c_THIntTensor_bitxor
  c_cadd = T.c_THIntTensor_cadd
  c_csub = T.c_THIntTensor_csub
  c_cmul = T.c_THIntTensor_cmul
  c_cpow = T.c_THIntTensor_cpow
  c_cdiv = T.c_THIntTensor_cdiv
  c_clshift = T.c_THIntTensor_clshift
  c_crshift = T.c_THIntTensor_crshift
  c_cfmod = T.c_THIntTensor_cfmod
  c_cremainder = T.c_THIntTensor_cremainder
  c_cbitand = T.c_THIntTensor_cbitand
  c_cbitor = T.c_THIntTensor_cbitor
  c_cbitxor = T.c_THIntTensor_cbitxor
  c_addcmul = T.c_THIntTensor_addcmul
  c_addcdiv = T.c_THIntTensor_addcdiv
  c_addmv = T.c_THIntTensor_addmv
  c_addmm = T.c_THIntTensor_addmm
  c_addr = T.c_THIntTensor_addr
  c_addbmm = T.c_THIntTensor_addbmm
  c_baddbmm = T.c_THIntTensor_baddbmm
  c_match = T.c_THIntTensor_match
  c_numel = T.c_THIntTensor_numel
  c_max = T.c_THIntTensor_max
  c_min = T.c_THIntTensor_min
  c_kthvalue = T.c_THIntTensor_kthvalue
  c_mode = T.c_THIntTensor_mode
  c_median = T.c_THIntTensor_median
  c_sum = T.c_THIntTensor_sum
  c_prod = T.c_THIntTensor_prod
  c_cumsum = T.c_THIntTensor_cumsum
  c_cumprod = T.c_THIntTensor_cumprod
  c_sign = T.c_THIntTensor_sign
  c_trace = T.c_THIntTensor_trace
  c_cross = T.c_THIntTensor_cross
  c_cmax = T.c_THIntTensor_cmax
  c_cmin = T.c_THIntTensor_cmin
  c_cmaxValue = T.c_THIntTensor_cmaxValue
  c_cminValue = T.c_THIntTensor_cminValue
  c_zeros = T.c_THIntTensor_zeros
  c_zerosLike = T.c_THIntTensor_zerosLike
  c_ones = T.c_THIntTensor_ones
  c_onesLike = T.c_THIntTensor_onesLike
  c_diag = T.c_THIntTensor_diag
  c_eye = T.c_THIntTensor_eye
  c_arange = T.c_THIntTensor_arange
  c_range = T.c_THIntTensor_range
  c_randperm = T.c_THIntTensor_randperm
  c_reshape = T.c_THIntTensor_reshape
  c_sort = T.c_THIntTensor_sort
  c_topk = T.c_THIntTensor_topk
  c_tril = T.c_THIntTensor_tril
  c_triu = T.c_THIntTensor_triu
  c_cat = T.c_THIntTensor_cat
  c_catArray = T.c_THIntTensor_catArray
  c_equal = T.c_THIntTensor_equal
  c_ltValue = T.c_THIntTensor_ltValue
  c_leValue = T.c_THIntTensor_leValue
  c_gtValue = T.c_THIntTensor_gtValue
  c_geValue = T.c_THIntTensor_geValue
  c_neValue = T.c_THIntTensor_neValue
  c_eqValue = T.c_THIntTensor_eqValue
  c_ltValueT = T.c_THIntTensor_ltValueT
  c_leValueT = T.c_THIntTensor_leValueT
  c_gtValueT = T.c_THIntTensor_gtValueT
  c_geValueT = T.c_THIntTensor_geValueT
  c_neValueT = T.c_THIntTensor_neValueT
  c_eqValueT = T.c_THIntTensor_eqValueT
  c_ltTensor = T.c_THIntTensor_ltTensor
  c_leTensor = T.c_THIntTensor_leTensor
  c_gtTensor = T.c_THIntTensor_gtTensor
  c_geTensor = T.c_THIntTensor_geTensor
  c_neTensor = T.c_THIntTensor_neTensor
  c_eqTensor = T.c_THIntTensor_eqTensor
  c_ltTensorT = T.c_THIntTensor_ltTensorT
  c_leTensorT = T.c_THIntTensor_leTensorT
  c_gtTensorT = T.c_THIntTensor_gtTensorT
  c_geTensorT = T.c_THIntTensor_geTensorT
  c_neTensorT = T.c_THIntTensor_neTensorT
  c_eqTensorT = T.c_THIntTensor_eqTensorT

instance GenericNegativeOps CTHIntTensor where
  c_neg = T.c_THIntTensor_neg
  c_abs = T.c_THIntTensor_abs

instance GenericMath CTHLongTensor where
  c_fill = T.c_THLongTensor_fill
  c_zero = T.c_THLongTensor_zero
  c_maskedFill = T.c_THLongTensor_maskedFill
  c_maskedCopy = T.c_THLongTensor_maskedCopy
  c_maskedSelect = T.c_THLongTensor_maskedSelect
  c_nonzero = T.c_THLongTensor_nonzero
  c_indexSelect = T.c_THLongTensor_indexSelect
  c_indexCopy = T.c_THLongTensor_indexCopy
  c_indexAdd = T.c_THLongTensor_indexAdd
  c_indexFill = T.c_THLongTensor_indexFill
  c_take = T.c_THLongTensor_take
  c_put = T.c_THLongTensor_put
  c_gather = T.c_THLongTensor_gather
  c_scatter = T.c_THLongTensor_scatter
  c_scatterAdd = T.c_THLongTensor_scatterAdd
  c_scatterFill = T.c_THLongTensor_scatterFill
  c_dot = T.c_THLongTensor_dot
  c_minall = T.c_THLongTensor_minall
  c_maxall = T.c_THLongTensor_maxall
  c_medianall = T.c_THLongTensor_medianall
  c_sumall = T.c_THLongTensor_sumall
  c_prodall = T.c_THLongTensor_prodall
  c_add = T.c_THLongTensor_add
  c_sub = T.c_THLongTensor_sub
  c_add_scaled = T.c_THLongTensor_add_scaled
  c_sub_scaled = T.c_THLongTensor_sub_scaled
  c_mul = T.c_THLongTensor_mul
  c_div = T.c_THLongTensor_div
  c_lshift = T.c_THLongTensor_lshift
  c_rshift = T.c_THLongTensor_rshift
  c_fmod = T.c_THLongTensor_fmod
  c_remainder = T.c_THLongTensor_remainder
  c_clamp = T.c_THLongTensor_clamp
  c_bitand = T.c_THLongTensor_bitand
  c_bitor = T.c_THLongTensor_bitor
  c_bitxor = T.c_THLongTensor_bitxor
  c_cadd = T.c_THLongTensor_cadd
  c_csub = T.c_THLongTensor_csub
  c_cmul = T.c_THLongTensor_cmul
  c_cpow = T.c_THLongTensor_cpow
  c_cdiv = T.c_THLongTensor_cdiv
  c_clshift = T.c_THLongTensor_clshift
  c_crshift = T.c_THLongTensor_crshift
  c_cfmod = T.c_THLongTensor_cfmod
  c_cremainder = T.c_THLongTensor_cremainder
  c_cbitand = T.c_THLongTensor_cbitand
  c_cbitor = T.c_THLongTensor_cbitor
  c_cbitxor = T.c_THLongTensor_cbitxor
  c_addcmul = T.c_THLongTensor_addcmul
  c_addcdiv = T.c_THLongTensor_addcdiv
  c_addmv = T.c_THLongTensor_addmv
  c_addmm = T.c_THLongTensor_addmm
  c_addr = T.c_THLongTensor_addr
  c_addbmm = T.c_THLongTensor_addbmm
  c_baddbmm = T.c_THLongTensor_baddbmm
  c_match = T.c_THLongTensor_match
  c_numel = T.c_THLongTensor_numel
  c_max = T.c_THLongTensor_max
  c_min = T.c_THLongTensor_min
  c_kthvalue = T.c_THLongTensor_kthvalue
  c_mode = T.c_THLongTensor_mode
  c_median = T.c_THLongTensor_median
  c_sum = T.c_THLongTensor_sum
  c_prod = T.c_THLongTensor_prod
  c_cumsum = T.c_THLongTensor_cumsum
  c_cumprod = T.c_THLongTensor_cumprod
  c_sign = T.c_THLongTensor_sign
  c_trace = T.c_THLongTensor_trace
  c_cross = T.c_THLongTensor_cross
  c_cmax = T.c_THLongTensor_cmax
  c_cmin = T.c_THLongTensor_cmin
  c_cmaxValue = T.c_THLongTensor_cmaxValue
  c_cminValue = T.c_THLongTensor_cminValue
  c_zeros = T.c_THLongTensor_zeros
  c_zerosLike = T.c_THLongTensor_zerosLike
  c_ones = T.c_THLongTensor_ones
  c_onesLike = T.c_THLongTensor_onesLike
  c_diag = T.c_THLongTensor_diag
  c_eye = T.c_THLongTensor_eye
  c_arange = T.c_THLongTensor_arange
  c_range = T.c_THLongTensor_range
  c_randperm = T.c_THLongTensor_randperm
  c_reshape = T.c_THLongTensor_reshape
  c_sort = T.c_THLongTensor_sort
  c_topk = T.c_THLongTensor_topk
  c_tril = T.c_THLongTensor_tril
  c_triu = T.c_THLongTensor_triu
  c_cat = T.c_THLongTensor_cat
  c_catArray = T.c_THLongTensor_catArray
  c_equal = T.c_THLongTensor_equal
  c_ltValue = T.c_THLongTensor_ltValue
  c_leValue = T.c_THLongTensor_leValue
  c_gtValue = T.c_THLongTensor_gtValue
  c_geValue = T.c_THLongTensor_geValue
  c_neValue = T.c_THLongTensor_neValue
  c_eqValue = T.c_THLongTensor_eqValue
  c_ltValueT = T.c_THLongTensor_ltValueT
  c_leValueT = T.c_THLongTensor_leValueT
  c_gtValueT = T.c_THLongTensor_gtValueT
  c_geValueT = T.c_THLongTensor_geValueT
  c_neValueT = T.c_THLongTensor_neValueT
  c_eqValueT = T.c_THLongTensor_eqValueT
  c_ltTensor = T.c_THLongTensor_ltTensor
  c_leTensor = T.c_THLongTensor_leTensor
  c_gtTensor = T.c_THLongTensor_gtTensor
  c_geTensor = T.c_THLongTensor_geTensor
  c_neTensor = T.c_THLongTensor_neTensor
  c_eqTensor = T.c_THLongTensor_eqTensor
  c_ltTensorT = T.c_THLongTensor_ltTensorT
  c_leTensorT = T.c_THLongTensor_leTensorT
  c_gtTensorT = T.c_THLongTensor_gtTensorT
  c_geTensorT = T.c_THLongTensor_geTensorT
  c_neTensorT = T.c_THLongTensor_neTensorT
  c_eqTensorT = T.c_THLongTensor_eqTensorT

instance GenericNegativeOps CTHLongTensor where
  c_neg = T.c_THLongTensor_neg
  c_abs = T.c_THLongTensor_abs

instance GenericMath CTHShortTensor where
  c_fill = T.c_THShortTensor_fill
  c_zero = T.c_THShortTensor_zero
  c_maskedFill = T.c_THShortTensor_maskedFill
  c_maskedCopy = T.c_THShortTensor_maskedCopy
  c_maskedSelect = T.c_THShortTensor_maskedSelect
  c_nonzero = T.c_THShortTensor_nonzero
  c_indexSelect = T.c_THShortTensor_indexSelect
  c_indexCopy = T.c_THShortTensor_indexCopy
  c_indexAdd = T.c_THShortTensor_indexAdd
  c_indexFill = T.c_THShortTensor_indexFill
  c_take = T.c_THShortTensor_take
  c_put = T.c_THShortTensor_put
  c_gather = T.c_THShortTensor_gather
  c_scatter = T.c_THShortTensor_scatter
  c_scatterAdd = T.c_THShortTensor_scatterAdd
  c_scatterFill = T.c_THShortTensor_scatterFill
  c_dot = T.c_THShortTensor_dot
  c_minall = T.c_THShortTensor_minall
  c_maxall = T.c_THShortTensor_maxall
  c_medianall = T.c_THShortTensor_medianall
  c_sumall = T.c_THShortTensor_sumall
  c_prodall = T.c_THShortTensor_prodall
  c_add = T.c_THShortTensor_add
  c_sub = T.c_THShortTensor_sub
  c_add_scaled = T.c_THShortTensor_add_scaled
  c_sub_scaled = T.c_THShortTensor_sub_scaled
  c_mul = T.c_THShortTensor_mul
  c_div = T.c_THShortTensor_div
  c_lshift = T.c_THShortTensor_lshift
  c_rshift = T.c_THShortTensor_rshift
  c_fmod = T.c_THShortTensor_fmod
  c_remainder = T.c_THShortTensor_remainder
  c_clamp = T.c_THShortTensor_clamp
  c_bitand = T.c_THShortTensor_bitand
  c_bitor = T.c_THShortTensor_bitor
  c_bitxor = T.c_THShortTensor_bitxor
  c_cadd = T.c_THShortTensor_cadd
  c_csub = T.c_THShortTensor_csub
  c_cmul = T.c_THShortTensor_cmul
  c_cpow = T.c_THShortTensor_cpow
  c_cdiv = T.c_THShortTensor_cdiv
  c_clshift = T.c_THShortTensor_clshift
  c_crshift = T.c_THShortTensor_crshift
  c_cfmod = T.c_THShortTensor_cfmod
  c_cremainder = T.c_THShortTensor_cremainder
  c_cbitand = T.c_THShortTensor_cbitand
  c_cbitor = T.c_THShortTensor_cbitor
  c_cbitxor = T.c_THShortTensor_cbitxor
  c_addcmul = T.c_THShortTensor_addcmul
  c_addcdiv = T.c_THShortTensor_addcdiv
  c_addmv = T.c_THShortTensor_addmv
  c_addmm = T.c_THShortTensor_addmm
  c_addr = T.c_THShortTensor_addr
  c_addbmm = T.c_THShortTensor_addbmm
  c_baddbmm = T.c_THShortTensor_baddbmm
  c_match = T.c_THShortTensor_match
  c_numel = T.c_THShortTensor_numel
  c_max = T.c_THShortTensor_max
  c_min = T.c_THShortTensor_min
  c_kthvalue = T.c_THShortTensor_kthvalue
  c_mode = T.c_THShortTensor_mode
  c_median = T.c_THShortTensor_median
  c_sum = T.c_THShortTensor_sum
  c_prod = T.c_THShortTensor_prod
  c_cumsum = T.c_THShortTensor_cumsum
  c_cumprod = T.c_THShortTensor_cumprod
  c_sign = T.c_THShortTensor_sign
  c_trace = T.c_THShortTensor_trace
  c_cross = T.c_THShortTensor_cross
  c_cmax = T.c_THShortTensor_cmax
  c_cmin = T.c_THShortTensor_cmin
  c_cmaxValue = T.c_THShortTensor_cmaxValue
  c_cminValue = T.c_THShortTensor_cminValue
  c_zeros = T.c_THShortTensor_zeros
  c_zerosLike = T.c_THShortTensor_zerosLike
  c_ones = T.c_THShortTensor_ones
  c_onesLike = T.c_THShortTensor_onesLike
  c_diag = T.c_THShortTensor_diag
  c_eye = T.c_THShortTensor_eye
  c_arange = T.c_THShortTensor_arange
  c_range = T.c_THShortTensor_range
  c_randperm = T.c_THShortTensor_randperm
  c_reshape = T.c_THShortTensor_reshape
  c_sort = T.c_THShortTensor_sort
  c_topk = T.c_THShortTensor_topk
  c_tril = T.c_THShortTensor_tril
  c_triu = T.c_THShortTensor_triu
  c_cat = T.c_THShortTensor_cat
  c_catArray = T.c_THShortTensor_catArray
  c_equal = T.c_THShortTensor_equal
  c_ltValue = T.c_THShortTensor_ltValue
  c_leValue = T.c_THShortTensor_leValue
  c_gtValue = T.c_THShortTensor_gtValue
  c_geValue = T.c_THShortTensor_geValue
  c_neValue = T.c_THShortTensor_neValue
  c_eqValue = T.c_THShortTensor_eqValue
  c_ltValueT = T.c_THShortTensor_ltValueT
  c_leValueT = T.c_THShortTensor_leValueT
  c_gtValueT = T.c_THShortTensor_gtValueT
  c_geValueT = T.c_THShortTensor_geValueT
  c_neValueT = T.c_THShortTensor_neValueT
  c_eqValueT = T.c_THShortTensor_eqValueT
  c_ltTensor = T.c_THShortTensor_ltTensor
  c_leTensor = T.c_THShortTensor_leTensor
  c_gtTensor = T.c_THShortTensor_gtTensor
  c_geTensor = T.c_THShortTensor_geTensor
  c_neTensor = T.c_THShortTensor_neTensor
  c_eqTensor = T.c_THShortTensor_eqTensor
  c_ltTensorT = T.c_THShortTensor_ltTensorT
  c_leTensorT = T.c_THShortTensor_leTensorT
  c_gtTensorT = T.c_THShortTensor_gtTensorT
  c_geTensorT = T.c_THShortTensor_geTensorT
  c_neTensorT = T.c_THShortTensor_neTensorT
  c_eqTensorT = T.c_THShortTensor_eqTensorT

instance GenericNegativeOps CTHShortTensor where
  c_neg = T.c_THShortTensor_neg
  c_abs = T.c_THShortTensor_abs

