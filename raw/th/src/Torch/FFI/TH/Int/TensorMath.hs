{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.TensorMath
  ( c_fill
  , c_zero
  , c_maskedFill
  , c_maskedCopy
  , c_maskedSelect
  , c_nonzero
  , c_indexSelect
  , c_indexCopy
  , c_indexAdd
  , c_indexFill
  , c_take
  , c_put
  , c_gather
  , c_scatter
  , c_scatterAdd
  , c_scatterFill
  , c_dot
  , c_minall
  , c_maxall
  , c_medianall
  , c_sumall
  , c_prodall
  , c_neg
  , c_add
  , c_sub
  , c_add_scaled
  , c_sub_scaled
  , c_mul
  , c_div
  , c_lshift
  , c_rshift
  , c_fmod
  , c_remainder
  , c_clamp
  , c_bitand
  , c_bitor
  , c_bitxor
  , c_cadd
  , c_csub
  , c_cmul
  , c_cpow
  , c_cdiv
  , c_clshift
  , c_crshift
  , c_cfmod
  , c_cremainder
  , c_cbitand
  , c_cbitor
  , c_cbitxor
  , c_addcmul
  , c_addcdiv
  , c_addmv
  , c_addmm
  , c_addr
  , c_addbmm
  , c_baddbmm
  , c_match
  , c_numel
  , c_max
  , c_min
  , c_kthvalue
  , c_mode
  , c_median
  , c_sum
  , c_prod
  , c_cumsum
  , c_cumprod
  , c_sign
  , c_trace
  , c_cross
  , c_cmax
  , c_cmin
  , c_cmaxValue
  , c_cminValue
  , c_zeros
  , c_zerosLike
  , c_ones
  , c_onesLike
  , c_diag
  , c_eye
  , c_arange
  , c_range
  , c_randperm
  , c_reshape
  , c_sort
  , c_topk
  , c_tril
  , c_triu
  , c_cat
  , c_catArray
  , c_equal
  , c_ltValue
  , c_leValue
  , c_gtValue
  , c_geValue
  , c_neValue
  , c_eqValue
  , c_ltValueT
  , c_leValueT
  , c_gtValueT
  , c_geValueT
  , c_neValueT
  , c_eqValueT
  , c_ltTensor
  , c_leTensor
  , c_gtTensor
  , c_geTensor
  , c_neTensor
  , c_eqTensor
  , c_ltTensorT
  , c_leTensorT
  , c_gtTensorT
  , c_geTensorT
  , c_neTensorT
  , c_eqTensorT
  , c_abs
  , p_fill
  , p_zero
  , p_maskedFill
  , p_maskedCopy
  , p_maskedSelect
  , p_nonzero
  , p_indexSelect
  , p_indexCopy
  , p_indexAdd
  , p_indexFill
  , p_take
  , p_put
  , p_gather
  , p_scatter
  , p_scatterAdd
  , p_scatterFill
  , p_dot
  , p_minall
  , p_maxall
  , p_medianall
  , p_sumall
  , p_prodall
  , p_neg
  , p_add
  , p_sub
  , p_add_scaled
  , p_sub_scaled
  , p_mul
  , p_div
  , p_lshift
  , p_rshift
  , p_fmod
  , p_remainder
  , p_clamp
  , p_bitand
  , p_bitor
  , p_bitxor
  , p_cadd
  , p_csub
  , p_cmul
  , p_cpow
  , p_cdiv
  , p_clshift
  , p_crshift
  , p_cfmod
  , p_cremainder
  , p_cbitand
  , p_cbitor
  , p_cbitxor
  , p_addcmul
  , p_addcdiv
  , p_addmv
  , p_addmm
  , p_addr
  , p_addbmm
  , p_baddbmm
  , p_match
  , p_numel
  , p_max
  , p_min
  , p_kthvalue
  , p_mode
  , p_median
  , p_sum
  , p_prod
  , p_cumsum
  , p_cumprod
  , p_sign
  , p_trace
  , p_cross
  , p_cmax
  , p_cmin
  , p_cmaxValue
  , p_cminValue
  , p_zeros
  , p_zerosLike
  , p_ones
  , p_onesLike
  , p_diag
  , p_eye
  , p_arange
  , p_range
  , p_randperm
  , p_reshape
  , p_sort
  , p_topk
  , p_tril
  , p_triu
  , p_cat
  , p_catArray
  , p_equal
  , p_ltValue
  , p_leValue
  , p_gtValue
  , p_geValue
  , p_neValue
  , p_eqValue
  , p_ltValueT
  , p_leValueT
  , p_gtValueT
  , p_geValueT
  , p_neValueT
  , p_eqValueT
  , p_ltTensor
  , p_leTensor
  , p_gtTensor
  , p_geTensor
  , p_neTensor
  , p_eqTensor
  , p_ltTensorT
  , p_leTensorT
  , p_gtTensorT
  , p_geTensorT
  , p_neTensorT
  , p_eqTensorT
  , p_abs
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_fill :  r_ value -> void
foreign import ccall "THTensorMath.h THIntTensor_fill"
  c_fill :: Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_zero :  r_ -> void
foreign import ccall "THTensorMath.h THIntTensor_zero"
  c_zero :: Ptr (CTHIntTensor) -> IO (())

-- | c_maskedFill :  tensor mask value -> void
foreign import ccall "THTensorMath.h THIntTensor_maskedFill"
  c_maskedFill :: Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (())

-- | c_maskedCopy :  tensor mask src -> void
foreign import ccall "THTensorMath.h THIntTensor_maskedCopy"
  c_maskedCopy :: Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_maskedSelect :  tensor src mask -> void
foreign import ccall "THTensorMath.h THIntTensor_maskedSelect"
  c_maskedSelect :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_nonzero :  subscript tensor -> void
foreign import ccall "THTensorMath.h THIntTensor_nonzero"
  c_nonzero :: Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_indexSelect :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THIntTensor_indexSelect"
  c_indexSelect :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_indexCopy :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensor_indexCopy"
  c_indexCopy :: Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_indexAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensor_indexAdd"
  c_indexAdd :: Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_indexFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THIntTensor_indexFill"
  c_indexFill :: Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> CInt -> IO (())

-- | c_take :  tensor src index -> void
foreign import ccall "THTensorMath.h THIntTensor_take"
  c_take :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_put :  tensor index src accumulate -> void
foreign import ccall "THTensorMath.h THIntTensor_put"
  c_put :: Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_gather :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THIntTensor_gather"
  c_gather :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_scatter :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensor_scatter"
  c_scatter :: Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_scatterAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensor_scatterAdd"
  c_scatterAdd :: Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_scatterFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THIntTensor_scatterFill"
  c_scatterFill :: Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> CInt -> IO (())

-- | c_dot :  t src -> accreal
foreign import ccall "THTensorMath.h THIntTensor_dot"
  c_dot :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (CLong)

-- | c_minall :  t -> real
foreign import ccall "THTensorMath.h THIntTensor_minall"
  c_minall :: Ptr (CTHIntTensor) -> IO (CInt)

-- | c_maxall :  t -> real
foreign import ccall "THTensorMath.h THIntTensor_maxall"
  c_maxall :: Ptr (CTHIntTensor) -> IO (CInt)

-- | c_medianall :  t -> real
foreign import ccall "THTensorMath.h THIntTensor_medianall"
  c_medianall :: Ptr (CTHIntTensor) -> IO (CInt)

-- | c_sumall :  t -> accreal
foreign import ccall "THTensorMath.h THIntTensor_sumall"
  c_sumall :: Ptr (CTHIntTensor) -> IO (CLong)

-- | c_prodall :  t -> accreal
foreign import ccall "THTensorMath.h THIntTensor_prodall"
  c_prodall :: Ptr (CTHIntTensor) -> IO (CLong)

-- | c_neg :  self src -> void
foreign import ccall "THTensorMath.h THIntTensor_neg"
  c_neg :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_add :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_add"
  c_add :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_sub :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_sub"
  c_sub :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_add_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THIntTensor_add_scaled"
  c_add_scaled :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (())

-- | c_sub_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THIntTensor_sub_scaled"
  c_sub_scaled :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (())

-- | c_mul :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_mul"
  c_mul :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_div :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_div"
  c_div :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_lshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_lshift"
  c_lshift :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_rshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_rshift"
  c_rshift :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_fmod :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_fmod"
  c_fmod :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_remainder :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_remainder"
  c_remainder :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_clamp :  r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THIntTensor_clamp"
  c_clamp :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (())

-- | c_bitand :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_bitand"
  c_bitand :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_bitor :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_bitor"
  c_bitor :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_bitxor :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_bitxor"
  c_bitxor :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_cadd :  r_ t value src -> void
foreign import ccall "THTensorMath.h THIntTensor_cadd"
  c_cadd :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> IO (())

-- | c_csub :  self src1 value src2 -> void
foreign import ccall "THTensorMath.h THIntTensor_csub"
  c_csub :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> IO (())

-- | c_cmul :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cmul"
  c_cmul :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_cpow :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cpow"
  c_cpow :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_cdiv :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cdiv"
  c_cdiv :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_clshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_clshift"
  c_clshift :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_crshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_crshift"
  c_crshift :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_cfmod :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cfmod"
  c_cfmod :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_cremainder :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cremainder"
  c_cremainder :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_cbitand :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cbitand"
  c_cbitand :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_cbitor :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cbitor"
  c_cbitor :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_cbitxor :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cbitxor"
  c_cbitxor :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_addcmul :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addcmul"
  c_addcmul :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_addcdiv :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addcdiv"
  c_addcdiv :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_addmv :  r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THIntTensor_addmv"
  c_addmv :: Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_addmm :  r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addmm"
  c_addmm :: Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_addr :  r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addr"
  c_addr :: Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_addbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addbmm"
  c_addbmm :: Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_baddbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THIntTensor_baddbmm"
  c_baddbmm :: Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_match :  r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THIntTensor_match"
  c_match :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_numel :  t -> ptrdiff_t
foreign import ccall "THTensorMath.h THIntTensor_numel"
  c_numel :: Ptr (CTHIntTensor) -> IO (CPtrdiff)

-- | c_max :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_max"
  c_max :: Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (())

-- | c_min :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_min"
  c_min :: Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (())

-- | c_kthvalue :  values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_kthvalue"
  c_kthvalue :: Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CLLong -> CInt -> CInt -> IO (())

-- | c_mode :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_mode"
  c_mode :: Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (())

-- | c_median :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_median"
  c_median :: Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (())

-- | c_sum :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_sum"
  c_sum :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (())

-- | c_prod :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_prod"
  c_prod :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (())

-- | c_cumsum :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_cumsum"
  c_cumsum :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_cumprod :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_cumprod"
  c_cumprod :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_sign :  r_ t -> void
foreign import ccall "THTensorMath.h THIntTensor_sign"
  c_sign :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_trace :  t -> accreal
foreign import ccall "THTensorMath.h THIntTensor_trace"
  c_trace :: Ptr (CTHIntTensor) -> IO (CLong)

-- | c_cross :  r_ a b dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_cross"
  c_cross :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_cmax :  r t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cmax"
  c_cmax :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_cmin :  r t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cmin"
  c_cmin :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_cmaxValue :  r t value -> void
foreign import ccall "THTensorMath.h THIntTensor_cmaxValue"
  c_cmaxValue :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_cminValue :  r t value -> void
foreign import ccall "THTensorMath.h THIntTensor_cminValue"
  c_cminValue :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_zeros :  r_ size -> void
foreign import ccall "THTensorMath.h THIntTensor_zeros"
  c_zeros :: Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_zerosLike :  r_ input -> void
foreign import ccall "THTensorMath.h THIntTensor_zerosLike"
  c_zerosLike :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_ones :  r_ size -> void
foreign import ccall "THTensorMath.h THIntTensor_ones"
  c_ones :: Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_onesLike :  r_ input -> void
foreign import ccall "THTensorMath.h THIntTensor_onesLike"
  c_onesLike :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_diag :  r_ t k -> void
foreign import ccall "THTensorMath.h THIntTensor_diag"
  c_diag :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_eye :  r_ n m -> void
foreign import ccall "THTensorMath.h THIntTensor_eye"
  c_eye :: Ptr (CTHIntTensor) -> CLLong -> CLLong -> IO (())

-- | c_arange :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THIntTensor_arange"
  c_arange :: Ptr (CTHIntTensor) -> CLong -> CLong -> CLong -> IO (())

-- | c_range :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THIntTensor_range"
  c_range :: Ptr (CTHIntTensor) -> CLong -> CLong -> CLong -> IO (())

-- | c_randperm :  r_ _generator n -> void
foreign import ccall "THTensorMath.h THIntTensor_randperm"
  c_randperm :: Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (())

-- | c_reshape :  r_ t size -> void
foreign import ccall "THTensorMath.h THIntTensor_reshape"
  c_reshape :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_sort :  rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THIntTensor_sort"
  c_sort :: Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (())

-- | c_topk :  rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THIntTensor_topk"
  c_topk :: Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (())

-- | c_tril :  r_ t k -> void
foreign import ccall "THTensorMath.h THIntTensor_tril"
  c_tril :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CLLong -> IO (())

-- | c_triu :  r_ t k -> void
foreign import ccall "THTensorMath.h THIntTensor_triu"
  c_triu :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CLLong -> IO (())

-- | c_cat :  r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_cat"
  c_cat :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_catArray :  result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_catArray"
  c_catArray :: Ptr (CTHIntTensor) -> Ptr (Ptr (CTHIntTensor)) -> CInt -> CInt -> IO (())

-- | c_equal :  ta tb -> int
foreign import ccall "THTensorMath.h THIntTensor_equal"
  c_equal :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (CInt)

-- | c_ltValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_ltValue"
  c_ltValue :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_leValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_leValue"
  c_leValue :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_gtValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_gtValue"
  c_gtValue :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_geValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_geValue"
  c_geValue :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_neValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_neValue"
  c_neValue :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_eqValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_eqValue"
  c_eqValue :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_ltValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_ltValueT"
  c_ltValueT :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_leValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_leValueT"
  c_leValueT :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_gtValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_gtValueT"
  c_gtValueT :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_geValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_geValueT"
  c_geValueT :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_neValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_neValueT"
  c_neValueT :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_eqValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_eqValueT"
  c_eqValueT :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (())

-- | c_ltTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_ltTensor"
  c_ltTensor :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_leTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_leTensor"
  c_leTensor :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_gtTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_gtTensor"
  c_gtTensor :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_geTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_geTensor"
  c_geTensor :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_neTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_neTensor"
  c_neTensor :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_eqTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_eqTensor"
  c_eqTensor :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_ltTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_ltTensorT"
  c_ltTensorT :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_leTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_leTensorT"
  c_leTensorT :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_gtTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_gtTensorT"
  c_gtTensorT :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_geTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_geTensorT"
  c_geTensorT :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_neTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_neTensorT"
  c_neTensorT :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_eqTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_eqTensorT"
  c_eqTensorT :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_abs :  r_ t -> void
foreign import ccall "THTensorMath.h THIntTensor_abs"
  c_abs :: Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | p_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &THIntTensor_fill"
  p_fill :: FunPtr (Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &THIntTensor_zero"
  p_zero :: FunPtr (Ptr (CTHIntTensor) -> IO (()))

-- | p_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &THIntTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (()))

-- | p_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &THIntTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &THIntTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &THIntTensor_nonzero"
  p_nonzero :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THIntTensor_indexSelect"
  p_indexSelect :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THIntTensor_indexCopy"
  p_indexCopy :: FunPtr (Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THIntTensor_indexAdd"
  p_indexAdd :: FunPtr (Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THIntTensor_indexFill"
  p_indexFill :: FunPtr (Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> CInt -> IO (()))

-- | p_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &THIntTensor_take"
  p_take :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &THIntTensor_put"
  p_put :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THIntTensor_gather"
  p_gather :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THIntTensor_scatter"
  p_scatter :: FunPtr (Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THIntTensor_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THIntTensor_scatterFill"
  p_scatterFill :: FunPtr (Ptr (CTHIntTensor) -> CInt -> Ptr (CTHLongTensor) -> CInt -> IO (()))

-- | p_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &THIntTensor_dot"
  p_dot :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (CLong))

-- | p_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THIntTensor_minall"
  p_minall :: FunPtr (Ptr (CTHIntTensor) -> IO (CInt))

-- | p_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THIntTensor_maxall"
  p_maxall :: FunPtr (Ptr (CTHIntTensor) -> IO (CInt))

-- | p_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THIntTensor_medianall"
  p_medianall :: FunPtr (Ptr (CTHIntTensor) -> IO (CInt))

-- | p_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THIntTensor_sumall"
  p_sumall :: FunPtr (Ptr (CTHIntTensor) -> IO (CLong))

-- | p_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THIntTensor_prodall"
  p_prodall :: FunPtr (Ptr (CTHIntTensor) -> IO (CLong))

-- | p_neg : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THIntTensor_neg"
  p_neg :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_add"
  p_add :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_sub"
  p_sub :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THIntTensor_add_scaled"
  p_add_scaled :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (()))

-- | p_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THIntTensor_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (()))

-- | p_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_mul"
  p_mul :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_div"
  p_div :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_lshift"
  p_lshift :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_rshift"
  p_rshift :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_fmod"
  p_fmod :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_remainder"
  p_remainder :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &THIntTensor_clamp"
  p_clamp :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (()))

-- | p_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_bitand"
  p_bitand :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_bitor"
  p_bitor :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_bitxor"
  p_bitxor :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cadd"
  p_cadd :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> IO (()))

-- | p_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_csub"
  p_csub :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> IO (()))

-- | p_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cmul"
  p_cmul :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cpow"
  p_cpow :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cdiv"
  p_cdiv :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_clshift"
  p_clshift :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_crshift"
  p_crshift :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cfmod"
  p_cfmod :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cremainder"
  p_cremainder :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cbitand"
  p_cbitand :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cbitor"
  p_cbitor :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cbitxor"
  p_cbitxor :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_addcmul"
  p_addcmul :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_addcdiv"
  p_addcdiv :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &THIntTensor_addmv"
  p_addmv :: FunPtr (Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_addmm"
  p_addmm :: FunPtr (Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_addr"
  p_addr :: FunPtr (Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_addbmm"
  p_addbmm :: FunPtr (Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &THIntTensor_match"
  p_match :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &THIntTensor_numel"
  p_numel :: FunPtr (Ptr (CTHIntTensor) -> IO (CPtrdiff))

-- | p_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_max"
  p_max :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (()))

-- | p_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_min"
  p_min :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (()))

-- | p_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_kthvalue"
  p_kthvalue :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CLLong -> CInt -> CInt -> IO (()))

-- | p_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_mode"
  p_mode :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (()))

-- | p_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_median"
  p_median :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (()))

-- | p_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_sum"
  p_sum :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (()))

-- | p_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_prod"
  p_prod :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (()))

-- | p_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THIntTensor_cumsum"
  p_cumsum :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THIntTensor_cumprod"
  p_cumprod :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THIntTensor_sign"
  p_sign :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THIntTensor_trace"
  p_trace :: FunPtr (Ptr (CTHIntTensor) -> IO (CLong))

-- | p_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &THIntTensor_cross"
  p_cross :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cmax"
  p_cmax :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cmin"
  p_cmin :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_cminValue"
  p_cminValue :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THIntTensor_zeros"
  p_zeros :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THIntTensor_zerosLike"
  p_zerosLike :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THIntTensor_ones"
  p_ones :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THIntTensor_onesLike"
  p_onesLike :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THIntTensor_diag"
  p_diag :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &THIntTensor_eye"
  p_eye :: FunPtr (Ptr (CTHIntTensor) -> CLLong -> CLLong -> IO (()))

-- | p_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THIntTensor_arange"
  p_arange :: FunPtr (Ptr (CTHIntTensor) -> CLong -> CLong -> CLong -> IO (()))

-- | p_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THIntTensor_range"
  p_range :: FunPtr (Ptr (CTHIntTensor) -> CLong -> CLong -> CLong -> IO (()))

-- | p_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &THIntTensor_randperm"
  p_randperm :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (()))

-- | p_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &THIntTensor_reshape"
  p_reshape :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &THIntTensor_sort"
  p_sort :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CInt -> CInt -> IO (()))

-- | p_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &THIntTensor_topk"
  p_topk :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (()))

-- | p_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THIntTensor_tril"
  p_tril :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CLLong -> IO (()))

-- | p_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THIntTensor_triu"
  p_triu :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CLLong -> IO (()))

-- | p_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &THIntTensor_cat"
  p_cat :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &THIntTensor_catArray"
  p_catArray :: FunPtr (Ptr (CTHIntTensor) -> Ptr (Ptr (CTHIntTensor)) -> CInt -> CInt -> IO (()))

-- | p_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &THIntTensor_equal"
  p_equal :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (CInt))

-- | p_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_ltValue"
  p_ltValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_leValue"
  p_leValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_gtValue"
  p_gtValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_geValue"
  p_geValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_neValue"
  p_neValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_eqValue"
  p_eqValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_ltValueT"
  p_ltValueT :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_leValueT"
  p_leValueT :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_gtValueT"
  p_gtValueT :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_geValueT"
  p_geValueT :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_neValueT"
  p_neValueT :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_eqValueT"
  p_eqValueT :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> IO (()))

-- | p_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_leTensor"
  p_leTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_geTensor"
  p_geTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_neTensor"
  p_neTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_abs : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THIntTensor_abs"
  p_abs :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))