{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.TensorMath
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
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_fill :  r_ value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_fill"
  c_fill :: Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_zero :  r_ -> void
foreign import ccall "THTensorMath.h c_THTensorChar_zero"
  c_zero :: Ptr (CTHCharTensor) -> IO (())

-- | c_maskedFill :  tensor mask value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_maskedFill"
  c_maskedFill :: Ptr (CTHCharTensor) -> Ptr (CTHByteTensor) -> CChar -> IO (())

-- | c_maskedCopy :  tensor mask src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_maskedCopy"
  c_maskedCopy :: Ptr (CTHCharTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_maskedSelect :  tensor src mask -> void
foreign import ccall "THTensorMath.h c_THTensorChar_maskedSelect"
  c_maskedSelect :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_nonzero :  subscript tensor -> void
foreign import ccall "THTensorMath.h c_THTensorChar_nonzero"
  c_nonzero :: Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_indexSelect :  tensor src dim index -> void
foreign import ccall "THTensorMath.h c_THTensorChar_indexSelect"
  c_indexSelect :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_indexCopy :  tensor dim index src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_indexCopy"
  c_indexCopy :: Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_indexAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_indexAdd"
  c_indexAdd :: Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_indexFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h c_THTensorChar_indexFill"
  c_indexFill :: Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> CChar -> IO (())

-- | c_take :  tensor src index -> void
foreign import ccall "THTensorMath.h c_THTensorChar_take"
  c_take :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_put :  tensor index src accumulate -> void
foreign import ccall "THTensorMath.h c_THTensorChar_put"
  c_put :: Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (())

-- | c_gather :  tensor src dim index -> void
foreign import ccall "THTensorMath.h c_THTensorChar_gather"
  c_gather :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_scatter :  tensor dim index src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_scatter"
  c_scatter :: Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_scatterAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_scatterAdd"
  c_scatterAdd :: Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_scatterFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h c_THTensorChar_scatterFill"
  c_scatterFill :: Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> CChar -> IO (())

-- | c_dot :  t src -> accreal
foreign import ccall "THTensorMath.h c_THTensorChar_dot"
  c_dot :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (CLong)

-- | c_minall :  t -> real
foreign import ccall "THTensorMath.h c_THTensorChar_minall"
  c_minall :: Ptr (CTHCharTensor) -> IO (CChar)

-- | c_maxall :  t -> real
foreign import ccall "THTensorMath.h c_THTensorChar_maxall"
  c_maxall :: Ptr (CTHCharTensor) -> IO (CChar)

-- | c_medianall :  t -> real
foreign import ccall "THTensorMath.h c_THTensorChar_medianall"
  c_medianall :: Ptr (CTHCharTensor) -> IO (CChar)

-- | c_sumall :  t -> accreal
foreign import ccall "THTensorMath.h c_THTensorChar_sumall"
  c_sumall :: Ptr (CTHCharTensor) -> IO (CLong)

-- | c_prodall :  t -> accreal
foreign import ccall "THTensorMath.h c_THTensorChar_prodall"
  c_prodall :: Ptr (CTHCharTensor) -> IO (CLong)

-- | c_add :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_add"
  c_add :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_sub :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_sub"
  c_sub :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_add_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h c_THTensorChar_add_scaled"
  c_add_scaled :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> CChar -> IO (())

-- | c_sub_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h c_THTensorChar_sub_scaled"
  c_sub_scaled :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> CChar -> IO (())

-- | c_mul :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_mul"
  c_mul :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_div :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_div"
  c_div :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_lshift :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_lshift"
  c_lshift :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_rshift :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_rshift"
  c_rshift :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_fmod :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_fmod"
  c_fmod :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_remainder :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_remainder"
  c_remainder :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_clamp :  r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_clamp"
  c_clamp :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> CChar -> IO (())

-- | c_bitand :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_bitand"
  c_bitand :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_bitor :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_bitor"
  c_bitor :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_bitxor :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_bitxor"
  c_bitxor :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_cadd :  r_ t value src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cadd"
  c_cadd :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> IO (())

-- | c_csub :  self src1 value src2 -> void
foreign import ccall "THTensorMath.h c_THTensorChar_csub"
  c_csub :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> IO (())

-- | c_cmul :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cmul"
  c_cmul :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_cpow :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cpow"
  c_cpow :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_cdiv :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cdiv"
  c_cdiv :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_clshift :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_clshift"
  c_clshift :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_crshift :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_crshift"
  c_crshift :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_cfmod :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cfmod"
  c_cfmod :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_cremainder :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cremainder"
  c_cremainder :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_cbitand :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cbitand"
  c_cbitand :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_cbitor :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cbitor"
  c_cbitor :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_cbitxor :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cbitxor"
  c_cbitxor :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_addcmul :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h c_THTensorChar_addcmul"
  c_addcmul :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_addcdiv :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h c_THTensorChar_addcdiv"
  c_addcdiv :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_addmv :  r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h c_THTensorChar_addmv"
  c_addmv :: Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_addmm :  r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h c_THTensorChar_addmm"
  c_addmm :: Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_addr :  r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h c_THTensorChar_addr"
  c_addr :: Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_addbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h c_THTensorChar_addbmm"
  c_addbmm :: Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_baddbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h c_THTensorChar_baddbmm"
  c_baddbmm :: Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_match :  r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h c_THTensorChar_match"
  c_match :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_numel :  t -> ptrdiff_t
foreign import ccall "THTensorMath.h c_THTensorChar_numel"
  c_numel :: Ptr (CTHCharTensor) -> IO (CPtrdiff)

-- | c_max :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorChar_max"
  c_max :: Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (())

-- | c_min :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorChar_min"
  c_min :: Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (())

-- | c_kthvalue :  values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorChar_kthvalue"
  c_kthvalue :: Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CLLong -> CInt -> CInt -> IO (())

-- | c_mode :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorChar_mode"
  c_mode :: Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (())

-- | c_median :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorChar_median"
  c_median :: Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (())

-- | c_sum :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorChar_sum"
  c_sum :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (())

-- | c_prod :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorChar_prod"
  c_prod :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (())

-- | c_cumsum :  r_ t dimension -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cumsum"
  c_cumsum :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (())

-- | c_cumprod :  r_ t dimension -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cumprod"
  c_cumprod :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (())

-- | c_sign :  r_ t -> void
foreign import ccall "THTensorMath.h c_THTensorChar_sign"
  c_sign :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_trace :  t -> accreal
foreign import ccall "THTensorMath.h c_THTensorChar_trace"
  c_trace :: Ptr (CTHCharTensor) -> IO (CLong)

-- | c_cross :  r_ a b dimension -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cross"
  c_cross :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (())

-- | c_cmax :  r t src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cmax"
  c_cmax :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_cmin :  r t src -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cmin"
  c_cmin :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_cmaxValue :  r t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cmaxValue"
  c_cmaxValue :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_cminValue :  r t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cminValue"
  c_cminValue :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_zeros :  r_ size -> void
foreign import ccall "THTensorMath.h c_THTensorChar_zeros"
  c_zeros :: Ptr (CTHCharTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_zerosLike :  r_ input -> void
foreign import ccall "THTensorMath.h c_THTensorChar_zerosLike"
  c_zerosLike :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_ones :  r_ size -> void
foreign import ccall "THTensorMath.h c_THTensorChar_ones"
  c_ones :: Ptr (CTHCharTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_onesLike :  r_ input -> void
foreign import ccall "THTensorMath.h c_THTensorChar_onesLike"
  c_onesLike :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_diag :  r_ t k -> void
foreign import ccall "THTensorMath.h c_THTensorChar_diag"
  c_diag :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (())

-- | c_eye :  r_ n m -> void
foreign import ccall "THTensorMath.h c_THTensorChar_eye"
  c_eye :: Ptr (CTHCharTensor) -> CLLong -> CLLong -> IO (())

-- | c_arange :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h c_THTensorChar_arange"
  c_arange :: Ptr (CTHCharTensor) -> CLong -> CLong -> CLong -> IO (())

-- | c_range :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h c_THTensorChar_range"
  c_range :: Ptr (CTHCharTensor) -> CLong -> CLong -> CLong -> IO (())

-- | c_randperm :  r_ _generator n -> void
foreign import ccall "THTensorMath.h c_THTensorChar_randperm"
  c_randperm :: Ptr (CTHCharTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (())

-- | c_reshape :  r_ t size -> void
foreign import ccall "THTensorMath.h c_THTensorChar_reshape"
  c_reshape :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_sort :  rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h c_THTensorChar_sort"
  c_sort :: Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (())

-- | c_topk :  rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h c_THTensorChar_topk"
  c_topk :: Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (())

-- | c_tril :  r_ t k -> void
foreign import ccall "THTensorMath.h c_THTensorChar_tril"
  c_tril :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CLLong -> IO (())

-- | c_triu :  r_ t k -> void
foreign import ccall "THTensorMath.h c_THTensorChar_triu"
  c_triu :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CLLong -> IO (())

-- | c_cat :  r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h c_THTensorChar_cat"
  c_cat :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (())

-- | c_catArray :  result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h c_THTensorChar_catArray"
  c_catArray :: Ptr (CTHCharTensor) -> Ptr (Ptr (CTHCharTensor)) -> CInt -> CInt -> IO (())

-- | c_equal :  ta tb -> int
foreign import ccall "THTensorMath.h c_THTensorChar_equal"
  c_equal :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (CInt)

-- | c_ltValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_ltValue"
  c_ltValue :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_leValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_leValue"
  c_leValue :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_gtValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_gtValue"
  c_gtValue :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_geValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_geValue"
  c_geValue :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_neValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_neValue"
  c_neValue :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_eqValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_eqValue"
  c_eqValue :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_ltValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_ltValueT"
  c_ltValueT :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_leValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_leValueT"
  c_leValueT :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_gtValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_gtValueT"
  c_gtValueT :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_geValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_geValueT"
  c_geValueT :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_neValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_neValueT"
  c_neValueT :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_eqValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorChar_eqValueT"
  c_eqValueT :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_ltTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorChar_ltTensor"
  c_ltTensor :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_leTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorChar_leTensor"
  c_leTensor :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_gtTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorChar_gtTensor"
  c_gtTensor :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_geTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorChar_geTensor"
  c_geTensor :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_neTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorChar_neTensor"
  c_neTensor :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_eqTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorChar_eqTensor"
  c_eqTensor :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_ltTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorChar_ltTensorT"
  c_ltTensorT :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_leTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorChar_leTensorT"
  c_leTensorT :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_gtTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorChar_gtTensorT"
  c_gtTensorT :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_geTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorChar_geTensorT"
  c_geTensorT :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_neTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorChar_neTensorT"
  c_neTensorT :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_eqTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorChar_eqTensorT"
  c_eqTensorT :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | p_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_fill"
  p_fill :: FunPtr (Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_zero"
  p_zero :: FunPtr (Ptr (CTHCharTensor) -> IO (()))

-- | p_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_maskedFill"
  p_maskedFill :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHByteTensor) -> CChar -> IO (()))

-- | p_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_nonzero"
  p_nonzero :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_indexSelect"
  p_indexSelect :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_indexCopy"
  p_indexCopy :: FunPtr (Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_indexAdd"
  p_indexAdd :: FunPtr (Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_indexFill"
  p_indexFill :: FunPtr (Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> CChar -> IO (()))

-- | p_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_take"
  p_take :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_put"
  p_put :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (()))

-- | p_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_gather"
  p_gather :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_scatter"
  p_scatter :: FunPtr (Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_scatterFill"
  p_scatterFill :: FunPtr (Ptr (CTHCharTensor) -> CInt -> Ptr (CTHLongTensor) -> CChar -> IO (()))

-- | p_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &p_THTensorChar_dot"
  p_dot :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (CLong))

-- | p_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &p_THTensorChar_minall"
  p_minall :: FunPtr (Ptr (CTHCharTensor) -> IO (CChar))

-- | p_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &p_THTensorChar_maxall"
  p_maxall :: FunPtr (Ptr (CTHCharTensor) -> IO (CChar))

-- | p_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &p_THTensorChar_medianall"
  p_medianall :: FunPtr (Ptr (CTHCharTensor) -> IO (CChar))

-- | p_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &p_THTensorChar_sumall"
  p_sumall :: FunPtr (Ptr (CTHCharTensor) -> IO (CLong))

-- | p_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &p_THTensorChar_prodall"
  p_prodall :: FunPtr (Ptr (CTHCharTensor) -> IO (CLong))

-- | p_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_add"
  p_add :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_sub"
  p_sub :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_add_scaled"
  p_add_scaled :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> CChar -> IO (()))

-- | p_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> CChar -> IO (()))

-- | p_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_mul"
  p_mul :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_div"
  p_div :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_lshift"
  p_lshift :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_rshift"
  p_rshift :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_fmod"
  p_fmod :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_remainder"
  p_remainder :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_clamp"
  p_clamp :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> CChar -> IO (()))

-- | p_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_bitand"
  p_bitand :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_bitor"
  p_bitor :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_bitxor"
  p_bitxor :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cadd"
  p_cadd :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> IO (()))

-- | p_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_csub"
  p_csub :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> IO (()))

-- | p_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cmul"
  p_cmul :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cpow"
  p_cpow :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cdiv"
  p_cdiv :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_clshift"
  p_clshift :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_crshift"
  p_crshift :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cfmod"
  p_cfmod :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cremainder"
  p_cremainder :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cbitand"
  p_cbitand :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cbitor"
  p_cbitor :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cbitxor"
  p_cbitxor :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_addcmul"
  p_addcmul :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_addcdiv"
  p_addcdiv :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_addmv"
  p_addmv :: FunPtr (Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_addmm"
  p_addmm :: FunPtr (Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_addr"
  p_addr :: FunPtr (Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_addbmm"
  p_addbmm :: FunPtr (Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_baddbmm"
  p_baddbmm :: FunPtr (Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_match"
  p_match :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &p_THTensorChar_numel"
  p_numel :: FunPtr (Ptr (CTHCharTensor) -> IO (CPtrdiff))

-- | p_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_max"
  p_max :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (()))

-- | p_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_min"
  p_min :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (()))

-- | p_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_kthvalue"
  p_kthvalue :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CLLong -> CInt -> CInt -> IO (()))

-- | p_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_mode"
  p_mode :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (()))

-- | p_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_median"
  p_median :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (()))

-- | p_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_sum"
  p_sum :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (()))

-- | p_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_prod"
  p_prod :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (()))

-- | p_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cumsum"
  p_cumsum :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (()))

-- | p_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cumprod"
  p_cumprod :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (()))

-- | p_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_sign"
  p_sign :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &p_THTensorChar_trace"
  p_trace :: FunPtr (Ptr (CTHCharTensor) -> IO (CLong))

-- | p_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cross"
  p_cross :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (()))

-- | p_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cmax"
  p_cmax :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cmin"
  p_cmin :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cminValue"
  p_cminValue :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_zeros"
  p_zeros :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_zerosLike"
  p_zerosLike :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_ones"
  p_ones :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_onesLike"
  p_onesLike :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_diag"
  p_diag :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (()))

-- | p_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_eye"
  p_eye :: FunPtr (Ptr (CTHCharTensor) -> CLLong -> CLLong -> IO (()))

-- | p_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_arange"
  p_arange :: FunPtr (Ptr (CTHCharTensor) -> CLong -> CLong -> CLong -> IO (()))

-- | p_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_range"
  p_range :: FunPtr (Ptr (CTHCharTensor) -> CLong -> CLong -> CLong -> IO (()))

-- | p_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_randperm"
  p_randperm :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (()))

-- | p_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_reshape"
  p_reshape :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_sort"
  p_sort :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CInt -> CInt -> IO (()))

-- | p_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_topk"
  p_topk :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (()))

-- | p_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_tril"
  p_tril :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CLLong -> IO (()))

-- | p_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_triu"
  p_triu :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CLLong -> IO (()))

-- | p_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_cat"
  p_cat :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (()))

-- | p_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_catArray"
  p_catArray :: FunPtr (Ptr (CTHCharTensor) -> Ptr (Ptr (CTHCharTensor)) -> CInt -> CInt -> IO (()))

-- | p_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &p_THTensorChar_equal"
  p_equal :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (CInt))

-- | p_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_ltValue"
  p_ltValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_leValue"
  p_leValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_gtValue"
  p_gtValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_geValue"
  p_geValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_neValue"
  p_neValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_eqValue"
  p_eqValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_ltValueT"
  p_ltValueT :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_leValueT"
  p_leValueT :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_gtValueT"
  p_gtValueT :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_geValueT"
  p_geValueT :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_neValueT"
  p_neValueT :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_eqValueT"
  p_eqValueT :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_ltTensor"
  p_ltTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_leTensor"
  p_leTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_gtTensor"
  p_gtTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_geTensor"
  p_geTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_neTensor"
  p_neTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_eqTensor"
  p_eqTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_leTensorT"
  p_leTensorT :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_geTensorT"
  p_geTensorT :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_neTensorT"
  p_neTensorT :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorChar_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))