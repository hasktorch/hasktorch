
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module NativeFunctions where

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map

import Foreign.C.String
import Foreign.C.Types
import Foreign

data Scalar
data Tensor
data TensorOptions
data TensorList
data IndexTensor
data IntList
data StdArray a b
data ScalarType
data SparseTensorRef

data StdString
data Generator
data Device
data Storage

C.context $ C.cppCtx <> mempty {
    C.ctxTypesTable = Map.fromList [
        (C.TypeName "at::Scalar", [t|Scalar|])
      , (C.TypeName "at::Tensor", [t|Tensor|])
      , (C.TypeName "at::TensorOptions", [t|TensorOptions|])
      , (C.TypeName "at::TensorList", [t|TensorList|])
      , (C.TypeName "at::IndexTensor", [t|IndexTensor|])
      , (C.TypeName "at::IntList", [t|IntList|])
      , (C.TypeName "at::ScalarType", [t|ScalarType|])
      , (C.TypeName "at::SparseTensorRef", [t|SparseTensorRef|])
      , (C.TypeName "at::Storage", [t|Storage|])
      , (C.TypeName "at::Device", [t|Device|])
      , (C.TypeName "at::Generator", [t|Generator|])
      , (C.TypeName "std::string", [t|StdString|])
      , (C.TypeName "std::array<bool,2>", [t|StdArray CBool 2|])
      , (C.TypeName "std::array<bool,3>", [t|StdArray CBool 3|])
      , (C.TypeName "std::array<bool,4>", [t|StdArray CBool 4|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor>", [t|(Tensor,Tensor)|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor>", [t|(Tensor,Tensor,Tensor)|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>", [t|(Tensor,Tensor,Tensor,Tensor)|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>", [t|(Tensor,Tensor,Tensor,Tensor,Tensor)|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::TensorList>", [t|(Tensor,Tensor,Tensor,TensorList)|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,double,int64_t>", [t|(Tensor,Tensor,CDouble,Int64)|])
    ]
}

C.include "<ATen/ATen.h>"

_cast_Byte_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
_cast_Byte_tb _self _non_blocking = [C.block| at::Tensor* { return new at::Tensor(at::native::_cast_Byte(*$(at::Tensor* _self), $(bool _non_blocking))); }|]

_cast_Char_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
_cast_Char_tb _self _non_blocking = [C.block| at::Tensor* { return new at::Tensor(at::native::_cast_Char(*$(at::Tensor* _self), $(bool _non_blocking))); }|]

_cast_Double_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
_cast_Double_tb _self _non_blocking = [C.block| at::Tensor* { return new at::Tensor(at::native::_cast_Double(*$(at::Tensor* _self), $(bool _non_blocking))); }|]

_cast_Float_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
_cast_Float_tb _self _non_blocking = [C.block| at::Tensor* { return new at::Tensor(at::native::_cast_Float(*$(at::Tensor* _self), $(bool _non_blocking))); }|]

_cast_Int_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
_cast_Int_tb _self _non_blocking = [C.block| at::Tensor* { return new at::Tensor(at::native::_cast_Int(*$(at::Tensor* _self), $(bool _non_blocking))); }|]

_cast_Long_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
_cast_Long_tb _self _non_blocking = [C.block| at::Tensor* { return new at::Tensor(at::native::_cast_Long(*$(at::Tensor* _self), $(bool _non_blocking))); }|]

_cast_Short_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
_cast_Short_tb _self _non_blocking = [C.block| at::Tensor* { return new at::Tensor(at::native::_cast_Short(*$(at::Tensor* _self), $(bool _non_blocking))); }|]

_cast_Half_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
_cast_Half_tb _self _non_blocking = [C.block| at::Tensor* { return new at::Tensor(at::native::_cast_Half(*$(at::Tensor* _self), $(bool _non_blocking))); }|]

_cudnn_ctc_loss_ttlllb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
_cudnn_ctc_loss_ttlllb _log_probs _targets _input_lengths _target_lengths _blank _deterministic = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::_cudnn_ctc_loss(*$(at::Tensor* _log_probs), *$(at::Tensor* _targets), *$(at::IntList* _input_lengths), *$(at::IntList* _target_lengths), $(int64_t _blank), $(bool _deterministic))); }|]

_cudnn_rnn_flatten_weight_llllllbb :: Ptr TensorList -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
_cudnn_rnn_flatten_weight_llllllbb _weight_arr _weight_stride0 _input_size _mode _hidden_size _num_layers _batch_first _bidirectional = [C.block| at::Tensor* { return new at::Tensor(at::native::_cudnn_rnn_flatten_weight(*$(at::TensorList* _weight_arr), $(int64_t _weight_stride0), $(int64_t _input_size), $(int64_t _mode), $(int64_t _hidden_size), $(int64_t _num_layers), $(bool _batch_first), $(bool _bidirectional))); }|]

_cudnn_rnn_tlltttlllbdbblt :: Ptr Tensor -> Ptr TensorList -> Int64 -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> Int64 -> CBool -> CDouble -> CBool -> CBool -> Ptr IntList -> Ptr Tensor -> IO (Ptr (Tensor,Tensor,Tensor,Tensor,Tensor))
_cudnn_rnn_tlltttlllbdbblt _input _weight _weight_stride0 _weight_buf _hx _cx _mode _hidden_size _num_layers _batch_first _dropout _train _bidirectional _batch_sizes _dropout_state = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>(at::native::_cudnn_rnn(*$(at::Tensor* _input), *$(at::TensorList* _weight), $(int64_t _weight_stride0), *$(at::Tensor* _weight_buf), *$(at::Tensor* _hx), *$(at::Tensor* _cx), $(int64_t _mode), $(int64_t _hidden_size), $(int64_t _num_layers), $(bool _batch_first), $(double _dropout), $(bool _train), $(bool _bidirectional), *$(at::IntList* _batch_sizes), *$(at::Tensor* _dropout_state))); }|]

_cudnn_rnn_backward_tlltttttttlllbdbbltta :: Ptr Tensor -> Ptr TensorList -> Int64 -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> Int64 -> CBool -> CDouble -> CBool -> CBool -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr (StdArray CBool 4) -> IO (Ptr (Tensor,Tensor,Tensor,TensorList))
_cudnn_rnn_backward_tlltttttttlllbdbbltta _input _weight _weight_stride0 _weight_buf _hx _cx _output _grad_output _grad_hy _grad_cy _mode _hidden_size _num_layers _batch_first _dropout _train _bidirectional _batch_sizes _dropout_state _reserve _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor,at::TensorList>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor,at::TensorList>(at::native::_cudnn_rnn_backward(*$(at::Tensor* _input), *$(at::TensorList* _weight), $(int64_t _weight_stride0), *$(at::Tensor* _weight_buf), *$(at::Tensor* _hx), *$(at::Tensor* _cx), *$(at::Tensor* _output), *$(at::Tensor* _grad_output), *$(at::Tensor* _grad_hy), *$(at::Tensor* _grad_cy), $(int64_t _mode), $(int64_t _hidden_size), $(int64_t _num_layers), $(bool _batch_first), $(double _dropout), $(bool _train), $(bool _bidirectional), *$(at::IntList* _batch_sizes), *$(at::Tensor* _dropout_state), *$(at::Tensor* _reserve), *$(std::array<bool,4>* _output_mask))); }|]

_cudnn_init_dropout_state_dblo :: CDouble -> CBool -> Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
_cudnn_init_dropout_state_dblo _dropout _train _dropout_seed _options = [C.block| at::Tensor* { return new at::Tensor(at::native::_cudnn_init_dropout_state($(double _dropout), $(bool _train), $(int64_t _dropout_seed), *$(at::TensorOptions* _options))); }|]

fused_dropout_cuda_tdp :: Ptr Tensor -> CDouble -> Ptr Generator -> IO (Ptr (Tensor,Tensor))
fused_dropout_cuda_tdp _self _p _generator = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::fused_dropout_cuda(*$(at::Tensor* _self), $(double _p), $(at::Generator * _generator))); }|]

masked_scale_cuda_ttd :: Ptr Tensor -> Ptr Tensor -> CDouble -> IO (Ptr Tensor)
masked_scale_cuda_ttd _self _mask _scale = [C.block| at::Tensor* { return new at::Tensor(at::native::masked_scale_cuda(*$(at::Tensor* _self), *$(at::Tensor* _mask), $(double _scale))); }|]

_reshape_from_tensor_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_reshape_from_tensor_tt _self _shape = [C.block| at::Tensor* { return new at::Tensor(at::native::_reshape_from_tensor(*$(at::Tensor* _self), *$(at::Tensor* _shape))); }|]

_shape_as_tensor_t :: Ptr Tensor -> IO (Ptr Tensor)
_shape_as_tensor_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_shape_as_tensor(*$(at::Tensor* _self))); }|]

dropout_tdb :: Ptr Tensor -> CDouble -> CBool -> IO (Ptr Tensor)
dropout_tdb _input _p _train = [C.block| at::Tensor* { return new at::Tensor(at::native::dropout(*$(at::Tensor* _input), $(double _p), $(bool _train))); }|]

dropout__tdb :: Ptr Tensor -> CDouble -> CBool -> IO (Ptr Tensor)
dropout__tdb _self _p _train = [C.block| at::Tensor* { return new at::Tensor(at::native::dropout_(*$(at::Tensor* _self), $(double _p), $(bool _train))); }|]

feature_dropout_tdb :: Ptr Tensor -> CDouble -> CBool -> IO (Ptr Tensor)
feature_dropout_tdb _input _p _train = [C.block| at::Tensor* { return new at::Tensor(at::native::feature_dropout(*$(at::Tensor* _input), $(double _p), $(bool _train))); }|]

feature_dropout__tdb :: Ptr Tensor -> CDouble -> CBool -> IO (Ptr Tensor)
feature_dropout__tdb _self _p _train = [C.block| at::Tensor* { return new at::Tensor(at::native::feature_dropout_(*$(at::Tensor* _self), $(double _p), $(bool _train))); }|]

alpha_dropout_tdb :: Ptr Tensor -> CDouble -> CBool -> IO (Ptr Tensor)
alpha_dropout_tdb _input _p _train = [C.block| at::Tensor* { return new at::Tensor(at::native::alpha_dropout(*$(at::Tensor* _input), $(double _p), $(bool _train))); }|]

alpha_dropout__tdb :: Ptr Tensor -> CDouble -> CBool -> IO (Ptr Tensor)
alpha_dropout__tdb _self _p _train = [C.block| at::Tensor* { return new at::Tensor(at::native::alpha_dropout_(*$(at::Tensor* _self), $(double _p), $(bool _train))); }|]

feature_alpha_dropout_tdb :: Ptr Tensor -> CDouble -> CBool -> IO (Ptr Tensor)
feature_alpha_dropout_tdb _input _p _train = [C.block| at::Tensor* { return new at::Tensor(at::native::feature_alpha_dropout(*$(at::Tensor* _input), $(double _p), $(bool _train))); }|]

feature_alpha_dropout__tdb :: Ptr Tensor -> CDouble -> CBool -> IO (Ptr Tensor)
feature_alpha_dropout__tdb _self _p _train = [C.block| at::Tensor* { return new at::Tensor(at::native::feature_alpha_dropout_(*$(at::Tensor* _self), $(double _p), $(bool _train))); }|]

abs_t :: Ptr Tensor -> IO (Ptr Tensor)
abs_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::abs(*$(at::Tensor* _self))); }|]

_abs__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_abs__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_abs__cpu(*$(at::Tensor* _self))); }|]

_abs__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_abs__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_abs__cuda(*$(at::Tensor* _self))); }|]

_abs_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_abs_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_abs_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_abs_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_abs_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_abs_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

acos_t :: Ptr Tensor -> IO (Ptr Tensor)
acos_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::acos(*$(at::Tensor* _self))); }|]

_acos__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_acos__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_acos__cpu(*$(at::Tensor* _self))); }|]

_acos__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_acos__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_acos__cuda(*$(at::Tensor* _self))); }|]

_acos_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_acos_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_acos_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_acos_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_acos_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_acos_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

avg_pool1d_tlllbb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> CBool -> IO (Ptr Tensor)
avg_pool1d_tlllbb _self _kernel_size _stride _padding _ceil_mode _count_include_pad = [C.block| at::Tensor* { return new at::Tensor(at::native::avg_pool1d(*$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), $(bool _ceil_mode), $(bool _count_include_pad))); }|]

adaptive_avg_pool1d_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
adaptive_avg_pool1d_tl _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool1d(*$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

adaptive_max_pool1d_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr (Tensor,Tensor))
adaptive_max_pool1d_tl _self _output_size = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::adaptive_max_pool1d(*$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

add_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
add_tts _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::add(*$(at::Tensor* _self), *$(at::Tensor* _other), *$(at::Scalar* _alpha))); }|]

add__tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
add__tts _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::add_(*$(at::Tensor* _self), *$(at::Tensor* _other), *$(at::Scalar* _alpha))); }|]

add_out_ttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
add_out_ttts _result _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::add_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other), *$(at::Scalar* _alpha))); }|]

add_tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
add_tss _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::add(*$(at::Tensor* _self), *$(at::Scalar* _other), *$(at::Scalar* _alpha))); }|]

add__tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
add__tss _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::add_(*$(at::Tensor* _self), *$(at::Scalar* _other), *$(at::Scalar* _alpha))); }|]

addmv_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
addmv_tttss _self _mat _vec _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::addmv(*$(at::Tensor* _self), *$(at::Tensor* _mat), *$(at::Tensor* _vec), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

addmv__tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
addmv__tttss _self _mat _vec _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::addmv_(*$(at::Tensor* _self), *$(at::Tensor* _mat), *$(at::Tensor* _vec), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

addmv_out_ttttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
addmv_out_ttttss _result _self _mat _vec _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::addmv_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _mat), *$(at::Tensor* _vec), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

addr_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
addr_tttss _self _vec1 _vec2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::addr(*$(at::Tensor* _self), *$(at::Tensor* _vec1), *$(at::Tensor* _vec2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

addr__tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
addr__tttss _self _vec1 _vec2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::addr_(*$(at::Tensor* _self), *$(at::Tensor* _vec1), *$(at::Tensor* _vec2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

addr_out_ttttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
addr_out_ttttss _result _self _vec1 _vec2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::addr_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _vec1), *$(at::Tensor* _vec2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

affine_grid_generator_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
affine_grid_generator_tl _theta _size = [C.block| at::Tensor* { return new at::Tensor(at::native::affine_grid_generator(*$(at::Tensor* _theta), *$(at::IntList* _size))); }|]

affine_grid_generator_backward_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
affine_grid_generator_backward_tl _grad _size = [C.block| at::Tensor* { return new at::Tensor(at::native::affine_grid_generator_backward(*$(at::Tensor* _grad), *$(at::IntList* _size))); }|]

all_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
all_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::all(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

all_out_ttlb :: Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
all_out_ttlb _result _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::all_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

allclose_ttddb :: Ptr Tensor -> Ptr Tensor -> CDouble -> CDouble -> CBool -> IO (CBool)
allclose_ttddb _self _other _rtol _atol _equal_nan = [C.block| bool { return (at::native::allclose(*$(at::Tensor* _self), *$(at::Tensor* _other), $(double _rtol), $(double _atol), $(bool _equal_nan))); }|]

any_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
any_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::any(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

any_out_ttlb :: Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
any_out_ttlb _result _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::any_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

arange_so :: Ptr Scalar -> Ptr TensorOptions -> IO (Ptr Tensor)
arange_so _end _options = [C.block| at::Tensor* { return new at::Tensor(at::native::arange(*$(at::Scalar* _end), *$(at::TensorOptions* _options))); }|]

arange_sso :: Ptr Scalar -> Ptr Scalar -> Ptr TensorOptions -> IO (Ptr Tensor)
arange_sso _start _end _options = [C.block| at::Tensor* { return new at::Tensor(at::native::arange(*$(at::Scalar* _start), *$(at::Scalar* _end), *$(at::TensorOptions* _options))); }|]

arange_ssso :: Ptr Scalar -> Ptr Scalar -> Ptr Scalar -> Ptr TensorOptions -> IO (Ptr Tensor)
arange_ssso _start _end _step _options = [C.block| at::Tensor* { return new at::Tensor(at::native::arange(*$(at::Scalar* _start), *$(at::Scalar* _end), *$(at::Scalar* _step), *$(at::TensorOptions* _options))); }|]

arange_out_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
arange_out_ts _result _end = [C.block| at::Tensor* { return new at::Tensor(at::native::arange_out(*$(at::Tensor* _result), *$(at::Scalar* _end))); }|]

arange_cpu_out_tsss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
arange_cpu_out_tsss _result _start _end _step = [C.block| at::Tensor* { return new at::Tensor(at::native::arange_cpu_out(*$(at::Tensor* _result), *$(at::Scalar* _start), *$(at::Scalar* _end), *$(at::Scalar* _step))); }|]

arange_cuda_out_tsss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
arange_cuda_out_tsss _result _start _end _step = [C.block| at::Tensor* { return new at::Tensor(at::native::arange_cuda_out(*$(at::Tensor* _result), *$(at::Scalar* _start), *$(at::Scalar* _end), *$(at::Scalar* _step))); }|]

_dim_arange_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
_dim_arange_tl _like _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::_dim_arange(*$(at::Tensor* _like), $(int64_t _dim))); }|]

argmax_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
argmax_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::argmax(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

argmax_t :: Ptr Tensor -> IO (Ptr Tensor)
argmax_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::argmax(*$(at::Tensor* _self))); }|]

_argmax_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
_argmax_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::_argmax(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

argmin_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
argmin_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::argmin(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

argmin_t :: Ptr Tensor -> IO (Ptr Tensor)
argmin_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::argmin(*$(at::Tensor* _self))); }|]

_argmin_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
_argmin_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::_argmin(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

as_strided_tlll :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Int64 -> IO (Ptr Tensor)
as_strided_tlll _self _size _stride _storage_offset = [C.block| at::Tensor* { return new at::Tensor(at::native::as_strided(*$(at::Tensor* _self), *$(at::IntList* _size), *$(at::IntList* _stride), $(int64_t _storage_offset))); }|]

as_strided__tlll :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Int64 -> IO (Ptr Tensor)
as_strided__tlll _self _size _stride _storage_offset = [C.block| at::Tensor* { return new at::Tensor(at::native::as_strided_(*$(at::Tensor* _self), *$(at::IntList* _size), *$(at::IntList* _stride), $(int64_t _storage_offset))); }|]

asin_t :: Ptr Tensor -> IO (Ptr Tensor)
asin_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::asin(*$(at::Tensor* _self))); }|]

_asin__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_asin__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_asin__cpu(*$(at::Tensor* _self))); }|]

_asin__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_asin__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_asin__cuda(*$(at::Tensor* _self))); }|]

_asin_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_asin_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_asin_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_asin_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_asin_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_asin_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

atan_t :: Ptr Tensor -> IO (Ptr Tensor)
atan_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::atan(*$(at::Tensor* _self))); }|]

_atan__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_atan__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_atan__cpu(*$(at::Tensor* _self))); }|]

_atan__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_atan__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_atan__cuda(*$(at::Tensor* _self))); }|]

_atan_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_atan_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_atan_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_atan_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_atan_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_atan_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

baddbmm_cpu_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
baddbmm_cpu_tttss _self _batch1 _batch2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::baddbmm_cpu(*$(at::Tensor* _self), *$(at::Tensor* _batch1), *$(at::Tensor* _batch2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

baddbmm_cuda_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
baddbmm_cuda_tttss _self _batch1 _batch2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::baddbmm_cuda(*$(at::Tensor* _self), *$(at::Tensor* _batch1), *$(at::Tensor* _batch2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

baddbmm__cpu_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
baddbmm__cpu_tttss _self _batch1 _batch2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::baddbmm__cpu(*$(at::Tensor* _self), *$(at::Tensor* _batch1), *$(at::Tensor* _batch2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

baddbmm__cuda_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
baddbmm__cuda_tttss _self _batch1 _batch2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::baddbmm__cuda(*$(at::Tensor* _self), *$(at::Tensor* _batch1), *$(at::Tensor* _batch2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

_baddbmm_mkl__tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
_baddbmm_mkl__tttss _self _batch1 _batch2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::_baddbmm_mkl_(*$(at::Tensor* _self), *$(at::Tensor* _batch1), *$(at::Tensor* _batch2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

baddbmm_out_cpu_ttttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
baddbmm_out_cpu_ttttss _result _self _batch1 _batch2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::baddbmm_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _batch1), *$(at::Tensor* _batch2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

baddbmm_out_cuda_ttttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
baddbmm_out_cuda_ttttss _result _self _batch1 _batch2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::baddbmm_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _batch1), *$(at::Tensor* _batch2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

bartlett_window_lo :: Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
bartlett_window_lo _window_length _options = [C.block| at::Tensor* { return new at::Tensor(at::native::bartlett_window($(int64_t _window_length), *$(at::TensorOptions* _options))); }|]

bartlett_window_lbo :: Int64 -> CBool -> Ptr TensorOptions -> IO (Ptr Tensor)
bartlett_window_lbo _window_length _periodic _options = [C.block| at::Tensor* { return new at::Tensor(at::native::bartlett_window($(int64_t _window_length), $(bool _periodic), *$(at::TensorOptions* _options))); }|]

batch_norm_tttttbddb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CDouble -> CDouble -> CBool -> IO (Ptr Tensor)
batch_norm_tttttbddb _input _weight _bias _running_mean _running_var _training _momentum _eps _cudnn_enabled = [C.block| at::Tensor* { return new at::Tensor(at::native::batch_norm(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::Tensor* _running_mean), *$(at::Tensor* _running_var), $(bool _training), $(double _momentum), $(double _eps), $(bool _cudnn_enabled))); }|]

bernoulli_tp :: Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
bernoulli_tp _self _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::bernoulli(*$(at::Tensor* _self), $(at::Generator * _generator))); }|]

bernoulli_out_ttp :: Ptr Tensor -> Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
bernoulli_out_ttp _result _self _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::bernoulli_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(at::Generator * _generator))); }|]

bernoulli_tensor_cpu__ttp :: Ptr Tensor -> Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
bernoulli_tensor_cpu__ttp _self _p _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::bernoulli_tensor_cpu_(*$(at::Tensor* _self), *$(at::Tensor* _p), $(at::Generator * _generator))); }|]

bernoulli_tensor_cuda__ttp :: Ptr Tensor -> Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
bernoulli_tensor_cuda__ttp _self _p _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::bernoulli_tensor_cuda_(*$(at::Tensor* _self), *$(at::Tensor* _p), $(at::Generator * _generator))); }|]

bernoulli_scalar_cpu__tdp :: Ptr Tensor -> CDouble -> Ptr Generator -> IO (Ptr Tensor)
bernoulli_scalar_cpu__tdp _self _p _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::bernoulli_scalar_cpu_(*$(at::Tensor* _self), $(double _p), $(at::Generator * _generator))); }|]

bernoulli_scalar_cuda__tdp :: Ptr Tensor -> CDouble -> Ptr Generator -> IO (Ptr Tensor)
bernoulli_scalar_cuda__tdp _self _p _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::bernoulli_scalar_cuda_(*$(at::Tensor* _self), $(double _p), $(at::Generator * _generator))); }|]

bernoulli_tdp :: Ptr Tensor -> CDouble -> Ptr Generator -> IO (Ptr Tensor)
bernoulli_tdp _self _p _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::bernoulli(*$(at::Tensor* _self), $(double _p), $(at::Generator * _generator))); }|]

bilinear_tttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
bilinear_tttt _input1 _input2 _weight _bias = [C.block| at::Tensor* { return new at::Tensor(at::native::bilinear(*$(at::Tensor* _input1), *$(at::Tensor* _input2), *$(at::Tensor* _weight), *$(at::Tensor* _bias))); }|]

binary_cross_entropy_with_logits_ttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
binary_cross_entropy_with_logits_ttttl _self _target _weight _pos_weight _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::binary_cross_entropy_with_logits(*$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), *$(at::Tensor* _pos_weight), $(int64_t _reduction))); }|]

binary_cross_entropy_with_logits_backward_tttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
binary_cross_entropy_with_logits_backward_tttttl _grad_output _self _target _weight _pos_weight _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::binary_cross_entropy_with_logits_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), *$(at::Tensor* _pos_weight), $(int64_t _reduction))); }|]

_bincount_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
_bincount_cpu_ttl _self _weights _minlength = [C.block| at::Tensor* { return new at::Tensor(at::native::_bincount_cpu(*$(at::Tensor* _self), *$(at::Tensor* _weights), $(int64_t _minlength))); }|]

_bincount_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
_bincount_cuda_ttl _self _weights _minlength = [C.block| at::Tensor* { return new at::Tensor(at::native::_bincount_cuda(*$(at::Tensor* _self), *$(at::Tensor* _weights), $(int64_t _minlength))); }|]

blackman_window_lo :: Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
blackman_window_lo _window_length _options = [C.block| at::Tensor* { return new at::Tensor(at::native::blackman_window($(int64_t _window_length), *$(at::TensorOptions* _options))); }|]

blackman_window_lbo :: Int64 -> CBool -> Ptr TensorOptions -> IO (Ptr Tensor)
blackman_window_lbo _window_length _periodic _options = [C.block| at::Tensor* { return new at::Tensor(at::native::blackman_window($(int64_t _window_length), $(bool _periodic), *$(at::TensorOptions* _options))); }|]

bmm_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
bmm_cpu_tt _self _mat2 = [C.block| at::Tensor* { return new at::Tensor(at::native::bmm_cpu(*$(at::Tensor* _self), *$(at::Tensor* _mat2))); }|]

bmm_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
bmm_cuda_tt _self _mat2 = [C.block| at::Tensor* { return new at::Tensor(at::native::bmm_cuda(*$(at::Tensor* _self), *$(at::Tensor* _mat2))); }|]

bmm_out_cpu_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
bmm_out_cpu_ttt _result _self _mat2 = [C.block| at::Tensor* { return new at::Tensor(at::native::bmm_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _mat2))); }|]

bmm_out_cuda_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
bmm_out_cuda_ttt _result _self _mat2 = [C.block| at::Tensor* { return new at::Tensor(at::native::bmm_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _mat2))); }|]

broadcast_tensors_l :: Ptr TensorList -> IO (Ptr TensorList)
broadcast_tensors_l _tensors = [C.block| at::TensorList* { return new at::TensorList(at::native::broadcast_tensors(*$(at::TensorList* _tensors))); }|]

cat_ll :: Ptr TensorList -> Int64 -> IO (Ptr Tensor)
cat_ll _tensors _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::cat(*$(at::TensorList* _tensors), $(int64_t _dim))); }|]

cat_out_tll :: Ptr Tensor -> Ptr TensorList -> Int64 -> IO (Ptr Tensor)
cat_out_tll _result _tensors _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::cat_out(*$(at::Tensor* _result), *$(at::TensorList* _tensors), $(int64_t _dim))); }|]

ceil_t :: Ptr Tensor -> IO (Ptr Tensor)
ceil_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::ceil(*$(at::Tensor* _self))); }|]

_ceil__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_ceil__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_ceil__cpu(*$(at::Tensor* _self))); }|]

_ceil__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_ceil__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_ceil__cuda(*$(at::Tensor* _self))); }|]

_ceil_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_ceil_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_ceil_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_ceil_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_ceil_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_ceil_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

chain_matmul_l :: Ptr TensorList -> IO (Ptr Tensor)
chain_matmul_l _matrices = [C.block| at::Tensor* { return new at::Tensor(at::native::chain_matmul(*$(at::TensorList* _matrices))); }|]

chunk_tll :: Ptr Tensor -> Int64 -> Int64 -> IO (Ptr TensorList)
chunk_tll _self _chunks _dim = [C.block| at::TensorList* { return new at::TensorList(at::native::chunk(*$(at::Tensor* _self), $(int64_t _chunks), $(int64_t _dim))); }|]

clamp_tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
clamp_tss _self _min _max = [C.block| at::Tensor* { return new at::Tensor(at::native::clamp(*$(at::Tensor* _self), *$(at::Scalar* _min), *$(at::Scalar* _max))); }|]

_clamp__cpu_tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
_clamp__cpu_tss _self _min _max = [C.block| at::Tensor* { return new at::Tensor(at::native::_clamp__cpu(*$(at::Tensor* _self), *$(at::Scalar* _min), *$(at::Scalar* _max))); }|]

_clamp__cuda_tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
_clamp__cuda_tss _self _min _max = [C.block| at::Tensor* { return new at::Tensor(at::native::_clamp__cuda(*$(at::Tensor* _self), *$(at::Scalar* _min), *$(at::Scalar* _max))); }|]

_clamp_out_cpu_ttss :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
_clamp_out_cpu_ttss _result _self _min _max = [C.block| at::Tensor* { return new at::Tensor(at::native::_clamp_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _min), *$(at::Scalar* _max))); }|]

_clamp_out_cuda_ttss :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
_clamp_out_cuda_ttss _result _self _min _max = [C.block| at::Tensor* { return new at::Tensor(at::native::_clamp_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _min), *$(at::Scalar* _max))); }|]

clamp_max_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
clamp_max_ts _self _max = [C.block| at::Tensor* { return new at::Tensor(at::native::clamp_max(*$(at::Tensor* _self), *$(at::Scalar* _max))); }|]

_clamp_max__cpu_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
_clamp_max__cpu_ts _self _max = [C.block| at::Tensor* { return new at::Tensor(at::native::_clamp_max__cpu(*$(at::Tensor* _self), *$(at::Scalar* _max))); }|]

_clamp_max__cuda_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
_clamp_max__cuda_ts _self _max = [C.block| at::Tensor* { return new at::Tensor(at::native::_clamp_max__cuda(*$(at::Tensor* _self), *$(at::Scalar* _max))); }|]

_clamp_max_out_cpu_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
_clamp_max_out_cpu_tts _result _self _max = [C.block| at::Tensor* { return new at::Tensor(at::native::_clamp_max_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _max))); }|]

_clamp_max_out_cuda_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
_clamp_max_out_cuda_tts _result _self _max = [C.block| at::Tensor* { return new at::Tensor(at::native::_clamp_max_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _max))); }|]

clamp_min_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
clamp_min_ts _self _min = [C.block| at::Tensor* { return new at::Tensor(at::native::clamp_min(*$(at::Tensor* _self), *$(at::Scalar* _min))); }|]

_clamp_min__cpu_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
_clamp_min__cpu_ts _self _min = [C.block| at::Tensor* { return new at::Tensor(at::native::_clamp_min__cpu(*$(at::Tensor* _self), *$(at::Scalar* _min))); }|]

_clamp_min__cuda_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
_clamp_min__cuda_ts _self _min = [C.block| at::Tensor* { return new at::Tensor(at::native::_clamp_min__cuda(*$(at::Tensor* _self), *$(at::Scalar* _min))); }|]

_clamp_min_out_cpu_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
_clamp_min_out_cpu_tts _result _self _min = [C.block| at::Tensor* { return new at::Tensor(at::native::_clamp_min_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _min))); }|]

_clamp_min_out_cuda_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
_clamp_min_out_cuda_tts _result _self _min = [C.block| at::Tensor* { return new at::Tensor(at::native::_clamp_min_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _min))); }|]

cudnn_is_acceptable_t :: Ptr Tensor -> IO (CBool)
cudnn_is_acceptable_t _self = [C.block| bool { return (at::native::cudnn_is_acceptable(*$(at::Tensor* _self))); }|]

constant_pad_nd_tls :: Ptr Tensor -> Ptr IntList -> Ptr Scalar -> IO (Ptr Tensor)
constant_pad_nd_tls _self _pad _value = [C.block| at::Tensor* { return new at::Tensor(at::native::constant_pad_nd(*$(at::Tensor* _self), *$(at::IntList* _pad), *$(at::Scalar* _value))); }|]

contiguous_t :: Ptr Tensor -> IO (Ptr Tensor)
contiguous_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::contiguous(*$(at::Tensor* _self))); }|]

convolution_tttlllbll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> Ptr IntList -> Int64 -> IO (Ptr Tensor)
convolution_tttlllbll _input _weight _bias _stride _padding _dilation _transposed _output_padding _groups = [C.block| at::Tensor* { return new at::Tensor(at::native::convolution(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _transposed), *$(at::IntList* _output_padding), $(int64_t _groups))); }|]

_convolution_tttlllbllbbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> Ptr IntList -> Int64 -> CBool -> CBool -> CBool -> IO (Ptr Tensor)
_convolution_tttlllbllbbb _input _weight _bias _stride _padding _dilation _transposed _output_padding _groups _benchmark _deterministic _cudnn_enabled = [C.block| at::Tensor* { return new at::Tensor(at::native::_convolution(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _transposed), *$(at::IntList* _output_padding), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic), $(bool _cudnn_enabled))); }|]

_convolution_nogroup_tttlllbl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> Ptr IntList -> IO (Ptr Tensor)
_convolution_nogroup_tttlllbl _input _weight _bias _stride _padding _dilation _transposed _output_padding = [C.block| at::Tensor* { return new at::Tensor(at::native::_convolution_nogroup(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _transposed), *$(at::IntList* _output_padding))); }|]

_convolution_double_backward_ttttttlllbllbbba :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> Ptr IntList -> Int64 -> CBool -> CBool -> CBool -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
_convolution_double_backward_ttttttlllbllbbba _ggI _ggW _ggb _gO _weight _self _stride _padding _dilation _transposed _output_padding _groups _benchmark _deterministic _cudnn_enabled _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::_convolution_double_backward(*$(at::Tensor* _ggI), *$(at::Tensor* _ggW), *$(at::Tensor* _ggb), *$(at::Tensor* _gO), *$(at::Tensor* _weight), *$(at::Tensor* _self), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _transposed), *$(at::IntList* _output_padding), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic), $(bool _cudnn_enabled), *$(std::array<bool,3>* _output_mask))); }|]

conv1d_tttllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> IO (Ptr Tensor)
conv1d_tttllll _input _weight _bias _stride _padding _dilation _groups = [C.block| at::Tensor* { return new at::Tensor(at::native::conv1d(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(int64_t _groups))); }|]

conv2d_tttllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> IO (Ptr Tensor)
conv2d_tttllll _input _weight _bias _stride _padding _dilation _groups = [C.block| at::Tensor* { return new at::Tensor(at::native::conv2d(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(int64_t _groups))); }|]

conv3d_tttllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> IO (Ptr Tensor)
conv3d_tttllll _input _weight _bias _stride _padding _dilation _groups = [C.block| at::Tensor* { return new at::Tensor(at::native::conv3d(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(int64_t _groups))); }|]

conv_tbc_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
conv_tbc_tttl _self _weight _bias _pad = [C.block| at::Tensor* { return new at::Tensor(at::native::conv_tbc(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::Tensor* _bias), $(int64_t _pad))); }|]

conv_tbc_backward_ttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr (Tensor,Tensor,Tensor))
conv_tbc_backward_ttttl _self _input _weight _bias _pad = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::conv_tbc_backward(*$(at::Tensor* _self), *$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), $(int64_t _pad))); }|]

conv_transpose1d_tttlllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> Ptr IntList -> IO (Ptr Tensor)
conv_transpose1d_tttlllll _input _weight _bias _stride _padding _output_padding _groups _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::conv_transpose1d(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), $(int64_t _groups), *$(at::IntList* _dilation))); }|]

conv_transpose2d_tttlllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> Ptr IntList -> IO (Ptr Tensor)
conv_transpose2d_tttlllll _input _weight _bias _stride _padding _output_padding _groups _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::conv_transpose2d(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), $(int64_t _groups), *$(at::IntList* _dilation))); }|]

conv_transpose3d_tttlllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> Ptr IntList -> IO (Ptr Tensor)
conv_transpose3d_tttlllll _input _weight _bias _stride _padding _output_padding _groups _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::conv_transpose3d(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), $(int64_t _groups), *$(at::IntList* _dilation))); }|]

_s_copy__cpu_ttb :: Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
_s_copy__cpu_ttb _self _src _non_blocking = [C.block| at::Tensor* { return new at::Tensor(at::native::_s_copy__cpu(*$(at::Tensor* _self), *$(at::Tensor* _src), $(bool _non_blocking))); }|]

_s_copy__cuda_ttb :: Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
_s_copy__cuda_ttb _self _src _non_blocking = [C.block| at::Tensor* { return new at::Tensor(at::native::_s_copy__cuda(*$(at::Tensor* _self), *$(at::Tensor* _src), $(bool _non_blocking))); }|]

_s_copy_from_cuda_ttb :: Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
_s_copy_from_cuda_ttb _self _dst _non_blocking = [C.block| at::Tensor* { return new at::Tensor(at::native::_s_copy_from_cuda(*$(at::Tensor* _self), *$(at::Tensor* _dst), $(bool _non_blocking))); }|]

_copy_same_type__cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO ()
_copy_same_type__cpu_tt _self _src = [C.block| void {  (at::native::_copy_same_type__cpu(*$(at::Tensor* _self), *$(at::Tensor* _src))); }|]

cos_t :: Ptr Tensor -> IO (Ptr Tensor)
cos_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::cos(*$(at::Tensor* _self))); }|]

_cos__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_cos__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_cos__cpu(*$(at::Tensor* _self))); }|]

_cos__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_cos__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_cos__cuda(*$(at::Tensor* _self))); }|]

_cos_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_cos_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_cos_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_cos_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_cos_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_cos_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

cosh_t :: Ptr Tensor -> IO (Ptr Tensor)
cosh_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::cosh(*$(at::Tensor* _self))); }|]

_cosh__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_cosh__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_cosh__cpu(*$(at::Tensor* _self))); }|]

_cosh__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_cosh__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_cosh__cuda(*$(at::Tensor* _self))); }|]

_cosh_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_cosh_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_cosh_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_cosh_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_cosh_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_cosh_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

cosine_embedding_loss_tttdl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CDouble -> Int64 -> IO (Ptr Tensor)
cosine_embedding_loss_tttdl _input1 _input2 _target _margin _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::cosine_embedding_loss(*$(at::Tensor* _input1), *$(at::Tensor* _input2), *$(at::Tensor* _target), $(double _margin), $(int64_t _reduction))); }|]

cudnn_affine_grid_generator_forward_tllll :: Ptr Tensor -> Int64 -> Int64 -> Int64 -> Int64 -> IO (Ptr Tensor)
cudnn_affine_grid_generator_forward_tllll _theta _N _C _H _W = [C.block| at::Tensor* { return new at::Tensor(at::native::cudnn_affine_grid_generator_forward(*$(at::Tensor* _theta), $(int64_t _N), $(int64_t _C), $(int64_t _H), $(int64_t _W))); }|]

cudnn_affine_grid_generator_backward_tllll :: Ptr Tensor -> Int64 -> Int64 -> Int64 -> Int64 -> IO (Ptr Tensor)
cudnn_affine_grid_generator_backward_tllll _grad _N _C _H _W = [C.block| at::Tensor* { return new at::Tensor(at::native::cudnn_affine_grid_generator_backward(*$(at::Tensor* _grad), $(int64_t _N), $(int64_t _C), $(int64_t _H), $(int64_t _W))); }|]

cudnn_batch_norm_tttttbdd :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CDouble -> CDouble -> IO (Ptr (Tensor,Tensor,Tensor))
cudnn_batch_norm_tttttbdd _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::cudnn_batch_norm(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::Tensor* _running_mean), *$(at::Tensor* _running_var), $(bool _training), $(double _exponential_average_factor), $(double _epsilon))); }|]

cudnn_batch_norm_backward_tttttttd :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CDouble -> IO (Ptr (Tensor,Tensor,Tensor))
cudnn_batch_norm_backward_tttttttd _input _grad_output _weight _running_mean _running_var _save_mean _save_var _epsilon = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::cudnn_batch_norm_backward(*$(at::Tensor* _input), *$(at::Tensor* _grad_output), *$(at::Tensor* _weight), *$(at::Tensor* _running_mean), *$(at::Tensor* _running_var), *$(at::Tensor* _save_mean), *$(at::Tensor* _save_var), $(double _epsilon))); }|]

cudnn_convolution_tttllllbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
cudnn_convolution_tttllllbb _self _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = [C.block| at::Tensor* { return new at::Tensor(at::native::cudnn_convolution(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic))); }|]

cudnn_convolution_backward_input_lttllllbb :: Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
cudnn_convolution_backward_input_lttllllbb _self_size _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic = [C.block| at::Tensor* { return new at::Tensor(at::native::cudnn_convolution_backward_input(*$(at::IntList* _self_size), *$(at::Tensor* _grad_output), *$(at::Tensor* _weight), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic))); }|]

cudnn_convolution_backward_tttllllbba :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
cudnn_convolution_backward_tttllllbba _self _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::cudnn_convolution_backward(*$(at::Tensor* _self), *$(at::Tensor* _grad_output), *$(at::Tensor* _weight), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic), *$(std::array<bool,3>* _output_mask))); }|]

cudnn_convolution_backward_bias_t :: Ptr Tensor -> IO (Ptr Tensor)
cudnn_convolution_backward_bias_t _grad_output = [C.block| at::Tensor* { return new at::Tensor(at::native::cudnn_convolution_backward_bias(*$(at::Tensor* _grad_output))); }|]

cudnn_convolution_backward_weight_lttllllbb :: Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
cudnn_convolution_backward_weight_lttllllbb _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic = [C.block| at::Tensor* { return new at::Tensor(at::native::cudnn_convolution_backward_weight(*$(at::IntList* _weight_size), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic))); }|]

cudnn_convolution_transpose_tttlllllbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
cudnn_convolution_transpose_tttlllllbb _self _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic = [C.block| at::Tensor* { return new at::Tensor(at::native::cudnn_convolution_transpose(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic))); }|]

cudnn_convolution_transpose_backward_tttlllllbba :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
cudnn_convolution_transpose_backward_tttlllllbba _self _grad_output _weight _padding _output_padding _stride _dilation _groups _benchmark _deterministic _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::cudnn_convolution_transpose_backward(*$(at::Tensor* _self), *$(at::Tensor* _grad_output), *$(at::Tensor* _weight), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic), *$(std::array<bool,3>* _output_mask))); }|]

cudnn_convolution_transpose_backward_input_ttllllbb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
cudnn_convolution_transpose_backward_input_ttllllbb _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic = [C.block| at::Tensor* { return new at::Tensor(at::native::cudnn_convolution_transpose_backward_input(*$(at::Tensor* _grad_output), *$(at::Tensor* _weight), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic))); }|]

cudnn_convolution_transpose_backward_weight_lttllllbb :: Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
cudnn_convolution_transpose_backward_weight_lttllllbb _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic = [C.block| at::Tensor* { return new at::Tensor(at::native::cudnn_convolution_transpose_backward_weight(*$(at::IntList* _weight_size), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic))); }|]

cudnn_grid_sampler_forward_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
cudnn_grid_sampler_forward_tt _self _grid = [C.block| at::Tensor* { return new at::Tensor(at::native::cudnn_grid_sampler_forward(*$(at::Tensor* _self), *$(at::Tensor* _grid))); }|]

cudnn_grid_sampler_backward_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
cudnn_grid_sampler_backward_ttt _self _grid _grad_output = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::cudnn_grid_sampler_backward(*$(at::Tensor* _self), *$(at::Tensor* _grid), *$(at::Tensor* _grad_output))); }|]

cumsum_tls :: Ptr Tensor -> Int64 -> Ptr ScalarType -> IO (Ptr Tensor)
cumsum_tls _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::cumsum(*$(at::Tensor* _self), $(int64_t _dim), *$(at::ScalarType* _dtype))); }|]

cumsum_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
cumsum_tl _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::cumsum(*$(at::Tensor* _self), $(int64_t _dim))); }|]

cumsum_out_ttls :: Ptr Tensor -> Ptr Tensor -> Int64 -> Ptr ScalarType -> IO (Ptr Tensor)
cumsum_out_ttls _result _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::cumsum_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim), *$(at::ScalarType* _dtype))); }|]

cumsum_out_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
cumsum_out_ttl _result _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::cumsum_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim))); }|]

cumprod_tls :: Ptr Tensor -> Int64 -> Ptr ScalarType -> IO (Ptr Tensor)
cumprod_tls _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::cumprod(*$(at::Tensor* _self), $(int64_t _dim), *$(at::ScalarType* _dtype))); }|]

cumprod_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
cumprod_tl _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::cumprod(*$(at::Tensor* _self), $(int64_t _dim))); }|]

cumprod_out_ttls :: Ptr Tensor -> Ptr Tensor -> Int64 -> Ptr ScalarType -> IO (Ptr Tensor)
cumprod_out_ttls _result _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::cumprod_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim), *$(at::ScalarType* _dtype))); }|]

cumprod_out_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
cumprod_out_ttl _result _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::cumprod_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim))); }|]

ctc_loss_ttllll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Int64 -> Int64 -> IO (Ptr Tensor)
ctc_loss_ttllll _log_probs _targets _input_lengths _target_lengths _blank _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::ctc_loss(*$(at::Tensor* _log_probs), *$(at::Tensor* _targets), *$(at::IntList* _input_lengths), *$(at::IntList* _target_lengths), $(int64_t _blank), $(int64_t _reduction))); }|]

ctc_loss_ttttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
ctc_loss_ttttll _log_probs _targets _input_lengths _target_lengths _blank _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::ctc_loss(*$(at::Tensor* _log_probs), *$(at::Tensor* _targets), *$(at::Tensor* _input_lengths), *$(at::Tensor* _target_lengths), $(int64_t _blank), $(int64_t _reduction))); }|]

ctc_loss_cpu_ttlll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Int64 -> IO (Ptr (Tensor,Tensor))
ctc_loss_cpu_ttlll _log_probs _targets _input_lengths _target_lengths _blank = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::ctc_loss_cpu(*$(at::Tensor* _log_probs), *$(at::Tensor* _targets), *$(at::IntList* _input_lengths), *$(at::IntList* _target_lengths), $(int64_t _blank))); }|]

ctc_loss_gpu_ttlll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Int64 -> IO (Ptr (Tensor,Tensor))
ctc_loss_gpu_ttlll _log_probs _targets _input_lengths _target_lengths _blank = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::ctc_loss_gpu(*$(at::Tensor* _log_probs), *$(at::Tensor* _targets), *$(at::IntList* _input_lengths), *$(at::IntList* _target_lengths), $(int64_t _blank))); }|]

ctc_loss_backward_cpu_tttllttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
ctc_loss_backward_cpu_tttllttl _grad _log_probs _targets _input_lengths _target_lengths _neg_log_likelihood _log_alpha _blank = [C.block| at::Tensor* { return new at::Tensor(at::native::ctc_loss_backward_cpu(*$(at::Tensor* _grad), *$(at::Tensor* _log_probs), *$(at::Tensor* _targets), *$(at::IntList* _input_lengths), *$(at::IntList* _target_lengths), *$(at::Tensor* _neg_log_likelihood), *$(at::Tensor* _log_alpha), $(int64_t _blank))); }|]

ctc_loss_backward_gpu_tttllttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
ctc_loss_backward_gpu_tttllttl _grad _log_probs _targets _input_lengths _target_lengths _neg_log_likelihood _log_alpha _blank = [C.block| at::Tensor* { return new at::Tensor(at::native::ctc_loss_backward_gpu(*$(at::Tensor* _grad), *$(at::Tensor* _log_probs), *$(at::Tensor* _targets), *$(at::IntList* _input_lengths), *$(at::IntList* _target_lengths), *$(at::Tensor* _neg_log_likelihood), *$(at::Tensor* _log_alpha), $(int64_t _blank))); }|]

det_t :: Ptr Tensor -> IO (Ptr Tensor)
det_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::det(*$(at::Tensor* _self))); }|]

diag_embed_tlll :: Ptr Tensor -> Int64 -> Int64 -> Int64 -> IO (Ptr Tensor)
diag_embed_tlll _self _offset _dim1 _dim2 = [C.block| at::Tensor* { return new at::Tensor(at::native::diag_embed(*$(at::Tensor* _self), $(int64_t _offset), $(int64_t _dim1), $(int64_t _dim2))); }|]

diagflat_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
diagflat_tl _self _offset = [C.block| at::Tensor* { return new at::Tensor(at::native::diagflat(*$(at::Tensor* _self), $(int64_t _offset))); }|]

diagonal_tlll :: Ptr Tensor -> Int64 -> Int64 -> Int64 -> IO (Ptr Tensor)
diagonal_tlll _self _offset _dim1 _dim2 = [C.block| at::Tensor* { return new at::Tensor(at::native::diagonal(*$(at::Tensor* _self), $(int64_t _offset), $(int64_t _dim1), $(int64_t _dim2))); }|]

div_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
div_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::div(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

div__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
div__tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::div_(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

div_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
div_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::div_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

div_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
div_ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::div(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

div__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
div__ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::div_(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

dot_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
dot_tt _self _tensor = [C.block| at::Tensor* { return new at::Tensor(at::native::dot(*$(at::Tensor* _self), *$(at::Tensor* _tensor))); }|]

dot_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
dot_out_ttt _result _self _tensor = [C.block| at::Tensor* { return new at::Tensor(at::native::dot_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _tensor))); }|]

einsum_sl :: Ptr StdString -> Ptr TensorList -> IO (Ptr Tensor)
einsum_sl _equation _tensors = [C.block| at::Tensor* { return new at::Tensor(at::native::einsum(*$(std::string* _equation), *$(at::TensorList* _tensors))); }|]

embedding_ttlbb :: Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
embedding_ttlbb _weight _indices _padding_idx _scale_grad_by_freq _sparse = [C.block| at::Tensor* { return new at::Tensor(at::native::embedding(*$(at::Tensor* _weight), *$(at::Tensor* _indices), $(int64_t _padding_idx), $(bool _scale_grad_by_freq), $(bool _sparse))); }|]

embedding_backward_ttllbb :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
embedding_backward_ttllbb _grad _indices _num_weights _padding_idx _scale_grad_by_freq _sparse = [C.block| at::Tensor* { return new at::Tensor(at::native::embedding_backward(*$(at::Tensor* _grad), *$(at::Tensor* _indices), $(int64_t _num_weights), $(int64_t _padding_idx), $(bool _scale_grad_by_freq), $(bool _sparse))); }|]

embedding_dense_backward_cpu_ttllb :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> CBool -> IO (Ptr Tensor)
embedding_dense_backward_cpu_ttllb _grad _indices _num_weights _padding_idx _scale_grad_by_freq = [C.block| at::Tensor* { return new at::Tensor(at::native::embedding_dense_backward_cpu(*$(at::Tensor* _grad), *$(at::Tensor* _indices), $(int64_t _num_weights), $(int64_t _padding_idx), $(bool _scale_grad_by_freq))); }|]

embedding_dense_backward_cuda_ttllb :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> CBool -> IO (Ptr Tensor)
embedding_dense_backward_cuda_ttllb _grad _indices _num_weights _padding_idx _scale_grad_by_freq = [C.block| at::Tensor* { return new at::Tensor(at::native::embedding_dense_backward_cuda(*$(at::Tensor* _grad), *$(at::Tensor* _indices), $(int64_t _num_weights), $(int64_t _padding_idx), $(bool _scale_grad_by_freq))); }|]

embedding_renorm_cpu__ttdd :: Ptr Tensor -> Ptr Tensor -> CDouble -> CDouble -> IO (Ptr Tensor)
embedding_renorm_cpu__ttdd _self _indices _max_norm _norm_type = [C.block| at::Tensor* { return new at::Tensor(at::native::embedding_renorm_cpu_(*$(at::Tensor* _self), *$(at::Tensor* _indices), $(double _max_norm), $(double _norm_type))); }|]

embedding_renorm_cuda__ttdd :: Ptr Tensor -> Ptr Tensor -> CDouble -> CDouble -> IO (Ptr Tensor)
embedding_renorm_cuda__ttdd _self _indices _max_norm _norm_type = [C.block| at::Tensor* { return new at::Tensor(at::native::embedding_renorm_cuda_(*$(at::Tensor* _self), *$(at::Tensor* _indices), $(double _max_norm), $(double _norm_type))); }|]

embedding_sparse_backward_ttllb :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> CBool -> IO (Ptr Tensor)
embedding_sparse_backward_ttllb _grad _indices _num_weights _padding_idx _scale_grad_by_freq = [C.block| at::Tensor* { return new at::Tensor(at::native::embedding_sparse_backward(*$(at::Tensor* _grad), *$(at::Tensor* _indices), $(int64_t _num_weights), $(int64_t _padding_idx), $(bool _scale_grad_by_freq))); }|]

embedding_bag_tttblb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor,Tensor,Tensor))
embedding_bag_tttblb _weight _indices _offsets _scale_grad_by_freq _mode _sparse = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>(at::native::embedding_bag(*$(at::Tensor* _weight), *$(at::Tensor* _indices), *$(at::Tensor* _offsets), $(bool _scale_grad_by_freq), $(int64_t _mode), $(bool _sparse))); }|]

_embedding_bag_cpu_tttblb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor,Tensor,Tensor))
_embedding_bag_cpu_tttblb _weight _indices _offsets _scale_grad_by_freq _mode _sparse = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>(at::native::_embedding_bag_cpu(*$(at::Tensor* _weight), *$(at::Tensor* _indices), *$(at::Tensor* _offsets), $(bool _scale_grad_by_freq), $(int64_t _mode), $(bool _sparse))); }|]

_embedding_bag_cuda_tttblb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor,Tensor,Tensor))
_embedding_bag_cuda_tttblb _weight _indices _offsets _scale_grad_by_freq _mode _sparse = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>(at::native::_embedding_bag_cuda(*$(at::Tensor* _weight), *$(at::Tensor* _indices), *$(at::Tensor* _offsets), $(bool _scale_grad_by_freq), $(int64_t _mode), $(bool _sparse))); }|]

_embedding_bag_backward_ttttttlblb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> Int64 -> CBool -> IO (Ptr Tensor)
_embedding_bag_backward_ttttttlblb _grad _indices _offsets _offset2bag _bag_size _maximum_indices _num_weights _scale_grad_by_freq _mode _sparse = [C.block| at::Tensor* { return new at::Tensor(at::native::_embedding_bag_backward(*$(at::Tensor* _grad), *$(at::Tensor* _indices), *$(at::Tensor* _offsets), *$(at::Tensor* _offset2bag), *$(at::Tensor* _bag_size), *$(at::Tensor* _maximum_indices), $(int64_t _num_weights), $(bool _scale_grad_by_freq), $(int64_t _mode), $(bool _sparse))); }|]

_embedding_bag_sparse_backward_tttttlbl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> Int64 -> IO (Ptr Tensor)
_embedding_bag_sparse_backward_tttttlbl _grad _indices _offsets _offset2bag _bag_size _num_weights _scale_grad_by_freq _mode = [C.block| at::Tensor* { return new at::Tensor(at::native::_embedding_bag_sparse_backward(*$(at::Tensor* _grad), *$(at::Tensor* _indices), *$(at::Tensor* _offsets), *$(at::Tensor* _offset2bag), *$(at::Tensor* _bag_size), $(int64_t _num_weights), $(bool _scale_grad_by_freq), $(int64_t _mode))); }|]

_embedding_bag_dense_backward_cpu_ttttttlbl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> Int64 -> IO (Ptr Tensor)
_embedding_bag_dense_backward_cpu_ttttttlbl _grad _indices _offsets _offset2bag _bag_size _maximum_indices _num_weights _scale_grad_by_freq _mode = [C.block| at::Tensor* { return new at::Tensor(at::native::_embedding_bag_dense_backward_cpu(*$(at::Tensor* _grad), *$(at::Tensor* _indices), *$(at::Tensor* _offsets), *$(at::Tensor* _offset2bag), *$(at::Tensor* _bag_size), *$(at::Tensor* _maximum_indices), $(int64_t _num_weights), $(bool _scale_grad_by_freq), $(int64_t _mode))); }|]

_embedding_bag_dense_backward_cuda_ttttttlbl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> Int64 -> IO (Ptr Tensor)
_embedding_bag_dense_backward_cuda_ttttttlbl _grad _indices _offsets _offset2bag _bag_size _maximum_indices _num_weights _scale_grad_by_freq _mode = [C.block| at::Tensor* { return new at::Tensor(at::native::_embedding_bag_dense_backward_cuda(*$(at::Tensor* _grad), *$(at::Tensor* _indices), *$(at::Tensor* _offsets), *$(at::Tensor* _offset2bag), *$(at::Tensor* _bag_size), *$(at::Tensor* _maximum_indices), $(int64_t _num_weights), $(bool _scale_grad_by_freq), $(int64_t _mode))); }|]

empty_cpu_lo :: Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
empty_cpu_lo _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::empty_cpu(*$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

empty_cuda_lo :: Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
empty_cuda_lo _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::empty_cuda(*$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

empty_sparse_lo :: Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
empty_sparse_lo _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::empty_sparse(*$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

resize_cpu__tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
resize_cpu__tl _self _size = [C.block| at::Tensor* { return new at::Tensor(at::native::resize_cpu_(*$(at::Tensor* _self), *$(at::IntList* _size))); }|]

resize_cuda__tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
resize_cuda__tl _self _size = [C.block| at::Tensor* { return new at::Tensor(at::native::resize_cuda_(*$(at::Tensor* _self), *$(at::IntList* _size))); }|]

empty_out_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
empty_out_tl _result _size = [C.block| at::Tensor* { return new at::Tensor(at::native::empty_out(*$(at::Tensor* _result), *$(at::IntList* _size))); }|]

empty_like_t :: Ptr Tensor -> IO (Ptr Tensor)
empty_like_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::empty_like(*$(at::Tensor* _self))); }|]

empty_like_to :: Ptr Tensor -> Ptr TensorOptions -> IO (Ptr Tensor)
empty_like_to _self _options = [C.block| at::Tensor* { return new at::Tensor(at::native::empty_like(*$(at::Tensor* _self), *$(at::TensorOptions* _options))); }|]

empty_strided_cpu_llo :: Ptr IntList -> Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
empty_strided_cpu_llo _size _stride _options = [C.block| at::Tensor* { return new at::Tensor(at::native::empty_strided_cpu(*$(at::IntList* _size), *$(at::IntList* _stride), *$(at::TensorOptions* _options))); }|]

empty_strided_cuda_llo :: Ptr IntList -> Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
empty_strided_cuda_llo _size _stride _options = [C.block| at::Tensor* { return new at::Tensor(at::native::empty_strided_cuda(*$(at::IntList* _size), *$(at::IntList* _stride), *$(at::TensorOptions* _options))); }|]

erf_t :: Ptr Tensor -> IO (Ptr Tensor)
erf_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::erf(*$(at::Tensor* _self))); }|]

_erf__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_erf__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_erf__cpu(*$(at::Tensor* _self))); }|]

_erf__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_erf__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_erf__cuda(*$(at::Tensor* _self))); }|]

_erf_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_erf_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_erf_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_erf_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_erf_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_erf_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

erfc_t :: Ptr Tensor -> IO (Ptr Tensor)
erfc_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::erfc(*$(at::Tensor* _self))); }|]

_erfc__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_erfc__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_erfc__cpu(*$(at::Tensor* _self))); }|]

_erfc__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_erfc__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_erfc__cuda(*$(at::Tensor* _self))); }|]

_erfc_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_erfc_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_erfc_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_erfc_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_erfc_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_erfc_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

exp_t :: Ptr Tensor -> IO (Ptr Tensor)
exp_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::exp(*$(at::Tensor* _self))); }|]

_exp__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_exp__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_exp__cpu(*$(at::Tensor* _self))); }|]

_exp__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_exp__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_exp__cuda(*$(at::Tensor* _self))); }|]

_exp_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_exp_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_exp_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_exp_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_exp_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_exp_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

expm1_t :: Ptr Tensor -> IO (Ptr Tensor)
expm1_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::expm1(*$(at::Tensor* _self))); }|]

_expm1__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_expm1__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_expm1__cpu(*$(at::Tensor* _self))); }|]

_expm1__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_expm1__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_expm1__cuda(*$(at::Tensor* _self))); }|]

_expm1_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_expm1_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_expm1_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_expm1_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_expm1_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_expm1_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

expand_tlb :: Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
expand_tlb _self _size _implicit = [C.block| at::Tensor* { return new at::Tensor(at::native::expand(*$(at::Tensor* _self), *$(at::IntList* _size), $(bool _implicit))); }|]

expand_as_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
expand_as_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::expand_as(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

eye_lo :: Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
eye_lo _n _options = [C.block| at::Tensor* { return new at::Tensor(at::native::eye($(int64_t _n), *$(at::TensorOptions* _options))); }|]

eye_llo :: Int64 -> Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
eye_llo _n _m _options = [C.block| at::Tensor* { return new at::Tensor(at::native::eye($(int64_t _n), $(int64_t _m), *$(at::TensorOptions* _options))); }|]

eye_out_cpu_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
eye_out_cpu_tl _result _n = [C.block| at::Tensor* { return new at::Tensor(at::native::eye_out_cpu(*$(at::Tensor* _result), $(int64_t _n))); }|]

eye_out_cuda_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
eye_out_cuda_tl _result _n = [C.block| at::Tensor* { return new at::Tensor(at::native::eye_out_cuda(*$(at::Tensor* _result), $(int64_t _n))); }|]

eye_out_cpu_tll :: Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
eye_out_cpu_tll _result _n _m = [C.block| at::Tensor* { return new at::Tensor(at::native::eye_out_cpu(*$(at::Tensor* _result), $(int64_t _n), $(int64_t _m))); }|]

eye_out_cuda_tll :: Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
eye_out_cuda_tll _result _n _m = [C.block| at::Tensor* { return new at::Tensor(at::native::eye_out_cuda(*$(at::Tensor* _result), $(int64_t _n), $(int64_t _m))); }|]

flatten_tll :: Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
flatten_tll _self _start_dim _end_dim = [C.block| at::Tensor* { return new at::Tensor(at::native::flatten(*$(at::Tensor* _self), $(int64_t _start_dim), $(int64_t _end_dim))); }|]

fill__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
fill__ts _self _value = [C.block| at::Tensor* { return new at::Tensor(at::native::fill_(*$(at::Tensor* _self), *$(at::Scalar* _value))); }|]

fill__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
fill__tt _self _value = [C.block| at::Tensor* { return new at::Tensor(at::native::fill_(*$(at::Tensor* _self), *$(at::Tensor* _value))); }|]

floor_t :: Ptr Tensor -> IO (Ptr Tensor)
floor_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::floor(*$(at::Tensor* _self))); }|]

_floor__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_floor__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_floor__cpu(*$(at::Tensor* _self))); }|]

_floor__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_floor__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_floor__cuda(*$(at::Tensor* _self))); }|]

_floor_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_floor_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_floor_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_floor_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_floor_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_floor_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

full_lso :: Ptr IntList -> Ptr Scalar -> Ptr TensorOptions -> IO (Ptr Tensor)
full_lso _size _fill_value _options = [C.block| at::Tensor* { return new at::Tensor(at::native::full(*$(at::IntList* _size), *$(at::Scalar* _fill_value), *$(at::TensorOptions* _options))); }|]

full_out_tls :: Ptr Tensor -> Ptr IntList -> Ptr Scalar -> IO (Ptr Tensor)
full_out_tls _result _size _fill_value = [C.block| at::Tensor* { return new at::Tensor(at::native::full_out(*$(at::Tensor* _result), *$(at::IntList* _size), *$(at::Scalar* _fill_value))); }|]

full_like_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
full_like_ts _self _fill_value = [C.block| at::Tensor* { return new at::Tensor(at::native::full_like(*$(at::Tensor* _self), *$(at::Scalar* _fill_value))); }|]

full_like_tso :: Ptr Tensor -> Ptr Scalar -> Ptr TensorOptions -> IO (Ptr Tensor)
full_like_tso _self _fill_value _options = [C.block| at::Tensor* { return new at::Tensor(at::native::full_like(*$(at::Tensor* _self), *$(at::Scalar* _fill_value), *$(at::TensorOptions* _options))); }|]

grid_sampler_ttll :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
grid_sampler_ttll _input _grid _interpolation_mode _padding_mode = [C.block| at::Tensor* { return new at::Tensor(at::native::grid_sampler(*$(at::Tensor* _input), *$(at::Tensor* _grid), $(int64_t _interpolation_mode), $(int64_t _padding_mode))); }|]

grid_sampler_2d_cpu_ttll :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
grid_sampler_2d_cpu_ttll _input _grid _interpolation_mode _padding_mode = [C.block| at::Tensor* { return new at::Tensor(at::native::grid_sampler_2d_cpu(*$(at::Tensor* _input), *$(at::Tensor* _grid), $(int64_t _interpolation_mode), $(int64_t _padding_mode))); }|]

grid_sampler_2d_cuda_ttll :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
grid_sampler_2d_cuda_ttll _input _grid _interpolation_mode _padding_mode = [C.block| at::Tensor* { return new at::Tensor(at::native::grid_sampler_2d_cuda(*$(at::Tensor* _input), *$(at::Tensor* _grid), $(int64_t _interpolation_mode), $(int64_t _padding_mode))); }|]

grid_sampler_2d_backward_cpu_tttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr (Tensor,Tensor))
grid_sampler_2d_backward_cpu_tttll _grad_output _input _grid _interpolation_mode _padding_mode = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::grid_sampler_2d_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _input), *$(at::Tensor* _grid), $(int64_t _interpolation_mode), $(int64_t _padding_mode))); }|]

grid_sampler_2d_backward_cuda_tttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr (Tensor,Tensor))
grid_sampler_2d_backward_cuda_tttll _grad_output _input _grid _interpolation_mode _padding_mode = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::grid_sampler_2d_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _input), *$(at::Tensor* _grid), $(int64_t _interpolation_mode), $(int64_t _padding_mode))); }|]

grid_sampler_3d_cpu_ttll :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
grid_sampler_3d_cpu_ttll _input _grid _interpolation_mode _padding_mode = [C.block| at::Tensor* { return new at::Tensor(at::native::grid_sampler_3d_cpu(*$(at::Tensor* _input), *$(at::Tensor* _grid), $(int64_t _interpolation_mode), $(int64_t _padding_mode))); }|]

grid_sampler_3d_cuda_ttll :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
grid_sampler_3d_cuda_ttll _input _grid _interpolation_mode _padding_mode = [C.block| at::Tensor* { return new at::Tensor(at::native::grid_sampler_3d_cuda(*$(at::Tensor* _input), *$(at::Tensor* _grid), $(int64_t _interpolation_mode), $(int64_t _padding_mode))); }|]

grid_sampler_3d_backward_cpu_tttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr (Tensor,Tensor))
grid_sampler_3d_backward_cpu_tttll _grad_output _input _grid _interpolation_mode _padding_mode = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::grid_sampler_3d_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _input), *$(at::Tensor* _grid), $(int64_t _interpolation_mode), $(int64_t _padding_mode))); }|]

grid_sampler_3d_backward_cuda_tttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr (Tensor,Tensor))
grid_sampler_3d_backward_cuda_tttll _grad_output _input _grid _interpolation_mode _padding_mode = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::grid_sampler_3d_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _input), *$(at::Tensor* _grid), $(int64_t _interpolation_mode), $(int64_t _padding_mode))); }|]

hann_window_lo :: Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
hann_window_lo _window_length _options = [C.block| at::Tensor* { return new at::Tensor(at::native::hann_window($(int64_t _window_length), *$(at::TensorOptions* _options))); }|]

hann_window_lbo :: Int64 -> CBool -> Ptr TensorOptions -> IO (Ptr Tensor)
hann_window_lbo _window_length _periodic _options = [C.block| at::Tensor* { return new at::Tensor(at::native::hann_window($(int64_t _window_length), $(bool _periodic), *$(at::TensorOptions* _options))); }|]

hamming_window_lo :: Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
hamming_window_lo _window_length _options = [C.block| at::Tensor* { return new at::Tensor(at::native::hamming_window($(int64_t _window_length), *$(at::TensorOptions* _options))); }|]

hamming_window_lbo :: Int64 -> CBool -> Ptr TensorOptions -> IO (Ptr Tensor)
hamming_window_lbo _window_length _periodic _options = [C.block| at::Tensor* { return new at::Tensor(at::native::hamming_window($(int64_t _window_length), $(bool _periodic), *$(at::TensorOptions* _options))); }|]

hamming_window_lbdo :: Int64 -> CBool -> CDouble -> Ptr TensorOptions -> IO (Ptr Tensor)
hamming_window_lbdo _window_length _periodic _alpha _options = [C.block| at::Tensor* { return new at::Tensor(at::native::hamming_window($(int64_t _window_length), $(bool _periodic), $(double _alpha), *$(at::TensorOptions* _options))); }|]

hamming_window_lbddo :: Int64 -> CBool -> CDouble -> CDouble -> Ptr TensorOptions -> IO (Ptr Tensor)
hamming_window_lbddo _window_length _periodic _alpha _beta _options = [C.block| at::Tensor* { return new at::Tensor(at::native::hamming_window($(int64_t _window_length), $(bool _periodic), $(double _alpha), $(double _beta), *$(at::TensorOptions* _options))); }|]

hinge_embedding_loss_ttdl :: Ptr Tensor -> Ptr Tensor -> CDouble -> Int64 -> IO (Ptr Tensor)
hinge_embedding_loss_ttdl _self _target _margin _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::hinge_embedding_loss(*$(at::Tensor* _self), *$(at::Tensor* _target), $(double _margin), $(int64_t _reduction))); }|]

ger_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
ger_tt _self _vec2 = [C.block| at::Tensor* { return new at::Tensor(at::native::ger(*$(at::Tensor* _self), *$(at::Tensor* _vec2))); }|]

ger_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
ger_out_ttt _result _self _vec2 = [C.block| at::Tensor* { return new at::Tensor(at::native::ger_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _vec2))); }|]

gesv_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
gesv_tt _self _A = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::gesv(*$(at::Tensor* _self), *$(at::Tensor* _A))); }|]

gesv_out_tttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
gesv_out_tttt _solution _lu _self _A = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::gesv_out(*$(at::Tensor* _solution), *$(at::Tensor* _lu), *$(at::Tensor* _self), *$(at::Tensor* _A))); }|]

_gesv_helper_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
_gesv_helper_cpu_tt _self _A = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::_gesv_helper_cpu(*$(at::Tensor* _self), *$(at::Tensor* _A))); }|]

_gesv_helper_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
_gesv_helper_cuda_tt _self _A = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::_gesv_helper_cuda(*$(at::Tensor* _self), *$(at::Tensor* _A))); }|]

group_norm_tlttdb :: Ptr Tensor -> Int64 -> Ptr Tensor -> Ptr Tensor -> CDouble -> CBool -> IO (Ptr Tensor)
group_norm_tlttdb _input _num_groups _weight _bias _eps _cudnn_enabled = [C.block| at::Tensor* { return new at::Tensor(at::native::group_norm(*$(at::Tensor* _input), $(int64_t _num_groups), *$(at::Tensor* _weight), *$(at::Tensor* _bias), $(double _eps), $(bool _cudnn_enabled))); }|]

fft_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
fft_tlb _self _signal_ndim _normalized = [C.block| at::Tensor* { return new at::Tensor(at::native::fft(*$(at::Tensor* _self), $(int64_t _signal_ndim), $(bool _normalized))); }|]

ifft_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
ifft_tlb _self _signal_ndim _normalized = [C.block| at::Tensor* { return new at::Tensor(at::native::ifft(*$(at::Tensor* _self), $(int64_t _signal_ndim), $(bool _normalized))); }|]

rfft_tlbb :: Ptr Tensor -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
rfft_tlbb _self _signal_ndim _normalized _onesided = [C.block| at::Tensor* { return new at::Tensor(at::native::rfft(*$(at::Tensor* _self), $(int64_t _signal_ndim), $(bool _normalized), $(bool _onesided))); }|]

irfft_tlbbl :: Ptr Tensor -> Int64 -> CBool -> CBool -> Ptr IntList -> IO (Ptr Tensor)
irfft_tlbbl _self _signal_ndim _normalized _onesided _signal_sizes = [C.block| at::Tensor* { return new at::Tensor(at::native::irfft(*$(at::Tensor* _self), $(int64_t _signal_ndim), $(bool _normalized), $(bool _onesided), *$(at::IntList* _signal_sizes))); }|]

_fft_mkl_tlbbblbbl :: Ptr Tensor -> Int64 -> CBool -> CBool -> CBool -> Ptr IntList -> CBool -> CBool -> Ptr IntList -> IO (Ptr Tensor)
_fft_mkl_tlbbblbbl _self _signal_ndim _complex_input _complex_output _inverse _checked_signal_sizes _normalized _onesided _output_sizes = [C.block| at::Tensor* { return new at::Tensor(at::native::_fft_mkl(*$(at::Tensor* _self), $(int64_t _signal_ndim), $(bool _complex_input), $(bool _complex_output), $(bool _inverse), *$(at::IntList* _checked_signal_sizes), $(bool _normalized), $(bool _onesided), *$(at::IntList* _output_sizes))); }|]

_fft_cufft_tlbbblbbl :: Ptr Tensor -> Int64 -> CBool -> CBool -> CBool -> Ptr IntList -> CBool -> CBool -> Ptr IntList -> IO (Ptr Tensor)
_fft_cufft_tlbbblbbl _self _signal_ndim _complex_input _complex_output _inverse _checked_signal_sizes _normalized _onesided _output_sizes = [C.block| at::Tensor* { return new at::Tensor(at::native::_fft_cufft(*$(at::Tensor* _self), $(int64_t _signal_ndim), $(bool _complex_input), $(bool _complex_output), $(bool _inverse), *$(at::IntList* _checked_signal_sizes), $(bool _normalized), $(bool _onesided), *$(at::IntList* _output_sizes))); }|]

_cufft_get_plan_cache_size_ :: IO (Int64)
_cufft_get_plan_cache_size_  = [C.block| int64_t { return (at::native::_cufft_get_plan_cache_size()); }|]

_cufft_get_plan_cache_max_size_ :: IO (Int64)
_cufft_get_plan_cache_max_size_  = [C.block| int64_t { return (at::native::_cufft_get_plan_cache_max_size()); }|]

_cufft_set_plan_cache_max_size_l :: Int64 -> IO ()
_cufft_set_plan_cache_max_size_l _max_size = [C.block| void {  (at::native::_cufft_set_plan_cache_max_size($(int64_t _max_size))); }|]

_cufft_clear_plan_cache_ :: IO ()
_cufft_clear_plan_cache_  = [C.block| void {  (at::native::_cufft_clear_plan_cache()); }|]

index_tl :: Ptr Tensor -> Ptr TensorList -> IO (Ptr Tensor)
index_tl _self _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::index(*$(at::Tensor* _self), *$(at::TensorList* _indices))); }|]

index_copy__tltt :: Ptr Tensor -> Int64 -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
index_copy__tltt _self _dim _index _source = [C.block| at::Tensor* { return new at::Tensor(at::native::index_copy_(*$(at::Tensor* _self), $(int64_t _dim), *$(at::Tensor* _index), *$(at::Tensor* _source))); }|]

index_put_tltb :: Ptr Tensor -> Ptr TensorList -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
index_put_tltb _self _indices _values _accumulate = [C.block| at::Tensor* { return new at::Tensor(at::native::index_put(*$(at::Tensor* _self), *$(at::TensorList* _indices), *$(at::Tensor* _values), $(bool _accumulate))); }|]

index_put__tltb :: Ptr Tensor -> Ptr TensorList -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
index_put__tltb _self _indices _values _accumulate = [C.block| at::Tensor* { return new at::Tensor(at::native::index_put_(*$(at::Tensor* _self), *$(at::TensorList* _indices), *$(at::Tensor* _values), $(bool _accumulate))); }|]

instance_norm_tttttbddb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CDouble -> CDouble -> CBool -> IO (Ptr Tensor)
instance_norm_tttttbddb _input _weight _bias _running_mean _running_var _use_input_stats _momentum _eps _cudnn_enabled = [C.block| at::Tensor* { return new at::Tensor(at::native::instance_norm(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::Tensor* _running_mean), *$(at::Tensor* _running_var), $(bool _use_input_stats), $(double _momentum), $(double _eps), $(bool _cudnn_enabled))); }|]

inverse_t :: Ptr Tensor -> IO (Ptr Tensor)
inverse_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::inverse(*$(at::Tensor* _self))); }|]

inverse_out_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
inverse_out_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::inverse_out(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_inverse_helper_cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_inverse_helper_cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_inverse_helper_cpu(*$(at::Tensor* _self))); }|]

_inverse_helper_cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_inverse_helper_cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_inverse_helper_cuda(*$(at::Tensor* _self))); }|]

isclose_ttddb :: Ptr Tensor -> Ptr Tensor -> CDouble -> CDouble -> CBool -> IO (Ptr Tensor)
isclose_ttddb _self _other _rtol _atol _equal_nan = [C.block| at::Tensor* { return new at::Tensor(at::native::isclose(*$(at::Tensor* _self), *$(at::Tensor* _other), $(double _rtol), $(double _atol), $(bool _equal_nan))); }|]

is_distributed_t :: Ptr Tensor -> IO (CBool)
is_distributed_t _self = [C.block| bool { return (at::native::is_distributed(*$(at::Tensor* _self))); }|]

is_floating_point_t :: Ptr Tensor -> IO (CBool)
is_floating_point_t _self = [C.block| bool { return (at::native::is_floating_point(*$(at::Tensor* _self))); }|]

is_complex_t :: Ptr Tensor -> IO (CBool)
is_complex_t _self = [C.block| bool { return (at::native::is_complex(*$(at::Tensor* _self))); }|]

is_nonzero_t :: Ptr Tensor -> IO (CBool)
is_nonzero_t _self = [C.block| bool { return (at::native::is_nonzero(*$(at::Tensor* _self))); }|]

is_same_size_tt :: Ptr Tensor -> Ptr Tensor -> IO (CBool)
is_same_size_tt _self _other = [C.block| bool { return (at::native::is_same_size(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

is_signed_t :: Ptr Tensor -> IO (CBool)
is_signed_t _self = [C.block| bool { return (at::native::is_signed(*$(at::Tensor* _self))); }|]

kl_div_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
kl_div_ttl _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::kl_div(*$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

kl_div_backward_cpu_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
kl_div_backward_cpu_tttl _grad_output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::kl_div_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

kl_div_backward_cuda_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
kl_div_backward_cuda_tttl _grad_output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::kl_div_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

kthvalue_tllb :: Ptr Tensor -> Int64 -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
kthvalue_tllb _self _k _dim _keepdim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::kthvalue(*$(at::Tensor* _self), $(int64_t _k), $(int64_t _dim), $(bool _keepdim))); }|]

kthvalue_out_tttllb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
kthvalue_out_tttllb _values _indices _self _k _dim _keepdim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::kthvalue_out(*$(at::Tensor* _values), *$(at::Tensor* _indices), *$(at::Tensor* _self), $(int64_t _k), $(int64_t _dim), $(bool _keepdim))); }|]

layer_norm_tlttdb :: Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> CDouble -> CBool -> IO (Ptr Tensor)
layer_norm_tlttdb _input _normalized_shape _weight _bias _eps _cudnn_enable = [C.block| at::Tensor* { return new at::Tensor(at::native::layer_norm(*$(at::Tensor* _input), *$(at::IntList* _normalized_shape), *$(at::Tensor* _weight), *$(at::Tensor* _bias), $(double _eps), $(bool _cudnn_enable))); }|]

linear_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
linear_ttt _input _weight _bias = [C.block| at::Tensor* { return new at::Tensor(at::native::linear(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias))); }|]

fbgemm_linear_int8_weight_ttttsst :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Tensor -> IO (Ptr Tensor)
fbgemm_linear_int8_weight_ttttsst _input _weight _packed _col_offsets _weight_scale _weight_zero_point _bias = [C.block| at::Tensor* { return new at::Tensor(at::native::fbgemm_linear_int8_weight(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _packed), *$(at::Tensor* _col_offsets), *$(at::Scalar* _weight_scale), *$(at::Scalar* _weight_zero_point), *$(at::Tensor* _bias))); }|]

fbgemm_linear_quantize_weight_t :: Ptr Tensor -> IO (Ptr (Tensor,Tensor,CDouble,Int64))
fbgemm_linear_quantize_weight_t _input = [C.block| std::tuple<at::Tensor,at::Tensor,double,int64_t>* { return new std::tuple<at::Tensor,at::Tensor,double,int64_t>(at::native::fbgemm_linear_quantize_weight(*$(at::Tensor* _input))); }|]

fbgemm_pack_quantized_matrix_tll :: Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
fbgemm_pack_quantized_matrix_tll _input _K _N = [C.block| at::Tensor* { return new at::Tensor(at::native::fbgemm_pack_quantized_matrix(*$(at::Tensor* _input), $(int64_t _K), $(int64_t _N))); }|]

fbgemm_is_cpu_supported_ :: IO (CBool)
fbgemm_is_cpu_supported_  = [C.block| bool { return (at::native::fbgemm_is_cpu_supported()); }|]

linspace_sslo :: Ptr Scalar -> Ptr Scalar -> Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
linspace_sslo _start _end _steps _options = [C.block| at::Tensor* { return new at::Tensor(at::native::linspace(*$(at::Scalar* _start), *$(at::Scalar* _end), $(int64_t _steps), *$(at::TensorOptions* _options))); }|]

linspace_cpu_out_tssl :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Int64 -> IO (Ptr Tensor)
linspace_cpu_out_tssl _result _start _end _steps = [C.block| at::Tensor* { return new at::Tensor(at::native::linspace_cpu_out(*$(at::Tensor* _result), *$(at::Scalar* _start), *$(at::Scalar* _end), $(int64_t _steps))); }|]

linspace_cuda_out_tssl :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Int64 -> IO (Ptr Tensor)
linspace_cuda_out_tssl _result _start _end _steps = [C.block| at::Tensor* { return new at::Tensor(at::native::linspace_cuda_out(*$(at::Tensor* _result), *$(at::Scalar* _start), *$(at::Scalar* _end), $(int64_t _steps))); }|]

log_t :: Ptr Tensor -> IO (Ptr Tensor)
log_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::log(*$(at::Tensor* _self))); }|]

_log__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_log__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log__cpu(*$(at::Tensor* _self))); }|]

_log__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_log__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log__cuda(*$(at::Tensor* _self))); }|]

_log_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_log_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_log_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_log_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

log10_t :: Ptr Tensor -> IO (Ptr Tensor)
log10_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::log10(*$(at::Tensor* _self))); }|]

_log10__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_log10__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log10__cpu(*$(at::Tensor* _self))); }|]

_log10__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_log10__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log10__cuda(*$(at::Tensor* _self))); }|]

_log10_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_log10_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log10_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_log10_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_log10_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log10_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

log1p_t :: Ptr Tensor -> IO (Ptr Tensor)
log1p_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::log1p(*$(at::Tensor* _self))); }|]

_log1p__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_log1p__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log1p__cpu(*$(at::Tensor* _self))); }|]

_log1p__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_log1p__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log1p__cuda(*$(at::Tensor* _self))); }|]

log1p_sparse__t :: Ptr Tensor -> IO (Ptr Tensor)
log1p_sparse__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::log1p_sparse_(*$(at::Tensor* _self))); }|]

_log1p_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_log1p_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log1p_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_log1p_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_log1p_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log1p_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

log1p_out_sparse_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
log1p_out_sparse_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::log1p_out_sparse(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

log2_t :: Ptr Tensor -> IO (Ptr Tensor)
log2_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::log2(*$(at::Tensor* _self))); }|]

_log2__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_log2__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log2__cpu(*$(at::Tensor* _self))); }|]

_log2__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_log2__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log2__cuda(*$(at::Tensor* _self))); }|]

_log2_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_log2_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log2_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_log2_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_log2_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_log2_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

logdet_t :: Ptr Tensor -> IO (Ptr Tensor)
logdet_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::logdet(*$(at::Tensor* _self))); }|]

logspace_sslo :: Ptr Scalar -> Ptr Scalar -> Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
logspace_sslo _start _end _steps _options = [C.block| at::Tensor* { return new at::Tensor(at::native::logspace(*$(at::Scalar* _start), *$(at::Scalar* _end), $(int64_t _steps), *$(at::TensorOptions* _options))); }|]

logspace_cpu_out_tssl :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Int64 -> IO (Ptr Tensor)
logspace_cpu_out_tssl _result _start _end _steps = [C.block| at::Tensor* { return new at::Tensor(at::native::logspace_cpu_out(*$(at::Tensor* _result), *$(at::Scalar* _start), *$(at::Scalar* _end), $(int64_t _steps))); }|]

logspace_cuda_out_tssl :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Int64 -> IO (Ptr Tensor)
logspace_cuda_out_tssl _result _start _end _steps = [C.block| at::Tensor* { return new at::Tensor(at::native::logspace_cuda_out(*$(at::Tensor* _result), *$(at::Scalar* _start), *$(at::Scalar* _end), $(int64_t _steps))); }|]

log_softmax_tls :: Ptr Tensor -> Int64 -> Ptr ScalarType -> IO (Ptr Tensor)
log_softmax_tls _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::log_softmax(*$(at::Tensor* _self), $(int64_t _dim), *$(at::ScalarType* _dtype))); }|]

log_softmax_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
log_softmax_tl _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::log_softmax(*$(at::Tensor* _self), $(int64_t _dim))); }|]

log_softmax_cpu_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
log_softmax_cpu_tlb _self _dim _half_to_float = [C.block| at::Tensor* { return new at::Tensor(at::native::log_softmax_cpu(*$(at::Tensor* _self), $(int64_t _dim), $(bool _half_to_float))); }|]

log_softmax_cuda_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
log_softmax_cuda_tlb _self _dim _half_to_float = [C.block| at::Tensor* { return new at::Tensor(at::native::log_softmax_cuda(*$(at::Tensor* _self), $(int64_t _dim), $(bool _half_to_float))); }|]

log_softmax_backward_cpu_ttlt :: Ptr Tensor -> Ptr Tensor -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
log_softmax_backward_cpu_ttlt _grad_output _output _dim _self = [C.block| at::Tensor* { return new at::Tensor(at::native::log_softmax_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _output), $(int64_t _dim), *$(at::Tensor* _self))); }|]

log_softmax_backward_cuda_ttlt :: Ptr Tensor -> Ptr Tensor -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
log_softmax_backward_cuda_ttlt _grad_output _output _dim _self = [C.block| at::Tensor* { return new at::Tensor(at::native::log_softmax_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _output), $(int64_t _dim), *$(at::Tensor* _self))); }|]

logsumexp_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
logsumexp_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::logsumexp(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

logsumexp_out_ttlb :: Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
logsumexp_out_ttlb _result _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::logsumexp_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

margin_ranking_loss_tttdl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CDouble -> Int64 -> IO (Ptr Tensor)
margin_ranking_loss_tttdl _input1 _input2 _target _margin _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::margin_ranking_loss(*$(at::Tensor* _input1), *$(at::Tensor* _input2), *$(at::Tensor* _target), $(double _margin), $(int64_t _reduction))); }|]

matmul_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
matmul_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::matmul(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

matmul_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
matmul_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::matmul_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

matrix_rank_tdb :: Ptr Tensor -> CDouble -> CBool -> IO (Ptr Tensor)
matrix_rank_tdb _self _tol _symmetric = [C.block| at::Tensor* { return new at::Tensor(at::native::matrix_rank(*$(at::Tensor* _self), $(double _tol), $(bool _symmetric))); }|]

matrix_rank_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
matrix_rank_tb _self _symmetric = [C.block| at::Tensor* { return new at::Tensor(at::native::matrix_rank(*$(at::Tensor* _self), $(bool _symmetric))); }|]

matrix_power_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
matrix_power_tl _self _n = [C.block| at::Tensor* { return new at::Tensor(at::native::matrix_power(*$(at::Tensor* _self), $(int64_t _n))); }|]

max_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
max_tlb _self _dim _keepdim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::max(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

max_out_tttlb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
max_out_tttlb _max _max_values _self _dim _keepdim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::max_out(*$(at::Tensor* _max), *$(at::Tensor* _max_values), *$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

max_values_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
max_values_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::max_values(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

max_pool1d_with_indices_tllllb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr (Tensor,Tensor))
max_pool1d_with_indices_tllllb _self _kernel_size _stride _padding _dilation _ceil_mode = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::max_pool1d_with_indices(*$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _ceil_mode))); }|]

max_pool1d_tllllb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr Tensor)
max_pool1d_tllllb _self _kernel_size _stride _padding _dilation _ceil_mode = [C.block| at::Tensor* { return new at::Tensor(at::native::max_pool1d(*$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _ceil_mode))); }|]

max_pool2d_tllllb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr Tensor)
max_pool2d_tllllb _self _kernel_size _stride _padding _dilation _ceil_mode = [C.block| at::Tensor* { return new at::Tensor(at::native::max_pool2d(*$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _ceil_mode))); }|]

max_pool3d_tllllb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr Tensor)
max_pool3d_tllllb _self _kernel_size _stride _padding _dilation _ceil_mode = [C.block| at::Tensor* { return new at::Tensor(at::native::max_pool3d(*$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _ceil_mode))); }|]

mean_ts :: Ptr Tensor -> Ptr ScalarType -> IO (Ptr Tensor)
mean_ts _self _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::mean(*$(at::Tensor* _self), *$(at::ScalarType* _dtype))); }|]

mean_t :: Ptr Tensor -> IO (Ptr Tensor)
mean_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::mean(*$(at::Tensor* _self))); }|]

mean_tlbs :: Ptr Tensor -> Ptr IntList -> CBool -> Ptr ScalarType -> IO (Ptr Tensor)
mean_tlbs _self _dim _keepdim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::mean(*$(at::Tensor* _self), *$(at::IntList* _dim), $(bool _keepdim), *$(at::ScalarType* _dtype))); }|]

mean_tlb :: Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
mean_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::mean(*$(at::Tensor* _self), *$(at::IntList* _dim), $(bool _keepdim))); }|]

mean_tls :: Ptr Tensor -> Ptr IntList -> Ptr ScalarType -> IO (Ptr Tensor)
mean_tls _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::mean(*$(at::Tensor* _self), *$(at::IntList* _dim), *$(at::ScalarType* _dtype))); }|]

mean_out_ttlbs :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> CBool -> Ptr ScalarType -> IO (Ptr Tensor)
mean_out_ttlbs _result _self _dim _keepdim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::mean_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::IntList* _dim), $(bool _keepdim), *$(at::ScalarType* _dtype))); }|]

mean_out_ttlb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
mean_out_ttlb _result _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::mean_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::IntList* _dim), $(bool _keepdim))); }|]

mean_out_ttls :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr ScalarType -> IO (Ptr Tensor)
mean_out_ttls _result _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::mean_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::IntList* _dim), *$(at::ScalarType* _dtype))); }|]

median_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
median_tlb _self _dim _keepdim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::median(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

median_out_tttlb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
median_out_tttlb _values _indices _self _dim _keepdim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::median_out(*$(at::Tensor* _values), *$(at::Tensor* _indices), *$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

min_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
min_tlb _self _dim _keepdim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::min(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

min_out_tttlb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
min_out_tttlb _min _min_indices _self _dim _keepdim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::min_out(*$(at::Tensor* _min), *$(at::Tensor* _min_indices), *$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

min_values_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
min_values_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::min_values(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

mkldnn_convolution_tttllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> IO (Ptr Tensor)
mkldnn_convolution_tttllll _self _weight _bias _padding _stride _dilation _groups = [C.block| at::Tensor* { return new at::Tensor(at::native::mkldnn_convolution(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups))); }|]

mkldnn_convolution_backward_input_lttllllb :: Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> IO (Ptr Tensor)
mkldnn_convolution_backward_input_lttllllb _self_size _grad_output _weight _padding _stride _dilation _groups _bias_defined = [C.block| at::Tensor* { return new at::Tensor(at::native::mkldnn_convolution_backward_input(*$(at::IntList* _self_size), *$(at::Tensor* _grad_output), *$(at::Tensor* _weight), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _bias_defined))); }|]

mkldnn_convolution_backward_weights_lttllllb :: Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
mkldnn_convolution_backward_weights_lttllllb _weight_size _grad_output _self _padding _stride _dilation _groups _bias_defined = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::mkldnn_convolution_backward_weights(*$(at::IntList* _weight_size), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _bias_defined))); }|]

mkldnn_convolution_backward_tttlllla :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
mkldnn_convolution_backward_tttlllla _self _grad_output _weight _padding _stride _dilation _groups _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::mkldnn_convolution_backward(*$(at::Tensor* _self), *$(at::Tensor* _grad_output), *$(at::Tensor* _weight), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), *$(std::array<bool,3>* _output_mask))); }|]

miopen_batch_norm_tttttbdd :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CDouble -> CDouble -> IO (Ptr (Tensor,Tensor,Tensor))
miopen_batch_norm_tttttbdd _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::miopen_batch_norm(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::Tensor* _running_mean), *$(at::Tensor* _running_var), $(bool _training), $(double _exponential_average_factor), $(double _epsilon))); }|]

miopen_batch_norm_backward_tttttttd :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CDouble -> IO (Ptr (Tensor,Tensor,Tensor))
miopen_batch_norm_backward_tttttttd _input _grad_output _weight _running_mean _running_var _save_mean _save_var _epsilon = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::miopen_batch_norm_backward(*$(at::Tensor* _input), *$(at::Tensor* _grad_output), *$(at::Tensor* _weight), *$(at::Tensor* _running_mean), *$(at::Tensor* _running_var), *$(at::Tensor* _save_mean), *$(at::Tensor* _save_var), $(double _epsilon))); }|]

miopen_convolution_tttllllbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
miopen_convolution_tttllllbb _self _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = [C.block| at::Tensor* { return new at::Tensor(at::native::miopen_convolution(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic))); }|]

miopen_convolution_backward_input_lttllllbb :: Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
miopen_convolution_backward_input_lttllllbb _self_size _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic = [C.block| at::Tensor* { return new at::Tensor(at::native::miopen_convolution_backward_input(*$(at::IntList* _self_size), *$(at::Tensor* _grad_output), *$(at::Tensor* _weight), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic))); }|]

miopen_convolution_backward_tttllllbba :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
miopen_convolution_backward_tttllllbba _self _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::miopen_convolution_backward(*$(at::Tensor* _self), *$(at::Tensor* _grad_output), *$(at::Tensor* _weight), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic), *$(std::array<bool,3>* _output_mask))); }|]

miopen_convolution_backward_bias_t :: Ptr Tensor -> IO (Ptr Tensor)
miopen_convolution_backward_bias_t _grad_output = [C.block| at::Tensor* { return new at::Tensor(at::native::miopen_convolution_backward_bias(*$(at::Tensor* _grad_output))); }|]

miopen_convolution_backward_weight_lttllllbb :: Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
miopen_convolution_backward_weight_lttllllbb _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic = [C.block| at::Tensor* { return new at::Tensor(at::native::miopen_convolution_backward_weight(*$(at::IntList* _weight_size), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic))); }|]

miopen_convolution_transpose_tttlllllbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
miopen_convolution_transpose_tttlllllbb _self _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic = [C.block| at::Tensor* { return new at::Tensor(at::native::miopen_convolution_transpose(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic))); }|]

miopen_convolution_transpose_backward_tttlllllbba :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
miopen_convolution_transpose_backward_tttlllllbba _self _grad_output _weight _padding _output_padding _stride _dilation _groups _benchmark _deterministic _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::miopen_convolution_transpose_backward(*$(at::Tensor* _self), *$(at::Tensor* _grad_output), *$(at::Tensor* _weight), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic), *$(std::array<bool,3>* _output_mask))); }|]

miopen_convolution_transpose_backward_input_ttllllbb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
miopen_convolution_transpose_backward_input_ttllllbb _grad_output _weight _padding _stride _dilation _groups _benchmark _deterministic = [C.block| at::Tensor* { return new at::Tensor(at::native::miopen_convolution_transpose_backward_input(*$(at::Tensor* _grad_output), *$(at::Tensor* _weight), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic))); }|]

miopen_convolution_transpose_backward_weight_lttllllbb :: Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
miopen_convolution_transpose_backward_weight_lttllllbb _weight_size _grad_output _self _padding _stride _dilation _groups _benchmark _deterministic = [C.block| at::Tensor* { return new at::Tensor(at::native::miopen_convolution_transpose_backward_weight(*$(at::IntList* _weight_size), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding), *$(at::IntList* _stride), *$(at::IntList* _dilation), $(int64_t _groups), $(bool _benchmark), $(bool _deterministic))); }|]

mm_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
mm_tt _self _mat2 = [C.block| at::Tensor* { return new at::Tensor(at::native::mm(*$(at::Tensor* _self), *$(at::Tensor* _mat2))); }|]

mm_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
mm_out_ttt _result _self _mat2 = [C.block| at::Tensor* { return new at::Tensor(at::native::mm_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _mat2))); }|]

_sparse_mm_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_sparse_mm_tt _sparse _dense = [C.block| at::Tensor* { return new at::Tensor(at::native::_sparse_mm(*$(at::Tensor* _sparse), *$(at::Tensor* _dense))); }|]

mode_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
mode_tlb _self _dim _keepdim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::mode(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

mode_out_tttlb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
mode_out_tttlb _values _indices _self _dim _keepdim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::mode_out(*$(at::Tensor* _values), *$(at::Tensor* _indices), *$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

mul_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
mul_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::mul(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

mul__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
mul__tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::mul_(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

mul_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
mul_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::mul_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

mul_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
mul_ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::mul(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

mul__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
mul__ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::mul_(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

mv_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
mv_tt _self _vec = [C.block| at::Tensor* { return new at::Tensor(at::native::mv(*$(at::Tensor* _self), *$(at::Tensor* _vec))); }|]

mv_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
mv_out_ttt _result _self _vec = [C.block| at::Tensor* { return new at::Tensor(at::native::mv_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _vec))); }|]

mvlgamma_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
mvlgamma_tl _self _p = [C.block| at::Tensor* { return new at::Tensor(at::native::mvlgamma(*$(at::Tensor* _self), $(int64_t _p))); }|]

mvlgamma__tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
mvlgamma__tl _self _p = [C.block| at::Tensor* { return new at::Tensor(at::native::mvlgamma_(*$(at::Tensor* _self), $(int64_t _p))); }|]

narrow_copy_dense_tlll :: Ptr Tensor -> Int64 -> Int64 -> Int64 -> IO (Ptr Tensor)
narrow_copy_dense_tlll _self _dim _start _length = [C.block| at::Tensor* { return new at::Tensor(at::native::narrow_copy_dense(*$(at::Tensor* _self), $(int64_t _dim), $(int64_t _start), $(int64_t _length))); }|]

narrow_copy_sparse_tlll :: Ptr Tensor -> Int64 -> Int64 -> Int64 -> IO (Ptr Tensor)
narrow_copy_sparse_tlll _self _dim _start _length = [C.block| at::Tensor* { return new at::Tensor(at::native::narrow_copy_sparse(*$(at::Tensor* _self), $(int64_t _dim), $(int64_t _start), $(int64_t _length))); }|]

narrow_tlll :: Ptr Tensor -> Int64 -> Int64 -> Int64 -> IO (Ptr Tensor)
narrow_tlll _self _dim _start _length = [C.block| at::Tensor* { return new at::Tensor(at::native::narrow(*$(at::Tensor* _self), $(int64_t _dim), $(int64_t _start), $(int64_t _length))); }|]

batch_norm_cpu_tttttbdd :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CDouble -> CDouble -> IO (Ptr (Tensor,Tensor,Tensor))
batch_norm_cpu_tttttbdd _input _weight _bias _running_mean _running_var _training _momentum _eps = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::batch_norm_cpu(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::Tensor* _running_mean), *$(at::Tensor* _running_var), $(bool _training), $(double _momentum), $(double _eps))); }|]

batch_norm_cuda_tttttbdd :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CDouble -> CDouble -> IO (Ptr (Tensor,Tensor,Tensor))
batch_norm_cuda_tttttbdd _input _weight _bias _running_mean _running_var _training _momentum _eps = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::batch_norm_cuda(*$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _bias), *$(at::Tensor* _running_mean), *$(at::Tensor* _running_var), $(bool _training), $(double _momentum), $(double _eps))); }|]

batch_norm_backward_cpu_tttttttbda :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CDouble -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
batch_norm_backward_cpu_tttttttbda _grad_out _input _weight _running_mean _running_var _save_mean _save_invstd _train _eps _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::batch_norm_backward_cpu(*$(at::Tensor* _grad_out), *$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _running_mean), *$(at::Tensor* _running_var), *$(at::Tensor* _save_mean), *$(at::Tensor* _save_invstd), $(bool _train), $(double _eps), *$(std::array<bool,3>* _output_mask))); }|]

batch_norm_backward_cuda_tttttttbda :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CDouble -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
batch_norm_backward_cuda_tttttttbda _grad_out _input _weight _running_mean _running_var _save_mean _save_invstd _train _eps _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::batch_norm_backward_cuda(*$(at::Tensor* _grad_out), *$(at::Tensor* _input), *$(at::Tensor* _weight), *$(at::Tensor* _running_mean), *$(at::Tensor* _running_var), *$(at::Tensor* _save_mean), *$(at::Tensor* _save_invstd), $(bool _train), $(double _eps), *$(std::array<bool,3>* _output_mask))); }|]

ones_lo :: Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
ones_lo _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::ones(*$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

ones_out_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
ones_out_tl _result _size = [C.block| at::Tensor* { return new at::Tensor(at::native::ones_out(*$(at::Tensor* _result), *$(at::IntList* _size))); }|]

ones_like_t :: Ptr Tensor -> IO (Ptr Tensor)
ones_like_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::ones_like(*$(at::Tensor* _self))); }|]

ones_like_to :: Ptr Tensor -> Ptr TensorOptions -> IO (Ptr Tensor)
ones_like_to _self _options = [C.block| at::Tensor* { return new at::Tensor(at::native::ones_like(*$(at::Tensor* _self), *$(at::TensorOptions* _options))); }|]

pairwise_distance_ttddb :: Ptr Tensor -> Ptr Tensor -> CDouble -> CDouble -> CBool -> IO (Ptr Tensor)
pairwise_distance_ttddb _x1 _x2 _p _eps _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::pairwise_distance(*$(at::Tensor* _x1), *$(at::Tensor* _x2), $(double _p), $(double _eps), $(bool _keepdim))); }|]

pdist_td :: Ptr Tensor -> CDouble -> IO (Ptr Tensor)
pdist_td _self _p = [C.block| at::Tensor* { return new at::Tensor(at::native::pdist(*$(at::Tensor* _self), $(double _p))); }|]

_pdist_forward_td :: Ptr Tensor -> CDouble -> IO (Ptr Tensor)
_pdist_forward_td _self _p = [C.block| at::Tensor* { return new at::Tensor(at::native::_pdist_forward(*$(at::Tensor* _self), $(double _p))); }|]

_pdist_backward_ttdt :: Ptr Tensor -> Ptr Tensor -> CDouble -> Ptr Tensor -> IO (Ptr Tensor)
_pdist_backward_ttdt _grad _self _p _pdist = [C.block| at::Tensor* { return new at::Tensor(at::native::_pdist_backward(*$(at::Tensor* _grad), *$(at::Tensor* _self), $(double _p), *$(at::Tensor* _pdist))); }|]

cosine_similarity_ttld :: Ptr Tensor -> Ptr Tensor -> Int64 -> CDouble -> IO (Ptr Tensor)
cosine_similarity_ttld _x1 _x2 _dim _eps = [C.block| at::Tensor* { return new at::Tensor(at::native::cosine_similarity(*$(at::Tensor* _x1), *$(at::Tensor* _x2), $(int64_t _dim), $(double _eps))); }|]

permute_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
permute_tl _self _dims = [C.block| at::Tensor* { return new at::Tensor(at::native::permute(*$(at::Tensor* _self), *$(at::IntList* _dims))); }|]

pixel_shuffle_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
pixel_shuffle_tl _self _upscale_factor = [C.block| at::Tensor* { return new at::Tensor(at::native::pixel_shuffle(*$(at::Tensor* _self), $(int64_t _upscale_factor))); }|]

pin_memory_t :: Ptr Tensor -> IO (Ptr Tensor)
pin_memory_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::pin_memory(*$(at::Tensor* _self))); }|]

pinverse_td :: Ptr Tensor -> CDouble -> IO (Ptr Tensor)
pinverse_td _self _rcond = [C.block| at::Tensor* { return new at::Tensor(at::native::pinverse(*$(at::Tensor* _self), $(double _rcond))); }|]

scalar_tensor_so :: Ptr Scalar -> Ptr TensorOptions -> IO (Ptr Tensor)
scalar_tensor_so _s _options = [C.block| at::Tensor* { return new at::Tensor(at::native::scalar_tensor(*$(at::Scalar* _s), *$(at::TensorOptions* _options))); }|]

rand_lo :: Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
rand_lo _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::rand(*$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

rand_lpo :: Ptr IntList -> Ptr Generator -> Ptr TensorOptions -> IO (Ptr Tensor)
rand_lpo _size _generator _options = [C.block| at::Tensor* { return new at::Tensor(at::native::rand(*$(at::IntList* _size), $(at::Generator * _generator), *$(at::TensorOptions* _options))); }|]

rand_out_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
rand_out_tl _result _size = [C.block| at::Tensor* { return new at::Tensor(at::native::rand_out(*$(at::Tensor* _result), *$(at::IntList* _size))); }|]

rand_out_tlp :: Ptr Tensor -> Ptr IntList -> Ptr Generator -> IO (Ptr Tensor)
rand_out_tlp _result _size _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::rand_out(*$(at::Tensor* _result), *$(at::IntList* _size), $(at::Generator * _generator))); }|]

rand_like_t :: Ptr Tensor -> IO (Ptr Tensor)
rand_like_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::rand_like(*$(at::Tensor* _self))); }|]

rand_like_to :: Ptr Tensor -> Ptr TensorOptions -> IO (Ptr Tensor)
rand_like_to _self _options = [C.block| at::Tensor* { return new at::Tensor(at::native::rand_like(*$(at::Tensor* _self), *$(at::TensorOptions* _options))); }|]

randint_llo :: Int64 -> Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
randint_llo _high _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::randint($(int64_t _high), *$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

randint_llpo :: Int64 -> Ptr IntList -> Ptr Generator -> Ptr TensorOptions -> IO (Ptr Tensor)
randint_llpo _high _size _generator _options = [C.block| at::Tensor* { return new at::Tensor(at::native::randint($(int64_t _high), *$(at::IntList* _size), $(at::Generator * _generator), *$(at::TensorOptions* _options))); }|]

randint_lllo :: Int64 -> Int64 -> Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
randint_lllo _low _high _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::randint($(int64_t _low), $(int64_t _high), *$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

randint_lllpo :: Int64 -> Int64 -> Ptr IntList -> Ptr Generator -> Ptr TensorOptions -> IO (Ptr Tensor)
randint_lllpo _low _high _size _generator _options = [C.block| at::Tensor* { return new at::Tensor(at::native::randint($(int64_t _low), $(int64_t _high), *$(at::IntList* _size), $(at::Generator * _generator), *$(at::TensorOptions* _options))); }|]

randint_out_tll :: Ptr Tensor -> Int64 -> Ptr IntList -> IO (Ptr Tensor)
randint_out_tll _result _high _size = [C.block| at::Tensor* { return new at::Tensor(at::native::randint_out(*$(at::Tensor* _result), $(int64_t _high), *$(at::IntList* _size))); }|]

randint_out_tllp :: Ptr Tensor -> Int64 -> Ptr IntList -> Ptr Generator -> IO (Ptr Tensor)
randint_out_tllp _result _high _size _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::randint_out(*$(at::Tensor* _result), $(int64_t _high), *$(at::IntList* _size), $(at::Generator * _generator))); }|]

randint_out_tlll :: Ptr Tensor -> Int64 -> Int64 -> Ptr IntList -> IO (Ptr Tensor)
randint_out_tlll _result _low _high _size = [C.block| at::Tensor* { return new at::Tensor(at::native::randint_out(*$(at::Tensor* _result), $(int64_t _low), $(int64_t _high), *$(at::IntList* _size))); }|]

randint_out_tlllp :: Ptr Tensor -> Int64 -> Int64 -> Ptr IntList -> Ptr Generator -> IO (Ptr Tensor)
randint_out_tlllp _result _low _high _size _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::randint_out(*$(at::Tensor* _result), $(int64_t _low), $(int64_t _high), *$(at::IntList* _size), $(at::Generator * _generator))); }|]

randint_like_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
randint_like_tl _self _high = [C.block| at::Tensor* { return new at::Tensor(at::native::randint_like(*$(at::Tensor* _self), $(int64_t _high))); }|]

randint_like_tll :: Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
randint_like_tll _self _low _high = [C.block| at::Tensor* { return new at::Tensor(at::native::randint_like(*$(at::Tensor* _self), $(int64_t _low), $(int64_t _high))); }|]

randint_like_tlo :: Ptr Tensor -> Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
randint_like_tlo _self _high _options = [C.block| at::Tensor* { return new at::Tensor(at::native::randint_like(*$(at::Tensor* _self), $(int64_t _high), *$(at::TensorOptions* _options))); }|]

randint_like_tllo :: Ptr Tensor -> Int64 -> Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
randint_like_tllo _self _low _high _options = [C.block| at::Tensor* { return new at::Tensor(at::native::randint_like(*$(at::Tensor* _self), $(int64_t _low), $(int64_t _high), *$(at::TensorOptions* _options))); }|]

randn_lo :: Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
randn_lo _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::randn(*$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

randn_lpo :: Ptr IntList -> Ptr Generator -> Ptr TensorOptions -> IO (Ptr Tensor)
randn_lpo _size _generator _options = [C.block| at::Tensor* { return new at::Tensor(at::native::randn(*$(at::IntList* _size), $(at::Generator * _generator), *$(at::TensorOptions* _options))); }|]

randn_out_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
randn_out_tl _result _size = [C.block| at::Tensor* { return new at::Tensor(at::native::randn_out(*$(at::Tensor* _result), *$(at::IntList* _size))); }|]

randn_out_tlp :: Ptr Tensor -> Ptr IntList -> Ptr Generator -> IO (Ptr Tensor)
randn_out_tlp _result _size _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::randn_out(*$(at::Tensor* _result), *$(at::IntList* _size), $(at::Generator * _generator))); }|]

randn_like_t :: Ptr Tensor -> IO (Ptr Tensor)
randn_like_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::randn_like(*$(at::Tensor* _self))); }|]

randn_like_to :: Ptr Tensor -> Ptr TensorOptions -> IO (Ptr Tensor)
randn_like_to _self _options = [C.block| at::Tensor* { return new at::Tensor(at::native::randn_like(*$(at::Tensor* _self), *$(at::TensorOptions* _options))); }|]

randperm_lo :: Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
randperm_lo _n _options = [C.block| at::Tensor* { return new at::Tensor(at::native::randperm($(int64_t _n), *$(at::TensorOptions* _options))); }|]

randperm_lpo :: Int64 -> Ptr Generator -> Ptr TensorOptions -> IO (Ptr Tensor)
randperm_lpo _n _generator _options = [C.block| at::Tensor* { return new at::Tensor(at::native::randperm($(int64_t _n), $(at::Generator * _generator), *$(at::TensorOptions* _options))); }|]

randperm_out_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
randperm_out_tl _result _n = [C.block| at::Tensor* { return new at::Tensor(at::native::randperm_out(*$(at::Tensor* _result), $(int64_t _n))); }|]

randperm_out_cpu_tlp :: Ptr Tensor -> Int64 -> Ptr Generator -> IO (Ptr Tensor)
randperm_out_cpu_tlp _result _n _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::randperm_out_cpu(*$(at::Tensor* _result), $(int64_t _n), $(at::Generator * _generator))); }|]

randperm_out_cuda_tlp :: Ptr Tensor -> Int64 -> Ptr Generator -> IO (Ptr Tensor)
randperm_out_cuda_tlp _result _n _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::randperm_out_cuda(*$(at::Tensor* _result), $(int64_t _n), $(at::Generator * _generator))); }|]

range_ssso :: Ptr Scalar -> Ptr Scalar -> Ptr Scalar -> Ptr TensorOptions -> IO (Ptr Tensor)
range_ssso _start _end _step _options = [C.block| at::Tensor* { return new at::Tensor(at::native::range(*$(at::Scalar* _start), *$(at::Scalar* _end), *$(at::Scalar* _step), *$(at::TensorOptions* _options))); }|]

range_sso :: Ptr Scalar -> Ptr Scalar -> Ptr TensorOptions -> IO (Ptr Tensor)
range_sso _start _end _options = [C.block| at::Tensor* { return new at::Tensor(at::native::range(*$(at::Scalar* _start), *$(at::Scalar* _end), *$(at::TensorOptions* _options))); }|]

range_cpu_out_tsss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
range_cpu_out_tsss _result _start _end _step = [C.block| at::Tensor* { return new at::Tensor(at::native::range_cpu_out(*$(at::Tensor* _result), *$(at::Scalar* _start), *$(at::Scalar* _end), *$(at::Scalar* _step))); }|]

range_cuda_out_tsss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
range_cuda_out_tsss _result _start _end _step = [C.block| at::Tensor* { return new at::Tensor(at::native::range_cuda_out(*$(at::Tensor* _result), *$(at::Scalar* _start), *$(at::Scalar* _end), *$(at::Scalar* _step))); }|]

repeat_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
repeat_tl _self _repeats = [C.block| at::Tensor* { return new at::Tensor(at::native::repeat(*$(at::Tensor* _self), *$(at::IntList* _repeats))); }|]

reshape_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reshape_tl _self _shape = [C.block| at::Tensor* { return new at::Tensor(at::native::reshape(*$(at::Tensor* _self), *$(at::IntList* _shape))); }|]

reshape_as_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
reshape_as_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::reshape_as(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

roiPooling2d_forward_cpu_ttlld :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> CDouble -> IO (Ptr (Tensor,Tensor))
roiPooling2d_forward_cpu_ttlld _input _rois _pooledHeight _pooledWidth _spatialScale = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::RoiPooling2d_forward_cpu(*$(at::Tensor* _input), *$(at::Tensor* _rois), $(int64_t _pooledHeight), $(int64_t _pooledWidth), $(double _spatialScale))); }|]

roiPooling2d_forward_cuda_ttlld :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> CDouble -> IO (Ptr (Tensor,Tensor))
roiPooling2d_forward_cuda_ttlld _input _rois _pooledHeight _pooledWidth _spatialScale = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::RoiPooling2d_forward_cuda(*$(at::Tensor* _input), *$(at::Tensor* _rois), $(int64_t _pooledHeight), $(int64_t _pooledWidth), $(double _spatialScale))); }|]

roiPooling2d_backward_cpu_ttlldtt :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> CDouble -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
roiPooling2d_backward_cpu_ttlldtt _input _rois _pooledHeight _pooledWidth _spatialScale _gradOutput _argmaxes = [C.block| at::Tensor* { return new at::Tensor(at::native::RoiPooling2d_backward_cpu(*$(at::Tensor* _input), *$(at::Tensor* _rois), $(int64_t _pooledHeight), $(int64_t _pooledWidth), $(double _spatialScale), *$(at::Tensor* _gradOutput), *$(at::Tensor* _argmaxes))); }|]

roiPooling2d_backward_cuda_ttlldtt :: Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> CDouble -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
roiPooling2d_backward_cuda_ttlldtt _input _rois _pooledHeight _pooledWidth _spatialScale _gradOutput _argmaxes = [C.block| at::Tensor* { return new at::Tensor(at::native::RoiPooling2d_backward_cuda(*$(at::Tensor* _input), *$(at::Tensor* _rois), $(int64_t _pooledHeight), $(int64_t _pooledWidth), $(double _spatialScale), *$(at::Tensor* _gradOutput), *$(at::Tensor* _argmaxes))); }|]

round_t :: Ptr Tensor -> IO (Ptr Tensor)
round_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::round(*$(at::Tensor* _self))); }|]

_round__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_round__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_round__cpu(*$(at::Tensor* _self))); }|]

_round__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_round__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_round__cuda(*$(at::Tensor* _self))); }|]

_round_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_round_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_round_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_round_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_round_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_round_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

rrelu_tssbp :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> CBool -> Ptr Generator -> IO (Ptr Tensor)
rrelu_tssbp _self _lower _upper _training _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::rrelu(*$(at::Tensor* _self), *$(at::Scalar* _lower), *$(at::Scalar* _upper), $(bool _training), $(at::Generator * _generator))); }|]

rrelu__tssbp :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> CBool -> Ptr Generator -> IO (Ptr Tensor)
rrelu__tssbp _self _lower _upper _training _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::rrelu_(*$(at::Tensor* _self), *$(at::Scalar* _lower), *$(at::Scalar* _upper), $(bool _training), $(at::Generator * _generator))); }|]

relu_t :: Ptr Tensor -> IO (Ptr Tensor)
relu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::relu(*$(at::Tensor* _self))); }|]

relu__t :: Ptr Tensor -> IO (Ptr Tensor)
relu__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::relu_(*$(at::Tensor* _self))); }|]

prelu_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
prelu_cpu_tt _self _weight = [C.block| at::Tensor* { return new at::Tensor(at::native::prelu_cpu(*$(at::Tensor* _self), *$(at::Tensor* _weight))); }|]

prelu_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
prelu_cuda_tt _self _weight = [C.block| at::Tensor* { return new at::Tensor(at::native::prelu_cuda(*$(at::Tensor* _self), *$(at::Tensor* _weight))); }|]

prelu_backward_cpu_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
prelu_backward_cpu_ttt _grad_output _self _weight = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::prelu_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight))); }|]

prelu_backward_cuda_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
prelu_backward_cuda_ttt _grad_output _self _weight = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::prelu_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight))); }|]

hardshrink_cpu_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
hardshrink_cpu_ts _self _lambd = [C.block| at::Tensor* { return new at::Tensor(at::native::hardshrink_cpu(*$(at::Tensor* _self), *$(at::Scalar* _lambd))); }|]

hardshrink_cuda_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
hardshrink_cuda_ts _self _lambd = [C.block| at::Tensor* { return new at::Tensor(at::native::hardshrink_cuda(*$(at::Tensor* _self), *$(at::Scalar* _lambd))); }|]

hardshrink_backward_cpu_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
hardshrink_backward_cpu_tts _grad_out _self _lambd = [C.block| at::Tensor* { return new at::Tensor(at::native::hardshrink_backward_cpu(*$(at::Tensor* _grad_out), *$(at::Tensor* _self), *$(at::Scalar* _lambd))); }|]

hardshrink_backward_cuda_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
hardshrink_backward_cuda_tts _grad_out _self _lambd = [C.block| at::Tensor* { return new at::Tensor(at::native::hardshrink_backward_cuda(*$(at::Tensor* _grad_out), *$(at::Tensor* _self), *$(at::Scalar* _lambd))); }|]

rsqrt_t :: Ptr Tensor -> IO (Ptr Tensor)
rsqrt_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::rsqrt(*$(at::Tensor* _self))); }|]

_rsqrt__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_rsqrt__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_rsqrt__cpu(*$(at::Tensor* _self))); }|]

_rsqrt__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_rsqrt__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_rsqrt__cuda(*$(at::Tensor* _self))); }|]

_rsqrt_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_rsqrt_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_rsqrt_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_rsqrt_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_rsqrt_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_rsqrt_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

select_tll :: Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
select_tll _self _dim _index = [C.block| at::Tensor* { return new at::Tensor(at::native::select(*$(at::Tensor* _self), $(int64_t _dim), $(int64_t _index))); }|]

selu_t :: Ptr Tensor -> IO (Ptr Tensor)
selu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::selu(*$(at::Tensor* _self))); }|]

selu__t :: Ptr Tensor -> IO (Ptr Tensor)
selu__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::selu_(*$(at::Tensor* _self))); }|]

celu_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
celu_ts _self _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::celu(*$(at::Tensor* _self), *$(at::Scalar* _alpha))); }|]

celu__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
celu__ts _self _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::celu_(*$(at::Tensor* _self), *$(at::Scalar* _alpha))); }|]

sigmoid_t :: Ptr Tensor -> IO (Ptr Tensor)
sigmoid_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::sigmoid(*$(at::Tensor* _self))); }|]

_sigmoid__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_sigmoid__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sigmoid__cpu(*$(at::Tensor* _self))); }|]

_sigmoid__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_sigmoid__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sigmoid__cuda(*$(at::Tensor* _self))); }|]

_sigmoid_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_sigmoid_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sigmoid_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_sigmoid_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_sigmoid_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sigmoid_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

sin_t :: Ptr Tensor -> IO (Ptr Tensor)
sin_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::sin(*$(at::Tensor* _self))); }|]

_sin__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_sin__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sin__cpu(*$(at::Tensor* _self))); }|]

_sin__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_sin__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sin__cuda(*$(at::Tensor* _self))); }|]

_sin_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_sin_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sin_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_sin_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_sin_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sin_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

sinh_t :: Ptr Tensor -> IO (Ptr Tensor)
sinh_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::sinh(*$(at::Tensor* _self))); }|]

_sinh__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_sinh__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sinh__cpu(*$(at::Tensor* _self))); }|]

_sinh__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_sinh__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sinh__cuda(*$(at::Tensor* _self))); }|]

_sinh_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_sinh_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sinh_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_sinh_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_sinh_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sinh_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

detach_t :: Ptr Tensor -> IO (Ptr Tensor)
detach_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::detach(*$(at::Tensor* _self))); }|]

detach__t :: Ptr Tensor -> IO (Ptr Tensor)
detach__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::detach_(*$(at::Tensor* _self))); }|]

size_tl :: Ptr Tensor -> Int64 -> IO (Int64)
size_tl _self _dim = [C.block| int64_t { return (at::native::size(*$(at::Tensor* _self), $(int64_t _dim))); }|]

slice_tllll :: Ptr Tensor -> Int64 -> Int64 -> Int64 -> Int64 -> IO (Ptr Tensor)
slice_tllll _self _dim _start _end _step = [C.block| at::Tensor* { return new at::Tensor(at::native::slice(*$(at::Tensor* _self), $(int64_t _dim), $(int64_t _start), $(int64_t _end), $(int64_t _step))); }|]

slogdet_t :: Ptr Tensor -> IO (Ptr (Tensor,Tensor))
slogdet_t _self = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::slogdet(*$(at::Tensor* _self))); }|]

smm_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
smm_tt _self _mat2 = [C.block| at::Tensor* { return new at::Tensor(at::native::smm(*$(at::Tensor* _self), *$(at::Tensor* _mat2))); }|]

softmax_tls :: Ptr Tensor -> Int64 -> Ptr ScalarType -> IO (Ptr Tensor)
softmax_tls _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::softmax(*$(at::Tensor* _self), $(int64_t _dim), *$(at::ScalarType* _dtype))); }|]

softmax_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
softmax_tl _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::softmax(*$(at::Tensor* _self), $(int64_t _dim))); }|]

softmax_cpu_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
softmax_cpu_tlb _self _dim _half_to_float = [C.block| at::Tensor* { return new at::Tensor(at::native::softmax_cpu(*$(at::Tensor* _self), $(int64_t _dim), $(bool _half_to_float))); }|]

softmax_cuda_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
softmax_cuda_tlb _self _dim _half_to_float = [C.block| at::Tensor* { return new at::Tensor(at::native::softmax_cuda(*$(at::Tensor* _self), $(int64_t _dim), $(bool _half_to_float))); }|]

softmax_backward_cpu_ttlt :: Ptr Tensor -> Ptr Tensor -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
softmax_backward_cpu_ttlt _grad_output _output _dim _self = [C.block| at::Tensor* { return new at::Tensor(at::native::softmax_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _output), $(int64_t _dim), *$(at::Tensor* _self))); }|]

softmax_backward_cuda_ttlt :: Ptr Tensor -> Ptr Tensor -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
softmax_backward_cuda_ttlt _grad_output _output _dim _self = [C.block| at::Tensor* { return new at::Tensor(at::native::softmax_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _output), $(int64_t _dim), *$(at::Tensor* _self))); }|]

add_out_sparse_cpu_ttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
add_out_sparse_cpu_ttts _result _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::add_out_sparse_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other), *$(at::Scalar* _alpha))); }|]

add_out_sparse_cuda_ttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
add_out_sparse_cuda_ttts _result _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::add_out_sparse_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other), *$(at::Scalar* _alpha))); }|]

add_out_dense_sparse_cpu_ttrs :: Ptr Tensor -> Ptr Tensor -> Ptr SparseTensorRef -> Ptr Scalar -> IO (Ptr Tensor)
add_out_dense_sparse_cpu_ttrs _result _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::add_out_dense_sparse_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::SparseTensorRef* _other), *$(at::Scalar* _alpha))); }|]

add_out_dense_sparse_cuda_ttrs :: Ptr Tensor -> Ptr Tensor -> Ptr SparseTensorRef -> Ptr Scalar -> IO (Ptr Tensor)
add_out_dense_sparse_cuda_ttrs _result _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::add_out_dense_sparse_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::SparseTensorRef* _other), *$(at::Scalar* _alpha))); }|]

div_out_sparse_zerodim_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
div_out_sparse_zerodim_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::div_out_sparse_zerodim(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

div_out_sparse_scalar_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
div_out_sparse_scalar_tts _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::div_out_sparse_scalar(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

mul_out_sparse_cpu_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
mul_out_sparse_cpu_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::mul_out_sparse_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

mul_out_sparse_cuda_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
mul_out_sparse_cuda_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::mul_out_sparse_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

mul_out_sparse_zerodim_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
mul_out_sparse_zerodim_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::mul_out_sparse_zerodim(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

mul_out_sparse_scalar_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
mul_out_sparse_scalar_tts _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::mul_out_sparse_scalar(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

split_tll :: Ptr Tensor -> Int64 -> Int64 -> IO (Ptr TensorList)
split_tll _self _split_size _dim = [C.block| at::TensorList* { return new at::TensorList(at::native::split(*$(at::Tensor* _self), $(int64_t _split_size), $(int64_t _dim))); }|]

split_with_sizes_tll :: Ptr Tensor -> Ptr IntList -> Int64 -> IO (Ptr TensorList)
split_with_sizes_tll _self _split_sizes _dim = [C.block| at::TensorList* { return new at::TensorList(at::native::split_with_sizes(*$(at::Tensor* _self), *$(at::IntList* _split_sizes), $(int64_t _dim))); }|]

squeeze_t :: Ptr Tensor -> IO (Ptr Tensor)
squeeze_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::squeeze(*$(at::Tensor* _self))); }|]

squeeze_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
squeeze_tl _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::squeeze(*$(at::Tensor* _self), $(int64_t _dim))); }|]

squeeze__t :: Ptr Tensor -> IO (Ptr Tensor)
squeeze__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::squeeze_(*$(at::Tensor* _self))); }|]

squeeze__tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
squeeze__tl _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::squeeze_(*$(at::Tensor* _self), $(int64_t _dim))); }|]

sspaddmm_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
sspaddmm_tttss _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::sspaddmm(*$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

_sspaddmm_out_only_sparse_ttttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
_sspaddmm_out_only_sparse_ttttss _result _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::_sspaddmm_out_only_sparse(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

_sspaddmm_out_only_sparse_cuda_ttttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
_sspaddmm_out_only_sparse_cuda_ttttss _result _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::_sspaddmm_out_only_sparse_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

_sspaddmm_out_cpu_ttttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
_sspaddmm_out_cpu_ttttss _result _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::_sspaddmm_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

_sspaddmm_out_cuda_ttttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
_sspaddmm_out_cuda_ttttss _result _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::_sspaddmm_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

stack_ll :: Ptr TensorList -> Int64 -> IO (Ptr Tensor)
stack_ll _tensors _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::stack(*$(at::TensorList* _tensors), $(int64_t _dim))); }|]

stack_out_tll :: Ptr Tensor -> Ptr TensorList -> Int64 -> IO (Ptr Tensor)
stack_out_tll _result _tensors _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::stack_out(*$(at::Tensor* _result), *$(at::TensorList* _tensors), $(int64_t _dim))); }|]

stft_tllltbb :: Ptr Tensor -> Int64 -> Int64 -> Int64 -> Ptr Tensor -> CBool -> CBool -> IO (Ptr Tensor)
stft_tllltbb _self _n_fft _hop_length _win_length _window _normalized _onesided = [C.block| at::Tensor* { return new at::Tensor(at::native::stft(*$(at::Tensor* _self), $(int64_t _n_fft), $(int64_t _hop_length), $(int64_t _win_length), *$(at::Tensor* _window), $(bool _normalized), $(bool _onesided))); }|]

stride_tl :: Ptr Tensor -> Int64 -> IO (Int64)
stride_tl _self _dim = [C.block| int64_t { return (at::native::stride(*$(at::Tensor* _self), $(int64_t _dim))); }|]

sum_ts :: Ptr Tensor -> Ptr ScalarType -> IO (Ptr Tensor)
sum_ts _self _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::sum(*$(at::Tensor* _self), *$(at::ScalarType* _dtype))); }|]

sum_t :: Ptr Tensor -> IO (Ptr Tensor)
sum_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::sum(*$(at::Tensor* _self))); }|]

sum_tlbs :: Ptr Tensor -> Ptr IntList -> CBool -> Ptr ScalarType -> IO (Ptr Tensor)
sum_tlbs _self _dim _keepdim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::sum(*$(at::Tensor* _self), *$(at::IntList* _dim), $(bool _keepdim), *$(at::ScalarType* _dtype))); }|]

sum_tlb :: Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
sum_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::sum(*$(at::Tensor* _self), *$(at::IntList* _dim), $(bool _keepdim))); }|]

sum_tls :: Ptr Tensor -> Ptr IntList -> Ptr ScalarType -> IO (Ptr Tensor)
sum_tls _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::sum(*$(at::Tensor* _self), *$(at::IntList* _dim), *$(at::ScalarType* _dtype))); }|]

sum_out_ttlbs :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> CBool -> Ptr ScalarType -> IO (Ptr Tensor)
sum_out_ttlbs _result _self _dim _keepdim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::sum_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::IntList* _dim), $(bool _keepdim), *$(at::ScalarType* _dtype))); }|]

sum_out_ttlb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
sum_out_ttlb _result _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::sum_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::IntList* _dim), $(bool _keepdim))); }|]

sum_out_ttls :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr ScalarType -> IO (Ptr Tensor)
sum_out_ttls _result _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::sum_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::IntList* _dim), *$(at::ScalarType* _dtype))); }|]

sum_to_size_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
sum_to_size_tl _self _size = [C.block| at::Tensor* { return new at::Tensor(at::native::sum_to_size(*$(at::Tensor* _self), *$(at::IntList* _size))); }|]

sqrt_t :: Ptr Tensor -> IO (Ptr Tensor)
sqrt_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::sqrt(*$(at::Tensor* _self))); }|]

_sqrt__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_sqrt__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sqrt__cpu(*$(at::Tensor* _self))); }|]

_sqrt__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_sqrt__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sqrt__cuda(*$(at::Tensor* _self))); }|]

_sqrt_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_sqrt_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sqrt_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_sqrt_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_sqrt_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sqrt_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

std_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
std_tb _self _unbiased = [C.block| at::Tensor* { return new at::Tensor(at::native::std(*$(at::Tensor* _self), $(bool _unbiased))); }|]

std_tlbb :: Ptr Tensor -> Ptr IntList -> CBool -> CBool -> IO (Ptr Tensor)
std_tlbb _self _dim _unbiased _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::std(*$(at::Tensor* _self), *$(at::IntList* _dim), $(bool _unbiased), $(bool _keepdim))); }|]

std_out_ttlbb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> CBool -> CBool -> IO (Ptr Tensor)
std_out_ttlbb _result _self _dim _unbiased _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::std_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::IntList* _dim), $(bool _unbiased), $(bool _keepdim))); }|]

prod_ts :: Ptr Tensor -> Ptr ScalarType -> IO (Ptr Tensor)
prod_ts _self _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::prod(*$(at::Tensor* _self), *$(at::ScalarType* _dtype))); }|]

prod_t :: Ptr Tensor -> IO (Ptr Tensor)
prod_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::prod(*$(at::Tensor* _self))); }|]

prod_tlbs :: Ptr Tensor -> Int64 -> CBool -> Ptr ScalarType -> IO (Ptr Tensor)
prod_tlbs _self _dim _keepdim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::prod(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim), *$(at::ScalarType* _dtype))); }|]

prod_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
prod_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::prod(*$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

prod_tls :: Ptr Tensor -> Int64 -> Ptr ScalarType -> IO (Ptr Tensor)
prod_tls _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::prod(*$(at::Tensor* _self), $(int64_t _dim), *$(at::ScalarType* _dtype))); }|]

prod_out_ttlbs :: Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> Ptr ScalarType -> IO (Ptr Tensor)
prod_out_ttlbs _result _self _dim _keepdim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::prod_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim), *$(at::ScalarType* _dtype))); }|]

prod_out_ttlb :: Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> IO (Ptr Tensor)
prod_out_ttlb _result _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::prod_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim), $(bool _keepdim))); }|]

prod_out_ttls :: Ptr Tensor -> Ptr Tensor -> Int64 -> Ptr ScalarType -> IO (Ptr Tensor)
prod_out_ttls _result _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::prod_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim), *$(at::ScalarType* _dtype))); }|]

t_t :: Ptr Tensor -> IO (Ptr Tensor)
t_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::t(*$(at::Tensor* _self))); }|]

t__t :: Ptr Tensor -> IO (Ptr Tensor)
t__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::t_(*$(at::Tensor* _self))); }|]

tan_t :: Ptr Tensor -> IO (Ptr Tensor)
tan_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::tan(*$(at::Tensor* _self))); }|]

_tan__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_tan__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_tan__cpu(*$(at::Tensor* _self))); }|]

_tan__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_tan__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_tan__cuda(*$(at::Tensor* _self))); }|]

_tan_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_tan_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_tan_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_tan_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_tan_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_tan_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

tanh_t :: Ptr Tensor -> IO (Ptr Tensor)
tanh_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::tanh(*$(at::Tensor* _self))); }|]

_tanh__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_tanh__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_tanh__cpu(*$(at::Tensor* _self))); }|]

_tanh__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_tanh__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_tanh__cuda(*$(at::Tensor* _self))); }|]

_tanh_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_tanh_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_tanh_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_tanh_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_tanh_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_tanh_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

tensordot_ttll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
tensordot_ttll _self _other _dims_self _dims_other = [C.block| at::Tensor* { return new at::Tensor(at::native::tensordot(*$(at::Tensor* _self), *$(at::Tensor* _other), *$(at::IntList* _dims_self), *$(at::IntList* _dims_other))); }|]

threshold_tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
threshold_tss _self _threshold _value = [C.block| at::Tensor* { return new at::Tensor(at::native::threshold(*$(at::Tensor* _self), *$(at::Scalar* _threshold), *$(at::Scalar* _value))); }|]

threshold__tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
threshold__tss _self _threshold _value = [C.block| at::Tensor* { return new at::Tensor(at::native::threshold_(*$(at::Tensor* _self), *$(at::Scalar* _threshold), *$(at::Scalar* _value))); }|]

threshold_out_ttss :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
threshold_out_ttss _result _self _threshold _value = [C.block| at::Tensor* { return new at::Tensor(at::native::threshold_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _threshold), *$(at::Scalar* _value))); }|]

threshold_backward_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
threshold_backward_tts _grad_output _self _threshold = [C.block| at::Tensor* { return new at::Tensor(at::native::threshold_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Scalar* _threshold))); }|]

transpose_tll :: Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
transpose_tll _self _dim0 _dim1 = [C.block| at::Tensor* { return new at::Tensor(at::native::transpose(*$(at::Tensor* _self), $(int64_t _dim0), $(int64_t _dim1))); }|]

transpose__tll :: Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
transpose__tll _self _dim0 _dim1 = [C.block| at::Tensor* { return new at::Tensor(at::native::transpose_(*$(at::Tensor* _self), $(int64_t _dim0), $(int64_t _dim1))); }|]

one_hot_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
one_hot_tl _self _num_classes = [C.block| at::Tensor* { return new at::Tensor(at::native::one_hot(*$(at::Tensor* _self), $(int64_t _num_classes))); }|]

flip_cpu_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
flip_cpu_tl _self _dims = [C.block| at::Tensor* { return new at::Tensor(at::native::flip_cpu(*$(at::Tensor* _self), *$(at::IntList* _dims))); }|]

flip_cuda_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
flip_cuda_tl _self _dims = [C.block| at::Tensor* { return new at::Tensor(at::native::flip_cuda(*$(at::Tensor* _self), *$(at::IntList* _dims))); }|]

roll_cpu_tll :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
roll_cpu_tll _self _shifts _dims = [C.block| at::Tensor* { return new at::Tensor(at::native::roll_cpu(*$(at::Tensor* _self), *$(at::IntList* _shifts), *$(at::IntList* _dims))); }|]

roll_cuda_tll :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
roll_cuda_tll _self _shifts _dims = [C.block| at::Tensor* { return new at::Tensor(at::native::roll_cuda(*$(at::Tensor* _self), *$(at::IntList* _shifts), *$(at::IntList* _dims))); }|]

rot90_tll :: Ptr Tensor -> Int64 -> Ptr IntList -> IO (Ptr Tensor)
rot90_tll _self _k _dims = [C.block| at::Tensor* { return new at::Tensor(at::native::rot90(*$(at::Tensor* _self), $(int64_t _k), *$(at::IntList* _dims))); }|]

_trilinear_tttlllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Int64 -> IO (Ptr Tensor)
_trilinear_tttlllll _i1 _i2 _i3 _expand1 _expand2 _expand3 _sumdim _unroll_dim = [C.block| at::Tensor* { return new at::Tensor(at::native::_trilinear(*$(at::Tensor* _i1), *$(at::Tensor* _i2), *$(at::Tensor* _i3), *$(at::IntList* _expand1), *$(at::IntList* _expand2), *$(at::IntList* _expand3), *$(at::IntList* _sumdim), $(int64_t _unroll_dim))); }|]

triplet_margin_loss_tttdddbl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CDouble -> CDouble -> CDouble -> CBool -> Int64 -> IO (Ptr Tensor)
triplet_margin_loss_tttdddbl _anchor _positive _negative _margin _p _eps _swap _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::triplet_margin_loss(*$(at::Tensor* _anchor), *$(at::Tensor* _positive), *$(at::Tensor* _negative), $(double _margin), $(double _p), $(double _eps), $(bool _swap), $(int64_t _reduction))); }|]

trunc_t :: Ptr Tensor -> IO (Ptr Tensor)
trunc_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::trunc(*$(at::Tensor* _self))); }|]

_trunc__cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
_trunc__cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_trunc__cpu(*$(at::Tensor* _self))); }|]

_trunc__cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
_trunc__cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_trunc__cuda(*$(at::Tensor* _self))); }|]

_trunc_out_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_trunc_out_cpu_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_trunc_out_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

_trunc_out_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_trunc_out_cuda_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_trunc_out_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

type_as_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
type_as_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::type_as(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

_unique_cpu_tbb :: Ptr Tensor -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
_unique_cpu_tbb _self _sorted _return_inverse = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::_unique_cpu(*$(at::Tensor* _self), $(bool _sorted), $(bool _return_inverse))); }|]

_unique_cuda_tbb :: Ptr Tensor -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
_unique_cuda_tbb _self _sorted _return_inverse = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::_unique_cuda(*$(at::Tensor* _self), $(bool _sorted), $(bool _return_inverse))); }|]

_unique_dim_tlbb :: Ptr Tensor -> Int64 -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
_unique_dim_tlbb _self _dim _sorted _return_inverse = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::_unique_dim(*$(at::Tensor* _self), $(int64_t _dim), $(bool _sorted), $(bool _return_inverse))); }|]

_unsafe_view_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
_unsafe_view_tl _self _size = [C.block| at::Tensor* { return new at::Tensor(at::native::_unsafe_view(*$(at::Tensor* _self), *$(at::IntList* _size))); }|]

unsqueeze_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
unsqueeze_tl _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::unsqueeze(*$(at::Tensor* _self), $(int64_t _dim))); }|]

unsqueeze__tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
unsqueeze__tl _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::unsqueeze_(*$(at::Tensor* _self), $(int64_t _dim))); }|]

var_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
var_tb _self _unbiased = [C.block| at::Tensor* { return new at::Tensor(at::native::var(*$(at::Tensor* _self), $(bool _unbiased))); }|]

var_tlbb :: Ptr Tensor -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
var_tlbb _self _dim _unbiased _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::var(*$(at::Tensor* _self), $(int64_t _dim), $(bool _unbiased), $(bool _keepdim))); }|]

var_out_ttlbb :: Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> CBool -> IO (Ptr Tensor)
var_out_ttlbb _result _self _dim _unbiased _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::var_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim), $(bool _unbiased), $(bool _keepdim))); }|]

view_as_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
view_as_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::view_as(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

where_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
where_ttt _condition _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::where(*$(at::Tensor* _condition), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

_s_where_cpu_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_s_where_cpu_ttt _condition _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::_s_where_cpu(*$(at::Tensor* _condition), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

_s_where_cuda_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_s_where_cuda_ttt _condition _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::_s_where_cuda(*$(at::Tensor* _condition), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

norm_except_dim_tll :: Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
norm_except_dim_tll _v _pow _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::norm_except_dim(*$(at::Tensor* _v), $(int64_t _pow), $(int64_t _dim))); }|]

_weight_norm_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
_weight_norm_ttl _v _g _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::_weight_norm(*$(at::Tensor* _v), *$(at::Tensor* _g), $(int64_t _dim))); }|]

weight_norm_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr (Tensor,Tensor))
weight_norm_cuda_ttl _v _g _dim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::weight_norm_cuda(*$(at::Tensor* _v), *$(at::Tensor* _g), $(int64_t _dim))); }|]

weight_norm_cuda_backward_ttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr (Tensor,Tensor))
weight_norm_cuda_backward_ttttl _grad_w _saved_v _saved_g _saved_norms _dim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::weight_norm_cuda_backward(*$(at::Tensor* _grad_w), *$(at::Tensor* _saved_v), *$(at::Tensor* _saved_g), *$(at::Tensor* _saved_norms), $(int64_t _dim))); }|]

_weight_norm_differentiable_backward_ttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr (Tensor,Tensor))
_weight_norm_differentiable_backward_ttttl _grad_w _saved_v _saved_g _saved_norms _dim = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::_weight_norm_differentiable_backward(*$(at::Tensor* _grad_w), *$(at::Tensor* _saved_v), *$(at::Tensor* _saved_g), *$(at::Tensor* _saved_norms), $(int64_t _dim))); }|]

zeros_lo :: Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
zeros_lo _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::zeros(*$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

zeros_out_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
zeros_out_tl _result _size = [C.block| at::Tensor* { return new at::Tensor(at::native::zeros_out(*$(at::Tensor* _result), *$(at::IntList* _size))); }|]

zeros_like_t :: Ptr Tensor -> IO (Ptr Tensor)
zeros_like_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::zeros_like(*$(at::Tensor* _self))); }|]

zeros_like_to :: Ptr Tensor -> Ptr TensorOptions -> IO (Ptr Tensor)
zeros_like_to _self _options = [C.block| at::Tensor* { return new at::Tensor(at::native::zeros_like(*$(at::Tensor* _self), *$(at::TensorOptions* _options))); }|]

_standard_gamma_grad_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_standard_gamma_grad_cpu_tt _self _output = [C.block| at::Tensor* { return new at::Tensor(at::native::_standard_gamma_grad_cpu(*$(at::Tensor* _self), *$(at::Tensor* _output))); }|]

_standard_gamma_grad_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_standard_gamma_grad_cuda_tt _self _output = [C.block| at::Tensor* { return new at::Tensor(at::native::_standard_gamma_grad_cuda(*$(at::Tensor* _self), *$(at::Tensor* _output))); }|]

_s_gamma_cpu_tp :: Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
_s_gamma_cpu_tp _self _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::_s_gamma_cpu(*$(at::Tensor* _self), $(at::Generator * _generator))); }|]

_s_gamma_cuda_tp :: Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
_s_gamma_cuda_tp _self _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::_s_gamma_cuda(*$(at::Tensor* _self), $(at::Generator * _generator))); }|]

_s_poisson_cpu_tp :: Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
_s_poisson_cpu_tp _self _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::_s_poisson_cpu(*$(at::Tensor* _self), $(at::Generator * _generator))); }|]

_s_poisson_cuda_tp :: Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
_s_poisson_cuda_tp _self _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::_s_poisson_cuda(*$(at::Tensor* _self), $(at::Generator * _generator))); }|]

norm_sparse_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
norm_sparse_ts _self _p = [C.block| at::Tensor* { return new at::Tensor(at::native::norm_sparse(*$(at::Tensor* _self), *$(at::Scalar* _p))); }|]

_sparse_sum_t :: Ptr Tensor -> IO (Ptr Tensor)
_sparse_sum_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_sparse_sum(*$(at::Tensor* _self))); }|]

_sparse_sum_ts :: Ptr Tensor -> Ptr ScalarType -> IO (Ptr Tensor)
_sparse_sum_ts _self _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::_sparse_sum(*$(at::Tensor* _self), *$(at::ScalarType* _dtype))); }|]

_sparse_sum_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
_sparse_sum_tl _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::_sparse_sum(*$(at::Tensor* _self), *$(at::IntList* _dim))); }|]

_sparse_sum_tls :: Ptr Tensor -> Ptr IntList -> Ptr ScalarType -> IO (Ptr Tensor)
_sparse_sum_tls _self _dim _dtype = [C.block| at::Tensor* { return new at::Tensor(at::native::_sparse_sum(*$(at::Tensor* _self), *$(at::IntList* _dim), *$(at::ScalarType* _dtype))); }|]

_sparse_sum_backward_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
_sparse_sum_backward_cpu_ttl _grad _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::_sparse_sum_backward_cpu(*$(at::Tensor* _grad), *$(at::Tensor* _self), *$(at::IntList* _dim))); }|]

_sparse_sum_backward_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
_sparse_sum_backward_cuda_ttl _grad _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::_sparse_sum_backward_cuda(*$(at::Tensor* _grad), *$(at::Tensor* _self), *$(at::IntList* _dim))); }|]

norm_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
norm_ts _self _p = [C.block| at::Tensor* { return new at::Tensor(at::native::norm(*$(at::Tensor* _self), *$(at::Scalar* _p))); }|]

norm_tslb :: Ptr Tensor -> Ptr Scalar -> Int64 -> CBool -> IO (Ptr Tensor)
norm_tslb _self _p _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::norm(*$(at::Tensor* _self), *$(at::Scalar* _p), $(int64_t _dim), $(bool _keepdim))); }|]

norm_out_ttslb :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Int64 -> CBool -> IO (Ptr Tensor)
norm_out_ttslb _result _self _p _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::norm_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _p), $(int64_t _dim), $(bool _keepdim))); }|]

frobenius_norm_t :: Ptr Tensor -> IO (Ptr Tensor)
frobenius_norm_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::frobenius_norm(*$(at::Tensor* _self))); }|]

frobenius_norm_tlb :: Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
frobenius_norm_tlb _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::frobenius_norm(*$(at::Tensor* _self), *$(at::IntList* _dim), $(bool _keepdim))); }|]

frobenius_norm_out_ttlb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
frobenius_norm_out_ttlb _result _self _dim _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::frobenius_norm_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::IntList* _dim), $(bool _keepdim))); }|]

nuclear_norm_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
nuclear_norm_tb _self _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::nuclear_norm(*$(at::Tensor* _self), $(bool _keepdim))); }|]

nuclear_norm_out_ttb :: Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
nuclear_norm_out_ttb _result _self _keepdim = [C.block| at::Tensor* { return new at::Tensor(at::native::nuclear_norm_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(bool _keepdim))); }|]

clone_sparse_t :: Ptr Tensor -> IO (Ptr Tensor)
clone_sparse_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::clone_sparse(*$(at::Tensor* _self))); }|]

clone_t :: Ptr Tensor -> IO (Ptr Tensor)
clone_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::clone(*$(at::Tensor* _self))); }|]

resize_as_sparse__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
resize_as_sparse__tt _self _the_template = [C.block| at::Tensor* { return new at::Tensor(at::native::resize_as_sparse_(*$(at::Tensor* _self), *$(at::Tensor* _the_template))); }|]

resize_as__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
resize_as__tt _self _the_template = [C.block| at::Tensor* { return new at::Tensor(at::native::resize_as_(*$(at::Tensor* _self), *$(at::Tensor* _the_template))); }|]

pow_out_sparse_scalar_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
pow_out_sparse_scalar_tts _result _self _exponent = [C.block| at::Tensor* { return new at::Tensor(at::native::pow_out_sparse_scalar(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _exponent))); }|]

pow_sparse_scalar_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
pow_sparse_scalar_ts _self _exponent = [C.block| at::Tensor* { return new at::Tensor(at::native::pow_sparse_scalar(*$(at::Tensor* _self), *$(at::Scalar* _exponent))); }|]

pow_out_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
pow_out_tts _result _self _exponent = [C.block| at::Tensor* { return new at::Tensor(at::native::pow_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _exponent))); }|]

pow_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
pow_ts _self _exponent = [C.block| at::Tensor* { return new at::Tensor(at::native::pow(*$(at::Tensor* _self), *$(at::Scalar* _exponent))); }|]

zero_sparse__t :: Ptr Tensor -> IO (Ptr Tensor)
zero_sparse__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::zero_sparse_(*$(at::Tensor* _self))); }|]

zero__t :: Ptr Tensor -> IO (Ptr Tensor)
zero__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::zero_(*$(at::Tensor* _self))); }|]

sub_out_ttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
sub_out_ttts _result _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::sub_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other), *$(at::Scalar* _alpha))); }|]

sub_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
sub_tts _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::sub(*$(at::Tensor* _self), *$(at::Tensor* _other), *$(at::Scalar* _alpha))); }|]

sub__tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
sub__tts _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::sub_(*$(at::Tensor* _self), *$(at::Tensor* _other), *$(at::Scalar* _alpha))); }|]

sub_tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
sub_tss _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::sub(*$(at::Tensor* _self), *$(at::Scalar* _other), *$(at::Scalar* _alpha))); }|]

sub__tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
sub__tss _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::sub_(*$(at::Tensor* _self), *$(at::Scalar* _other), *$(at::Scalar* _alpha))); }|]

rsub_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
rsub_tts _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::rsub(*$(at::Tensor* _self), *$(at::Tensor* _other), *$(at::Scalar* _alpha))); }|]

rsub_tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
rsub_tss _self _other _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::rsub(*$(at::Tensor* _self), *$(at::Scalar* _other), *$(at::Scalar* _alpha))); }|]

s_addmm_out_sparse_dense_cpu_ttttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
s_addmm_out_sparse_dense_cpu_ttttss _result _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::s_addmm_out_sparse_dense_cpu(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

s_addmm_out_sparse_dense_cuda_ttttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
s_addmm_out_sparse_dense_cuda_ttttss _result _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::s_addmm_out_sparse_dense_cuda(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

s_addmm_sparse_dense_cpu_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
s_addmm_sparse_dense_cpu_tttss _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::s_addmm_sparse_dense_cpu(*$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

s_addmm_sparse_dense_cuda_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
s_addmm_sparse_dense_cuda_tttss _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::s_addmm_sparse_dense_cuda(*$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

s_addmm_sparse_dense_cpu__tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
s_addmm_sparse_dense_cpu__tttss _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::s_addmm_sparse_dense_cpu_(*$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

s_addmm_sparse_dense_cuda__tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
s_addmm_sparse_dense_cuda__tttss _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::s_addmm_sparse_dense_cuda_(*$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

_sparse_addmm_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
_sparse_addmm_tttss _self _sparse _dense _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::_sparse_addmm(*$(at::Tensor* _self), *$(at::Tensor* _sparse), *$(at::Tensor* _dense), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

addmm_out_ttttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
addmm_out_ttttss _result _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::addmm_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

addmm_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
addmm_tttss _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::addmm(*$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

addmm__tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
addmm__tttss _self _mat1 _mat2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::addmm_(*$(at::Tensor* _self), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

sparse_coo_tensor_lo :: Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
sparse_coo_tensor_lo _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::sparse_coo_tensor(*$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

sparse_coo_tensor_tto :: Ptr Tensor -> Ptr Tensor -> Ptr TensorOptions -> IO (Ptr Tensor)
sparse_coo_tensor_tto _indices _values _options = [C.block| at::Tensor* { return new at::Tensor(at::native::sparse_coo_tensor(*$(at::Tensor* _indices), *$(at::Tensor* _values), *$(at::TensorOptions* _options))); }|]

sparse_coo_tensor_ttlo :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
sparse_coo_tensor_ttlo _indices _values _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::sparse_coo_tensor(*$(at::Tensor* _indices), *$(at::Tensor* _values), *$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

_sparse_coo_tensor_unsafe_ttlo :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
_sparse_coo_tensor_unsafe_ttlo _indices _values _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::_sparse_coo_tensor_unsafe(*$(at::Tensor* _indices), *$(at::Tensor* _values), *$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

new_with_dims_sparse_lllo :: Int64 -> Int64 -> Ptr IntList -> Ptr TensorOptions -> IO (Ptr Tensor)
new_with_dims_sparse_lllo _sparse_dim _dense_dim _size _options = [C.block| at::Tensor* { return new at::Tensor(at::native::new_with_dims_sparse($(int64_t _sparse_dim), $(int64_t _dense_dim), *$(at::IntList* _size), *$(at::TensorOptions* _options))); }|]

new_with_dims_and_tensor_sparse_llltto :: Int64 -> Int64 -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr TensorOptions -> IO (Ptr Tensor)
new_with_dims_and_tensor_sparse_llltto _sparse_dim _dense_dim _size _indices _values _options = [C.block| at::Tensor* { return new at::Tensor(at::native::new_with_dims_and_tensor_sparse($(int64_t _sparse_dim), $(int64_t _dense_dim), *$(at::IntList* _size), *$(at::Tensor* _indices), *$(at::Tensor* _values), *$(at::TensorOptions* _options))); }|]

sparse_resize__tlll :: Ptr Tensor -> Ptr IntList -> Int64 -> Int64 -> IO (Ptr Tensor)
sparse_resize__tlll _self _size _sparse_dim _dense_dim = [C.block| at::Tensor* { return new at::Tensor(at::native::sparse_resize_(*$(at::Tensor* _self), *$(at::IntList* _size), $(int64_t _sparse_dim), $(int64_t _dense_dim))); }|]

sparse_resize_and_clear__tlll :: Ptr Tensor -> Ptr IntList -> Int64 -> Int64 -> IO (Ptr Tensor)
sparse_resize_and_clear__tlll _self _size _sparse_dim _dense_dim = [C.block| at::Tensor* { return new at::Tensor(at::native::sparse_resize_and_clear_(*$(at::Tensor* _self), *$(at::IntList* _size), $(int64_t _sparse_dim), $(int64_t _dense_dim))); }|]

sparse_mask_cpu_tr :: Ptr Tensor -> Ptr SparseTensorRef -> IO (Ptr Tensor)
sparse_mask_cpu_tr _self _mask = [C.block| at::Tensor* { return new at::Tensor(at::native::sparse_mask_cpu(*$(at::Tensor* _self), *$(at::SparseTensorRef* _mask))); }|]

sparse_mask_cuda_tr :: Ptr Tensor -> Ptr SparseTensorRef -> IO (Ptr Tensor)
sparse_mask_cuda_tr _self _mask = [C.block| at::Tensor* { return new at::Tensor(at::native::sparse_mask_cuda(*$(at::Tensor* _self), *$(at::SparseTensorRef* _mask))); }|]

sparse_to_dense_t :: Ptr Tensor -> IO (Ptr Tensor)
sparse_to_dense_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::sparse_to_dense(*$(at::Tensor* _self))); }|]

sparse_dim_sparse_t :: Ptr Tensor -> IO (Int64)
sparse_dim_sparse_t _self = [C.block| int64_t { return (at::native::sparse_dim_sparse(*$(at::Tensor* _self))); }|]

dense_dim_sparse_t :: Ptr Tensor -> IO (Int64)
dense_dim_sparse_t _self = [C.block| int64_t { return (at::native::dense_dim_sparse(*$(at::Tensor* _self))); }|]

_nnz_sparse_t :: Ptr Tensor -> IO (Int64)
_nnz_sparse_t _self = [C.block| int64_t { return (at::native::_nnz_sparse(*$(at::Tensor* _self))); }|]

coalesce_sparse_cpu_t :: Ptr Tensor -> IO (Ptr Tensor)
coalesce_sparse_cpu_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::coalesce_sparse_cpu(*$(at::Tensor* _self))); }|]

coalesce_sparse_cuda_t :: Ptr Tensor -> IO (Ptr Tensor)
coalesce_sparse_cuda_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::coalesce_sparse_cuda(*$(at::Tensor* _self))); }|]

is_coalesced_sparse_t :: Ptr Tensor -> IO (CBool)
is_coalesced_sparse_t _self = [C.block| bool { return (at::native::is_coalesced_sparse(*$(at::Tensor* _self))); }|]

_indices_sparse_t :: Ptr Tensor -> IO (Ptr Tensor)
_indices_sparse_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_indices_sparse(*$(at::Tensor* _self))); }|]

_values_sparse_t :: Ptr Tensor -> IO (Ptr Tensor)
_values_sparse_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::_values_sparse(*$(at::Tensor* _self))); }|]

_coalesced_sparse__tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
_coalesced_sparse__tb _self _coalesced = [C.block| at::Tensor* { return new at::Tensor(at::native::_coalesced_sparse_(*$(at::Tensor* _self), $(bool _coalesced))); }|]

indices_sparse_t :: Ptr Tensor -> IO (Ptr Tensor)
indices_sparse_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::indices_sparse(*$(at::Tensor* _self))); }|]

values_sparse_t :: Ptr Tensor -> IO (Ptr Tensor)
values_sparse_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::values_sparse(*$(at::Tensor* _self))); }|]

hspmm_out_sparse_cpu_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
hspmm_out_sparse_cpu_ttt _result _mat1 _mat2 = [C.block| at::Tensor* { return new at::Tensor(at::native::hspmm_out_sparse_cpu(*$(at::Tensor* _result), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2))); }|]

hspmm_out_sparse_cuda_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
hspmm_out_sparse_cuda_ttt _result _mat1 _mat2 = [C.block| at::Tensor* { return new at::Tensor(at::native::hspmm_out_sparse_cuda(*$(at::Tensor* _result), *$(at::Tensor* _mat1), *$(at::Tensor* _mat2))); }|]

hspmm_sparse_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
hspmm_sparse_cpu_tt _mat1 _mat2 = [C.block| at::Tensor* { return new at::Tensor(at::native::hspmm_sparse_cpu(*$(at::Tensor* _mat1), *$(at::Tensor* _mat2))); }|]

hspmm_sparse_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
hspmm_sparse_cuda_tt _mat1 _mat2 = [C.block| at::Tensor* { return new at::Tensor(at::native::hspmm_sparse_cuda(*$(at::Tensor* _mat1), *$(at::Tensor* _mat2))); }|]

copy_sparse__ttb :: Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
copy_sparse__ttb _self _src _non_blocking = [C.block| at::Tensor* { return new at::Tensor(at::native::copy_sparse_(*$(at::Tensor* _self), *$(at::Tensor* _src), $(bool _non_blocking))); }|]

numel_t :: Ptr Tensor -> IO (Int64)
numel_t _self = [C.block| int64_t { return (at::native::numel(*$(at::Tensor* _self))); }|]

unbind_tl :: Ptr Tensor -> Int64 -> IO (Ptr TensorList)
unbind_tl _self _dim = [C.block| at::TensorList* { return new at::TensorList(at::native::unbind(*$(at::Tensor* _self), $(int64_t _dim))); }|]

dense_to_sparse_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
dense_to_sparse_tl _self _sparse_dim = [C.block| at::Tensor* { return new at::Tensor(at::native::dense_to_sparse(*$(at::Tensor* _self), $(int64_t _sparse_dim))); }|]

dense_to_sparse_t :: Ptr Tensor -> IO (Ptr Tensor)
dense_to_sparse_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::dense_to_sparse(*$(at::Tensor* _self))); }|]

to_tobb :: Ptr Tensor -> Ptr TensorOptions -> CBool -> CBool -> IO (Ptr Tensor)
to_tobb _self _options _non_blocking _copy = [C.block| at::Tensor* { return new at::Tensor(at::native::to(*$(at::Tensor* _self), *$(at::TensorOptions* _options), $(bool _non_blocking), $(bool _copy))); }|]

to_tdevicesbb :: Ptr Tensor -> Ptr Device -> Ptr ScalarType -> CBool -> CBool -> IO (Ptr Tensor)
to_tdevicesbb _self _device _dtype _non_blocking _copy = [C.block| at::Tensor* { return new at::Tensor(at::native::to(*$(at::Tensor* _self), *$(at::Device* _device), *$(at::ScalarType* _dtype), $(bool _non_blocking), $(bool _copy))); }|]

to_tsbb :: Ptr Tensor -> Ptr ScalarType -> CBool -> CBool -> IO (Ptr Tensor)
to_tsbb _self _dtype _non_blocking _copy = [C.block| at::Tensor* { return new at::Tensor(at::native::to(*$(at::Tensor* _self), *$(at::ScalarType* _dtype), $(bool _non_blocking), $(bool _copy))); }|]

to_ttbb :: Ptr Tensor -> Ptr Tensor -> CBool -> CBool -> IO (Ptr Tensor)
to_ttbb _self _other _non_blocking _copy = [C.block| at::Tensor* { return new at::Tensor(at::native::to(*$(at::Tensor* _self), *$(at::Tensor* _other), $(bool _non_blocking), $(bool _copy))); }|]

meshgrid_l :: Ptr TensorList -> IO (Ptr TensorList)
meshgrid_l _tensors = [C.block| at::TensorList* { return new at::TensorList(at::native::meshgrid(*$(at::TensorList* _tensors))); }|]

item_t :: Ptr Tensor -> IO (Ptr Scalar)
item_t _self = [C.block| at::Scalar* { return new at::Scalar(at::native::item(*$(at::Tensor* _self))); }|]

_local_scalar_dense_cpu_t :: Ptr Tensor -> IO (Ptr Scalar)
_local_scalar_dense_cpu_t _self = [C.block| at::Scalar* { return new at::Scalar(at::native::_local_scalar_dense_cpu(*$(at::Tensor* _self))); }|]

_local_scalar_dense_cuda_t :: Ptr Tensor -> IO (Ptr Scalar)
_local_scalar_dense_cuda_t _self = [C.block| at::Scalar* { return new at::Scalar(at::native::_local_scalar_dense_cuda(*$(at::Tensor* _self))); }|]

_thnn_fused_lstm_cell_cuda_ttttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor,Tensor))
_thnn_fused_lstm_cell_cuda_ttttt _input_gates _hidden_gates _cx _input_bias _hidden_bias = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::_thnn_fused_lstm_cell_cuda(*$(at::Tensor* _input_gates), *$(at::Tensor* _hidden_gates), *$(at::Tensor* _cx), *$(at::Tensor* _input_bias), *$(at::Tensor* _hidden_bias))); }|]

_thnn_fused_lstm_cell_backward_cuda_tttttb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr (Tensor,Tensor,Tensor,Tensor,Tensor))
_thnn_fused_lstm_cell_backward_cuda_tttttb _grad_hy _grad_cy _cx _cy _workspace _has_bias = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>(at::native::_thnn_fused_lstm_cell_backward_cuda(*$(at::Tensor* _grad_hy), *$(at::Tensor* _grad_cy), *$(at::Tensor* _cx), *$(at::Tensor* _cy), *$(at::Tensor* _workspace), $(bool _has_bias))); }|]

_thnn_fused_gru_cell_cuda_ttttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
_thnn_fused_gru_cell_cuda_ttttt _input_gates _hidden_gates _hx _input_bias _hidden_bias = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::_thnn_fused_gru_cell_cuda(*$(at::Tensor* _input_gates), *$(at::Tensor* _hidden_gates), *$(at::Tensor* _hx), *$(at::Tensor* _input_bias), *$(at::Tensor* _hidden_bias))); }|]

_thnn_fused_gru_cell_backward_cuda_ttb :: Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr (Tensor,Tensor,Tensor,Tensor,Tensor))
_thnn_fused_gru_cell_backward_cuda_ttb _grad_hy _workspace _has_bias = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>(at::native::_thnn_fused_gru_cell_backward_cuda(*$(at::Tensor* _grad_hy), *$(at::Tensor* _workspace), $(bool _has_bias))); }|]

lstm_tllbldbbb :: Ptr Tensor -> Ptr TensorList -> Ptr TensorList -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (Ptr (Tensor,Tensor,Tensor))
lstm_tllbldbbb _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::lstm(*$(at::Tensor* _input), *$(at::TensorList* _hx), *$(at::TensorList* _params), $(bool _has_biases), $(int64_t _num_layers), $(double _dropout), $(bool _train), $(bool _bidirectional), $(bool _batch_first))); }|]

lstm_ttllbldbb :: Ptr Tensor -> Ptr Tensor -> Ptr TensorList -> Ptr TensorList -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (Ptr (Tensor,Tensor,Tensor))
lstm_ttllbldbb _data _batch_sizes _hx _params _has_biases _num_layers _dropout _train _bidirectional = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::lstm(*$(at::Tensor* _data), *$(at::Tensor* _batch_sizes), *$(at::TensorList* _hx), *$(at::TensorList* _params), $(bool _has_biases), $(int64_t _num_layers), $(double _dropout), $(bool _train), $(bool _bidirectional))); }|]

gru_ttlbldbbb :: Ptr Tensor -> Ptr Tensor -> Ptr TensorList -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
gru_ttlbldbbb _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::gru(*$(at::Tensor* _input), *$(at::Tensor* _hx), *$(at::TensorList* _params), $(bool _has_biases), $(int64_t _num_layers), $(double _dropout), $(bool _train), $(bool _bidirectional), $(bool _batch_first))); }|]

gru_tttlbldbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr TensorList -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
gru_tttlbldbb _data _batch_sizes _hx _params _has_biases _num_layers _dropout _train _bidirectional = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::gru(*$(at::Tensor* _data), *$(at::Tensor* _batch_sizes), *$(at::Tensor* _hx), *$(at::TensorList* _params), $(bool _has_biases), $(int64_t _num_layers), $(double _dropout), $(bool _train), $(bool _bidirectional))); }|]

rnn_tanh_ttlbldbbb :: Ptr Tensor -> Ptr Tensor -> Ptr TensorList -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
rnn_tanh_ttlbldbbb _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::rnn_tanh(*$(at::Tensor* _input), *$(at::Tensor* _hx), *$(at::TensorList* _params), $(bool _has_biases), $(int64_t _num_layers), $(double _dropout), $(bool _train), $(bool _bidirectional), $(bool _batch_first))); }|]

rnn_tanh_tttlbldbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr TensorList -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
rnn_tanh_tttlbldbb _data _batch_sizes _hx _params _has_biases _num_layers _dropout _train _bidirectional = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::rnn_tanh(*$(at::Tensor* _data), *$(at::Tensor* _batch_sizes), *$(at::Tensor* _hx), *$(at::TensorList* _params), $(bool _has_biases), $(int64_t _num_layers), $(double _dropout), $(bool _train), $(bool _bidirectional))); }|]

rnn_relu_ttlbldbbb :: Ptr Tensor -> Ptr Tensor -> Ptr TensorList -> CBool -> Int64 -> CDouble -> CBool -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
rnn_relu_ttlbldbbb _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::rnn_relu(*$(at::Tensor* _input), *$(at::Tensor* _hx), *$(at::TensorList* _params), $(bool _has_biases), $(int64_t _num_layers), $(double _dropout), $(bool _train), $(bool _bidirectional), $(bool _batch_first))); }|]

rnn_relu_tttlbldbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr TensorList -> CBool -> Int64 -> CDouble -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
rnn_relu_tttlbldbb _data _batch_sizes _hx _params _has_biases _num_layers _dropout _train _bidirectional = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::rnn_relu(*$(at::Tensor* _data), *$(at::Tensor* _batch_sizes), *$(at::Tensor* _hx), *$(at::TensorList* _params), $(bool _has_biases), $(int64_t _num_layers), $(double _dropout), $(bool _train), $(bool _bidirectional))); }|]

lstm_cell_tltttt :: Ptr Tensor -> Ptr TensorList -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
lstm_cell_tltttt _input _hx _w_ih _w_hh _b_ih _b_hh = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::lstm_cell(*$(at::Tensor* _input), *$(at::TensorList* _hx), *$(at::Tensor* _w_ih), *$(at::Tensor* _w_hh), *$(at::Tensor* _b_ih), *$(at::Tensor* _b_hh))); }|]

gru_cell_tttttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
gru_cell_tttttt _input _hx _w_ih _w_hh _b_ih _b_hh = [C.block| at::Tensor* { return new at::Tensor(at::native::gru_cell(*$(at::Tensor* _input), *$(at::Tensor* _hx), *$(at::Tensor* _w_ih), *$(at::Tensor* _w_hh), *$(at::Tensor* _b_ih), *$(at::Tensor* _b_hh))); }|]

rnn_tanh_cell_tttttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
rnn_tanh_cell_tttttt _input _hx _w_ih _w_hh _b_ih _b_hh = [C.block| at::Tensor* { return new at::Tensor(at::native::rnn_tanh_cell(*$(at::Tensor* _input), *$(at::Tensor* _hx), *$(at::Tensor* _w_ih), *$(at::Tensor* _w_hh), *$(at::Tensor* _b_ih), *$(at::Tensor* _b_hh))); }|]

rnn_relu_cell_tttttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
rnn_relu_cell_tttttt _input _hx _w_ih _w_hh _b_ih _b_hh = [C.block| at::Tensor* { return new at::Tensor(at::native::rnn_relu_cell(*$(at::Tensor* _input), *$(at::Tensor* _hx), *$(at::Tensor* _w_ih), *$(at::Tensor* _w_hh), *$(at::Tensor* _b_ih), *$(at::Tensor* _b_hh))); }|]

_pack_padded_sequence_ttb :: Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr (Tensor,Tensor))
_pack_padded_sequence_ttb _input _lengths _batch_first = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::_pack_padded_sequence(*$(at::Tensor* _input), *$(at::Tensor* _lengths), $(bool _batch_first))); }|]

_pack_padded_sequence_backward_tltb :: Ptr Tensor -> Ptr IntList -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
_pack_padded_sequence_backward_tltb _grad _input_size _batch_sizes _batch_first = [C.block| at::Tensor* { return new at::Tensor(at::native::_pack_padded_sequence_backward(*$(at::Tensor* _grad), *$(at::IntList* _input_size), *$(at::Tensor* _batch_sizes), $(bool _batch_first))); }|]

_pad_packed_sequence_ttbsl :: Ptr Tensor -> Ptr Tensor -> CBool -> Ptr Scalar -> Int64 -> IO (Ptr (Tensor,Tensor))
_pad_packed_sequence_ttbsl _data _batch_sizes _batch_first _padding_value _total_length = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::_pad_packed_sequence(*$(at::Tensor* _data), *$(at::Tensor* _batch_sizes), $(bool _batch_first), *$(at::Scalar* _padding_value), $(int64_t _total_length))); }|]

data_ptr_t :: Ptr Tensor -> IO ()
data_ptr_t _self = [C.block| void {  (at::native::data_ptr(*$(at::Tensor* _self))); }|]

set__tstorage :: Ptr Tensor -> Ptr Storage -> IO (Ptr Tensor)
set__tstorage _self _source = [C.block| at::Tensor* { return new at::Tensor(at::native::set_(*$(at::Tensor* _self), *$(at::Storage* _source))); }|]

set__tstoragelll :: Ptr Tensor -> Ptr Storage -> Int64 -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
set__tstoragelll _self _source _storage_offset _size _stride = [C.block| at::Tensor* { return new at::Tensor(at::native::set_(*$(at::Tensor* _self), *$(at::Storage* _source), $(int64_t _storage_offset), *$(at::IntList* _size), *$(at::IntList* _stride))); }|]

set__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
set__tt _self _source = [C.block| at::Tensor* { return new at::Tensor(at::native::set_(*$(at::Tensor* _self), *$(at::Tensor* _source))); }|]

set__t :: Ptr Tensor -> IO (Ptr Tensor)
set__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::set_(*$(at::Tensor* _self))); }|]

is_set_to_tt :: Ptr Tensor -> Ptr Tensor -> IO (CBool)
is_set_to_tt _self _tensor = [C.block| bool { return (at::native::is_set_to(*$(at::Tensor* _self), *$(at::Tensor* _tensor))); }|]

masked_fill__tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
masked_fill__tts _self _mask _value = [C.block| at::Tensor* { return new at::Tensor(at::native::masked_fill_(*$(at::Tensor* _self), *$(at::Tensor* _mask), *$(at::Scalar* _value))); }|]

masked_fill__ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
masked_fill__ttt _self _mask _value = [C.block| at::Tensor* { return new at::Tensor(at::native::masked_fill_(*$(at::Tensor* _self), *$(at::Tensor* _mask), *$(at::Tensor* _value))); }|]

masked_scatter__ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
masked_scatter__ttt _self _mask _source = [C.block| at::Tensor* { return new at::Tensor(at::native::masked_scatter_(*$(at::Tensor* _self), *$(at::Tensor* _mask), *$(at::Tensor* _source))); }|]

view_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
view_tl _self _size = [C.block| at::Tensor* { return new at::Tensor(at::native::view(*$(at::Tensor* _self), *$(at::IntList* _size))); }|]

put__tttb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
put__tttb _self _index _source _accumulate = [C.block| at::Tensor* { return new at::Tensor(at::native::put_(*$(at::Tensor* _self), *$(at::Tensor* _index), *$(at::Tensor* _source), $(bool _accumulate))); }|]

index_add__tltt :: Ptr Tensor -> Int64 -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
index_add__tltt _self _dim _index _source = [C.block| at::Tensor* { return new at::Tensor(at::native::index_add_(*$(at::Tensor* _self), $(int64_t _dim), *$(at::Tensor* _index), *$(at::Tensor* _source))); }|]

index_fill__tlts :: Ptr Tensor -> Int64 -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
index_fill__tlts _self _dim _index _value = [C.block| at::Tensor* { return new at::Tensor(at::native::index_fill_(*$(at::Tensor* _self), $(int64_t _dim), *$(at::Tensor* _index), *$(at::Scalar* _value))); }|]

index_fill__tltt :: Ptr Tensor -> Int64 -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
index_fill__tltt _self _dim _index _value = [C.block| at::Tensor* { return new at::Tensor(at::native::index_fill_(*$(at::Tensor* _self), $(int64_t _dim), *$(at::Tensor* _index), *$(at::Tensor* _value))); }|]

scatter__tltt :: Ptr Tensor -> Int64 -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
scatter__tltt _self _dim _index _src = [C.block| at::Tensor* { return new at::Tensor(at::native::scatter_(*$(at::Tensor* _self), $(int64_t _dim), *$(at::Tensor* _index), *$(at::Tensor* _src))); }|]

scatter__tlts :: Ptr Tensor -> Int64 -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
scatter__tlts _self _dim _index _value = [C.block| at::Tensor* { return new at::Tensor(at::native::scatter_(*$(at::Tensor* _self), $(int64_t _dim), *$(at::Tensor* _index), *$(at::Scalar* _value))); }|]

scatter_add__tltt :: Ptr Tensor -> Int64 -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
scatter_add__tltt _self _dim _index _src = [C.block| at::Tensor* { return new at::Tensor(at::native::scatter_add_(*$(at::Tensor* _self), $(int64_t _dim), *$(at::Tensor* _index), *$(at::Tensor* _src))); }|]

lt__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
lt__ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::lt_(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

lt__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
lt__tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::lt_(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

gt__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
gt__ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::gt_(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

gt__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
gt__tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::gt_(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

le__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
le__ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::le_(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

le__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
le__tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::le_(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

ge__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
ge__ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::ge_(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

ge__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
ge__tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::ge_(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

eq__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
eq__ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::eq_(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

eq__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
eq__tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::eq_(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

ne__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
ne__ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::ne_(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

ne__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
ne__tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::ne_(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

__and___ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
__and___ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__and__(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

__and___tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
__and___tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__and__(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

__iand___ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
__iand___ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__iand__(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

__iand___tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
__iand___tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__iand__(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

__or___ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
__or___ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__or__(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

__or___tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
__or___tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__or__(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

__ior___ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
__ior___ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__ior__(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

__ior___tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
__ior___tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__ior__(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

__xor___ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
__xor___ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__xor__(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

__xor___tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
__xor___tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__xor__(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

__ixor___ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
__ixor___ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__ixor__(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

__ixor___tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
__ixor___tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__ixor__(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

__lshift___ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
__lshift___ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__lshift__(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

__lshift___tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
__lshift___tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__lshift__(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

__ilshift___ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
__ilshift___ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__ilshift__(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

__ilshift___tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
__ilshift___tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__ilshift__(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

__rshift___ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
__rshift___ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__rshift__(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

__rshift___tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
__rshift___tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__rshift__(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

__irshift___ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
__irshift___ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__irshift__(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

__irshift___tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
__irshift___tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::__irshift__(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

lgamma__t :: Ptr Tensor -> IO (Ptr Tensor)
lgamma__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::lgamma_(*$(at::Tensor* _self))); }|]

atan2__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
atan2__tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::atan2_(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

tril_cpu__tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
tril_cpu__tl _self _diagonal = [C.block| at::Tensor* { return new at::Tensor(at::native::tril_cpu_(*$(at::Tensor* _self), $(int64_t _diagonal))); }|]

tril_cuda__tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
tril_cuda__tl _self _diagonal = [C.block| at::Tensor* { return new at::Tensor(at::native::tril_cuda_(*$(at::Tensor* _self), $(int64_t _diagonal))); }|]

triu_cpu__tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
triu_cpu__tl _self _diagonal = [C.block| at::Tensor* { return new at::Tensor(at::native::triu_cpu_(*$(at::Tensor* _self), $(int64_t _diagonal))); }|]

triu_cuda__tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
triu_cuda__tl _self _diagonal = [C.block| at::Tensor* { return new at::Tensor(at::native::triu_cuda_(*$(at::Tensor* _self), $(int64_t _diagonal))); }|]

digamma__t :: Ptr Tensor -> IO (Ptr Tensor)
digamma__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::digamma_(*$(at::Tensor* _self))); }|]

polygamma__tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
polygamma__tl _self _n = [C.block| at::Tensor* { return new at::Tensor(at::native::polygamma_(*$(at::Tensor* _self), $(int64_t _n))); }|]

erfinv__t :: Ptr Tensor -> IO (Ptr Tensor)
erfinv__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::erfinv_(*$(at::Tensor* _self))); }|]

frac__t :: Ptr Tensor -> IO (Ptr Tensor)
frac__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::frac_(*$(at::Tensor* _self))); }|]

renorm__tsls :: Ptr Tensor -> Ptr Scalar -> Int64 -> Ptr Scalar -> IO (Ptr Tensor)
renorm__tsls _self _p _dim _maxnorm = [C.block| at::Tensor* { return new at::Tensor(at::native::renorm_(*$(at::Tensor* _self), *$(at::Scalar* _p), $(int64_t _dim), *$(at::Scalar* _maxnorm))); }|]

reciprocal__t :: Ptr Tensor -> IO (Ptr Tensor)
reciprocal__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::reciprocal_(*$(at::Tensor* _self))); }|]

neg__t :: Ptr Tensor -> IO (Ptr Tensor)
neg__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::neg_(*$(at::Tensor* _self))); }|]

pow__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
pow__ts _self _exponent = [C.block| at::Tensor* { return new at::Tensor(at::native::pow_(*$(at::Tensor* _self), *$(at::Scalar* _exponent))); }|]

pow__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
pow__tt _self _exponent = [C.block| at::Tensor* { return new at::Tensor(at::native::pow_(*$(at::Tensor* _self), *$(at::Tensor* _exponent))); }|]

lerp__tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
lerp__tts _self _end _weight = [C.block| at::Tensor* { return new at::Tensor(at::native::lerp_(*$(at::Tensor* _self), *$(at::Tensor* _end), *$(at::Scalar* _weight))); }|]

sign__t :: Ptr Tensor -> IO (Ptr Tensor)
sign__t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::sign_(*$(at::Tensor* _self))); }|]

fmod__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
fmod__ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::fmod_(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

fmod__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
fmod__tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::fmod_(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

remainder__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
remainder__ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::remainder_(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

remainder__tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
remainder__tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::remainder_(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

addbmm__tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
addbmm__tttss _self _batch1 _batch2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::addbmm_(*$(at::Tensor* _self), *$(at::Tensor* _batch1), *$(at::Tensor* _batch2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

addbmm_out_ttttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
addbmm_out_ttttss _result _self _batch1 _batch2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::addbmm_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _batch1), *$(at::Tensor* _batch2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

addbmm_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
addbmm_tttss _self _batch1 _batch2 _beta _alpha = [C.block| at::Tensor* { return new at::Tensor(at::native::addbmm(*$(at::Tensor* _self), *$(at::Tensor* _batch1), *$(at::Tensor* _batch2), *$(at::Scalar* _beta), *$(at::Scalar* _alpha))); }|]

addcmul__ttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
addcmul__ttts _self _tensor1 _tensor2 _value = [C.block| at::Tensor* { return new at::Tensor(at::native::addcmul_(*$(at::Tensor* _self), *$(at::Tensor* _tensor1), *$(at::Tensor* _tensor2), *$(at::Scalar* _value))); }|]

addcdiv__ttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
addcdiv__ttts _self _tensor1 _tensor2 _value = [C.block| at::Tensor* { return new at::Tensor(at::native::addcdiv_(*$(at::Tensor* _self), *$(at::Tensor* _tensor1), *$(at::Tensor* _tensor2), *$(at::Scalar* _value))); }|]

random__tllp :: Ptr Tensor -> Int64 -> Int64 -> Ptr Generator -> IO (Ptr Tensor)
random__tllp _self _from _to _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::random_(*$(at::Tensor* _self), $(int64_t _from), $(int64_t _to), $(at::Generator * _generator))); }|]

random__tlp :: Ptr Tensor -> Int64 -> Ptr Generator -> IO (Ptr Tensor)
random__tlp _self _to _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::random_(*$(at::Tensor* _self), $(int64_t _to), $(at::Generator * _generator))); }|]

random__tp :: Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
random__tp _self _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::random_(*$(at::Tensor* _self), $(at::Generator * _generator))); }|]

uniform__tddp :: Ptr Tensor -> CDouble -> CDouble -> Ptr Generator -> IO (Ptr Tensor)
uniform__tddp _self _from _to _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::uniform_(*$(at::Tensor* _self), $(double _from), $(double _to), $(at::Generator * _generator))); }|]

normal__tddp :: Ptr Tensor -> CDouble -> CDouble -> Ptr Generator -> IO (Ptr Tensor)
normal__tddp _self _mean _std _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::normal_(*$(at::Tensor* _self), $(double _mean), $(double _std), $(at::Generator * _generator))); }|]

cauchy__tddp :: Ptr Tensor -> CDouble -> CDouble -> Ptr Generator -> IO (Ptr Tensor)
cauchy__tddp _self _median _sigma _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::cauchy_(*$(at::Tensor* _self), $(double _median), $(double _sigma), $(at::Generator * _generator))); }|]

log_normal__tddp :: Ptr Tensor -> CDouble -> CDouble -> Ptr Generator -> IO (Ptr Tensor)
log_normal__tddp _self _mean _std _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::log_normal_(*$(at::Tensor* _self), $(double _mean), $(double _std), $(at::Generator * _generator))); }|]

exponential__tdp :: Ptr Tensor -> CDouble -> Ptr Generator -> IO (Ptr Tensor)
exponential__tdp _self _lambd _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::exponential_(*$(at::Tensor* _self), $(double _lambd), $(at::Generator * _generator))); }|]

geometric__tdp :: Ptr Tensor -> CDouble -> Ptr Generator -> IO (Ptr Tensor)
geometric__tdp _self _p _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::geometric_(*$(at::Tensor* _self), $(double _p), $(at::Generator * _generator))); }|]

diag_out_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
diag_out_ttl _result _self _diagonal = [C.block| at::Tensor* { return new at::Tensor(at::native::diag_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _diagonal))); }|]

diag_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
diag_tl _self _diagonal = [C.block| at::Tensor* { return new at::Tensor(at::native::diag(*$(at::Tensor* _self), $(int64_t _diagonal))); }|]

cross_out_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
cross_out_tttl _result _self _other _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::cross_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other), $(int64_t _dim))); }|]

cross_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
cross_ttl _self _other _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::cross(*$(at::Tensor* _self), *$(at::Tensor* _other), $(int64_t _dim))); }|]

triu_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
triu_tl _self _diagonal = [C.block| at::Tensor* { return new at::Tensor(at::native::triu(*$(at::Tensor* _self), $(int64_t _diagonal))); }|]

tril_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
tril_tl _self _diagonal = [C.block| at::Tensor* { return new at::Tensor(at::native::tril(*$(at::Tensor* _self), $(int64_t _diagonal))); }|]

tril_indices_cpu_lllo :: Int64 -> Int64 -> Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
tril_indices_cpu_lllo _row _col _offset _options = [C.block| at::Tensor* { return new at::Tensor(at::native::tril_indices_cpu($(int64_t _row), $(int64_t _col), $(int64_t _offset), *$(at::TensorOptions* _options))); }|]

tril_indices_cuda_lllo :: Int64 -> Int64 -> Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
tril_indices_cuda_lllo _row _col _offset _options = [C.block| at::Tensor* { return new at::Tensor(at::native::tril_indices_cuda($(int64_t _row), $(int64_t _col), $(int64_t _offset), *$(at::TensorOptions* _options))); }|]

triu_indices_cpu_lllo :: Int64 -> Int64 -> Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
triu_indices_cpu_lllo _row _col _offset _options = [C.block| at::Tensor* { return new at::Tensor(at::native::triu_indices_cpu($(int64_t _row), $(int64_t _col), $(int64_t _offset), *$(at::TensorOptions* _options))); }|]

triu_indices_cuda_lllo :: Int64 -> Int64 -> Int64 -> Ptr TensorOptions -> IO (Ptr Tensor)
triu_indices_cuda_lllo _row _col _offset _options = [C.block| at::Tensor* { return new at::Tensor(at::native::triu_indices_cuda($(int64_t _row), $(int64_t _col), $(int64_t _offset), *$(at::TensorOptions* _options))); }|]

trace_t :: Ptr Tensor -> IO (Ptr Tensor)
trace_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::trace(*$(at::Tensor* _self))); }|]

ne_out_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
ne_out_tts _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::ne_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

ne_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
ne_ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::ne(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

ne_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
ne_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::ne_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

ne_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
ne_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::ne(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

eq_out_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
eq_out_tts _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::eq_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

eq_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
eq_ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::eq(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

eq_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
eq_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::eq_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

eq_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
eq_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::eq(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

ge_out_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
ge_out_tts _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::ge_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

ge_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
ge_ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::ge(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

ge_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
ge_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::ge_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

ge_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
ge_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::ge(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

le_out_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
le_out_tts _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::le_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

le_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
le_ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::le(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

le_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
le_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::le_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

le_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
le_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::le(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

gt_out_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
gt_out_tts _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::gt_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

gt_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
gt_ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::gt(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

gt_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
gt_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::gt_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

gt_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
gt_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::gt(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

lt_out_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
lt_out_tts _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::lt_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

lt_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
lt_ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::lt(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

lt_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
lt_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::lt_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

lt_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
lt_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::lt(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

take_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
take_out_ttt _result _self _index = [C.block| at::Tensor* { return new at::Tensor(at::native::take_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _index))); }|]

take_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
take_tt _self _index = [C.block| at::Tensor* { return new at::Tensor(at::native::take(*$(at::Tensor* _self), *$(at::Tensor* _index))); }|]

index_select_out_ttlt :: Ptr Tensor -> Ptr Tensor -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
index_select_out_ttlt _result _self _dim _index = [C.block| at::Tensor* { return new at::Tensor(at::native::index_select_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim), *$(at::Tensor* _index))); }|]

index_select_tlt :: Ptr Tensor -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
index_select_tlt _self _dim _index = [C.block| at::Tensor* { return new at::Tensor(at::native::index_select(*$(at::Tensor* _self), $(int64_t _dim), *$(at::Tensor* _index))); }|]

masked_select_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
masked_select_out_ttt _result _self _mask = [C.block| at::Tensor* { return new at::Tensor(at::native::masked_select_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _mask))); }|]

masked_select_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
masked_select_tt _self _mask = [C.block| at::Tensor* { return new at::Tensor(at::native::masked_select(*$(at::Tensor* _self), *$(at::Tensor* _mask))); }|]

nonzero_out_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
nonzero_out_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::nonzero_out(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

nonzero_t :: Ptr Tensor -> IO (Ptr Tensor)
nonzero_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::nonzero(*$(at::Tensor* _self))); }|]

gather_out_ttlt :: Ptr Tensor -> Ptr Tensor -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
gather_out_ttlt _result _self _dim _index = [C.block| at::Tensor* { return new at::Tensor(at::native::gather_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _dim), *$(at::Tensor* _index))); }|]

gather_tlt :: Ptr Tensor -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
gather_tlt _self _dim _index = [C.block| at::Tensor* { return new at::Tensor(at::native::gather(*$(at::Tensor* _self), $(int64_t _dim), *$(at::Tensor* _index))); }|]

addcmul_out_tttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
addcmul_out_tttts _result _self _tensor1 _tensor2 _value = [C.block| at::Tensor* { return new at::Tensor(at::native::addcmul_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _tensor1), *$(at::Tensor* _tensor2), *$(at::Scalar* _value))); }|]

addcmul_ttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
addcmul_ttts _self _tensor1 _tensor2 _value = [C.block| at::Tensor* { return new at::Tensor(at::native::addcmul(*$(at::Tensor* _self), *$(at::Tensor* _tensor1), *$(at::Tensor* _tensor2), *$(at::Scalar* _value))); }|]

addcdiv_out_tttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
addcdiv_out_tttts _result _self _tensor1 _tensor2 _value = [C.block| at::Tensor* { return new at::Tensor(at::native::addcdiv_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _tensor1), *$(at::Tensor* _tensor2), *$(at::Scalar* _value))); }|]

addcdiv_ttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
addcdiv_ttts _self _tensor1 _tensor2 _value = [C.block| at::Tensor* { return new at::Tensor(at::native::addcdiv(*$(at::Tensor* _self), *$(at::Tensor* _tensor1), *$(at::Tensor* _tensor2), *$(at::Scalar* _value))); }|]

gels_out_tttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
gels_out_tttt _X _qr _self _A = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::gels_out(*$(at::Tensor* _X), *$(at::Tensor* _qr), *$(at::Tensor* _self), *$(at::Tensor* _A))); }|]

gels_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
gels_tt _self _A = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::gels(*$(at::Tensor* _self), *$(at::Tensor* _A))); }|]

trtrs_out_ttttbbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
trtrs_out_ttttbbb _X _M _self _A _upper _transpose _unitriangular = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::trtrs_out(*$(at::Tensor* _X), *$(at::Tensor* _M), *$(at::Tensor* _self), *$(at::Tensor* _A), $(bool _upper), $(bool _transpose), $(bool _unitriangular))); }|]

trtrs_ttbbb :: Ptr Tensor -> Ptr Tensor -> CBool -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
trtrs_ttbbb _self _A _upper _transpose _unitriangular = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::trtrs(*$(at::Tensor* _self), *$(at::Tensor* _A), $(bool _upper), $(bool _transpose), $(bool _unitriangular))); }|]

symeig_out_tttbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
symeig_out_tttbb _e _V _self _eigenvectors _upper = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::symeig_out(*$(at::Tensor* _e), *$(at::Tensor* _V), *$(at::Tensor* _self), $(bool _eigenvectors), $(bool _upper))); }|]

symeig_tbb :: Ptr Tensor -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
symeig_tbb _self _eigenvectors _upper = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::symeig(*$(at::Tensor* _self), $(bool _eigenvectors), $(bool _upper))); }|]

eig_out_tttb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr (Tensor,Tensor))
eig_out_tttb _e _v _self _eigenvectors = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::eig_out(*$(at::Tensor* _e), *$(at::Tensor* _v), *$(at::Tensor* _self), $(bool _eigenvectors))); }|]

eig_tb :: Ptr Tensor -> CBool -> IO (Ptr (Tensor,Tensor))
eig_tb _self _eigenvectors = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::eig(*$(at::Tensor* _self), $(bool _eigenvectors))); }|]

svd_out_ttttbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CBool -> IO (Ptr (Tensor,Tensor,Tensor))
svd_out_ttttbb _U _S _V _self _some _compute_uv = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::svd_out(*$(at::Tensor* _U), *$(at::Tensor* _S), *$(at::Tensor* _V), *$(at::Tensor* _self), $(bool _some), $(bool _compute_uv))); }|]

svd_tbb :: Ptr Tensor -> CBool -> CBool -> IO (Ptr (Tensor,Tensor,Tensor))
svd_tbb _self _some _compute_uv = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::svd(*$(at::Tensor* _self), $(bool _some), $(bool _compute_uv))); }|]

cholesky_out_ttb :: Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
cholesky_out_ttb _result _self _upper = [C.block| at::Tensor* { return new at::Tensor(at::native::cholesky_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(bool _upper))); }|]

cholesky_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
cholesky_tb _self _upper = [C.block| at::Tensor* { return new at::Tensor(at::native::cholesky(*$(at::Tensor* _self), $(bool _upper))); }|]

_cholesky_helper_cpu_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
_cholesky_helper_cpu_tb _self _upper = [C.block| at::Tensor* { return new at::Tensor(at::native::_cholesky_helper_cpu(*$(at::Tensor* _self), $(bool _upper))); }|]

_cholesky_helper_cuda_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
_cholesky_helper_cuda_tb _self _upper = [C.block| at::Tensor* { return new at::Tensor(at::native::_cholesky_helper_cuda(*$(at::Tensor* _self), $(bool _upper))); }|]

cholesky_solve_out_tttb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
cholesky_solve_out_tttb _result _self _input2 _upper = [C.block| at::Tensor* { return new at::Tensor(at::native::cholesky_solve_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _input2), $(bool _upper))); }|]

cholesky_solve_ttb :: Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
cholesky_solve_ttb _self _input2 _upper = [C.block| at::Tensor* { return new at::Tensor(at::native::cholesky_solve(*$(at::Tensor* _self), *$(at::Tensor* _input2), $(bool _upper))); }|]

_cholesky_solve_helper_cpu_ttb :: Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
_cholesky_solve_helper_cpu_ttb _self _A _upper = [C.block| at::Tensor* { return new at::Tensor(at::native::_cholesky_solve_helper_cpu(*$(at::Tensor* _self), *$(at::Tensor* _A), $(bool _upper))); }|]

_cholesky_solve_helper_cuda_ttb :: Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
_cholesky_solve_helper_cuda_ttb _self _A _upper = [C.block| at::Tensor* { return new at::Tensor(at::native::_cholesky_solve_helper_cuda(*$(at::Tensor* _self), *$(at::Tensor* _A), $(bool _upper))); }|]

potri_out_ttb :: Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr Tensor)
potri_out_ttb _result _self _upper = [C.block| at::Tensor* { return new at::Tensor(at::native::potri_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(bool _upper))); }|]

potri_tb :: Ptr Tensor -> CBool -> IO (Ptr Tensor)
potri_tb _self _upper = [C.block| at::Tensor* { return new at::Tensor(at::native::potri(*$(at::Tensor* _self), $(bool _upper))); }|]

pstrf_out_tttbs :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> Ptr Scalar -> IO (Ptr (Tensor,Tensor))
pstrf_out_tttbs _u _piv _self _upper _tol = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::pstrf_out(*$(at::Tensor* _u), *$(at::Tensor* _piv), *$(at::Tensor* _self), $(bool _upper), *$(at::Scalar* _tol))); }|]

pstrf_tbs :: Ptr Tensor -> CBool -> Ptr Scalar -> IO (Ptr (Tensor,Tensor))
pstrf_tbs _self _upper _tol = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::pstrf(*$(at::Tensor* _self), $(bool _upper), *$(at::Scalar* _tol))); }|]

qr_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
qr_out_ttt _Q _R _self = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::qr_out(*$(at::Tensor* _Q), *$(at::Tensor* _R), *$(at::Tensor* _self))); }|]

qr_t :: Ptr Tensor -> IO (Ptr (Tensor,Tensor))
qr_t _self = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::qr(*$(at::Tensor* _self))); }|]

geqrf_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
geqrf_out_ttt _result0 _result1 _self = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::geqrf_out(*$(at::Tensor* _result0), *$(at::Tensor* _result1), *$(at::Tensor* _self))); }|]

geqrf_t :: Ptr Tensor -> IO (Ptr (Tensor,Tensor))
geqrf_t _self = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::geqrf(*$(at::Tensor* _self))); }|]

orgqr_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
orgqr_out_ttt _result _self _input2 = [C.block| at::Tensor* { return new at::Tensor(at::native::orgqr_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _input2))); }|]

orgqr_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
orgqr_tt _self _input2 = [C.block| at::Tensor* { return new at::Tensor(at::native::orgqr(*$(at::Tensor* _self), *$(at::Tensor* _input2))); }|]

ormqr_out_ttttbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CBool -> IO (Ptr Tensor)
ormqr_out_ttttbb _result _self _input2 _input3 _left _transpose = [C.block| at::Tensor* { return new at::Tensor(at::native::ormqr_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _input2), *$(at::Tensor* _input3), $(bool _left), $(bool _transpose))); }|]

ormqr_tttbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> CBool -> IO (Ptr Tensor)
ormqr_tttbb _self _input2 _input3 _left _transpose = [C.block| at::Tensor* { return new at::Tensor(at::native::ormqr(*$(at::Tensor* _self), *$(at::Tensor* _input2), *$(at::Tensor* _input3), $(bool _left), $(bool _transpose))); }|]

btrifact_out_tttb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr (Tensor,Tensor))
btrifact_out_tttb _A_LU _pivots _self _pivot = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::btrifact_out(*$(at::Tensor* _A_LU), *$(at::Tensor* _pivots), *$(at::Tensor* _self), $(bool _pivot))); }|]

btrifact_tb :: Ptr Tensor -> CBool -> IO (Ptr (Tensor,Tensor))
btrifact_tb _self _pivot = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::btrifact(*$(at::Tensor* _self), $(bool _pivot))); }|]

btrifact_with_info_out_ttttb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> CBool -> IO (Ptr (Tensor,Tensor,Tensor))
btrifact_with_info_out_ttttb _A_LU _pivots _info _self _pivot = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::btrifact_with_info_out(*$(at::Tensor* _A_LU), *$(at::Tensor* _pivots), *$(at::Tensor* _info), *$(at::Tensor* _self), $(bool _pivot))); }|]

btrifact_with_info_tb :: Ptr Tensor -> CBool -> IO (Ptr (Tensor,Tensor,Tensor))
btrifact_with_info_tb _self _pivot = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::btrifact_with_info(*$(at::Tensor* _self), $(bool _pivot))); }|]

btrisolve_out_tttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
btrisolve_out_tttt _result _self _LU_data _LU_pivots = [C.block| at::Tensor* { return new at::Tensor(at::native::btrisolve_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _LU_data), *$(at::Tensor* _LU_pivots))); }|]

btrisolve_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
btrisolve_ttt _self _LU_data _LU_pivots = [C.block| at::Tensor* { return new at::Tensor(at::native::btrisolve(*$(at::Tensor* _self), *$(at::Tensor* _LU_data), *$(at::Tensor* _LU_pivots))); }|]

multinomial_out_ttlbp :: Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> Ptr Generator -> IO (Ptr Tensor)
multinomial_out_ttlbp _result _self _num_samples _replacement _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::multinomial_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _num_samples), $(bool _replacement), $(at::Generator * _generator))); }|]

multinomial_tlbp :: Ptr Tensor -> Int64 -> CBool -> Ptr Generator -> IO (Ptr Tensor)
multinomial_tlbp _self _num_samples _replacement _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::multinomial(*$(at::Tensor* _self), $(int64_t _num_samples), $(bool _replacement), $(at::Generator * _generator))); }|]

lgamma_out_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
lgamma_out_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::lgamma_out(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

lgamma_t :: Ptr Tensor -> IO (Ptr Tensor)
lgamma_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::lgamma(*$(at::Tensor* _self))); }|]

digamma_out_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
digamma_out_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::digamma_out(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

digamma_t :: Ptr Tensor -> IO (Ptr Tensor)
digamma_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::digamma(*$(at::Tensor* _self))); }|]

polygamma_out_tlt :: Ptr Tensor -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
polygamma_out_tlt _result _n _self = [C.block| at::Tensor* { return new at::Tensor(at::native::polygamma_out(*$(at::Tensor* _result), $(int64_t _n), *$(at::Tensor* _self))); }|]

polygamma_lt :: Int64 -> Ptr Tensor -> IO (Ptr Tensor)
polygamma_lt _n _self = [C.block| at::Tensor* { return new at::Tensor(at::native::polygamma($(int64_t _n), *$(at::Tensor* _self))); }|]

erfinv_out_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
erfinv_out_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::erfinv_out(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

erfinv_t :: Ptr Tensor -> IO (Ptr Tensor)
erfinv_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::erfinv(*$(at::Tensor* _self))); }|]

frac_out_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
frac_out_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::frac_out(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

frac_t :: Ptr Tensor -> IO (Ptr Tensor)
frac_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::frac(*$(at::Tensor* _self))); }|]

dist_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
dist_tts _self _other _p = [C.block| at::Tensor* { return new at::Tensor(at::native::dist(*$(at::Tensor* _self), *$(at::Tensor* _other), *$(at::Scalar* _p))); }|]

reciprocal_out_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
reciprocal_out_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::reciprocal_out(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

reciprocal_t :: Ptr Tensor -> IO (Ptr Tensor)
reciprocal_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::reciprocal(*$(at::Tensor* _self))); }|]

neg_out_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
neg_out_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::neg_out(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

neg_t :: Ptr Tensor -> IO (Ptr Tensor)
neg_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::neg(*$(at::Tensor* _self))); }|]

atan2_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
atan2_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::atan2_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

atan2_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
atan2_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::atan2(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

lerp_out_ttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
lerp_out_ttts _result _self _end _weight = [C.block| at::Tensor* { return new at::Tensor(at::native::lerp_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _end), *$(at::Scalar* _weight))); }|]

lerp_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
lerp_tts _self _end _weight = [C.block| at::Tensor* { return new at::Tensor(at::native::lerp(*$(at::Tensor* _self), *$(at::Tensor* _end), *$(at::Scalar* _weight))); }|]

histc_out_ttlss :: Ptr Tensor -> Ptr Tensor -> Int64 -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
histc_out_ttlss _result _self _bins _min _max = [C.block| at::Tensor* { return new at::Tensor(at::native::histc_out(*$(at::Tensor* _result), *$(at::Tensor* _self), $(int64_t _bins), *$(at::Scalar* _min), *$(at::Scalar* _max))); }|]

histc_tlss :: Ptr Tensor -> Int64 -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
histc_tlss _self _bins _min _max = [C.block| at::Tensor* { return new at::Tensor(at::native::histc(*$(at::Tensor* _self), $(int64_t _bins), *$(at::Scalar* _min), *$(at::Scalar* _max))); }|]

sign_out_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
sign_out_tt _result _self = [C.block| at::Tensor* { return new at::Tensor(at::native::sign_out(*$(at::Tensor* _result), *$(at::Tensor* _self))); }|]

sign_t :: Ptr Tensor -> IO (Ptr Tensor)
sign_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::sign(*$(at::Tensor* _self))); }|]

fmod_out_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
fmod_out_tts _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::fmod_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

fmod_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
fmod_ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::fmod(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

fmod_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
fmod_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::fmod_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

fmod_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
fmod_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::fmod(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

remainder_out_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
remainder_out_tts _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::remainder_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

remainder_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
remainder_ts _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::remainder(*$(at::Tensor* _self), *$(at::Scalar* _other))); }|]

remainder_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
remainder_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::remainder_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

remainder_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
remainder_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::remainder(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

min_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
min_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::min_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

min_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
min_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::min(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

min_t :: Ptr Tensor -> IO (Ptr Tensor)
min_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::min(*$(at::Tensor* _self))); }|]

max_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
max_out_ttt _result _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::max_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

max_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
max_tt _self _other = [C.block| at::Tensor* { return new at::Tensor(at::native::max(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

max_t :: Ptr Tensor -> IO (Ptr Tensor)
max_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::max(*$(at::Tensor* _self))); }|]

median_t :: Ptr Tensor -> IO (Ptr Tensor)
median_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::median(*$(at::Tensor* _self))); }|]

sort_out_tttlb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
sort_out_tttlb _values _indices _self _dim _descending = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::sort_out(*$(at::Tensor* _values), *$(at::Tensor* _indices), *$(at::Tensor* _self), $(int64_t _dim), $(bool _descending))); }|]

sort_tlb :: Ptr Tensor -> Int64 -> CBool -> IO (Ptr (Tensor,Tensor))
sort_tlb _self _dim _descending = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::sort(*$(at::Tensor* _self), $(int64_t _dim), $(bool _descending))); }|]

topk_out_tttllbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
topk_out_tttllbb _values _indices _self _k _dim _largest _sorted = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::topk_out(*$(at::Tensor* _values), *$(at::Tensor* _indices), *$(at::Tensor* _self), $(int64_t _k), $(int64_t _dim), $(bool _largest), $(bool _sorted))); }|]

topk_tllbb :: Ptr Tensor -> Int64 -> Int64 -> CBool -> CBool -> IO (Ptr (Tensor,Tensor))
topk_tllbb _self _k _dim _largest _sorted = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::topk(*$(at::Tensor* _self), $(int64_t _k), $(int64_t _dim), $(bool _largest), $(bool _sorted))); }|]

all_t :: Ptr Tensor -> IO (Ptr Tensor)
all_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::all(*$(at::Tensor* _self))); }|]

any_t :: Ptr Tensor -> IO (Ptr Tensor)
any_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::any(*$(at::Tensor* _self))); }|]

renorm_out_ttsls :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Int64 -> Ptr Scalar -> IO (Ptr Tensor)
renorm_out_ttsls _result _self _p _dim _maxnorm = [C.block| at::Tensor* { return new at::Tensor(at::native::renorm_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Scalar* _p), $(int64_t _dim), *$(at::Scalar* _maxnorm))); }|]

renorm_tsls :: Ptr Tensor -> Ptr Scalar -> Int64 -> Ptr Scalar -> IO (Ptr Tensor)
renorm_tsls _self _p _dim _maxnorm = [C.block| at::Tensor* { return new at::Tensor(at::native::renorm(*$(at::Tensor* _self), *$(at::Scalar* _p), $(int64_t _dim), *$(at::Scalar* _maxnorm))); }|]

unfold_tlll :: Ptr Tensor -> Int64 -> Int64 -> Int64 -> IO (Ptr Tensor)
unfold_tlll _self _dimension _size _step = [C.block| at::Tensor* { return new at::Tensor(at::native::unfold(*$(at::Tensor* _self), $(int64_t _dimension), $(int64_t _size), $(int64_t _step))); }|]

equal_tt :: Ptr Tensor -> Ptr Tensor -> IO (CBool)
equal_tt _self _other = [C.block| bool { return (at::native::equal(*$(at::Tensor* _self), *$(at::Tensor* _other))); }|]

pow_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
pow_out_ttt _result _self _exponent = [C.block| at::Tensor* { return new at::Tensor(at::native::pow_out(*$(at::Tensor* _result), *$(at::Tensor* _self), *$(at::Tensor* _exponent))); }|]

pow_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
pow_tt _self _exponent = [C.block| at::Tensor* { return new at::Tensor(at::native::pow(*$(at::Tensor* _self), *$(at::Tensor* _exponent))); }|]

pow_out_tst :: Ptr Tensor -> Ptr Scalar -> Ptr Tensor -> IO (Ptr Tensor)
pow_out_tst _result _self _exponent = [C.block| at::Tensor* { return new at::Tensor(at::native::pow_out(*$(at::Tensor* _result), *$(at::Scalar* _self), *$(at::Tensor* _exponent))); }|]

pow_st :: Ptr Scalar -> Ptr Tensor -> IO (Ptr Tensor)
pow_st _self _exponent = [C.block| at::Tensor* { return new at::Tensor(at::native::pow(*$(at::Scalar* _self), *$(at::Tensor* _exponent))); }|]

normal_out_ttdp :: Ptr Tensor -> Ptr Tensor -> CDouble -> Ptr Generator -> IO (Ptr Tensor)
normal_out_ttdp _output _mean _std _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::normal_out(*$(at::Tensor* _output), *$(at::Tensor* _mean), $(double _std), $(at::Generator * _generator))); }|]

normal_tdp :: Ptr Tensor -> CDouble -> Ptr Generator -> IO (Ptr Tensor)
normal_tdp _mean _std _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::normal(*$(at::Tensor* _mean), $(double _std), $(at::Generator * _generator))); }|]

normal_out_tdtp :: Ptr Tensor -> CDouble -> Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
normal_out_tdtp _output _mean _std _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::normal_out(*$(at::Tensor* _output), $(double _mean), *$(at::Tensor* _std), $(at::Generator * _generator))); }|]

normal_dtp :: CDouble -> Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
normal_dtp _mean _std _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::normal($(double _mean), *$(at::Tensor* _std), $(at::Generator * _generator))); }|]

normal_out_tttp :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
normal_out_tttp _output _mean _std _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::normal_out(*$(at::Tensor* _output), *$(at::Tensor* _mean), *$(at::Tensor* _std), $(at::Generator * _generator))); }|]

normal_ttp :: Ptr Tensor -> Ptr Tensor -> Ptr Generator -> IO (Ptr Tensor)
normal_ttp _mean _std _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::normal(*$(at::Tensor* _mean), *$(at::Tensor* _std), $(at::Generator * _generator))); }|]

alias_t :: Ptr Tensor -> IO (Ptr Tensor)
alias_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::alias(*$(at::Tensor* _self))); }|]

_dirichlet_grad_out_tttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_dirichlet_grad_out_tttt _output _x _alpha _total = [C.block| at::Tensor* { return new at::Tensor(at::native::_dirichlet_grad_out(*$(at::Tensor* _output), *$(at::Tensor* _x), *$(at::Tensor* _alpha), *$(at::Tensor* _total))); }|]

_dirichlet_grad_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
_dirichlet_grad_ttt _x _alpha _total = [C.block| at::Tensor* { return new at::Tensor(at::native::_dirichlet_grad(*$(at::Tensor* _x), *$(at::Tensor* _alpha), *$(at::Tensor* _total))); }|]

binary_cross_entropy_out_ttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
binary_cross_entropy_out_ttttl _output _self _target _weight _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::binary_cross_entropy_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction))); }|]

binary_cross_entropy_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
binary_cross_entropy_tttl _self _target _weight _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::binary_cross_entropy(*$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction))); }|]

binary_cross_entropy_backward_out_tttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
binary_cross_entropy_backward_out_tttttl _grad_input _grad_output _self _target _weight _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::binary_cross_entropy_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction))); }|]

binary_cross_entropy_backward_ttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
binary_cross_entropy_backward_ttttl _grad_output _self _target _weight _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::binary_cross_entropy_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction))); }|]

mse_loss_out_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
mse_loss_out_tttl _output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::mse_loss_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

mse_loss_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
mse_loss_ttl _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::mse_loss(*$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

mse_loss_backward_out_ttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
mse_loss_backward_out_ttttl _grad_input _grad_output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::mse_loss_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

mse_loss_backward_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
mse_loss_backward_tttl _grad_output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::mse_loss_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

l1_loss_out_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
l1_loss_out_tttl _output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::l1_loss_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

l1_loss_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
l1_loss_ttl _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::l1_loss(*$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

l1_loss_backward_out_ttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
l1_loss_backward_out_ttttl _grad_input _grad_output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::l1_loss_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

l1_loss_backward_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
l1_loss_backward_tttl _grad_output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::l1_loss_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

multi_margin_loss_out_tttsstl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
multi_margin_loss_out_tttsstl _output _self _target _p _margin _weight _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::multi_margin_loss_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Scalar* _p), *$(at::Scalar* _margin), *$(at::Tensor* _weight), $(int64_t _reduction))); }|]

multi_margin_loss_ttsstl :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
multi_margin_loss_ttsstl _self _target _p _margin _weight _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::multi_margin_loss(*$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Scalar* _p), *$(at::Scalar* _margin), *$(at::Tensor* _weight), $(int64_t _reduction))); }|]

multi_margin_loss_backward_out_ttttsstl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
multi_margin_loss_backward_out_ttttsstl _grad_input _grad_output _self _target _p _margin _weight _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::multi_margin_loss_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Scalar* _p), *$(at::Scalar* _margin), *$(at::Tensor* _weight), $(int64_t _reduction))); }|]

multi_margin_loss_backward_tttsstl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
multi_margin_loss_backward_tttsstl _grad_output _self _target _p _margin _weight _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::multi_margin_loss_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Scalar* _p), *$(at::Scalar* _margin), *$(at::Tensor* _weight), $(int64_t _reduction))); }|]

multilabel_margin_loss_out_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
multilabel_margin_loss_out_tttl _output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::multilabel_margin_loss_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

multilabel_margin_loss_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
multilabel_margin_loss_ttl _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::multilabel_margin_loss(*$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

multilabel_margin_loss_forward_out_ttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr (Tensor,Tensor))
multilabel_margin_loss_forward_out_ttttl _output _is_target _self _target _reduction = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::multilabel_margin_loss_forward_out(*$(at::Tensor* _output), *$(at::Tensor* _is_target), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

multilabel_margin_loss_forward_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr (Tensor,Tensor))
multilabel_margin_loss_forward_ttl _self _target _reduction = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::multilabel_margin_loss_forward(*$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

multilabel_margin_loss_backward_out_ttttlt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
multilabel_margin_loss_backward_out_ttttlt _grad_input _grad_output _self _target _reduction _is_target = [C.block| at::Tensor* { return new at::Tensor(at::native::multilabel_margin_loss_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction), *$(at::Tensor* _is_target))); }|]

multilabel_margin_loss_backward_tttlt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
multilabel_margin_loss_backward_tttlt _grad_output _self _target _reduction _is_target = [C.block| at::Tensor* { return new at::Tensor(at::native::multilabel_margin_loss_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction), *$(at::Tensor* _is_target))); }|]

nll_loss_out_ttttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
nll_loss_out_ttttll _output _self _target _weight _reduction _ignore_index = [C.block| at::Tensor* { return new at::Tensor(at::native::nll_loss_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction), $(int64_t _ignore_index))); }|]

nll_loss_tttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
nll_loss_tttll _self _target _weight _reduction _ignore_index = [C.block| at::Tensor* { return new at::Tensor(at::native::nll_loss(*$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction), $(int64_t _ignore_index))); }|]

nll_loss_forward_out_tttttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr (Tensor,Tensor))
nll_loss_forward_out_tttttll _output _total_weight _self _target _weight _reduction _ignore_index = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::nll_loss_forward_out(*$(at::Tensor* _output), *$(at::Tensor* _total_weight), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction), $(int64_t _ignore_index))); }|]

nll_loss_forward_tttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr (Tensor,Tensor))
nll_loss_forward_tttll _self _target _weight _reduction _ignore_index = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::nll_loss_forward(*$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction), $(int64_t _ignore_index))); }|]

nll_loss_backward_out_tttttllt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
nll_loss_backward_out_tttttllt _grad_input _grad_output _self _target _weight _reduction _ignore_index _total_weight = [C.block| at::Tensor* { return new at::Tensor(at::native::nll_loss_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction), $(int64_t _ignore_index), *$(at::Tensor* _total_weight))); }|]

nll_loss_backward_ttttllt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
nll_loss_backward_ttttllt _grad_output _self _target _weight _reduction _ignore_index _total_weight = [C.block| at::Tensor* { return new at::Tensor(at::native::nll_loss_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction), $(int64_t _ignore_index), *$(at::Tensor* _total_weight))); }|]

nll_loss2d_out_ttttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
nll_loss2d_out_ttttll _output _self _target _weight _reduction _ignore_index = [C.block| at::Tensor* { return new at::Tensor(at::native::nll_loss2d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction), $(int64_t _ignore_index))); }|]

nll_loss2d_tttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr Tensor)
nll_loss2d_tttll _self _target _weight _reduction _ignore_index = [C.block| at::Tensor* { return new at::Tensor(at::native::nll_loss2d(*$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction), $(int64_t _ignore_index))); }|]

nll_loss2d_forward_out_tttttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr (Tensor,Tensor))
nll_loss2d_forward_out_tttttll _output _total_weight _self _target _weight _reduction _ignore_index = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::nll_loss2d_forward_out(*$(at::Tensor* _output), *$(at::Tensor* _total_weight), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction), $(int64_t _ignore_index))); }|]

nll_loss2d_forward_tttll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> IO (Ptr (Tensor,Tensor))
nll_loss2d_forward_tttll _self _target _weight _reduction _ignore_index = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::nll_loss2d_forward(*$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction), $(int64_t _ignore_index))); }|]

nll_loss2d_backward_out_tttttllt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
nll_loss2d_backward_out_tttttllt _grad_input _grad_output _self _target _weight _reduction _ignore_index _total_weight = [C.block| at::Tensor* { return new at::Tensor(at::native::nll_loss2d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction), $(int64_t _ignore_index), *$(at::Tensor* _total_weight))); }|]

nll_loss2d_backward_ttttllt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> Int64 -> Ptr Tensor -> IO (Ptr Tensor)
nll_loss2d_backward_ttttllt _grad_output _self _target _weight _reduction _ignore_index _total_weight = [C.block| at::Tensor* { return new at::Tensor(at::native::nll_loss2d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), *$(at::Tensor* _weight), $(int64_t _reduction), $(int64_t _ignore_index), *$(at::Tensor* _total_weight))); }|]

smooth_l1_loss_out_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
smooth_l1_loss_out_tttl _output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::smooth_l1_loss_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

smooth_l1_loss_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
smooth_l1_loss_ttl _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::smooth_l1_loss(*$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

smooth_l1_loss_backward_out_ttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
smooth_l1_loss_backward_out_ttttl _grad_input _grad_output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::smooth_l1_loss_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

smooth_l1_loss_backward_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
smooth_l1_loss_backward_tttl _grad_output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::smooth_l1_loss_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

soft_margin_loss_out_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
soft_margin_loss_out_tttl _output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::soft_margin_loss_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

soft_margin_loss_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
soft_margin_loss_ttl _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::soft_margin_loss(*$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

soft_margin_loss_backward_out_ttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
soft_margin_loss_backward_out_ttttl _grad_input _grad_output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::soft_margin_loss_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

soft_margin_loss_backward_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
soft_margin_loss_backward_tttl _grad_output _self _target _reduction = [C.block| at::Tensor* { return new at::Tensor(at::native::soft_margin_loss_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _target), $(int64_t _reduction))); }|]

elu_out_ttsss :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
elu_out_ttsss _output _self _alpha _scale _input_scale = [C.block| at::Tensor* { return new at::Tensor(at::native::elu_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Scalar* _alpha), *$(at::Scalar* _scale), *$(at::Scalar* _input_scale))); }|]

elu_tsss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
elu_tsss _self _alpha _scale _input_scale = [C.block| at::Tensor* { return new at::Tensor(at::native::elu(*$(at::Tensor* _self), *$(at::Scalar* _alpha), *$(at::Scalar* _scale), *$(at::Scalar* _input_scale))); }|]

elu_backward_out_ttssst :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Scalar -> Ptr Tensor -> IO (Ptr Tensor)
elu_backward_out_ttssst _grad_input _grad_output _alpha _scale _input_scale _output = [C.block| at::Tensor* { return new at::Tensor(at::native::elu_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Scalar* _alpha), *$(at::Scalar* _scale), *$(at::Scalar* _input_scale), *$(at::Tensor* _output))); }|]

elu_backward_tssst :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Scalar -> Ptr Tensor -> IO (Ptr Tensor)
elu_backward_tssst _grad_output _alpha _scale _input_scale _output = [C.block| at::Tensor* { return new at::Tensor(at::native::elu_backward(*$(at::Tensor* _grad_output), *$(at::Scalar* _alpha), *$(at::Scalar* _scale), *$(at::Scalar* _input_scale), *$(at::Tensor* _output))); }|]

elu__tsss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
elu__tsss _self _alpha _scale _input_scale = [C.block| at::Tensor* { return new at::Tensor(at::native::elu_(*$(at::Tensor* _self), *$(at::Scalar* _alpha), *$(at::Scalar* _scale), *$(at::Scalar* _input_scale))); }|]

glu_out_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
glu_out_ttl _output _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::glu_out(*$(at::Tensor* _output), *$(at::Tensor* _self), $(int64_t _dim))); }|]

glu_tl :: Ptr Tensor -> Int64 -> IO (Ptr Tensor)
glu_tl _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::glu(*$(at::Tensor* _self), $(int64_t _dim))); }|]

glu_backward_out_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
glu_backward_out_tttl _grad_input _grad_output _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::glu_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), $(int64_t _dim))); }|]

glu_backward_ttl :: Ptr Tensor -> Ptr Tensor -> Int64 -> IO (Ptr Tensor)
glu_backward_ttl _grad_output _self _dim = [C.block| at::Tensor* { return new at::Tensor(at::native::glu_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), $(int64_t _dim))); }|]

hardtanh_out_ttss :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
hardtanh_out_ttss _output _self _min_val _max_val = [C.block| at::Tensor* { return new at::Tensor(at::native::hardtanh_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Scalar* _min_val), *$(at::Scalar* _max_val))); }|]

hardtanh_tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
hardtanh_tss _self _min_val _max_val = [C.block| at::Tensor* { return new at::Tensor(at::native::hardtanh(*$(at::Tensor* _self), *$(at::Scalar* _min_val), *$(at::Scalar* _max_val))); }|]

hardtanh_backward_out_tttss :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
hardtanh_backward_out_tttss _grad_input _grad_output _self _min_val _max_val = [C.block| at::Tensor* { return new at::Tensor(at::native::hardtanh_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Scalar* _min_val), *$(at::Scalar* _max_val))); }|]

hardtanh_backward_ttss :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
hardtanh_backward_ttss _grad_output _self _min_val _max_val = [C.block| at::Tensor* { return new at::Tensor(at::native::hardtanh_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Scalar* _min_val), *$(at::Scalar* _max_val))); }|]

hardtanh__tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
hardtanh__tss _self _min_val _max_val = [C.block| at::Tensor* { return new at::Tensor(at::native::hardtanh_(*$(at::Tensor* _self), *$(at::Scalar* _min_val), *$(at::Scalar* _max_val))); }|]

leaky_relu_out_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
leaky_relu_out_tts _output _self _negative_slope = [C.block| at::Tensor* { return new at::Tensor(at::native::leaky_relu_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Scalar* _negative_slope))); }|]

leaky_relu_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
leaky_relu_ts _self _negative_slope = [C.block| at::Tensor* { return new at::Tensor(at::native::leaky_relu(*$(at::Tensor* _self), *$(at::Scalar* _negative_slope))); }|]

leaky_relu_backward_out_ttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
leaky_relu_backward_out_ttts _grad_input _grad_output _self _negative_slope = [C.block| at::Tensor* { return new at::Tensor(at::native::leaky_relu_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Scalar* _negative_slope))); }|]

leaky_relu_backward_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
leaky_relu_backward_tts _grad_output _self _negative_slope = [C.block| at::Tensor* { return new at::Tensor(at::native::leaky_relu_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Scalar* _negative_slope))); }|]

leaky_relu__ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
leaky_relu__ts _self _negative_slope = [C.block| at::Tensor* { return new at::Tensor(at::native::leaky_relu_(*$(at::Tensor* _self), *$(at::Scalar* _negative_slope))); }|]

log_sigmoid_out_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
log_sigmoid_out_tt _output _self = [C.block| at::Tensor* { return new at::Tensor(at::native::log_sigmoid_out(*$(at::Tensor* _output), *$(at::Tensor* _self))); }|]

log_sigmoid_t :: Ptr Tensor -> IO (Ptr Tensor)
log_sigmoid_t _self = [C.block| at::Tensor* { return new at::Tensor(at::native::log_sigmoid(*$(at::Tensor* _self))); }|]

log_sigmoid_forward_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
log_sigmoid_forward_out_ttt _output _buffer _self = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::log_sigmoid_forward_out(*$(at::Tensor* _output), *$(at::Tensor* _buffer), *$(at::Tensor* _self))); }|]

log_sigmoid_forward_t :: Ptr Tensor -> IO (Ptr (Tensor,Tensor))
log_sigmoid_forward_t _self = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::log_sigmoid_forward(*$(at::Tensor* _self))); }|]

log_sigmoid_backward_out_tttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
log_sigmoid_backward_out_tttt _grad_input _grad_output _self _buffer = [C.block| at::Tensor* { return new at::Tensor(at::native::log_sigmoid_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _buffer))); }|]

log_sigmoid_backward_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
log_sigmoid_backward_ttt _grad_output _self _buffer = [C.block| at::Tensor* { return new at::Tensor(at::native::log_sigmoid_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _buffer))); }|]

rrelu_with_noise_out_tttssbp :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> CBool -> Ptr Generator -> IO (Ptr Tensor)
rrelu_with_noise_out_tttssbp _output _self _noise _lower _upper _training _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::rrelu_with_noise_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _noise), *$(at::Scalar* _lower), *$(at::Scalar* _upper), $(bool _training), $(at::Generator * _generator))); }|]

rrelu_with_noise_ttssbp :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> CBool -> Ptr Generator -> IO (Ptr Tensor)
rrelu_with_noise_ttssbp _self _noise _lower _upper _training _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::rrelu_with_noise(*$(at::Tensor* _self), *$(at::Tensor* _noise), *$(at::Scalar* _lower), *$(at::Scalar* _upper), $(bool _training), $(at::Generator * _generator))); }|]

rrelu_with_noise_backward_out_ttttssb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> CBool -> IO (Ptr Tensor)
rrelu_with_noise_backward_out_ttttssb _grad_input _grad_output _self _noise _lower _upper _training = [C.block| at::Tensor* { return new at::Tensor(at::native::rrelu_with_noise_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _noise), *$(at::Scalar* _lower), *$(at::Scalar* _upper), $(bool _training))); }|]

rrelu_with_noise_backward_tttssb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> CBool -> IO (Ptr Tensor)
rrelu_with_noise_backward_tttssb _grad_output _self _noise _lower _upper _training = [C.block| at::Tensor* { return new at::Tensor(at::native::rrelu_with_noise_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _noise), *$(at::Scalar* _lower), *$(at::Scalar* _upper), $(bool _training))); }|]

rrelu_with_noise__ttssbp :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> CBool -> Ptr Generator -> IO (Ptr Tensor)
rrelu_with_noise__ttssbp _self _noise _lower _upper _training _generator = [C.block| at::Tensor* { return new at::Tensor(at::native::rrelu_with_noise_(*$(at::Tensor* _self), *$(at::Tensor* _noise), *$(at::Scalar* _lower), *$(at::Scalar* _upper), $(bool _training), $(at::Generator * _generator))); }|]

softplus_out_ttss :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
softplus_out_ttss _output _self _beta _threshold = [C.block| at::Tensor* { return new at::Tensor(at::native::softplus_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Scalar* _beta), *$(at::Scalar* _threshold))); }|]

softplus_tss :: Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> IO (Ptr Tensor)
softplus_tss _self _beta _threshold = [C.block| at::Tensor* { return new at::Tensor(at::native::softplus(*$(at::Tensor* _self), *$(at::Scalar* _beta), *$(at::Scalar* _threshold))); }|]

softplus_backward_out_tttsst :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Tensor -> IO (Ptr Tensor)
softplus_backward_out_tttsst _grad_input _grad_output _self _beta _threshold _output = [C.block| at::Tensor* { return new at::Tensor(at::native::softplus_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Scalar* _beta), *$(at::Scalar* _threshold), *$(at::Tensor* _output))); }|]

softplus_backward_ttsst :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> Ptr Scalar -> Ptr Tensor -> IO (Ptr Tensor)
softplus_backward_ttsst _grad_output _self _beta _threshold _output = [C.block| at::Tensor* { return new at::Tensor(at::native::softplus_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Scalar* _beta), *$(at::Scalar* _threshold), *$(at::Tensor* _output))); }|]

softshrink_out_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
softshrink_out_tts _output _self _lambd = [C.block| at::Tensor* { return new at::Tensor(at::native::softshrink_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Scalar* _lambd))); }|]

softshrink_ts :: Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
softshrink_ts _self _lambd = [C.block| at::Tensor* { return new at::Tensor(at::native::softshrink(*$(at::Tensor* _self), *$(at::Scalar* _lambd))); }|]

softshrink_backward_out_ttts :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
softshrink_backward_out_ttts _grad_input _grad_output _self _lambd = [C.block| at::Tensor* { return new at::Tensor(at::native::softshrink_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Scalar* _lambd))); }|]

softshrink_backward_tts :: Ptr Tensor -> Ptr Tensor -> Ptr Scalar -> IO (Ptr Tensor)
softshrink_backward_tts _grad_output _self _lambd = [C.block| at::Tensor* { return new at::Tensor(at::native::softshrink_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Scalar* _lambd))); }|]

adaptive_avg_pool2d_out_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
adaptive_avg_pool2d_out_cpu_ttl _output _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool2d_out_cpu(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

adaptive_avg_pool2d_out_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
adaptive_avg_pool2d_out_cuda_ttl _output _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool2d_out_cuda(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

adaptive_avg_pool2d_cpu_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
adaptive_avg_pool2d_cpu_tl _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool2d_cpu(*$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

adaptive_avg_pool2d_cuda_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
adaptive_avg_pool2d_cuda_tl _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool2d_cuda(*$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

adaptive_avg_pool2d_backward_out_cpu_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
adaptive_avg_pool2d_backward_out_cpu_ttt _grad_input _grad_output _self = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool2d_backward_out_cpu(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self))); }|]

adaptive_avg_pool2d_backward_out_cuda_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
adaptive_avg_pool2d_backward_out_cuda_ttt _grad_input _grad_output _self = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool2d_backward_out_cuda(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self))); }|]

adaptive_avg_pool2d_backward_cpu_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
adaptive_avg_pool2d_backward_cpu_tt _grad_output _self = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool2d_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _self))); }|]

adaptive_avg_pool2d_backward_cuda_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
adaptive_avg_pool2d_backward_cuda_tt _grad_output _self = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool2d_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _self))); }|]

adaptive_avg_pool3d_out_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
adaptive_avg_pool3d_out_ttl _output _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool3d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

adaptive_avg_pool3d_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
adaptive_avg_pool3d_tl _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool3d(*$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

adaptive_avg_pool3d_backward_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
adaptive_avg_pool3d_backward_out_ttt _grad_input _grad_output _self = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool3d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self))); }|]

adaptive_avg_pool3d_backward_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
adaptive_avg_pool3d_backward_tt _grad_output _self = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_avg_pool3d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self))); }|]

adaptive_max_pool2d_out_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr (Tensor,Tensor))
adaptive_max_pool2d_out_tttl _output _indices _self _output_size = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::adaptive_max_pool2d_out(*$(at::Tensor* _output), *$(at::Tensor* _indices), *$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

adaptive_max_pool2d_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr (Tensor,Tensor))
adaptive_max_pool2d_tl _self _output_size = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::adaptive_max_pool2d(*$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

adaptive_max_pool2d_backward_out_tttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
adaptive_max_pool2d_backward_out_tttt _grad_input _grad_output _self _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_max_pool2d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _indices))); }|]

adaptive_max_pool2d_backward_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
adaptive_max_pool2d_backward_ttt _grad_output _self _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_max_pool2d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _indices))); }|]

adaptive_max_pool3d_out_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr (Tensor,Tensor))
adaptive_max_pool3d_out_tttl _output _indices _self _output_size = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::adaptive_max_pool3d_out(*$(at::Tensor* _output), *$(at::Tensor* _indices), *$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

adaptive_max_pool3d_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr (Tensor,Tensor))
adaptive_max_pool3d_tl _self _output_size = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::adaptive_max_pool3d(*$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

adaptive_max_pool3d_backward_out_tttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
adaptive_max_pool3d_backward_out_tttt _grad_input _grad_output _self _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_max_pool3d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _indices))); }|]

adaptive_max_pool3d_backward_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
adaptive_max_pool3d_backward_ttt _grad_output _self _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::adaptive_max_pool3d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _indices))); }|]

avg_pool2d_out_ttlllbb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> CBool -> IO (Ptr Tensor)
avg_pool2d_out_ttlllbb _output _self _kernel_size _stride _padding _ceil_mode _count_include_pad = [C.block| at::Tensor* { return new at::Tensor(at::native::avg_pool2d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), $(bool _ceil_mode), $(bool _count_include_pad))); }|]

avg_pool2d_tlllbb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> CBool -> IO (Ptr Tensor)
avg_pool2d_tlllbb _self _kernel_size _stride _padding _ceil_mode _count_include_pad = [C.block| at::Tensor* { return new at::Tensor(at::native::avg_pool2d(*$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), $(bool _ceil_mode), $(bool _count_include_pad))); }|]

avg_pool2d_backward_out_tttlllbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> CBool -> IO (Ptr Tensor)
avg_pool2d_backward_out_tttlllbb _grad_input _grad_output _self _kernel_size _stride _padding _ceil_mode _count_include_pad = [C.block| at::Tensor* { return new at::Tensor(at::native::avg_pool2d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), $(bool _ceil_mode), $(bool _count_include_pad))); }|]

avg_pool2d_backward_ttlllbb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> CBool -> IO (Ptr Tensor)
avg_pool2d_backward_ttlllbb _grad_output _self _kernel_size _stride _padding _ceil_mode _count_include_pad = [C.block| at::Tensor* { return new at::Tensor(at::native::avg_pool2d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), $(bool _ceil_mode), $(bool _count_include_pad))); }|]

avg_pool3d_out_ttlllbb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> CBool -> IO (Ptr Tensor)
avg_pool3d_out_ttlllbb _output _self _kernel_size _stride _padding _ceil_mode _count_include_pad = [C.block| at::Tensor* { return new at::Tensor(at::native::avg_pool3d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), $(bool _ceil_mode), $(bool _count_include_pad))); }|]

avg_pool3d_tlllbb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> CBool -> IO (Ptr Tensor)
avg_pool3d_tlllbb _self _kernel_size _stride _padding _ceil_mode _count_include_pad = [C.block| at::Tensor* { return new at::Tensor(at::native::avg_pool3d(*$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), $(bool _ceil_mode), $(bool _count_include_pad))); }|]

avg_pool3d_backward_out_tttlllbb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> CBool -> IO (Ptr Tensor)
avg_pool3d_backward_out_tttlllbb _grad_input _grad_output _self _kernel_size _stride _padding _ceil_mode _count_include_pad = [C.block| at::Tensor* { return new at::Tensor(at::native::avg_pool3d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), $(bool _ceil_mode), $(bool _count_include_pad))); }|]

avg_pool3d_backward_ttlllbb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> CBool -> IO (Ptr Tensor)
avg_pool3d_backward_ttlllbb _grad_output _self _kernel_size _stride _padding _ceil_mode _count_include_pad = [C.block| at::Tensor* { return new at::Tensor(at::native::avg_pool3d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), $(bool _ceil_mode), $(bool _count_include_pad))); }|]

fractional_max_pool2d_out_cpu_tttllt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
fractional_max_pool2d_out_cpu_tttllt _output _indices _self _kernel_size _output_size _random_samples = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::fractional_max_pool2d_out_cpu(*$(at::Tensor* _output), *$(at::Tensor* _indices), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _output_size), *$(at::Tensor* _random_samples))); }|]

fractional_max_pool2d_out_cuda_tttllt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
fractional_max_pool2d_out_cuda_tttllt _output _indices _self _kernel_size _output_size _random_samples = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::fractional_max_pool2d_out_cuda(*$(at::Tensor* _output), *$(at::Tensor* _indices), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _output_size), *$(at::Tensor* _random_samples))); }|]

fractional_max_pool2d_cpu_tllt :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
fractional_max_pool2d_cpu_tllt _self _kernel_size _output_size _random_samples = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::fractional_max_pool2d_cpu(*$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _output_size), *$(at::Tensor* _random_samples))); }|]

fractional_max_pool2d_cuda_tllt :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> IO (Ptr (Tensor,Tensor))
fractional_max_pool2d_cuda_tllt _self _kernel_size _output_size _random_samples = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::fractional_max_pool2d_cuda(*$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _output_size), *$(at::Tensor* _random_samples))); }|]

fractional_max_pool2d_backward_out_cpu_tttllt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> IO (Ptr Tensor)
fractional_max_pool2d_backward_out_cpu_tttllt _grad_input _grad_output _self _kernel_size _output_size _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::fractional_max_pool2d_backward_out_cpu(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _output_size), *$(at::Tensor* _indices))); }|]

fractional_max_pool2d_backward_out_cuda_tttllt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> IO (Ptr Tensor)
fractional_max_pool2d_backward_out_cuda_tttllt _grad_input _grad_output _self _kernel_size _output_size _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::fractional_max_pool2d_backward_out_cuda(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _output_size), *$(at::Tensor* _indices))); }|]

fractional_max_pool2d_backward_cpu_ttllt :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> IO (Ptr Tensor)
fractional_max_pool2d_backward_cpu_ttllt _grad_output _self _kernel_size _output_size _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::fractional_max_pool2d_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _output_size), *$(at::Tensor* _indices))); }|]

fractional_max_pool2d_backward_cuda_ttllt :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> IO (Ptr Tensor)
fractional_max_pool2d_backward_cuda_ttllt _grad_output _self _kernel_size _output_size _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::fractional_max_pool2d_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _output_size), *$(at::Tensor* _indices))); }|]

max_pool2d_with_indices_out_tttllllb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr (Tensor,Tensor))
max_pool2d_with_indices_out_tttllllb _output _indices _self _kernel_size _stride _padding _dilation _ceil_mode = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::max_pool2d_with_indices_out(*$(at::Tensor* _output), *$(at::Tensor* _indices), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _ceil_mode))); }|]

max_pool2d_with_indices_tllllb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr (Tensor,Tensor))
max_pool2d_with_indices_tllllb _self _kernel_size _stride _padding _dilation _ceil_mode = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::max_pool2d_with_indices(*$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _ceil_mode))); }|]

max_pool2d_with_indices_backward_out_tttllllbt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> Ptr Tensor -> IO (Ptr Tensor)
max_pool2d_with_indices_backward_out_tttllllbt _grad_input _grad_output _self _kernel_size _stride _padding _dilation _ceil_mode _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::max_pool2d_with_indices_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _ceil_mode), *$(at::Tensor* _indices))); }|]

max_pool2d_with_indices_backward_ttllllbt :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> Ptr Tensor -> IO (Ptr Tensor)
max_pool2d_with_indices_backward_ttllllbt _grad_output _self _kernel_size _stride _padding _dilation _ceil_mode _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::max_pool2d_with_indices_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _ceil_mode), *$(at::Tensor* _indices))); }|]

max_pool3d_with_indices_out_tttllllb :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr (Tensor,Tensor))
max_pool3d_with_indices_out_tttllllb _output _indices _self _kernel_size _stride _padding _dilation _ceil_mode = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::max_pool3d_with_indices_out(*$(at::Tensor* _output), *$(at::Tensor* _indices), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _ceil_mode))); }|]

max_pool3d_with_indices_tllllb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr (Tensor,Tensor))
max_pool3d_with_indices_tllllb _self _kernel_size _stride _padding _dilation _ceil_mode = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::max_pool3d_with_indices(*$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _ceil_mode))); }|]

max_pool3d_with_indices_backward_out_tttllllbt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> Ptr Tensor -> IO (Ptr Tensor)
max_pool3d_with_indices_backward_out_tttllllbt _grad_input _grad_output _self _kernel_size _stride _padding _dilation _ceil_mode _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::max_pool3d_with_indices_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _ceil_mode), *$(at::Tensor* _indices))); }|]

max_pool3d_with_indices_backward_ttllllbt :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> CBool -> Ptr Tensor -> IO (Ptr Tensor)
max_pool3d_with_indices_backward_ttllllbt _grad_output _self _kernel_size _stride _padding _dilation _ceil_mode _indices = [C.block| at::Tensor* { return new at::Tensor(at::native::max_pool3d_with_indices_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), $(bool _ceil_mode), *$(at::Tensor* _indices))); }|]

max_unpool2d_out_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
max_unpool2d_out_tttl _output _self _indices _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::max_unpool2d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _indices), *$(at::IntList* _output_size))); }|]

max_unpool2d_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
max_unpool2d_ttl _self _indices _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::max_unpool2d(*$(at::Tensor* _self), *$(at::Tensor* _indices), *$(at::IntList* _output_size))); }|]

max_unpool2d_backward_out_ttttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
max_unpool2d_backward_out_ttttl _grad_input _grad_output _self _indices _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::max_unpool2d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _indices), *$(at::IntList* _output_size))); }|]

max_unpool2d_backward_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
max_unpool2d_backward_tttl _grad_output _self _indices _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::max_unpool2d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _indices), *$(at::IntList* _output_size))); }|]

max_unpool3d_out_tttlll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
max_unpool3d_out_tttlll _output _self _indices _output_size _stride _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::max_unpool3d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _indices), *$(at::IntList* _output_size), *$(at::IntList* _stride), *$(at::IntList* _padding))); }|]

max_unpool3d_ttlll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
max_unpool3d_ttlll _self _indices _output_size _stride _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::max_unpool3d(*$(at::Tensor* _self), *$(at::Tensor* _indices), *$(at::IntList* _output_size), *$(at::IntList* _stride), *$(at::IntList* _padding))); }|]

max_unpool3d_backward_out_ttttlll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
max_unpool3d_backward_out_ttttlll _grad_input _grad_output _self _indices _output_size _stride _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::max_unpool3d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _indices), *$(at::IntList* _output_size), *$(at::IntList* _stride), *$(at::IntList* _padding))); }|]

max_unpool3d_backward_tttlll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
max_unpool3d_backward_tttlll _grad_output _self _indices _output_size _stride _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::max_unpool3d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _indices), *$(at::IntList* _output_size), *$(at::IntList* _stride), *$(at::IntList* _padding))); }|]

reflection_pad1d_out_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad1d_out_cpu_ttl _output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad1d_out_cpu(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad1d_out_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad1d_out_cuda_ttl _output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad1d_out_cuda(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad1d_cpu_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad1d_cpu_tl _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad1d_cpu(*$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad1d_cuda_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad1d_cuda_tl _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad1d_cuda(*$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad1d_backward_out_cpu_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad1d_backward_out_cpu_tttl _grad_input _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad1d_backward_out_cpu(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad1d_backward_out_cuda_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad1d_backward_out_cuda_tttl _grad_input _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad1d_backward_out_cuda(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad1d_backward_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad1d_backward_cpu_ttl _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad1d_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad1d_backward_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad1d_backward_cuda_ttl _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad1d_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad2d_out_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad2d_out_cpu_ttl _output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad2d_out_cpu(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad2d_out_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad2d_out_cuda_ttl _output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad2d_out_cuda(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad2d_cpu_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad2d_cpu_tl _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad2d_cpu(*$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad2d_cuda_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad2d_cuda_tl _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad2d_cuda(*$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad2d_backward_out_cpu_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad2d_backward_out_cpu_tttl _grad_input _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad2d_backward_out_cpu(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad2d_backward_out_cuda_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad2d_backward_out_cuda_tttl _grad_input _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad2d_backward_out_cuda(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad2d_backward_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad2d_backward_cpu_ttl _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad2d_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

reflection_pad2d_backward_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
reflection_pad2d_backward_cuda_ttl _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::reflection_pad2d_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad1d_out_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad1d_out_cpu_ttl _output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad1d_out_cpu(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad1d_out_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad1d_out_cuda_ttl _output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad1d_out_cuda(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad1d_cpu_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad1d_cpu_tl _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad1d_cpu(*$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad1d_cuda_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad1d_cuda_tl _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad1d_cuda(*$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad1d_backward_out_cpu_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad1d_backward_out_cpu_tttl _grad_input _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad1d_backward_out_cpu(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad1d_backward_out_cuda_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad1d_backward_out_cuda_tttl _grad_input _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad1d_backward_out_cuda(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad1d_backward_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad1d_backward_cpu_ttl _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad1d_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad1d_backward_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad1d_backward_cuda_ttl _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad1d_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad2d_out_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad2d_out_cpu_ttl _output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad2d_out_cpu(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad2d_out_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad2d_out_cuda_ttl _output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad2d_out_cuda(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad2d_cpu_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad2d_cpu_tl _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad2d_cpu(*$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad2d_cuda_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad2d_cuda_tl _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad2d_cuda(*$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad2d_backward_out_cpu_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad2d_backward_out_cpu_tttl _grad_input _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad2d_backward_out_cpu(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad2d_backward_out_cuda_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad2d_backward_out_cuda_tttl _grad_input _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad2d_backward_out_cuda(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad2d_backward_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad2d_backward_cpu_ttl _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad2d_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad2d_backward_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad2d_backward_cuda_ttl _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad2d_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad3d_out_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad3d_out_cpu_ttl _output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad3d_out_cpu(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad3d_out_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad3d_out_cuda_ttl _output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad3d_out_cuda(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad3d_cpu_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad3d_cpu_tl _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad3d_cpu(*$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad3d_cuda_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad3d_cuda_tl _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad3d_cuda(*$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad3d_backward_out_cpu_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad3d_backward_out_cpu_tttl _grad_input _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad3d_backward_out_cpu(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad3d_backward_out_cuda_tttl :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad3d_backward_out_cuda_tttl _grad_input _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad3d_backward_out_cuda(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad3d_backward_cpu_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad3d_backward_cpu_ttl _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad3d_backward_cpu(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

replication_pad3d_backward_cuda_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
replication_pad3d_backward_cuda_ttl _grad_output _self _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::replication_pad3d_backward_cuda(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::IntList* _padding))); }|]

upsample_linear1d_out_ttlb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_linear1d_out_ttlb _output _self _output_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_linear1d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _output_size), $(bool _align_corners))); }|]

upsample_linear1d_tlb :: Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_linear1d_tlb _self _output_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_linear1d(*$(at::Tensor* _self), *$(at::IntList* _output_size), $(bool _align_corners))); }|]

upsample_linear1d_backward_out_ttllb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_linear1d_backward_out_ttllb _grad_input _grad_output _output_size _input_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_linear1d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size), $(bool _align_corners))); }|]

upsample_linear1d_backward_tllb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_linear1d_backward_tllb _grad_output _output_size _input_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_linear1d_backward(*$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size), $(bool _align_corners))); }|]

upsample_bilinear2d_out_ttlb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_bilinear2d_out_ttlb _output _self _output_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_bilinear2d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _output_size), $(bool _align_corners))); }|]

upsample_bilinear2d_tlb :: Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_bilinear2d_tlb _self _output_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_bilinear2d(*$(at::Tensor* _self), *$(at::IntList* _output_size), $(bool _align_corners))); }|]

upsample_bilinear2d_backward_out_ttllb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_bilinear2d_backward_out_ttllb _grad_input _grad_output _output_size _input_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_bilinear2d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size), $(bool _align_corners))); }|]

upsample_bilinear2d_backward_tllb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_bilinear2d_backward_tllb _grad_output _output_size _input_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_bilinear2d_backward(*$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size), $(bool _align_corners))); }|]

upsample_bicubic2d_out_ttlb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_bicubic2d_out_ttlb _output _self _output_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_bicubic2d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _output_size), $(bool _align_corners))); }|]

upsample_bicubic2d_tlb :: Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_bicubic2d_tlb _self _output_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_bicubic2d(*$(at::Tensor* _self), *$(at::IntList* _output_size), $(bool _align_corners))); }|]

upsample_bicubic2d_backward_out_ttllb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_bicubic2d_backward_out_ttllb _grad_input _grad_output _output_size _input_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_bicubic2d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size), $(bool _align_corners))); }|]

upsample_bicubic2d_backward_tllb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_bicubic2d_backward_tllb _grad_output _output_size _input_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_bicubic2d_backward(*$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size), $(bool _align_corners))); }|]

upsample_trilinear3d_out_ttlb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_trilinear3d_out_ttlb _output _self _output_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_trilinear3d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _output_size), $(bool _align_corners))); }|]

upsample_trilinear3d_tlb :: Ptr Tensor -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_trilinear3d_tlb _self _output_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_trilinear3d(*$(at::Tensor* _self), *$(at::IntList* _output_size), $(bool _align_corners))); }|]

upsample_trilinear3d_backward_out_ttllb :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_trilinear3d_backward_out_ttllb _grad_input _grad_output _output_size _input_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_trilinear3d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size), $(bool _align_corners))); }|]

upsample_trilinear3d_backward_tllb :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> CBool -> IO (Ptr Tensor)
upsample_trilinear3d_backward_tllb _grad_output _output_size _input_size _align_corners = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_trilinear3d_backward(*$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size), $(bool _align_corners))); }|]

upsample_nearest1d_out_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
upsample_nearest1d_out_ttl _output _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_nearest1d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

upsample_nearest1d_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
upsample_nearest1d_tl _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_nearest1d(*$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

upsample_nearest1d_backward_out_ttll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
upsample_nearest1d_backward_out_ttll _grad_input _grad_output _output_size _input_size = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_nearest1d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size))); }|]

upsample_nearest1d_backward_tll :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
upsample_nearest1d_backward_tll _grad_output _output_size _input_size = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_nearest1d_backward(*$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size))); }|]

upsample_nearest2d_out_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
upsample_nearest2d_out_ttl _output _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_nearest2d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

upsample_nearest2d_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
upsample_nearest2d_tl _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_nearest2d(*$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

upsample_nearest2d_backward_out_ttll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
upsample_nearest2d_backward_out_ttll _grad_input _grad_output _output_size _input_size = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_nearest2d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size))); }|]

upsample_nearest2d_backward_tll :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
upsample_nearest2d_backward_tll _grad_output _output_size _input_size = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_nearest2d_backward(*$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size))); }|]

upsample_nearest3d_out_ttl :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
upsample_nearest3d_out_ttl _output _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_nearest3d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

upsample_nearest3d_tl :: Ptr Tensor -> Ptr IntList -> IO (Ptr Tensor)
upsample_nearest3d_tl _self _output_size = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_nearest3d(*$(at::Tensor* _self), *$(at::IntList* _output_size))); }|]

upsample_nearest3d_backward_out_ttll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
upsample_nearest3d_backward_out_ttll _grad_input _grad_output _output_size _input_size = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_nearest3d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size))); }|]

upsample_nearest3d_backward_tll :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
upsample_nearest3d_backward_tll _grad_output _output_size _input_size = [C.block| at::Tensor* { return new at::Tensor(at::native::upsample_nearest3d_backward(*$(at::Tensor* _grad_output), *$(at::IntList* _output_size), *$(at::IntList* _input_size))); }|]

sigmoid_backward_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
sigmoid_backward_out_ttt _grad_input _grad_output _output = [C.block| at::Tensor* { return new at::Tensor(at::native::sigmoid_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _output))); }|]

sigmoid_backward_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
sigmoid_backward_tt _grad_output _output = [C.block| at::Tensor* { return new at::Tensor(at::native::sigmoid_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _output))); }|]

tanh_backward_out_ttt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
tanh_backward_out_ttt _grad_input _grad_output _output = [C.block| at::Tensor* { return new at::Tensor(at::native::tanh_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_output), *$(at::Tensor* _output))); }|]

tanh_backward_tt :: Ptr Tensor -> Ptr Tensor -> IO (Ptr Tensor)
tanh_backward_tt _grad_output _output = [C.block| at::Tensor* { return new at::Tensor(at::native::tanh_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _output))); }|]

thnn_conv_transpose2d_out_tttltllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv_transpose2d_out_tttltllll _output _self _weight _kernel_size _bias _stride _padding _output_padding _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv_transpose2d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _dilation))); }|]

thnn_conv_transpose2d_ttltllll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv_transpose2d_ttltllll _self _weight _kernel_size _bias _stride _padding _output_padding _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv_transpose2d(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _dilation))); }|]

thnn_conv_transpose2d_forward_out_tttttltllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_transpose2d_forward_out_tttttltllll _output _columns _ones _self _weight _kernel_size _bias _stride _padding _output_padding _dilation = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_transpose2d_forward_out(*$(at::Tensor* _output), *$(at::Tensor* _columns), *$(at::Tensor* _ones), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _dilation))); }|]

thnn_conv_transpose2d_forward_ttltllll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_transpose2d_forward_ttltllll _self _weight _kernel_size _bias _stride _padding _output_padding _dilation = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_transpose2d_forward(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _dilation))); }|]

thnn_conv_transpose2d_backward_out_ttttttllllltt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_transpose2d_backward_out_ttttttllllltt _grad_input _grad_weight _grad_bias _grad_output _self _weight _kernel_size _stride _padding _output_padding _dilation _columns _ones = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_transpose2d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_weight), *$(at::Tensor* _grad_bias), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _dilation), *$(at::Tensor* _columns), *$(at::Tensor* _ones))); }|]

thnn_conv_transpose2d_backward_tttllllltta :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_transpose2d_backward_tttllllltta _grad_output _self _weight _kernel_size _stride _padding _output_padding _dilation _columns _ones _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_transpose2d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _dilation), *$(at::Tensor* _columns), *$(at::Tensor* _ones), *$(std::array<bool,3>* _output_mask))); }|]

thnn_conv_transpose3d_out_tttltllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv_transpose3d_out_tttltllll _output _self _weight _kernel_size _bias _stride _padding _output_padding _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv_transpose3d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _dilation))); }|]

thnn_conv_transpose3d_ttltllll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv_transpose3d_ttltllll _self _weight _kernel_size _bias _stride _padding _output_padding _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv_transpose3d(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _dilation))); }|]

thnn_conv_transpose3d_forward_out_tttttltllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_transpose3d_forward_out_tttttltllll _output _finput _fgrad_input _self _weight _kernel_size _bias _stride _padding _output_padding _dilation = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_transpose3d_forward_out(*$(at::Tensor* _output), *$(at::Tensor* _finput), *$(at::Tensor* _fgrad_input), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _dilation))); }|]

thnn_conv_transpose3d_forward_ttltllll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_transpose3d_forward_ttltllll _self _weight _kernel_size _bias _stride _padding _output_padding _dilation = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_transpose3d_forward(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _dilation))); }|]

thnn_conv_transpose3d_backward_out_ttttttllllltt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_transpose3d_backward_out_ttttttllllltt _grad_input _grad_weight _grad_bias _grad_output _self _weight _kernel_size _stride _padding _output_padding _dilation _finput _fgrad_input = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_transpose3d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_weight), *$(at::Tensor* _grad_bias), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _dilation), *$(at::Tensor* _finput), *$(at::Tensor* _fgrad_input))); }|]

thnn_conv_transpose3d_backward_tttllllltta :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_transpose3d_backward_tttllllltta _grad_output _self _weight _kernel_size _stride _padding _output_padding _dilation _finput _fgrad_input _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_transpose3d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _output_padding), *$(at::IntList* _dilation), *$(at::Tensor* _finput), *$(at::Tensor* _fgrad_input), *$(std::array<bool,3>* _output_mask))); }|]

thnn_conv2d_out_tttltll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv2d_out_tttltll _output _self _weight _kernel_size _bias _stride _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv2d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding))); }|]

thnn_conv2d_ttltll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv2d_ttltll _self _weight _kernel_size _bias _stride _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv2d(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding))); }|]

thnn_conv2d_forward_out_tttttltll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv2d_forward_out_tttttltll _output _finput _fgrad_input _self _weight _kernel_size _bias _stride _padding = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv2d_forward_out(*$(at::Tensor* _output), *$(at::Tensor* _finput), *$(at::Tensor* _fgrad_input), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding))); }|]

thnn_conv2d_forward_ttltll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv2d_forward_ttltll _self _weight _kernel_size _bias _stride _padding = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv2d_forward(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding))); }|]

thnn_conv2d_backward_out_ttttttllltt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv2d_backward_out_ttttttllltt _grad_input _grad_weight _grad_bias _grad_output _self _weight _kernel_size _stride _padding _finput _fgrad_input = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv2d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_weight), *$(at::Tensor* _grad_bias), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::Tensor* _finput), *$(at::Tensor* _fgrad_input))); }|]

thnn_conv2d_backward_tttllltta :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv2d_backward_tttllltta _grad_output _self _weight _kernel_size _stride _padding _finput _fgrad_input _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv2d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::Tensor* _finput), *$(at::Tensor* _fgrad_input), *$(std::array<bool,3>* _output_mask))); }|]

thnn_conv_depthwise2d_out_tttltlll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv_depthwise2d_out_tttltlll _output _self _weight _kernel_size _bias _stride _padding _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv_depthwise2d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_depthwise2d_ttltlll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv_depthwise2d_ttltlll _self _weight _kernel_size _bias _stride _padding _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv_depthwise2d(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_depthwise2d_forward_out_tttltlll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv_depthwise2d_forward_out_tttltlll _output _self _weight _kernel_size _bias _stride _padding _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv_depthwise2d_forward_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_depthwise2d_forward_ttltlll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv_depthwise2d_forward_ttltlll _self _weight _kernel_size _bias _stride _padding _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv_depthwise2d_forward(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_depthwise2d_backward_out_tttttllll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor))
thnn_conv_depthwise2d_backward_out_tttttllll _grad_input _grad_weight _grad_output _self _weight _kernel_size _stride _padding _dilation = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::thnn_conv_depthwise2d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_weight), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_depthwise2d_backward_tttlllla :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr (StdArray CBool 2) -> IO (Ptr (Tensor,Tensor))
thnn_conv_depthwise2d_backward_tttlllla _grad_output _self _weight _kernel_size _stride _padding _dilation _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::native::thnn_conv_depthwise2d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), *$(std::array<bool,2>* _output_mask))); }|]

thnn_conv3d_out_tttltll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv3d_out_tttltll _output _self _weight _kernel_size _bias _stride _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv3d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding))); }|]

thnn_conv3d_ttltll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv3d_ttltll _self _weight _kernel_size _bias _stride _padding = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv3d(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding))); }|]

thnn_conv3d_forward_out_tttttltll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv3d_forward_out_tttttltll _output _finput _fgrad_input _self _weight _kernel_size _bias _stride _padding = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv3d_forward_out(*$(at::Tensor* _output), *$(at::Tensor* _finput), *$(at::Tensor* _fgrad_input), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding))); }|]

thnn_conv3d_forward_ttltll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv3d_forward_ttltll _self _weight _kernel_size _bias _stride _padding = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv3d_forward(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding))); }|]

thnn_conv3d_backward_out_ttttttllltt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv3d_backward_out_ttttttllltt _grad_input _grad_weight _grad_bias _grad_output _self _weight _kernel_size _stride _padding _finput _fgrad_input = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv3d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_weight), *$(at::Tensor* _grad_bias), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::Tensor* _finput), *$(at::Tensor* _fgrad_input))); }|]

thnn_conv3d_backward_tttllltta :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv3d_backward_tttllltta _grad_output _self _weight _kernel_size _stride _padding _finput _fgrad_input _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv3d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::Tensor* _finput), *$(at::Tensor* _fgrad_input), *$(std::array<bool,3>* _output_mask))); }|]

thnn_conv_dilated2d_out_tttltlll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv_dilated2d_out_tttltlll _output _self _weight _kernel_size _bias _stride _padding _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv_dilated2d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_dilated2d_ttltlll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv_dilated2d_ttltlll _self _weight _kernel_size _bias _stride _padding _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv_dilated2d(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_dilated2d_forward_out_tttttltlll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_dilated2d_forward_out_tttttltlll _output _columns _ones _self _weight _kernel_size _bias _stride _padding _dilation = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_dilated2d_forward_out(*$(at::Tensor* _output), *$(at::Tensor* _columns), *$(at::Tensor* _ones), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_dilated2d_forward_ttltlll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_dilated2d_forward_ttltlll _self _weight _kernel_size _bias _stride _padding _dilation = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_dilated2d_forward(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_dilated2d_backward_out_ttttttlllltt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_dilated2d_backward_out_ttttttlllltt _grad_input _grad_weight _grad_bias _grad_output _self _weight _kernel_size _stride _padding _dilation _columns _ones = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_dilated2d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_weight), *$(at::Tensor* _grad_bias), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), *$(at::Tensor* _columns), *$(at::Tensor* _ones))); }|]

thnn_conv_dilated2d_backward_tttlllltta :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_dilated2d_backward_tttlllltta _grad_output _self _weight _kernel_size _stride _padding _dilation _columns _ones _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_dilated2d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), *$(at::Tensor* _columns), *$(at::Tensor* _ones), *$(std::array<bool,3>* _output_mask))); }|]

thnn_conv_dilated3d_out_tttltlll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv_dilated3d_out_tttltlll _output _self _weight _kernel_size _bias _stride _padding _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv_dilated3d_out(*$(at::Tensor* _output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_dilated3d_ttltlll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_conv_dilated3d_ttltlll _self _weight _kernel_size _bias _stride _padding _dilation = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_conv_dilated3d(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_dilated3d_forward_out_tttttltlll :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_dilated3d_forward_out_tttttltlll _output _columns _ones _self _weight _kernel_size _bias _stride _padding _dilation = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_dilated3d_forward_out(*$(at::Tensor* _output), *$(at::Tensor* _columns), *$(at::Tensor* _ones), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_dilated3d_forward_ttltlll :: Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_dilated3d_forward_ttltlll _self _weight _kernel_size _bias _stride _padding _dilation = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_dilated3d_forward(*$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::Tensor* _bias), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation))); }|]

thnn_conv_dilated3d_backward_out_ttttttlllltt :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_dilated3d_backward_out_ttttttlllltt _grad_input _grad_weight _grad_bias _grad_output _self _weight _kernel_size _stride _padding _dilation _columns _ones = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_dilated3d_backward_out(*$(at::Tensor* _grad_input), *$(at::Tensor* _grad_weight), *$(at::Tensor* _grad_bias), *$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), *$(at::Tensor* _columns), *$(at::Tensor* _ones))); }|]

thnn_conv_dilated3d_backward_tttlllltta :: Ptr Tensor -> Ptr Tensor -> Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr Tensor -> Ptr Tensor -> Ptr (StdArray CBool 3) -> IO (Ptr (Tensor,Tensor,Tensor))
thnn_conv_dilated3d_backward_tttlllltta _grad_output _self _weight _kernel_size _stride _padding _dilation _columns _ones _output_mask = [C.block| std::tuple<at::Tensor,at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor,at::Tensor>(at::native::thnn_conv_dilated3d_backward(*$(at::Tensor* _grad_output), *$(at::Tensor* _self), *$(at::Tensor* _weight), *$(at::IntList* _kernel_size), *$(at::IntList* _stride), *$(at::IntList* _padding), *$(at::IntList* _dilation), *$(at::Tensor* _columns), *$(at::Tensor* _ones), *$(std::array<bool,3>* _output_mask))); }|]

thnn_col2im_tlllll :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_col2im_tlllll _self _output_size _kernel_size _dilation _padding _stride = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_col2im(*$(at::Tensor* _self), *$(at::IntList* _output_size), *$(at::IntList* _kernel_size), *$(at::IntList* _dilation), *$(at::IntList* _padding), *$(at::IntList* _stride))); }|]

thnn_col2im_backward_tllll :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_col2im_backward_tllll _grad_output _kernel_size _dilation _padding _stride = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_col2im_backward(*$(at::Tensor* _grad_output), *$(at::IntList* _kernel_size), *$(at::IntList* _dilation), *$(at::IntList* _padding), *$(at::IntList* _stride))); }|]

thnn_im2col_tllll :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_im2col_tllll _self _kernel_size _dilation _padding _stride = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_im2col(*$(at::Tensor* _self), *$(at::IntList* _kernel_size), *$(at::IntList* _dilation), *$(at::IntList* _padding), *$(at::IntList* _stride))); }|]

thnn_im2col_backward_tlllll :: Ptr Tensor -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> Ptr IntList -> IO (Ptr Tensor)
thnn_im2col_backward_tlllll _grad_output _input_size _kernel_size _dilation _padding _stride = [C.block| at::Tensor* { return new at::Tensor(at::native::thnn_im2col_backward(*$(at::Tensor* _grad_output), *$(at::IntList* _input_size), *$(at::IntList* _kernel_size), *$(at::IntList* _dilation), *$(at::IntList* _padding), *$(at::IntList* _stride))); }|]


