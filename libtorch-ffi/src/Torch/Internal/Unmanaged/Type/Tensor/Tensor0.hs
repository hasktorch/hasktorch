
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.Tensor.Tensor0 where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }



C.include "<ATen/Tensor.h>"
C.include "<ATen/Functions.h>"
C.include "<ATen/TensorOperators.h>"
C.include "<vector>"



newTensor
  :: IO (Ptr Tensor)
newTensor  =
  [C.throwBlock| at::Tensor* { return new at::Tensor(
    );
  }|]

newTensor_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
newTensor_t _x =
  [C.throwBlock| at::Tensor* { return new at::Tensor(
    *$(at::Tensor* _x));
  }|]

tensor___dispatch_contiguous
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor___dispatch_contiguous _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).__dispatch_contiguous(
    ));
  }|]

tensor_backward_tbb
  :: Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> IO (())
tensor_backward_tbb _obj _gradient _keep_graph _create_graph =
  [C.throwBlock| void {  (*$(at::Tensor* _obj)).backward(
    *$(at::Tensor* _gradient)
  , $(bool _keep_graph)
  , $(bool _create_graph));
  }|]

tensor_contiguous
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_contiguous _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).contiguous(
    ));
  }|]

tensor_cpu
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cpu _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cpu(
    ));
  }|]

tensor_cuda
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cuda _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cuda(
    ));
  }|]

tensor_data_ptr
  :: Ptr Tensor
  -> IO (Ptr ())
tensor_data_ptr _obj =
  [C.throwBlock| void * { return (*$(at::Tensor* _obj)).data_ptr(
    );
  }|]

tensor_defined
  :: Ptr Tensor
  -> IO (CBool)
tensor_defined _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).defined(
    );
  }|]

tensor_dim
  :: Ptr Tensor
  -> IO (Int64)
tensor_dim _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).dim(
    );
  }|]

tensor_element_size
  :: Ptr Tensor
  -> IO (Int64)
tensor_element_size _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).element_size(
    );
  }|]

tensor_get_device
  :: Ptr Tensor
  -> IO (Int64)
tensor_get_device _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).get_device(
    );
  }|]

tensor_has_names
  :: Ptr Tensor
  -> IO (CBool)
tensor_has_names _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).has_names(
    );
  }|]

tensor_has_storage
  :: Ptr Tensor
  -> IO (CBool)
tensor_has_storage _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).has_storage(
    );
  }|]

tensor_hip
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_hip _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).hip(
    ));
  }|]

tensor_is_alias_of_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (CBool)
tensor_is_alias_of_t _obj _other =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_alias_of(
    *$(at::Tensor* _other));
  }|]

tensor_is_contiguous
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_contiguous _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_contiguous(
    );
  }|]

tensor_is_cuda
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_cuda _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_cuda(
    );
  }|]

tensor_is_hip
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_hip _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_hip(
    );
  }|]

tensor_is_meta
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_meta _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_meta(
    );
  }|]

tensor_is_metal
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_metal _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_metal(
    );
  }|]

tensor_is_mkldnn
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_mkldnn _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_mkldnn(
    );
  }|]

tensor_is_non_overlapping_and_dense
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_non_overlapping_and_dense _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_non_overlapping_and_dense(
    );
  }|]

tensor_is_quantized
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_quantized _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_quantized(
    );
  }|]

tensor_is_same_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (CBool)
tensor_is_same_t _obj _other =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_same(
    *$(at::Tensor* _other));
  }|]

tensor_is_sparse
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_sparse _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_sparse(
    );
  }|]

tensor_is_vulkan
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_vulkan _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_vulkan(
    );
  }|]

tensor_is_xpu
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_xpu _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_xpu(
    );
  }|]

tensor_item_double
  :: Ptr Tensor
  -> IO (CDouble)
tensor_item_double _obj =
  [C.throwBlock| double { return (*$(at::Tensor* _obj)).item<double>(
    );
  }|]

tensor_item_float
  :: Ptr Tensor
  -> IO (CFloat)
tensor_item_float _obj =
  [C.throwBlock| float { return (*$(at::Tensor* _obj)).item<float>(
    );
  }|]

tensor_item_int64_t
  :: Ptr Tensor
  -> IO (Int64)
tensor_item_int64_t _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).item<int64_t>(
    );
  }|]

tensor_itemsize
  :: Ptr Tensor
  -> IO (CSize)
tensor_itemsize _obj =
  [C.throwBlock| size_t { return (*$(at::Tensor* _obj)).itemsize(
    );
  }|]

tensor_layout
  :: Ptr Tensor
  -> IO (Layout)
tensor_layout _obj =
  [C.throwBlock| at::Layout { return (*$(at::Tensor* _obj)).layout(
    );
  }|]

tensor_metal
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_metal _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).metal(
    ));
  }|]

tensor_mutable_grad
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_mutable_grad _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).mutable_grad(
    ));
  }|]

tensor_nbytes
  :: Ptr Tensor
  -> IO (CSize)
tensor_nbytes _obj =
  [C.throwBlock| size_t { return (*$(at::Tensor* _obj)).nbytes(
    );
  }|]

tensor_ndimension
  :: Ptr Tensor
  -> IO (Int64)
tensor_ndimension _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).ndimension(
    );
  }|]

tensor_numel
  :: Ptr Tensor
  -> IO (Int64)
tensor_numel _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).numel(
    );
  }|]

tensor__imul__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (())
tensor__imul__s _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))*=(
    *$(at::Scalar* _other));
  }|]

tensor__imul__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (())
tensor__imul__t _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))*=(
    *$(at::Tensor* _other));
  }|]

tensor__iadd__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (())
tensor__iadd__s _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))+=(
    *$(at::Scalar* _other));
  }|]

tensor__iadd__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (())
tensor__iadd__t _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))+=(
    *$(at::Tensor* _other));
  }|]

tensor__isub__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (())
tensor__isub__s _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))-=(
    *$(at::Scalar* _other));
  }|]

tensor__isub__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (())
tensor__isub__t _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))-=(
    *$(at::Tensor* _other));
  }|]

tensor__idiv__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (())
tensor__idiv__s _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))/=(
    *$(at::Scalar* _other));
  }|]

tensor__idiv__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (())
tensor__idiv__t _obj _other =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))/=(
    *$(at::Tensor* _other));
  }|]

tensor__assign__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor__assign__t _obj _x =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))=(
    *$(at::Tensor* _x)));
  }|]

tensor__at__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor__at__s _obj _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))[(
    *$(at::Scalar* _index))]);
  }|]

tensor__at__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor__at__t _obj _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))[(
    *$(at::Tensor* _index))]);
  }|]

tensor__at__l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor__at__l _obj _index =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))[(
    $(int64_t _index))]);
  }|]

tensor_options
  :: Ptr Tensor
  -> IO (Ptr TensorOptions)
tensor_options _obj =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions((*$(at::Tensor* _obj)).options(
    ));
  }|]

tensor_print
  :: Ptr Tensor
  -> IO (())
tensor_print _obj =
  [C.throwBlock| void {  (*$(at::Tensor* _obj)).print(
    );
  }|]

tensor_requires_grad
  :: Ptr Tensor
  -> IO (CBool)
tensor_requires_grad _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).requires_grad(
    );
  }|]

tensor_reset
  :: Ptr Tensor
  -> IO (())
tensor_reset _obj =
  [C.throwBlock| void {  (*$(at::Tensor* _obj)).reset(
    );
  }|]

tensor_resize__l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_resize__l _obj _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).resize_(
    *$(std::vector<int64_t>* _size)));
  }|]

tensor_scalar_type
  :: Ptr Tensor
  -> IO (ScalarType)
tensor_scalar_type _obj =
  [C.throwBlock| at::ScalarType { return (*$(at::Tensor* _obj)).scalar_type(
    );
  }|]

tensor_set_requires_grad_b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_set_requires_grad_b _obj _requires_grad =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).set_requires_grad(
    $(bool _requires_grad)));
  }|]

tensor_size_l
  :: Ptr Tensor
  -> Int64
  -> IO (Int64)
tensor_size_l _obj _dim =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).size(
    $(int64_t _dim));
  }|]

tensor_sizes
  :: Ptr Tensor
  -> IO (Ptr IntArray)
tensor_sizes _obj =
  [C.throwBlock| std::vector<int64_t>* { return new std::vector<int64_t>((*$(at::Tensor* _obj)).sizes(
    ).vec());
  }|]

tensor_storage
  :: Ptr Tensor
  -> IO (Ptr Storage)
tensor_storage _obj =
  [C.throwBlock| at::Storage* { return new at::Storage((*$(at::Tensor* _obj)).storage(
    ));
  }|]

tensor_storage_offset
  :: Ptr Tensor
  -> IO (Int64)
tensor_storage_offset _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).storage_offset(
    );
  }|]

tensor_stride_l
  :: Ptr Tensor
  -> Int64
  -> IO (Int64)
tensor_stride_l _obj _dim =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).stride(
    $(int64_t _dim));
  }|]

tensor_strides
  :: Ptr Tensor
  -> IO (Ptr IntArray)
tensor_strides _obj =
  [C.throwBlock| std::vector<int64_t>* { return new std::vector<int64_t>((*$(at::Tensor* _obj)).strides(
    ).vec());
  }|]

tensor_to_Dsbb
  :: Ptr Tensor
  -> DeviceType
  -> ScalarType
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_to_Dsbb _obj _device _dtype _non_blocking _copy =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to(
    $(at::DeviceType _device)
  , $(at::ScalarType _dtype)
  , $(bool _non_blocking)
  , $(bool _copy)));
  }|]

tensor_to_sbb
  :: Ptr Tensor
  -> ScalarType
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_to_sbb _obj _dtype _non_blocking _copy =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to(
    $(at::ScalarType _dtype)
  , $(bool _non_blocking)
  , $(bool _copy)));
  }|]

tensor_to_tbb
  :: Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_to_tbb _obj _other _non_blocking _copy =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to(
    *$(at::Tensor* _other)
  , $(bool _non_blocking)
  , $(bool _copy)));
  }|]

tensor_to_obb
  :: Ptr Tensor
  -> Ptr TensorOptions
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
tensor_to_obb _obj _options _non_blocking _copy =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to(
    *$(at::TensorOptions* _options)
  , $(bool _non_blocking)
  , $(bool _copy)));
  }|]

tensor_toBackend_B
  :: Ptr Tensor
  -> Backend
  -> IO (Ptr Tensor)
tensor_toBackend_B _obj _b =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).toBackend(
    $(at::Backend _b)));
  }|]

tensor_toString
  :: Ptr Tensor
  -> IO (Ptr StdString)
tensor_toString _obj =
  [C.throwBlock| std::string* { return new std::string((*$(at::Tensor* _obj)).toString(
    ));
  }|]

tensor_toType_s
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
tensor_toType_s _obj _t =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).toType(
    $(at::ScalarType _t)));
  }|]

tensor_to_dense
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_to_dense _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_dense(
    ));
  }|]

tensor_to_mkldnn
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_to_mkldnn _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).to_mkldnn(
    ));
  }|]

tensor_use_count
  :: Ptr Tensor
  -> IO (CSize)
tensor_use_count _obj =
  [C.throwBlock| size_t { return (*$(at::Tensor* _obj)).use_count(
    );
  }|]

tensor_vulkan
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_vulkan _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).vulkan(
    ));
  }|]

tensor_weak_use_count
  :: Ptr Tensor
  -> IO (CSize)
tensor_weak_use_count _obj =
  [C.throwBlock| size_t { return (*$(at::Tensor* _obj)).weak_use_count(
    );
  }|]

tensor__backward_ltbb
  :: Ptr Tensor
  -> Ptr TensorList
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> IO (())
tensor__backward_ltbb _obj _inputs _gradient _retain_graph _create_graph =
  [C.throwBlock| void {  (*$(at::Tensor* _obj))._backward(
    *$(std::vector<at::Tensor>* _inputs)
  , *$(at::Tensor* _gradient)
  , $(bool _retain_graph)
  , $(bool _create_graph));
  }|]

tensor_set_data_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (())
tensor_set_data_t _obj _new_data =
  [C.throwBlock| void {  (*$(at::Tensor* _obj)).set_data(
    *$(at::Tensor* _new_data));
  }|]

tensor_data
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_data _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).data(
    ));
  }|]

tensor_is_leaf
  :: Ptr Tensor
  -> IO (CBool)
tensor_is_leaf _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).is_leaf(
    );
  }|]

tensor_output_nr
  :: Ptr Tensor
  -> IO (Int64)
tensor_output_nr _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).output_nr(
    );
  }|]

tensor__version
  :: Ptr Tensor
  -> IO (Int64)
tensor__version _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj))._version(
    );
  }|]

tensor_requires_grad__b
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_requires_grad__b _obj _requires_grad =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).requires_grad_(
    $(bool _requires_grad)));
  }|]

tensor_retain_grad
  :: Ptr Tensor
  -> IO (())
tensor_retain_grad _obj =
  [C.throwBlock| void {  (*$(at::Tensor* _obj)).retain_grad(
    );
  }|]

tensor_retains_grad
  :: Ptr Tensor
  -> IO (CBool)
tensor_retains_grad _obj =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).retains_grad(
    );
  }|]

tensor__fw_primal_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor__fw_primal_l _obj _level =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._fw_primal(
    $(int64_t _level)));
  }|]

tensor_rename__N
  :: Ptr Tensor
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
tensor_rename__N _obj _names =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).rename_(
    *$(std::vector<at::Dimname>* _names)));
  }|]

tensor_rename_N
  :: Ptr Tensor
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
tensor_rename_N _obj _names =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).rename(
    *$(std::vector<at::Dimname>* _names)));
  }|]

tensor_align_to_N
  :: Ptr Tensor
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
tensor_align_to_N _obj _names =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).align_to(
    *$(std::vector<at::Dimname>* _names)));
  }|]

tensor_align_to_Nl
  :: Ptr Tensor
  -> Ptr DimnameList
  -> Int64
  -> IO (Ptr Tensor)
tensor_align_to_Nl _obj _order _ellipsis_idx =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).align_to(
    *$(std::vector<at::Dimname>* _order)
  , $(int64_t _ellipsis_idx)));
  }|]

tensor_align_as_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_align_as_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).align_as(
    *$(at::Tensor* _other)));
  }|]

tensor_refine_names_N
  :: Ptr Tensor
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
tensor_refine_names_N _obj _names =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).refine_names(
    *$(std::vector<at::Dimname>* _names)));
  }|]

tensor_abs
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_abs _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).abs(
    ));
  }|]

tensor_abs_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_abs_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).abs_(
    ));
  }|]

tensor_absolute
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_absolute _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).absolute(
    ));
  }|]

tensor_absolute_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_absolute_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).absolute_(
    ));
  }|]

tensor_angle
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_angle _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).angle(
    ));
  }|]

tensor_sgn
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sgn _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sgn(
    ));
  }|]

tensor_sgn_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_sgn_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).sgn_(
    ));
  }|]

tensor_chalf_M
  :: Ptr Tensor
  -> MemoryFormat
  -> IO (Ptr Tensor)
tensor_chalf_M _obj _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).chalf(
    $(at::MemoryFormat _memory_format)));
  }|]

tensor__conj
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor__conj _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._conj(
    ));
  }|]

tensor_conj
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_conj _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).conj(
    ));
  }|]

tensor__conj_physical
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor__conj_physical _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._conj_physical(
    ));
  }|]

tensor_conj_physical
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_conj_physical _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).conj_physical(
    ));
  }|]

tensor_conj_physical_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_conj_physical_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).conj_physical_(
    ));
  }|]

tensor_resolve_conj
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_resolve_conj _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).resolve_conj(
    ));
  }|]

tensor_resolve_neg
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_resolve_neg _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).resolve_neg(
    ));
  }|]

tensor__neg_view
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor__neg_view _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._neg_view(
    ));
  }|]

tensor_acos
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_acos _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).acos(
    ));
  }|]

tensor_acos_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_acos_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).acos_(
    ));
  }|]

tensor_arccos
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_arccos _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).arccos(
    ));
  }|]

tensor_arccos_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_arccos_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).arccos_(
    ));
  }|]

tensor_add_ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_add_ts _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).add(
    *$(at::Tensor* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_add__ts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_add__ts _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).add_(
    *$(at::Tensor* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_add_ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_add_ss _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).add(
    *$(at::Scalar* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_add__ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_add__ss _obj _other _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).add_(
    *$(at::Scalar* _other)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_addmv_ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addmv_ttss _obj _mat _vec _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addmv(
    *$(at::Tensor* _mat)
  , *$(at::Tensor* _vec)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_addmv__ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addmv__ttss _obj _mat _vec _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addmv_(
    *$(at::Tensor* _mat)
  , *$(at::Tensor* _vec)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_addr_ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addr_ttss _obj _vec1 _vec2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addr(
    *$(at::Tensor* _vec1)
  , *$(at::Tensor* _vec2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_addr__ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_addr__ttss _obj _vec1 _vec2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).addr_(
    *$(at::Tensor* _vec1)
  , *$(at::Tensor* _vec2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor__is_all_true
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor__is_all_true _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._is_all_true(
    ));
  }|]

tensor__is_any_true
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor__is_any_true _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj))._is_any_true(
    ));
  }|]

tensor_all_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_all_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).all(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_all_nb
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr Tensor)
tensor_all_nb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).all(
    *$(at::Dimname* _dim)
  , $(bool _keepdim)));
  }|]

tensor_allclose_tddb
  :: Ptr Tensor
  -> Ptr Tensor
  -> CDouble
  -> CDouble
  -> CBool
  -> IO (CBool)
tensor_allclose_tddb _obj _other _rtol _atol _equal_nan =
  [C.throwBlock| bool { return (*$(at::Tensor* _obj)).allclose(
    *$(at::Tensor* _other)
  , $(double _rtol)
  , $(double _atol)
  , $(bool _equal_nan));
  }|]

tensor_any_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_any_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).any(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_any_nb
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr Tensor)
tensor_any_nb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).any(
    *$(at::Dimname* _dim)
  , $(bool _keepdim)));
  }|]

tensor_argmax_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_argmax_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).argmax(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_argmin_lb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
tensor_argmin_lb _obj _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).argmin(
    $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

tensor_acosh
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_acosh _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).acosh(
    ));
  }|]

tensor_acosh_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_acosh_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).acosh_(
    ));
  }|]

tensor_arccosh
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_arccosh _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).arccosh(
    ));
  }|]

tensor_arccosh_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_arccosh_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).arccosh_(
    ));
  }|]

tensor_asinh
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_asinh _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).asinh(
    ));
  }|]

tensor_asinh_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_asinh_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).asinh_(
    ));
  }|]

tensor_arcsinh
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_arcsinh _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).arcsinh(
    ));
  }|]

tensor_arcsinh_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_arcsinh_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).arcsinh_(
    ));
  }|]

tensor_atanh
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_atanh _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).atanh(
    ));
  }|]

tensor_atanh_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_atanh_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).atanh_(
    ));
  }|]

tensor_arctanh
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_arctanh _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).arctanh(
    ));
  }|]

tensor_arctanh_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_arctanh_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).arctanh_(
    ));
  }|]

tensor_as_strided_lll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Ptr IntArray
  -> Int64
  -> IO (Ptr Tensor)
tensor_as_strided_lll _obj _size _stride _storage_offset =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).as_strided(
    *$(std::vector<int64_t>* _size)
  , *$(std::vector<int64_t>* _stride)
  , $(int64_t _storage_offset)));
  }|]

tensor_as_strided__lll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Ptr IntArray
  -> Int64
  -> IO (Ptr Tensor)
tensor_as_strided__lll _obj _size _stride _storage_offset =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).as_strided_(
    *$(std::vector<int64_t>* _size)
  , *$(std::vector<int64_t>* _stride)
  , $(int64_t _storage_offset)));
  }|]

tensor_asin
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_asin _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).asin(
    ));
  }|]

tensor_asin_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_asin_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).asin_(
    ));
  }|]

tensor_arcsin
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_arcsin _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).arcsin(
    ));
  }|]

tensor_arcsin_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_arcsin_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).arcsin_(
    ));
  }|]

tensor_atan
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_atan _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).atan(
    ));
  }|]

tensor_atan_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_atan_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).atan_(
    ));
  }|]

tensor_arctan
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_arctan _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).arctan(
    ));
  }|]

tensor_arctan_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_arctan_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).arctan_(
    ));
  }|]

tensor_baddbmm_ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_baddbmm_ttss _obj _batch1 _batch2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).baddbmm(
    *$(at::Tensor* _batch1)
  , *$(at::Tensor* _batch2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_baddbmm__ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_baddbmm__ttss _obj _batch1 _batch2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).baddbmm_(
    *$(at::Tensor* _batch1)
  , *$(at::Tensor* _batch2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

tensor_bernoulli_G
  :: Ptr Tensor
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_bernoulli_G _obj _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bernoulli(
    *$(at::Generator* _generator)));
  }|]

tensor_bernoulli__tG
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_bernoulli__tG _obj _p _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bernoulli_(
    *$(at::Tensor* _p)
  , *$(at::Generator* _generator)));
  }|]

tensor_bernoulli__dG
  :: Ptr Tensor
  -> CDouble
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_bernoulli__dG _obj _p _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bernoulli_(
    $(double _p)
  , *$(at::Generator* _generator)));
  }|]

tensor_bernoulli_dG
  :: Ptr Tensor
  -> CDouble
  -> Ptr Generator
  -> IO (Ptr Tensor)
tensor_bernoulli_dG _obj _p _generator =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bernoulli(
    $(double _p)
  , *$(at::Generator* _generator)));
  }|]

tensor_bincount_tl
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
tensor_bincount_tl _obj _weights _minlength =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bincount(
    *$(at::Tensor* _weights)
  , $(int64_t _minlength)));
  }|]

tensor_bitwise_not
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_bitwise_not _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_not(
    ));
  }|]

tensor_bitwise_not_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_bitwise_not_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bitwise_not_(
    ));
  }|]

tensor_copysign_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_copysign_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).copysign(
    *$(at::Tensor* _other)));
  }|]

tensor_copysign__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_copysign__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).copysign_(
    *$(at::Tensor* _other)));
  }|]

tensor_copysign_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_copysign_s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).copysign(
    *$(at::Scalar* _other)));
  }|]

tensor_copysign__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_copysign__s _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).copysign_(
    *$(at::Scalar* _other)));
  }|]

tensor_logical_not
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_not _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_not(
    ));
  }|]

tensor_logical_not_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_not_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_not_(
    ));
  }|]

tensor_logical_xor_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_xor_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_xor(
    *$(at::Tensor* _other)));
  }|]

tensor_logical_xor__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_xor__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_xor_(
    *$(at::Tensor* _other)));
  }|]

tensor_logical_and_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_and_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_and(
    *$(at::Tensor* _other)));
  }|]

tensor_logical_and__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_and__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_and_(
    *$(at::Tensor* _other)));
  }|]

tensor_logical_or_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_or_t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_or(
    *$(at::Tensor* _other)));
  }|]

tensor_logical_or__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_logical_or__t _obj _other =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).logical_or_(
    *$(at::Tensor* _other)));
  }|]

tensor_bmm_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_bmm_t _obj _mat2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).bmm(
    *$(at::Tensor* _mat2)));
  }|]

tensor_broadcast_to_l
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensor_broadcast_to_l _obj _size =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).broadcast_to(
    *$(std::vector<int64_t>* _size)));
  }|]

tensor_ceil
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ceil _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ceil(
    ));
  }|]

tensor_ceil_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_ceil_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).ceil_(
    ));
  }|]

tensor_unsafe_chunk_ll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr TensorList)
tensor_unsafe_chunk_ll _obj _chunks _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>((*$(at::Tensor* _obj)).unsafe_chunk(
    $(int64_t _chunks)
  , $(int64_t _dim)));
  }|]

tensor_chunk_ll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr TensorList)
tensor_chunk_ll _obj _chunks _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>((*$(at::Tensor* _obj)).chunk(
    $(int64_t _chunks)
  , $(int64_t _dim)));
  }|]

tensor_tensor_split_tl
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> IO (Ptr TensorList)
tensor_tensor_split_tl _obj _tensor_indices_or_sections _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>((*$(at::Tensor* _obj)).tensor_split(
    *$(at::Tensor* _tensor_indices_or_sections)
  , $(int64_t _dim)));
  }|]

tensor_clamp_ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_clamp_ss _obj _min _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp(
    *$(at::Scalar* _min)
  , *$(at::Scalar* _max)));
  }|]

tensor_clamp_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_clamp_tt _obj _min _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp(
    *$(at::Tensor* _min)
  , *$(at::Tensor* _max)));
  }|]

tensor_clamp__ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_clamp__ss _obj _min _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_(
    *$(at::Scalar* _min)
  , *$(at::Scalar* _max)));
  }|]

tensor_clamp__tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_clamp__tt _obj _min _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_(
    *$(at::Tensor* _min)
  , *$(at::Tensor* _max)));
  }|]

tensor_clamp_max_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_clamp_max_s _obj _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_max(
    *$(at::Scalar* _max)));
  }|]

tensor_clamp_max_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_clamp_max_t _obj _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_max(
    *$(at::Tensor* _max)));
  }|]

tensor_clamp_max__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_clamp_max__s _obj _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_max_(
    *$(at::Scalar* _max)));
  }|]

tensor_clamp_max__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_clamp_max__t _obj _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_max_(
    *$(at::Tensor* _max)));
  }|]

tensor_clamp_min_s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_clamp_min_s _obj _min =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_min(
    *$(at::Scalar* _min)));
  }|]

tensor_clamp_min_t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_clamp_min_t _obj _min =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_min(
    *$(at::Tensor* _min)));
  }|]

tensor_clamp_min__s
  :: Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_clamp_min__s _obj _min =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_min_(
    *$(at::Scalar* _min)));
  }|]

tensor_clamp_min__t
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_clamp_min__t _obj _min =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clamp_min_(
    *$(at::Tensor* _min)));
  }|]

tensor_clip_ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_clip_ss _obj _min _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clip(
    *$(at::Scalar* _min)
  , *$(at::Scalar* _max)));
  }|]

tensor_clip_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_clip_tt _obj _min _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clip(
    *$(at::Tensor* _min)
  , *$(at::Tensor* _max)));
  }|]

tensor_clip__ss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
tensor_clip__ss _obj _min _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clip_(
    *$(at::Scalar* _min)
  , *$(at::Scalar* _max)));
  }|]

tensor_clip__tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_clip__tt _obj _min _max =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).clip_(
    *$(at::Tensor* _min)
  , *$(at::Tensor* _max)));
  }|]

tensor_contiguous_M
  :: Ptr Tensor
  -> MemoryFormat
  -> IO (Ptr Tensor)
tensor_contiguous_M _obj _memory_format =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).contiguous(
    $(at::MemoryFormat _memory_format)));
  }|]

tensor_copy__tb
  :: Ptr Tensor
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
tensor_copy__tb _obj _src _non_blocking =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).copy_(
    *$(at::Tensor* _src)
  , $(bool _non_blocking)));
  }|]

tensor_cos
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cos _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cos(
    ));
  }|]

tensor_cos_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cos_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cos_(
    ));
  }|]

tensor_cosh
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cosh _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cosh(
    ));
  }|]

tensor_cosh_
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cosh_ _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cosh_(
    ));
  }|]

tensor_cov_ltt
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_cov_ltt _obj _correction _fweights _aweights =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).cov(
    $(int64_t _correction)
  , *$(at::Tensor* _fweights)
  , *$(at::Tensor* _aweights)));
  }|]

tensor_corrcoef
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tensor_corrcoef _obj =
  [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).corrcoef(
    ));
  }|]

tensor_cummax_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_cummax_l _obj _dim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).cummax(
    $(int64_t _dim)));
  }|]

tensor_cummax_n
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_cummax_n _obj _dim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).cummax(
    *$(at::Dimname* _dim)));
  }|]

tensor_cummin_l
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
tensor_cummin_l _obj _dim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>((*$(at::Tensor* _obj)).cummin(
    $(int64_t _dim)));
  }|]

