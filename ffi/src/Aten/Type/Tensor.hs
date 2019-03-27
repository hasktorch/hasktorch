module Aten.Type.Tensor
       (Tensor(..), ITensor, upcastTensor, downcastTensor, newTensor,
        tensor_dim, tensor_storage_offset, tensor_defined, tensor_reset,
        tensor_cpu, tensor_cuda, tensor_print)
       where
import Aten.Type.Tensor.RawType
import Aten.Type.Tensor.Interface
import Aten.Type.Tensor.Implementation
