{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE OverloadedStrings #-}

module ParseFunctionSig where

import Data.Void (Void)
import GHC.Generics
import Text.Megaparsec as M
import Text.Megaparsec.Error as M
import Text.Megaparsec.Char as M

-- Examples:
-- - func: log10_(Tensor self) -> Tensor
-- - func: fft(Tensor self, int64_t signal_ndim, bool normalized=false) -> Tensor
-- - func: expand(Tensor self, IntList size, *, bool implicit=false) -> Tensor
-- - func: frobenius_norm_out(Tensor result, Tensor self, IntList[1] dim, bool keepdim=false) -> Tensor
-- - func: thnn_conv_dilated3d_forward(Tensor self, Tensor weight, IntList[3] kernel_size, Tensor? bias, IntList[3] stride, IntList[3] padding, IntList[3] dilation) -> (Tensor output, Tensor columns, Tensor ones)
-- - func: _cudnn_rnn_backward(Tensor input, TensorList weight, int64_t weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, BoolTensor? dropout_state, Tensor reserve, std::array<bool,4> output_mask) -> (Tensor, Tensor, Tensor, TensorList)
-- - func: einsum(std::string equation, TensorList tensors) -> Tensor
-- - func: empty(IntList size, TensorOptions options={}) -> Tensor
-- - func: conv3d(Tensor input, Tensor weight, Tensor? bias={}, IntList[3] stride=1, IntList[3] padding=0, IntList[3] dilation=1, int64_t groups=1) -> Tensor

data DefaultValue =
    ValBool Bool
    | ValInt Int
    | ValDouble Double
    | ValDict
    | AtKLong
    | ReductionMean
    | NullPtr -- nullptr 
    deriving Show

data Parameter  = Parameter {
    ptype :: Parsable
    , pname :: String
    , val :: Maybe DefaultValue
    } | Star  -- , *,  
    deriving Show

data Function  = Function {
    name :: String
    , parameters :: [Parameter]
    , retVal :: String
} deriving Show

data Parsable
    = Ptr Parsable
    | TenType TenType
    | CType CType
    | Tuple Parsable Parsable
    deriving (Show, Generic)

data CType
    = CBool
    | CVoid
    | CDouble
    | CInt64
    deriving (Eq, Show, Generic, Bounded, Enum)

data STLType
    = Array CType Int

data TenType = Scalar
    | Tensor
    | TensorQ -- Tensor?
    | TensorOptions
    | IntList { dim :: Maybe [Int] }
    deriving Show

type Parser = Parsec Void String

defBool :: Parser DefaultValue
defBool = do
  val <- string "true" <|> string "false"
  pure $ if val == "true" then ValBool True else ValBool False

