{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ParseFunctionSig where

--import Text.Megaparsec.Error as M

import Data.Aeson.Types ()
import Data.String.Conversions (cs)
import Data.Void (Void)
import Data.Yaml hiding (Array, Parser)
import GHC.Generics
import Text.Megaparsec as M
import Text.Megaparsec.Char as M
import Text.Megaparsec.Char.Lexer as L

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
-- - func: _cudnn_ctc_loss(Tensor log_probs, Tensor targets, IntList input_lengths, IntList target_lengths, int64_t blank, bool deterministic) -> (Tensor, Tensor)

data DefaultValue
  = ValBool Bool
  | ValInt Int
  | ValDouble Double
  | ValDict
  | ValArray
  | AtKLong
  | ReductionMean
  | NullPtr -- nullptr
  | ValNone
  deriving (Eq, Show)

data Parameter
  = Parameter
      { ptype :: Parsable,
        pname :: String,
        val :: Maybe DefaultValue
      }
  | Star -- , *,
  deriving (Eq, Show)

data Variants
  = VFunction
  | VMethod
  | VOperator
  deriving (Eq, Show)

data Function = Function
  { name :: String,
    parameters :: [Parameter],
    retType :: Parsable,
    variant :: Variants
  }
  deriving (Eq, Show)

type SignatureStr = String

type CppTypeStr = String

type HsTypeStr = String

data Parsable
  = Ptr Parsable
  | TenType TenType
  | DeviceType
  | GeneratorType
  | StorageType
  | CType CType
  | STLType STLType
  | ArrayRef CType
  | ArrayRefScalar
  | CppString
  | Tuple [Parsable]
  | CppClass SignatureStr CppTypeStr HsTypeStr
  | Backend
  | Layout
  | MemoryFormat
  | QScheme
  | ConstQuantizerPtr
  | Dimname
  | DimnameList
  | Symbol
  | IValue
  | Stream
  deriving (Eq, Show, Generic)

data CType
  = CBool
  | CVoid
  | CFloat
  | CDouble
  | CSize
  | CInt
  | CUInt8
  | CUInt16
  | CUInt32
  | CUInt64
  | CInt8
  | CInt16
  | CInt32
  | CInt64
  | CInt64Q
  | CString
  deriving (Eq, Show, Generic, Bounded, Enum)

data STLType
  = Array CType Int
  deriving (Eq, Show, Generic)

data TenType
  = Scalar
  | Tensor
  | TensorA -- Tensor(a)
  | TensorA' -- Tensor(a!)
  | TensorAQ -- Tensor(a)?
  | TensorAQ' -- Tensor(a!)?
  | TensorQ -- Tensor?
  | TensorAVector -- Tensor(a)[]
  | TensorOptions
  | TensorList
  | C10ListTensor
  | ITensorListRef
  | IntegerTensor
  | IndexTensor
  | BoolTensor
  | BoolTensorQ
  | ByteTensor
  | LongTensor
  | IntList {dim :: Maybe [Int]}
  | ScalarQ
  | ScalarType
  deriving (Eq, Show)

type Parser = Parsec Void String

cppClassList :: [(String, String, String)]
cppClassList = [("IntArray", "std::vector<int64_t>", "IntArray")]

defBool :: Parser DefaultValue
defBool = do
  val' <- string "true" <|> string "false" <|> string "True" <|> string "False"
  pure $ if val' == "true" || val' == "True" then ValBool True else ValBool False

defInt :: Parser DefaultValue
defInt = do
  val' <- pinteger
  pure $ ValInt (fromIntegral val')

defFloat :: Parser DefaultValue
defFloat = do
  val' <- L.scientific
  pure $ ValDouble (realToFrac val')

sc :: Parser ()
sc = L.space space1 empty empty

lexm :: Parser a -> Parser a
lexm = L.lexeme sc

parens :: Parser a -> Parser a
parens = between (string "(") (string ")")

pinteger :: Parser Integer
pinteger =
  (L.decimal)
    <|> ((string "-") >> L.decimal >>= \v -> pure (- v))

pfloat :: Parser Float
pfloat = L.float

rword :: String -> Parser ()
rword w = (lexm . try) (string w *> notFollowedBy alphaNumChar)

rws :: [String]
rws = []

identStart :: [Char]
identStart = ['a' .. 'z'] ++ ['A' .. 'Z'] ++ ['_']

identLetter :: [Char]
identLetter = ['a' .. 'z'] ++ ['A' .. 'Z'] ++ ['_'] ++ ['0' .. '9'] ++ [':', '<', '>']

-- | parser of identifier
--
-- >>> parseTest identifier "fft"
-- "fft"
-- >>> parseTest identifier "_fft"
-- "_fft"
identifier :: Parser String
identifier = (lexm . try) (p >>= check)
  where
    p = (:) <$> (oneOf identStart) <*> many (oneOf identLetter)
    check x =
      if x `elem` rws
        then fail $ "keyword " ++ show x ++ " cannot be an identifier"
        else return x

-- | parser of identifier
--
-- >>> parseTest typ "BoolTensor"
-- TenType BoolTensor
-- >>> parseTest typ "BoolTensor?"
-- TenType BoolTensorQ
-- >>> parseTest typ "ByteTensor"
-- TenType ByteTensor
-- >>> parseTest typ "Device"
-- DeviceType
-- >>> parseTest typ "Generator"
-- GeneratorType
-- >>> parseTest typ "IndexTensor"
-- TenType IndexTensor
-- >>> parseTest typ "IntegerTensor"
-- TenType IntegerTensor
-- >>> parseTest typ "IntArrayRef"
-- TenType (IntList {dim = Nothing})
-- >>> parseTest typ "IntList"
-- TenType (IntList {dim = Nothing})
-- >>> parseTest typ "IntList[1]"
-- TenType (IntList {dim = Just [1]})
-- >>> parseTest typ "int[]"
-- TenType (IntList {dim = Just []})
-- >>> parseTest typ "int[1]"
-- TenType (IntList {dim = Just [1]})
-- >>> parseTest typ "ScalarType"
-- TenType ScalarType
-- >>> parseTest typ "real"
-- TenType Scalar
-- >>> parseTest typ "accreal"
-- TenType Scalar
-- >>> parseTest typ "Scalar"
-- TenType Scalar
-- >>> parseTest typ "Scalar?"
-- TenType ScalarQ
-- >>> parseTest typ "ScalarType"
-- TenType ScalarType
-- >>> parseTest typ "Backend"
-- Backend
-- >>> parseTest typ "Layout"
-- Layout
-- >>> parseTest typ "MemoryFormat"
-- MemoryFormat
-- >>> parseTest typ "QScheme"
-- QScheme
-- >>> parseTest typ "Dimname"
-- Dimname
-- >>> parseTest typ "DimnameList"
-- DimnameList
-- >>> parseTest typ "Symbol"
-- Symbol
-- >>> parseTest typ "IValue"
-- IValue
-- >>> parseTest typ "Stream"
-- Stream
-- >>> parseTest typ "Storage"
-- StorageType
-- >>> parseTest typ "Tensor"
-- TenType Tensor
-- >>> parseTest typ "Tensor?"
-- TenType TensorQ
-- >>> parseTest typ "Tensor(a)"
-- TenType TensorA
-- >>> parseTest typ "Tensor(a!)"
-- TenType TensorA'
-- >>> parseTest typ "Tensor(a)[]"
-- TenType TensorAVector
-- >>> parseTest typ "Tensor[]"
-- TenType TensorList
-- >>> parseTest typ "Tensor?[]"
-- TenType TensorList
-- >>> parseTest typ "TensorList"
-- TenType TensorList
-- >>> parseTest typ "const c10::List<c10::optional<Tensor>> &"
-- TenType C10ListTensor
-- >>> parseTest typ "const at::ITensorListRef &"
-- TenType ITensorListRef
-- >>> parseTest typ "TensorOptions"
-- TenType TensorOptions
-- >>> parseTest typ "bool"
-- CType CBool
-- >>> parseTest typ "double"
-- CType CDouble
-- >>> parseTest typ "float"
-- CType CFloat
-- >>> parseTest typ "int"
-- CType CInt
-- >>> parseTest typ "int?"
-- CType CInt
-- >>> parseTest typ "int64_t"
-- CType CInt64
-- >>> parseTest typ "int64_t?"
-- CType CInt64Q
-- >>> parseTest typ "size_t"
-- CType CSize
-- >>> parseTest typ "std::array<bool,2>"
-- STLType (Array CBool 2)
-- >>> parseTest typ "bool[2]"
-- STLType (Array CBool 2)
-- >>> parseTest typ "std::string"
-- CppString
-- >>> parseTest typ "str"
-- CppString
-- >>> parseTest typ "char*"
-- CType CString
-- >>> parseTest typ "ArrayRef<double>"
-- ArrayRef CDouble
typ :: Parser Parsable
typ =
  tuple
    <|> idxtensor
    <|> booltensorq
    <|> booltensor
    <|> bytetensor
    <|> tensor
    <|> intlistDim
    <|> intlistNoDim
    <|> intpDim
    <|> intpNoDim
    <|> try intpDim'
    <|> try intpNoDim'
    <|> intpNoDim''
    <|> scalar
    <|> try stlbool
    <|> ctype
    <|> stl
    <|> try arrayrefScalar
    <|> arrayref
    <|> cppstring
    <|> cppclass
    <|> other
  where
    tuple = do
      _ <- lexm $ string "("
      val' <- (sepBy typ (lexm (string ",")))
      _ <- lexm $ string ")"
      pure $ Tuple val'
    other =
      ((lexm $ string "Backend") >> (pure $ Backend))
        <|> ((lexm $ try (string "at::Layout") <|> string "Layout") >> (pure $ Layout))
        <|> ((lexm $ try (string "at::MemoryFormat") <|> string "MemoryFormat") >> (pure $ MemoryFormat))
        <|> ((lexm $ try (string "at::QScheme") <|> string "QScheme") >> (pure $ QScheme))
        <|> ((lexm $ try (string "at::DimnameList") <|> string "DimnameList") >> (pure $ DimnameList))
        <|> try ((lexm $ try (string "at::Dimname") <|> string "Dimname") >> (pure $ Dimname))
        <|> ((lexm $ string "Symbol") >> (pure $ Symbol))
        <|> ((lexm $ try (string "at::Device") <|> string "Device") >> (pure $ DeviceType))
        <|> ((lexm $ try (string "at::Generator") <|> string "Generator") >> (pure $ GeneratorType))
        <|> ((lexm $ try (string "at::Storage") <|> string "Storage") >> (pure $ StorageType))
        <|> ((lexm $ string "ConstQuantizerPtr") >> (pure $ ConstQuantizerPtr))
        <|> ((lexm $ string "IValue") >> (pure $ IValue))
        <|> ((lexm $ try (string "at::Stream") <|> string "Stream") >> (pure $ Stream))
    cppclass = foldl (<|>) (fail "Can not parse cpptype.") $ map (\(sig, cpptype, hstype) -> ((lexm $ string sig) >> (pure $ CppClass sig cpptype hstype))) cppClassList
    scalar =
      ((lexm $ string "Scalar?") >> (pure $ TenType ScalarQ))
        <|> ((lexm $ try (string "at::ScalarType") <|> string "ScalarType") >> (pure $ TenType ScalarType))
        <|> ((lexm $ try (string "const at::Scalar &") <|> string "Scalar") >> (pure $ TenType Scalar))
        <|> ((lexm $ string "real") >> (pure $ TenType Scalar))
        <|> ((lexm $ string "accreal") >> (pure $ TenType Scalar))
    idxtensor = do
      _ <- lexm $ string "IndexTensor"
      pure $ TenType IndexTensor
    booltensor = do
      _ <- lexm $ string "BoolTensor"
      pure $ TenType BoolTensor
    booltensorq = do
      _ <- lexm $ string "BoolTensor?"
      pure $ TenType BoolTensorQ
    bytetensor = do
      _ <- lexm $ string "ByteTensor"
      pure $ TenType ByteTensor
    tensor =
      ((lexm $ string "IntegerTensor") >> (pure $ TenType IntegerTensor))
        <|> ((lexm $ try (string "at::TensorOptions") <|> string "TensorOptions") >> (pure $ TenType TensorOptions))
        <|> ((lexm $ try (string "at::TensorList") <|> string "TensorList") >> (pure $ TenType TensorList))
        <|> try ((lexm $ string "Tensor[]") >> (pure $ TenType TensorList))
        <|> try ((lexm $ string "Tensor?[]") >> (pure $ TenType TensorList))
        <|> try ((lexm $ try (string "const c10::List<c10::optional<at::Tensor>> &") <|> string "const c10::List<c10::optional<Tensor>> &") >> (pure $ TenType C10ListTensor))
        <|> try ((lexm $ string "const at::ITensorListRef &") >> (pure $ TenType ITensorListRef))
        <|> try ((lexm $ string "Tensor(a)[]") >> (pure $ TenType TensorAVector))
        <|> try ((lexm $ string "Tensor(a)") >> (pure $ TenType TensorA))
        <|> try ((lexm $ string "Tensor(a!)") >> (pure $ TenType TensorA'))
        <|> try ((lexm $ string "Tensor(b)") >> (pure $ TenType TensorA))
        <|> try ((lexm $ string "Tensor(b!)") >> (pure $ TenType TensorA'))
        <|> try ((lexm $ string "Tensor(c)") >> (pure $ TenType TensorA))
        <|> try ((lexm $ string "Tensor(c!)") >> (pure $ TenType TensorA'))
        <|> try ((lexm $ string "Tensor(d)") >> (pure $ TenType TensorA))
        <|> try ((lexm $ string "Tensor(d!)") >> (pure $ TenType TensorA'))
        <|> try ((lexm $ string "Tensor?(a)") >> (pure $ TenType TensorAQ))
        <|> try ((lexm $ string "Tensor?(a!)") >> (pure $ TenType TensorAQ'))
        <|> try ((lexm $ string "Tensor?(b)") >> (pure $ TenType TensorAQ))
        <|> try ((lexm $ string "Tensor?(b!)") >> (pure $ TenType TensorAQ'))
        <|> try ((lexm $ string "Tensor?(c)") >> (pure $ TenType TensorAQ))
        <|> try ((lexm $ string "Tensor?(c!)") >> (pure $ TenType TensorAQ'))
        <|> try ((lexm $ string "Tensor?(d)") >> (pure $ TenType TensorAQ))
        <|> try ((lexm $ string "Tensor?(d!)") >> (pure $ TenType TensorAQ'))
        <|> ((lexm $ string "Tensor?") >> (pure $ TenType TensorQ))
        <|> ((lexm $ try (string "at::Tensor") <|> string "Tensor") >> (pure $ TenType Tensor))
        <|> ((lexm $ string "LongTensor") >> (pure $ TenType LongTensor))
    intlistDim = do
      _ <- lexm $ string "IntList["
      val' <- (sepBy pinteger (lexm (string ",")))
      _ <- lexm $ string "]"
      pure $ TenType $ IntList (Just (map fromIntegral val'))
    intlistNoDim = do
      _ <- lexm $ string "IntList"
      pure $ TenType $ IntList Nothing
    intpDim = do
      _ <- lexm $ string "int["
      val' <- (sepBy pinteger (lexm (string ",")))
      _ <- lexm $ string "]"
      pure $ TenType $ IntList (Just (map fromIntegral val'))
    intpNoDim = do
      _ <- lexm $ string "int[]"
      pure $ TenType $ IntList Nothing
    intpDim' = do
      _ <- lexm $ try (string "at::IntArrayRef[") <|> string "IntArrayRef["
      val' <- (sepBy pinteger (lexm (string ",")))
      _ <- lexm $ string "]"
      pure $ TenType $ IntList (Just (map fromIntegral val'))
    intpNoDim' = do
      _ <- lexm $ try (string "at::IntArrayRef[]") <|> string "IntArrayRef[]"
      pure $ TenType $ IntList Nothing
    intpNoDim'' = do
      _ <- lexm $ try (string "at::IntArrayRef") <|> string "IntArrayRef"
      pure $ TenType $ IntList Nothing
    ctype =
      ((lexm $ string "bool") >> (pure $ CType CBool))
        <|> ((lexm $ string "char*") >> (pure $ CType CString))
        <|> ((lexm $ string "void*") >> (pure $ Ptr (CType CVoid)))
        <|> ((lexm $ string "void") >> (pure $ CType CVoid))
        <|> ((lexm $ string "float") >> (pure $ CType CFloat))
        <|> ((lexm $ string "double") >> (pure $ CType CDouble))
        <|> ((lexm $ string "size_t") >> (pure $ CType CSize))
        <|> try ((lexm $ string "int64_t?") >> (pure $ CType CInt64Q))
        <|> try ((lexm $ string "int64_t") >> (pure $ CType CInt64))
        <|> try ((lexm $ string "int32_t") >> (pure $ CType CInt32))
        <|> try ((lexm $ string "int16_t") >> (pure $ CType CInt16))
        <|> try ((lexm $ string "int8_t") >> (pure $ CType CInt8))
        <|> try ((lexm $ string "uint64_t") >> (pure $ CType CUInt64))
        <|> try ((lexm $ string "uint32_t") >> (pure $ CType CUInt32))
        <|> try ((lexm $ string "uint16_t") >> (pure $ CType CUInt16))
        <|> try ((lexm $ string "uint8_t") >> (pure $ CType CUInt8))
        <|> try ((lexm $ string "int?") >> (pure $ CType CInt))
        <|> ((lexm $ string "int") >> (pure $ CType CInt))
    stl = do
      _ <- lexm $ string "std::array<"
      val' <- ctype
      _ <- lexm $ string ","
      num <- pinteger
      _ <- lexm $ string ">"
      case val' of
        CType v -> pure $ STLType $ Array v (fromIntegral num)
        _ -> fail "Can not parse ctype."
    stlbool = do
      _ <- lexm $ string "bool["
      num <- pinteger
      _ <- lexm $ string "]"
      pure $ STLType $ Array CBool (fromIntegral num)
    arrayref = do
      _ <- lexm $ try (string "at::ArrayRef<") <|> string "ArrayRef<"
      val' <- ctype
      _ <- lexm $ string ">"
      case val' of
        CType v -> pure $ ArrayRef v
        _ -> fail "Can not parse ctype."
    arrayrefScalar = lexm $ string "at::ArrayRef<at::Scalar>" >> pure ArrayRefScalar
    cppstring =
      try ((lexm $ string "std::string") >> (pure $ CppString))
        <|> try ((lexm $ string "c10::string_view") >> (pure $ CppString))
        <|> ((lexm $ string "str") >> (pure $ CppString))

-- | parser of defaultValue
--
-- >>> parseTest defaultValue "-100"
-- ValInt (-100)
-- >>> parseTest defaultValue "20"
-- ValInt 20
-- >>> parseTest defaultValue "0.125"
-- ValDouble 0.125
-- >>> parseTest defaultValue "1e-8"
-- ValDouble 1.0e-8
-- >>> parseTest defaultValue "False"
-- ValBool False
-- >>> parseTest defaultValue "None"
-- ValNone
-- >>> parseTest defaultValue "Reduction::Mean"
-- ReductionMean
-- >>> parseTest defaultValue "Mean"
-- ReductionMean
-- >>> parseTest defaultValue "True"
-- ValBool True
-- >>> parseTest defaultValue "at::kLong"
-- AtKLong
-- >>> parseTest defaultValue "false"
-- ValBool False
-- >>> parseTest defaultValue "nullptr"
-- NullPtr
-- >>> parseTest defaultValue "true"
-- ValBool True
-- >>> parseTest defaultValue "{0,1}"
-- ValDict
-- >>> parseTest defaultValue "{}"
-- ValDict
-- >>> parseTest defaultValue "[0,1]"
-- ValArray
-- >>> parseTest defaultValue "[]"
-- ValArray
defaultValue :: Parser DefaultValue
defaultValue =
  try floatp
    <|> try intp
    <|> defBool
    <|> nullp
    <|> nonep
    <|> reductionp
    <|> reductionp'
    <|> atklongp
    <|> try dict01
    <|> dict
    <|> try ary01
    <|> ary
  where
    intp = do
      val' <- lexm $ pinteger :: Parser Integer
      pure $ ValInt (fromIntegral val')
    floatp = do
      v <- lexm $ L.float :: Parser Double
      pure $ ValDouble v
    nullp = do
      _ <- lexm $ string "nullptr"
      pure NullPtr
    reductionp = do
      _ <- lexm $ string "Reduction::Mean"
      pure ReductionMean
    reductionp' = do
      _ <- lexm $ string "Mean"
      pure ReductionMean
    atklongp = do
      _ <- lexm $ string "at::kLong"
      pure AtKLong
    dict = do
      _ <- lexm $ string "{}"
      pure ValDict
    dict01 = do
      _ <- lexm $ string "{0,1}"
      pure ValDict
    ary01 = do
      _ <- lexm $ string "[0,1]"
      pure ValArray
    ary = do
      _ <- lexm $ string "[]"
      pure ValArray
    nonep = do
      _ <- lexm $ string "None"
      pure ValNone

-- | parser of argument
--
-- >>> parseTest arg "*"
-- Star
-- >>> parseTest arg "Tensor self"
-- Parameter {ptype = TenType Tensor, pname = "self", val = Nothing}
-- >>> Right v = parse (sepBy arg (lexm (string ","))) "" "Tensor self, Tensor self"
-- >>> map ptype v
-- [TenType Tensor,TenType Tensor]
-- >>> Right v = parse (sepBy arg (lexm (string ","))) "" "Tensor self, Tensor? self"
-- >>> map ptype v
-- [TenType Tensor,TenType TensorQ]
arg :: Parser Parameter
arg = star <|> param
  where
    param = do
      -- ptype <- lexm $ identifier
      pt <- typ
      pn <- lexm $ identifier
      let withDefault = do
            _ <- lexm (string "=")
            v <- defaultValue
            pure (Just v)
      val' <- withDefault <|> (pure Nothing)
      pure $ Parameter pt pn val'
    star = do
      _ <- string "*"
      pure Star

-- | parser of argument
--
-- >>> parseTest rettype "Tensor"
-- TenType Tensor
-- >>> parseTest rettype "Tensor hoo"
-- TenType Tensor
-- >>> parseTest rettype "(Tensor hoo,Tensor bar)"
-- Tuple [TenType Tensor,TenType Tensor]
rettype :: Parser Parsable
rettype = tuple <|> single'
  where
    tuple = do
      _ <- lexm $ string "("
      val' <- (sepBy rettype (lexm (string ",")))
      _ <- lexm $ string ")"
      pure $ Tuple val'
    single' = do
      type' <- typ
      _ <- ((do v <- lexm (identifier); pure (Just v)) <|> (pure Nothing))
      pure type'

-- | parser of function
--
-- >>> parseTest func "log10_(Tensor self) -> Tensor"
-- Function {name = "log10_", parameters = [Parameter {ptype = TenType Tensor, pname = "self", val = Nothing}], retType = TenType Tensor, variant = VFunction}
-- >>> parseTest func "fft(Tensor self, int64_t signal_ndim, bool normalized=false) -> Tensor"
-- Function {name = "fft", parameters = [Parameter {ptype = TenType Tensor, pname = "self", val = Nothing},Parameter {ptype = CType CInt64, pname = "signal_ndim", val = Nothing},Parameter {ptype = CType CBool, pname = "normalized", val = Just (ValBool False)}], retType = TenType Tensor, variant = VFunction}
-- >>> parseTest func "frobenius_norm_out(Tensor result, Tensor self, IntList[1] dim, bool keepdim=false) -> Tensor"
-- Function {name = "frobenius_norm_out", parameters = [Parameter {ptype = TenType Tensor, pname = "result", val = Nothing},Parameter {ptype = TenType Tensor, pname = "self", val = Nothing},Parameter {ptype = TenType (IntList {dim = Just [1]}), pname = "dim", val = Nothing},Parameter {ptype = CType CBool, pname = "keepdim", val = Just (ValBool False)}], retType = TenType Tensor, variant = VFunction}
-- >>> parseTest func "thnn_conv_dilated3d_forward(Tensor self, Tensor weight, IntList[3] kernel_size, Tensor? bias, IntList[3] stride, IntList[3] padding, IntList[3] dilation) -> (Tensor output, Tensor columns, Tensor ones)"
-- Function {name = "thnn_conv_dilated3d_forward", parameters = [Parameter {ptype = TenType Tensor, pname = "self", val = Nothing},Parameter {ptype = TenType Tensor, pname = "weight", val = Nothing},Parameter {ptype = TenType (IntList {dim = Just [3]}), pname = "kernel_size", val = Nothing},Parameter {ptype = TenType TensorQ, pname = "bias", val = Nothing},Parameter {ptype = TenType (IntList {dim = Just [3]}), pname = "stride", val = Nothing},Parameter {ptype = TenType (IntList {dim = Just [3]}), pname = "padding", val = Nothing},Parameter {ptype = TenType (IntList {dim = Just [3]}), pname = "dilation", val = Nothing}], retType = Tuple [TenType Tensor,TenType Tensor,TenType Tensor], variant = VFunction}
func :: Parser Function
func = try operator <|> function
  where
    function = do
      fName <- identifier
      _ <- lexm $ string "("
      -- parse list of parameters
      args <- (sepBy arg (lexm (string ",")))
      _ <- lexm $ string ")"
      _ <- lexm $ string "->"
      retType' <- rettype
      pure $ Function fName args retType' VFunction
    operator = do
      _ <- lexm $ string "operator"
      fName <-
        try (string "+=")
          <|> try (string "-=")
          <|> try (string "*=")
          <|> try (string "/=")
          <|> string "="
          <|> string "-"
          <|> string "+"
          <|> string "*"
          <|> string "/"
          <|> string "[]"
      _ <- lexm $ string "("
      -- parse list of parameters
      args <- (sepBy arg (lexm (string ",")))
      _ <- lexm $ string ")"
      _ <- lexm $ string "->"
      retType' <- rettype
      pure $ Function fName args retType' VOperator

test :: IO ()
test = do
  --parseTest defBool "true"
  parseTest func "foo() -> Tensor"
  parseTest
    func
    "fft(Tensor self, int64_t signal_ndim, bool normalized=false) -> Tensor"

parseFuncSig :: String -> Either (ParseErrorBundle String Void) Function
parseFuncSig sig = parse func "" sig

instance FromJSON Parsable where
  parseJSON (String v) = do
    case parse typ "" (cs v) of
      Left err -> fail (errorBundlePretty err)
      Right p -> pure p
  parseJSON _ = fail "This type is not string-type."

instance FromJSON Function where
  parseJSON (String v) = do
    case parse func "" (cs v) of
      Left err -> fail (errorBundlePretty err)
      Right p -> pure p
  parseJSON _ = fail "This type is not function-type."
