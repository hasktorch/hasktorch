

{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE QuasiQuotes #-}
module RenderCommon where

import Text.Shakespeare.Text (st)
import Data.Char (toLower)
import Data.Text (Text)
import Data.String (fromString)
import qualified Data.Text as T
import qualified Data.List as L

import ParseFunctionSig as P
import ParseClass as PC

bra :: Text
bra = "["

cket :: Text
cket = "]"

------ To Cpp Type ------
tenTypeToCppType :: TenType -> Text
tenTypeToCppType tentype =
  case tentype of
    Scalar -> "at::Scalar"
    Tensor -> "at::Tensor"
    TensorA -> "at::Tensor"
    TensorA' -> "at::Tensor"
    TensorAQ -> "at::Tensor"
    TensorAQ' -> "at::Tensor"
    TensorQ -> "at::Tensor"
    TensorAVector -> "std::vector<at::Tensor>"
    TensorOptions -> "at::TensorOptions"
    TensorList -> "std::vector<at::Tensor>"
    IndexTensor -> "at::Tensor"
    IntegerTensor -> "at::Tensor"
    BoolTensor -> "at::Tensor"
    BoolTensorQ -> "at::Tensor"
    ByteTensor -> "at::Tensor"
    LongTensor -> "at::Tensor"
-- In libtorch, IntList is at::IntArrayRef. at::IntArrayRef is refernece-type supporting various array-types.
-- When we use at::IntArrayRef directly, it is diffcult to manage memory by GHC.
-- Because at::IntArrayRef do not managae refering memory.
-- For now, this codes uses std::vector<int64_t> instead of at::IntArrayRef.
--    IntList _ -> "at::IntArrayRef"
    IntList _ -> "std::vector<int64_t>"
    ScalarQ -> "at::Scalar"
    ScalarType -> "at::ScalarType"
    SparseTensorRef -> "at::SparseTensorRef"

ctypeToCppType :: CType -> Text
ctypeToCppType ct =
  case ct of
    CBool -> "bool"
    CVoid -> "void"
    CFloat -> "float"
    CDouble -> "double"
    CSize -> "size_t"
    CInt -> "int"
    CUInt8 -> "uint8_t"
    CUInt16 -> "uint16_t"
    CUInt32 -> "uint32_t"
    CUInt64 -> "uint64_t"
    CInt8 -> "int8_t"
    CInt16 -> "int16_t"
    CInt32 -> "int32_t"
    CInt64 -> "int64_t"
    CInt64Q -> "int64_t"

stltypeToCppType :: STLType -> Text
stltypeToCppType t =
  case t of
    P.Array ct len -> [st|std::array<#{ctypeToCppType ct},#{len}>|]


parsableToCppType :: Parsable -> Text
parsableToCppType parsable =
  case parsable of
    Ptr p -> parsableToCppType p <> " *"
    TenType t -> tenTypeToCppType t
    DeviceType -> "at::DeviceType"
    GeneratorType -> "at::Generator"
    StorageType -> "at::Storage"
    CType ct -> ctypeToCppType ct
    STLType t -> stltypeToCppType t
    CppString -> "std::string"
    Tuple parsables -> [st|std::tuple<#{T.intercalate "," (map parsableToCppType parsables)}>|]
    P.CppClass _ cpptype _ -> fromString cpptype
    Backend -> "at::Backend"
    Layout -> "at::Layout"


------ To Haskell Type ------
tenTypeToHsType :: TenType -> Text
tenTypeToHsType tentype =
  case tentype of
    Scalar -> "Scalar"
    Tensor -> "Tensor"
    TensorA -> "Tensor"
    TensorA' -> "Tensor"
    TensorAQ -> "Tensor"
    TensorAQ' -> "Tensor"
    TensorQ -> "Tensor"
    TensorAVector -> "TensorAVector"
    TensorOptions -> "TensorOptions"
    TensorList -> "TensorList"
    IntegerTensor -> "Tensor"
    IndexTensor -> "Tensor"
    BoolTensor -> "Tensor"
    BoolTensorQ -> "Tensor"
    ByteTensor -> "Tensor"
    LongTensor -> "Tensor"
--    IntList _ -> "IntArrayRef"
    IntList _ -> "IntArray"
    ScalarQ -> "Scalar"
    ScalarType -> "ScalarType"
    SparseTensorRef -> "SparseTensorRef"

stltypeToHsType :: STLType -> Text
stltypeToHsType t =
  case t of
    P.Array ct len -> [st|(StdArray #{ctypeToHsType ct} #{len})|]

ctypeToHsType :: CType -> Text
ctypeToHsType ct =
  case ct of
    CBool -> "CBool"
    CVoid -> "()"
    CFloat -> "CFloat"
    CDouble -> "CDouble"
    CSize -> "CSize"
    CInt -> "CInt"
    CUInt8 -> "Word8"
    CUInt16 -> "Word16"
    CUInt32 -> "Word32"
    CUInt64 -> "Word64"
    CInt8 -> "Int8"
    CInt16 -> "Int16"
    CInt32 -> "Int32"
    CInt64 -> "Int64"
    CInt64Q -> "Int64"

parsableToHsType :: Parsable -> Text
parsableToHsType parsable =
  case parsable of
    Ptr p -> "Ptr " <> parsableToHsType p
    TenType t -> tenTypeToHsType t
    DeviceType -> "DeviceType"
    GeneratorType -> "Generator"
    StorageType -> "Storage"
    CType ct -> ctypeToHsType ct
    STLType t -> stltypeToHsType t
    CppString -> "StdString"
    Tuple parsables -> [st|(#{T.intercalate "," (map parsableToHsType parsables)})|]
    P.CppClass _ _ hstype -> fromString hstype
    Backend -> "Backend"
    Layout -> "Layout"


------ To initial characters ------
tenTypeToInitial :: TenType -> Text
tenTypeToInitial tentype =
  case tentype of
    Scalar -> "s"
    Tensor -> "t"
    TensorA -> "t"
    TensorA' -> "T"
    TensorAQ -> "t"
    TensorAQ' -> "T"
    TensorQ -> "t"
    TensorAVector -> "v"
    TensorOptions -> "o"
    TensorList -> "l"
    IndexTensor -> "t"
    IntegerTensor -> "t"
    BoolTensor -> "t"
    BoolTensorQ -> "t"
    ByteTensor -> "t"
    LongTensor -> "L"
    IntList _ -> "l"
    ScalarQ -> "s"
    ScalarType -> "s"
    SparseTensorRef -> "r"

stltypeToInitial :: STLType -> Text
stltypeToInitial t =
  case t of
    P.Array _ _ -> "a"

ctypeToInitial :: CType -> Text
ctypeToInitial ct =
  case ct of
    CBool -> "b"
    CVoid -> "v"
    CFloat -> "f"
    CDouble -> "d"
    CSize -> "s"
    CInt -> "i"
    CUInt8 -> "B"
    CUInt16 -> "S"
    CUInt32 -> "I"
    CUInt64 -> "L"
    CInt8 -> "b"
    CInt16 -> "s"
    CInt32 -> "i"
    CInt64 -> "l"
    CInt64Q -> "l"

parsableToInitial :: Parsable -> Text
parsableToInitial parsable =
  case parsable of
    Ptr _ -> "p"
    TenType t -> tenTypeToInitial t
    DeviceType -> "D"
    GeneratorType -> "G"
    StorageType -> "S"
    CType ct -> ctypeToInitial ct
    STLType t -> stltypeToInitial t
    CppString -> "s"
    Tuple _ -> "t"
    P.CppClass _ _ _ -> "c"
    Backend -> "B"
    Layout -> "L"

isCType :: Parsable -> Bool
isCType p =
  case p of
    TenType ScalarType -> True
    DeviceType -> True
    Backend -> True
    Layout -> True
    CType _ -> True
    Ptr _ -> True
    _ -> False

isNotStar :: Parameter -> Bool
isNotStar p =
  case p of
    Star -> False
    Parameter _ _ _ -> True

retToCppType :: Parsable -> Text
retToCppType parsable =
  case parsable of
    Ptr p -> parsableToCppType p <> " *"
    TenType t -> tenTypeToCppType t
    DeviceType -> "DeviceType"
    GeneratorType -> "Generator"
    StorageType -> "Storage"
    CType ct -> ctypeToCppType ct
    STLType t -> stltypeToCppType t
    CppString -> "std::string"
    Tuple parsables -> [st|tuple<#{T.intercalate "," (map parsableToCppType parsables)}>|]
    P.CppClass _ cpptype _ -> fromString cpptype
    Backend -> "at::Backend"
    Layout -> "at::Layout"



toHsFuncName :: Bool -> String -> String
toHsFuncName is_constructor cpp_function_name = hsfuncname
  where
    hsfuncname' =
      if is_constructor
      then drop 3 $ toHsFuncName' cpp_function_name -- To drop the prefix string of 'new' 
      else toHsFuncName' cpp_function_name
    hsfuncname =
      case hsfuncname' of
        "=" -> "_assign_"
        "+=" -> "_iadd_"
        "-=" -> "_isub_"
        "*=" -> "_imul_"
        "/=" -> "_idiv_"
        "[]" -> "_at_"
        _ -> hsfuncname'
    replace [] = []
    replace ('<':xs') = '_':replace xs'
    replace ('>':xs') = replace xs'
    replace (',':xs') = '_':replace xs'
    replace (x':xs') = x':replace xs'
    toHsFuncName' :: String -> String
    toHsFuncName' [] = []
    toHsFuncName' (x:xs) = toLower x : replace xs


functionToCpp :: Bool -> Bool -> String -> String -> Function -> Text
functionToCpp is_managed add_type_initials prefix suffix fn =
  if is_managed then [st|
#{hsfuncname}#{type_initials}
  :: #{types}
#{hsfuncname}#{type_initials} = cast#{num_args} Unmanaged.#{hsfuncname}#{type_initials}
|]
  else [st|
#{hsfuncname}#{type_initials}
  :: #{types}
#{hsfuncname}#{type_initials} #{args} =
  #{bra}C.throwBlock| #{ret_type} { #{call_return} #{ret_wrapper}(#{prefix}#{P.name fn}#{suffix}(
    #{cargs}));
  }|#{cket}
|]
  where
    hsfuncname = toHsFuncName False (P.name fn)
    parameters' = filter isNotStar $ parameters fn
    num_args :: Int
    num_args = length parameters'
    args :: String
    args = L.intercalate " " $ map (\p -> "_" <> pname p) parameters'
    cargs :: Text
    cargs = T.intercalate "\n  , " $ flip map parameters' $ \p ->
      if isCType (ptype p)
      then [st|$(#{parsableToCppType (ptype p)} _#{pname p})|]
      else [st|*$(#{parsableToCppType (ptype p)}* _#{pname p})|]
    type_initials :: Text --- This is for avoding c++ overload arguments.
    type_initials =
      if add_type_initials && length parameters' > 0
      then "_" <> (mconcat $ flip map parameters' $ \p -> parsableToInitial (ptype p))
      else ""
    pointer :: Text
    pointer =
      if is_managed
      then "ForeignPtr"
      else "Ptr"
    types_list :: [Text]
    types_list = flip map parameters' $ \p ->
      if isCType (ptype p)
      then [st|#{parsableToHsType (ptype p)}|]
      else [st|#{pointer} #{parsableToHsType (ptype p)}|]
    types :: Text
    types = T.intercalate "\n  -> " $ types_list ++ [[st|IO (#{ret_hstype})|]]
    ret_type :: Text
    ret_type =
      if isCType (retType fn)
      then [st|#{parsableToCppType (retType fn)}|]
      else [st|#{parsableToCppType (retType fn)}*|]
    ret_hstype :: Text
    ret_hstype =
      if isCType (retType fn)
      then [st|#{parsableToHsType (retType fn)}|]
      else [st|#{pointer} #{parsableToHsType (retType fn)}|]
    ret_wrapper :: Text
    ret_wrapper =
      if isCType (retType fn)
      then ""
      else [st|new #{parsableToCppType (retType fn)}|]
    call_return :: Text
    call_return =
      case (retType fn) of
        CType CVoid -> ""
        _ -> "return"

methodToCpp :: PC.CppClassSpec -> Bool -> Bool -> Bool -> String -> String -> Function -> Text
methodToCpp class' is_constructor is_managed add_type_initials prefix suffix fn =
  case (is_managed,is_constructor) of
    (True,_) -> [st|
#{function_name}
  :: #{types}
#{function_name} = cast#{num_args} Unmanaged.#{function_name}
|]
    (False,True) -> [st|
#{function_name}
  :: #{types}
#{function_name} #{args} =
  #{bra}C.throwBlock| #{ret_type} { #{call_return} #{ret_wrapper cargs'}
  }|#{cket}
|]
    (False,False) -> [st|
#{function_name}
  :: #{types}
#{function_name} #{args} =
  #{bra}C.throwBlock| #{ret_type} { #{call_return} #{ret_wrapper cargs_with_obj}
  }|#{cket}
|]
  where
    function_name :: Text
    function_name =
      if is_constructor
      then [st|new#{(PC.hsname class')}#{(hsfuncname)}#{type_initials}|]
      else [st|#{toHsFuncName False (PC.hsname class')}_#{hsfuncname}#{type_initials}|]
    type_object :: Parameter
    type_object = Parameter (P.CppClass (PC.signature class')  (PC.cppname class')  (PC.hsname class') ) "obj" Nothing
    type_object_str :: Text
    type_object_str = [st|(*$(#{parsableToCppType (ptype type_object)}* _#{pname type_object}))|]
    hsfuncname = toHsFuncName is_constructor (P.name fn)
    op :: Text -> Text -> Text
    op fn' args' =
      case fn' of
        "=" -> "=" <> args'
        "+=" -> "+=" <> args'
        "-=" -> "-=" <> args'
        "*=" -> "*=" <> args'
        "/=" -> "/=" <> args'
        "[]" -> "[" <> args' <> "]"
        _ -> "." <> fn' <> args'
    cargs_with_obj = type_object_str <> op (fromString (prefix <> P.name fn <> suffix)) cargs
    parameters' =
      if is_constructor
      then (filter isNotStar $ parameters fn)
      else [type_object] <> (filter isNotStar $ parameters fn)
    parameters'' = (filter isNotStar $ parameters fn)
    num_args :: Int
    num_args = length parameters'
    args :: String
    args = L.intercalate " " $ map (\p -> "_" <> pname p) parameters'
    cargs'' :: Text
    cargs'' = T.intercalate "\n  , " $ flip map parameters'' $ \p ->
      if isCType (ptype p)
      then [st|$(#{parsableToCppType (ptype p)} _#{pname p})|]
      else [st|*$(#{parsableToCppType (ptype p)}* _#{pname p})|]
    cargs' :: Text
    cargs' = [st|
    #{cargs''}|]
    cargs :: Text
    cargs = [st|(
    #{cargs''})|]
    type_initials :: Text --- This is for avoding c++ overload arguments.
    type_initials =
      if add_type_initials && length parameters'' > 0
      then "_" <> (mconcat $ flip map parameters'' $ \p -> parsableToInitial (ptype p))
      else ""
    pointer :: Text
    pointer =
      if is_managed
      then "ForeignPtr"
      else "Ptr"
    types_list :: [Text]
    types_list = flip map parameters' $ \p ->
      if isCType (ptype p)
      then [st|#{parsableToHsType (ptype p)}|]
      else [st|#{pointer} #{parsableToHsType (ptype p)}|]
    types :: Text
    types = T.intercalate "\n  -> " $ types_list ++ [[st|IO (#{ret_hstype})|]]
    ret_type :: Text
    ret_type =
      if isCType (retType fn)
      then [st|#{parsableToCppType (retType fn)}|]
      else [st|#{parsableToCppType (retType fn)}*|]
    ret_hstype :: Text
    ret_hstype =
      if isCType (retType fn)
      then [st|#{parsableToHsType (retType fn)}|]
      else [st|#{pointer} #{parsableToHsType (retType fn)}|]
    isIntArrayRef (TenType (IntList _)) = True
    isIntArrayRef _ = False
    ret_wrapper :: Text -> Text
    ret_wrapper statement =
      if isCType (retType fn)
      then [st|#{statement};|]
      else if isIntArrayRef (retType fn)
           then [st|new #{parsableToCppType (retType fn)}(#{statement}.vec());|]
           else [st|new #{parsableToCppType (retType fn)}(#{statement});|]
    call_return :: Text
    call_return =
      case (retType fn) of
        CType CVoid -> ""
        _ -> "return"
