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
import qualified Data.Text as T
import qualified Data.List as L

import ParseFunctionSig as P

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
    TensorOptions -> "at::TensorOptions"
    TensorList -> "at::TensorList"
    IndexTensor -> "at::Tensor"
    BoolTensor -> "at::Tensor"
    BoolTensorQ -> "at::Tensor"
    LongTensor -> "at::Tensor"
    IntList _ -> "at::IntArrayRef"
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
    CInt -> "int"
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
    DeviceType -> "at::Device"
    GeneratorType -> "at::Generator"
    StorageType -> "at::Storage"
    CType ct -> ctypeToCppType ct
    STLType t -> stltypeToCppType t
    CppString -> "std::string"
    Tuple parsables -> [st|std::tuple<#{T.intercalate "," (map parsableToCppType parsables)}>|]


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
    TensorOptions -> "TensorOptions"
    TensorList -> "TensorList"
    IndexTensor -> "Tensor"
    BoolTensor -> "Tensor"
    BoolTensorQ -> "Tensor"
    LongTensor -> "Tensor"
    IntList _ -> "IntList"
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
    CVoid -> ""
    CFloat -> "CFloat"
    CDouble -> "CDouble"
    CInt -> "CInt"
    CInt64 -> "Int64"
    CInt64Q -> "Int64"

parsableToHsType :: Parsable -> Text
parsableToHsType parsable =
  case parsable of
    Ptr p -> "Ptr " <> parsableToHsType p
    TenType t -> tenTypeToHsType t
    DeviceType -> "Device"
    GeneratorType -> "Generator"
    StorageType -> "Storage"
    CType ct -> ctypeToHsType ct
    STLType t -> stltypeToHsType t
    CppString -> "StdString"
    Tuple parsables -> [st|(#{T.intercalate "," (map parsableToHsType parsables)})|]


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
    TensorOptions -> "o"
    TensorList -> "l"
    IndexTensor -> "t"
    BoolTensor -> "t"
    BoolTensorQ -> "t"
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
    CInt -> "i"
    CInt64 -> "l"
    CInt64Q -> "l"

parsableToInitial :: Parsable -> Text
parsableToInitial parsable =
  case parsable of
    Ptr _ -> "p"
    TenType t -> tenTypeToInitial t
    DeviceType -> "device"
    GeneratorType -> "generator"
    StorageType -> "storage"
    CType ct -> ctypeToInitial ct
    STLType t -> stltypeToInitial t
    CppString -> "s"
    Tuple _ -> "t"

isCType :: Parsable -> Bool
isCType p =
  case p of
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
    DeviceType -> "Device"
    GeneratorType -> "Generator"
    StorageType -> "Storage"
    CType ct -> ctypeToCppType ct
    STLType t -> stltypeToCppType t
    CppString -> "std::string"
    Tuple parsables -> [st|tuple<#{T.intercalate "," (map parsableToCppType parsables)}>|]

{-

From native_function.yaml
- func: add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  matches_jit_signature: True
  variants: function, method

- func: add_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
  matches_jit_signature: True
  variants: method

# For C++ only, until we have conversion from C++ numbers to Tensor
- func: add(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  matches_jit_signature: True

- func: add(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  matches_jit_signature: True
  variants: function, method

- func: add_(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
  matches_jit_signature: True
  variants: method

From NativeFunction.h(C++)
CAFFE2_API Tensor add(const Tensor & self, const Tensor & other, Scalar alpha=1);
CAFFE2_API Tensor & add_(Tensor & self, const Tensor & other, Scalar alpha=1);

-- see : https://github.com/pytorch/pytorch/blob/9101dfc57ccb6b6931b4e80233bbc64d9080d2e8/aten/src/ATen/native_parse.py#L159-L178
CAFFE2_API Tensor & add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha=1);

CAFFE2_API Tensor add(const Tensor & self, Scalar other, Scalar alpha=1);
CAFFE2_API Tensor & add_(Tensor & self, Scalar other, Scalar alpha=1);
-}

removeStarArgument :: Function -> Function
removeStarArgument fn =
  if arguments_out /= []
  then fn {parameters = new_params, name = (name fn) ++ "_out" }
  else fn
  where
    params = parameters fn
    splitByStar [] _ = ([],[])
    splitByStar (Star:xs) (y,y') = (y,y'++xs)
    splitByStar (x:xs) (y,y') = splitByStar xs (y++[x],y')
    (front_star,back_star) = splitByStar params ([],[])
    arguments_out = filter (\v -> ptype v == TenType TensorA') back_star
    arguments_other = filter (\v -> ptype v /= TenType TensorA') back_star
    new_params = arguments_out ++ front_star ++ arguments_other

functionToCpp :: Bool -> String -> Function -> Text
functionToCpp add_type_initials prefix fn = [st|
#{hsfuncname}#{type_initials}
  :: #{types}
#{hsfuncname}#{type_initials} #{args} =
  #{bra}C.block| #{ret_type} { #{call_return} #{ret_wrapper}(#{prefix}#{name fn}(
    #{cargs}));
  }|#{cket}
|]
  where
    renameFunc :: String -> String
    renameFunc [] = []
    renameFunc (x:xs) = toLower x : xs
    hsfuncname = renameFunc $ name fn
    parameters' = filter isNotStar $ parameters fn
    args :: String
    args = L.intercalate " " $ map (\p -> "_" <> pname p) parameters'
    cargs :: Text
    cargs = T.intercalate "\n  , " $ flip map parameters' $ \p ->
      if isCType (ptype p)
      then [st|$(#{parsableToCppType (ptype p)} _#{pname p})|]
      else [st|*$(#{parsableToCppType (ptype p)}* _#{pname p})|]
    type_initials :: Text --- This is for avoding c++ overload arguments.
    type_initials =
      if add_type_initials
      then "_" <> (mconcat $ flip map parameters' $ \p -> parsableToInitial (ptype p))
      else ""
    types_list :: [Text]
    types_list = flip map parameters' $ \p ->
      if isCType (ptype p)
      then [st|#{parsableToHsType (ptype p)}|]
      else [st|Ptr #{parsableToHsType (ptype p)}|]
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
      else [st|Ptr #{parsableToHsType (retType fn)}|]
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
