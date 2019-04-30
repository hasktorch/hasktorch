

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
    TensorList -> "at::TensorList"
    IndexTensor -> "at::Tensor"
    IntegerTensor -> "at::Tensor"
    BoolTensor -> "at::Tensor"
    BoolTensorQ -> "at::Tensor"
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

isCType :: Parsable -> Bool
isCType p =
  case p of
    TenType ScalarType -> True
    DeviceType -> True
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


--functionToCpp :: Bool -> String -> String -> Function -> Text
--functionToCpp = functionToCpp' False

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
  #{bra}C.block| #{ret_type} { #{call_return} #{ret_wrapper}(#{prefix}#{P.name fn}#{suffix}(
    #{cargs}));
  }|#{cket}
|]
  where
    renameFunc :: String -> String
    renameFunc [] = []
    renameFunc (x:xs) = toLower x : xs
    hsfuncname = renameFunc $ P.name fn
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
      if add_type_initials
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
    (True,True) -> [st|
new#{(PC.hsname class')}#{drop 3 (hsfuncname)}#{type_initials}
  :: #{types'}
new#{(PC.hsname class')}#{drop 3 (hsfuncname)}#{type_initials} = cast#{num_args'} Unmanaged.new#{(PC.hsname class')}#{drop 3 (hsfuncname)}#{type_initials}
|]
    (True,False) -> [st|
#{renameFunc (PC.hsname class')}_#{hsfuncname}#{type_initials}
  :: #{types}
#{renameFunc (PC.hsname class')}_#{hsfuncname}#{type_initials} = cast#{num_args} Unmanaged.#{renameFunc (PC.hsname class')}_#{hsfuncname}
|]
    (False,True) -> [st|
new#{(PC.hsname class')}#{drop 3 (hsfuncname)}#{type_initials}
  :: #{types'}
new#{(PC.hsname class')}#{drop 3 (hsfuncname)}#{type_initials} #{args'} =
  #{bra}C.block| #{ret_type} { #{call_return} #{ret_wrapper}(
    #{cargs});
  }|#{cket}
|]
    (False,False) -> [st|
#{renameFunc (PC.hsname class')}_#{hsfuncname}#{type_initials}
  :: #{types}
#{renameFunc (PC.hsname class')}_#{hsfuncname}#{type_initials} #{args} =
  #{bra}C.block| #{ret_type} { #{call_return} #{ret_wrapper}(#{type_object_str}->#{prefix}#{P.name fn}#{suffix}(
    #{cargs}));
  }|#{cket}
|]
  where
    type_object :: Parameter
    type_object = Parameter (P.CppClass (PC.signature class')  (PC.cppname class')  (PC.hsname class') ) "obj" Nothing
    type_object_str :: Text
    type_object_str = [st|$(#{parsableToCppType (ptype type_object)}* _#{pname type_object})|]
    renameFunc :: String -> String
    renameFunc [] = []
    renameFunc (x:xs) = toLower x : xs
    hsfuncname = renameFunc $ P.name fn
    parameters' = (filter isNotStar $ parameters fn)
    parameters'' = [type_object] <> parameters'
    num_args :: Int
    num_args = length parameters''
    num_args' :: Int
    num_args' = length parameters'
    args :: String
    args = L.intercalate " " $ map (\p -> "_" <> pname p) parameters''
    args' :: String
    args' = L.intercalate " " $ map (\p -> "_" <> pname p) parameters'
    cargs :: Text
    cargs = T.intercalate "\n  , " $ flip map parameters' $ \p ->
      if isCType (ptype p)
      then [st|$(#{parsableToCppType (ptype p)} _#{pname p})|]
      else [st|*$(#{parsableToCppType (ptype p)}* _#{pname p})|]
    type_initials :: Text --- This is for avoding c++ overload arguments.
    type_initials =
      if add_type_initials
      then "_" <> (mconcat $ flip map parameters'' $ \p -> parsableToInitial (ptype p))
      else ""
    pointer :: Text
    pointer =
      if is_managed
      then "ForeignPtr"
      else "Ptr"
    types_list :: [Text]
    types_list = flip map parameters'' $ \p ->
      if isCType (ptype p)
      then [st|#{parsableToHsType (ptype p)}|]
      else [st|#{pointer} #{parsableToHsType (ptype p)}|]
    types_list' :: [Text]
    types_list' = flip map parameters' $ \p ->
      if isCType (ptype p)
      then [st|#{parsableToHsType (ptype p)}|]
      else [st|#{pointer} #{parsableToHsType (ptype p)}|]
    types :: Text
    types = T.intercalate "\n  -> " $ types_list ++ [[st|IO (#{ret_hstype})|]]
    types' :: Text
    types' = T.intercalate "\n  -> " $ types_list' ++ [[st|IO (#{ret_hstype})|]]
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
