{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE QuasiQuotes #-}
module RenderNativeFunctions where

import Data.Yaml

import qualified Data.Yaml as Y
import Text.Shakespeare.Text (st,sbt)
import Data.Char (toLower)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.List as L

import ParseNativeFunctions
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
    TensorQ -> "at::Tensor"
    TensorOptions -> "at::TensorOptions"
    TensorList -> "at::TensorList"
    IndexTensor -> "at::IndexTensor"
    BoolTensor -> "at::Tensor"
    BoolTensorQ -> "at::Tensor"
    IntList _ -> "at::IntList"
    ScalarQ -> "at::Scalar"
    ScalarType -> "at::ScalarType"
    SparseTensorRef -> "at::SparseTensorRef"

ctypeToCppType :: CType -> Text
ctypeToCppType ct =
  case ct of
    CBool -> "bool"
    CVoid -> "void"
    CDouble -> "double"
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
    TensorQ -> "Tensor"
    TensorOptions -> "TensorOptions"
    TensorList -> "TensorList"
    IndexTensor -> "IndexTensor"
    BoolTensor -> "Tensor"
    BoolTensorQ -> "Tensor"
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
    CDouble -> "CDouble"
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
    TensorQ -> "t"
    TensorOptions -> "o"
    TensorList -> "l"
    IndexTensor -> "i"
    BoolTensor -> "t"
    BoolTensorQ -> "t"
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
    CDouble -> "d"
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

renameFunc :: String -> String
renameFunc [] = []
renameFunc (x:xs) = toLower x : xs

functionToCpp :: Function -> Text
functionToCpp fn =
  [sbt|#{hsfuncname}_#{type_initials} :: #{types}
      |#{hsfuncname}_#{type_initials} #{args} = #{bra}C.block| #{ret_type} { return #{ret_wrapper}(at::native::#{name fn}(#{cargs})); }|#{cket}
      |
      |]
  where
    hsfuncname = renameFunc $ name fn
    parameters' = filter isNotStar $ parameters fn
    args :: String
    args = L.intercalate " " $ map (\p -> "_" <> pname p) parameters'
    cargs :: Text
    cargs = T.intercalate ", " $ flip map parameters' $ \p ->
      if isCType (ptype p)
      then [st|$(#{parsableToCppType (ptype p)} _#{pname p})|]
      else [st|*$(#{parsableToCppType (ptype p)}* _#{pname p})|]
    type_initials :: Text
    type_initials = mconcat $ flip map parameters' $ \p ->parsableToInitial (ptype p)
    types_list :: [Text]
    types_list = flip map parameters' $ \p ->
      if isCType (ptype p)
      then [st|#{parsableToHsType (ptype p)}|]
      else [st|Ptr #{parsableToHsType (ptype p)}|]
    types :: Text
    types = T.intercalate " -> " $ types_list ++ [[st|IO (#{ret_hstype})|]]
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

renderFunctions :: [NativeFunction'] -> Text
renderFunctions nfs = mconcat $ flip map nfs $ \nf -> mconcat $ 
  case dispatch' nf of
    Nothing -> [functionToCpp $ func' nf]
    Just d -> map (\c -> functionToCpp $ (func' nf){name=c}) (uniqFunctions d)
  where
    uniqFunctions d = L.nub $ concat $
      [ case cpu d of
          Nothing -> []
          Just c -> [c]
      , case gpu d of
          Nothing -> []
          Just c -> [c]
      , case cuda d of
          Nothing -> []
          Just c -> [c]
      , case sparseCPU d of
          Nothing -> []
          Just c -> [c]
      , case sparseCUDA d of
          Nothing -> []
          Just c -> [c]
      ]


decodeAndCodeGen :: String -> IO ()
decodeAndCodeGen fileName = do
  funcs <- Y.decodeFileEither fileName :: IO (Either ParseException [NativeFunction'])
  case funcs of
    Left err' -> print err'
    Right fns -> do
      T.writeFile "ffi/NativeFunctions.hs" [st|
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
        (C.TypeName "at::Scalar", #{bra}t|Scalar|#{cket})
      , (C.TypeName "at::Tensor", #{bra}t|Tensor|#{cket})
      , (C.TypeName "at::TensorOptions", #{bra}t|TensorOptions|#{cket})
      , (C.TypeName "at::TensorList", #{bra}t|TensorList|#{cket})
      , (C.TypeName "at::IndexTensor", #{bra}t|IndexTensor|#{cket})
      , (C.TypeName "at::IntList", #{bra}t|IntList|#{cket})
      , (C.TypeName "at::ScalarType", #{bra}t|ScalarType|#{cket})
      , (C.TypeName "at::SparseTensorRef", #{bra}t|SparseTensorRef|#{cket})
      , (C.TypeName "at::Storage", #{bra}t|Storage|#{cket})
      , (C.TypeName "at::Device", #{bra}t|Device|#{cket})
      , (C.TypeName "at::Generator", #{bra}t|Generator|#{cket})
      , (C.TypeName "std::string", #{bra}t|StdString|#{cket})
      , (C.TypeName "std::array<bool,2>", #{bra}t|StdArray CBool 2|#{cket})
      , (C.TypeName "std::array<bool,3>", #{bra}t|StdArray CBool 3|#{cket})
      , (C.TypeName "std::array<bool,4>", #{bra}t|StdArray CBool 4|#{cket})
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor>", #{bra}t|(Tensor,Tensor)|#{cket})
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor>", #{bra}t|(Tensor,Tensor,Tensor)|#{cket})
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>", #{bra}t|(Tensor,Tensor,Tensor,Tensor)|#{cket})
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>", #{bra}t|(Tensor,Tensor,Tensor,Tensor,Tensor)|#{cket})
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::TensorList>", #{bra}t|(Tensor,Tensor,Tensor,TensorList)|#{cket})
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,double,int64_t>", #{bra}t|(Tensor,Tensor,CDouble,Int64)|#{cket})
    ]
}

C.include "<ATen/ATen.h>"

#{renderFunctions fns}
|]
