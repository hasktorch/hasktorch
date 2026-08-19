{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module RenderTuples where

import Data.List (nubBy)
import Data.Maybe (mapMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.Yaml (ParseException)
import qualified Data.Yaml as Y
import qualified ParseDeclarations as D
import ParseFunctionSig as P
import qualified ParseTuples as PT
import RenderCommon
import System.Directory (createDirectoryIfMissing)
import Text.Shakespeare.Text (st)

renderImport :: Bool -> Text
renderImport is_managed =
  if is_managed
    then
      [st|
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Type.Tuple as Unmanaged
|]
    else
      [st|
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/Tensor.h>"
C.include "<tuple>"
|]

tupleToCpp :: PT.Tuple -> Text
tupleToCpp (PT.Tuple parsables) = [st|std::tuple<#{T.intercalate "," (map parsableToCppType parsables)}>|]

tupleToHs :: PT.Tuple -> Text
tupleToHs (PT.Tuple parsables) = [st|StdTuple '(#{T.intercalate "," (map parsableToHsType parsables)})|]

tupleToHs' :: PT.Tuple -> Text
tupleToHs' (PT.Tuple parsables) = [st|#{T.intercalate "" (map parsableToHsType parsables)}|]

toHs :: P.Parsable -> Text
toHs typ_ =
  if isCType typ_
    then [st|#{parsableToHsType typ_}|]
    else [st|Ptr #{parsableToHsType typ_}|]

toManagedHs :: P.Parsable -> Text
toManagedHs typ_ =
  if isCType typ_
    then [st|#{parsableToHsType typ_}|]
    else [st|ForeignPtr #{parsableToHsType typ_}|]

toCpp :: P.Parsable -> Text
toCpp typ_ =
  if isCType typ_
    then [st|#{parsableToCppType typ_}|]
    else [st|#{parsableToCppType typ_}*|]

toCpp' :: P.Parsable -> Text
toCpp' typ_ =
  if isCType typ_
    then [st||]
    else [st|new #{parsableToCppType typ_}|]

renderCppObject :: PT.Tuple -> Text
renderCppObject typ_ =
  [st|

-----------------#{tupleToHs typ_}---------------------
|]

renderCppTuple2 :: PT.Tuple -> Text
renderCppTuple2 typ_@(PT.Tuple (a : b : _)) =
  [st|
instance CppTuple2 (Ptr #{withParens $ tupleToHs typ_}) where
  type A (Ptr #{withParens $ tupleToHs typ_}) = #{toHs a}
  type B (Ptr #{withParens $ tupleToHs typ_}) = #{toHs b}
  get0 v = #{bra}C.throwBlock| #{toCpp a} { return #{toCpp' a}(std::get<0>(*$(#{tupleToCpp typ_}* v)));}|#{cket}
  get1 v = #{bra}C.throwBlock| #{toCpp b} { return #{toCpp' b}(std::get<1>(*$(#{tupleToCpp typ_}* v)));}|#{cket}
|]
renderCppTuple2 _ = ""

renderCppTuple3 :: PT.Tuple -> Text
renderCppTuple3 typ_@(PT.Tuple (_ : _ : c : _)) =
  [st|
instance CppTuple3 (Ptr #{withParens $ tupleToHs typ_}) where
  type C (Ptr #{withParens $ tupleToHs typ_}) = #{toHs c}
  get2 v = #{bra}C.throwBlock| #{toCpp c} { return #{toCpp' c}(std::get<2>(*$(#{tupleToCpp typ_}* v)));}|#{cket}
|]
renderCppTuple3 _ = ""

renderCppTuple4 :: PT.Tuple -> Text
renderCppTuple4 typ_@(PT.Tuple (_ : _ : _ : d : _)) =
  [st|
instance CppTuple4 (Ptr #{withParens $ tupleToHs typ_}) where
  type D (Ptr #{withParens $ tupleToHs typ_}) = #{toHs d}
  get3 v = #{bra}C.throwBlock| #{toCpp d} { return #{toCpp' d}(std::get<3>(*$(#{tupleToCpp typ_}* v)));}|#{cket}
|]
renderCppTuple4 _ = ""

renderCppTuple5 :: PT.Tuple -> Text
renderCppTuple5 typ_@(PT.Tuple (_ : _ : _ : _ : e : _)) =
  [st|
instance CppTuple5 (Ptr #{withParens $ tupleToHs typ_}) where
  type E (Ptr #{withParens $ tupleToHs typ_}) = #{toHs e}
  get4 v = #{bra}C.throwBlock| #{toCpp e} { return #{toCpp' e}(std::get<4>(*$(#{tupleToCpp typ_}* v)));}|#{cket}
|]
renderCppTuple5 _ = ""

renderCppTuple6 :: PT.Tuple -> Text
renderCppTuple6 typ_@(PT.Tuple (_ : _ : _ : _ : _ : f : _)) =
  [st|
instance CppTuple6 (Ptr #{withParens $ tupleToHs typ_}) where
  type F (Ptr #{withParens $ tupleToHs typ_}) = #{toHs f}
  get5 v = #{bra}C.throwBlock| #{toCpp f} { return #{toCpp' f}(std::get<5>(*$(#{tupleToCpp typ_}* v)));}|#{cket}
|]
renderCppTuple6 _ = ""

renderCppTuple7 :: PT.Tuple -> Text
renderCppTuple7 typ_@(PT.Tuple (_ : _ : _ : _ : _ : _ : g : _)) =
  [st|
instance CppTuple7 (Ptr #{withParens $ tupleToHs typ_}) where
  type G (Ptr #{withParens $ tupleToHs typ_}) = #{toHs g}
  get6 v = #{bra}C.throwBlock| #{toCpp g} { return #{toCpp' g}(std::get<6>(*$(#{tupleToCpp typ_}* v)));}|#{cket}
|]
renderCppTuple7 _ = ""

renderCppTuple8 :: PT.Tuple -> Text
renderCppTuple8 typ_@(PT.Tuple (_ : _ : _ : _ : _ : _ : _ : h : _)) =
  [st|
instance CppTuple8 (Ptr #{withParens $ tupleToHs typ_}) where
  type H (Ptr #{withParens $ tupleToHs typ_}) = #{toHs h}
  get7 v = #{bra}C.throwBlock| #{toCpp h} { return #{toCpp' h}(std::get<7>(*$(#{tupleToCpp typ_}* v)));}|#{cket}
|]
renderCppTuple8 _ = ""

renderCppTuple9 :: PT.Tuple -> Text
renderCppTuple9 typ_@(PT.Tuple (_ : _ : _ : _ : _ : _ : _ : _ : i : _)) =
  [st|
instance CppTuple9 (Ptr #{withParens $ tupleToHs typ_}) where
  type I (Ptr #{withParens $ tupleToHs typ_}) = #{toHs i}
  get8 v = #{bra}C.throwBlock| #{toCpp i} { return #{toCpp' i}(std::get<8>(*$(#{tupleToCpp typ_}* v)));}|#{cket}
|]
renderCppTuple9 _ = ""

renderManagedCppTuple2 :: PT.Tuple -> Text
renderManagedCppTuple2 typ_@(PT.Tuple (a : b : _)) =
  [st|
instance CppTuple2 (ForeignPtr #{withParens $ tupleToHs typ_}) where
  type A (ForeignPtr #{withParens $ tupleToHs typ_}) = #{toManagedHs a}
  type B (ForeignPtr #{withParens $ tupleToHs typ_}) = #{toManagedHs b}
  get0 v = cast1 (get0 :: Ptr #{withParens $ tupleToHs typ_} -> IO (#{toHs a})) v
  get1 v = cast1 (get1 :: Ptr #{withParens $ tupleToHs typ_} -> IO (#{toHs b})) v
|]
renderManagedCppTuple2 _ = ""

renderManagedCppTuple3 :: PT.Tuple -> Text
renderManagedCppTuple3 typ_@(PT.Tuple (_ : _ : c : _)) =
  [st|
instance CppTuple3 (ForeignPtr #{withParens $ tupleToHs typ_}) where
  type C (ForeignPtr #{withParens $ tupleToHs typ_}) = #{toManagedHs c}
  get2 v = cast1 (get2 :: Ptr #{withParens $ tupleToHs typ_} -> IO (#{toHs c})) v
|]
renderManagedCppTuple3 _ = ""

renderManagedCppTuple4 :: PT.Tuple -> Text
renderManagedCppTuple4 typ_@(PT.Tuple (_ : _ : _ : d : _)) =
  [st|
instance CppTuple4 (ForeignPtr #{withParens $ tupleToHs typ_}) where
  type D (ForeignPtr #{withParens $ tupleToHs typ_}) = #{toManagedHs d}
  get3 v = cast1 (get3 :: Ptr #{withParens $ tupleToHs typ_} -> IO (#{toHs d})) v
|]
renderManagedCppTuple4 _ = ""

renderManagedCppTuple5 :: PT.Tuple -> Text
renderManagedCppTuple5 typ_@(PT.Tuple (_ : _ : _ : _ : e : _)) =
  [st|
instance CppTuple5 (ForeignPtr #{withParens $ tupleToHs typ_}) where
  type E (ForeignPtr #{withParens $ tupleToHs typ_}) = #{toManagedHs e}
  get4 v = cast1 (get4 :: Ptr #{withParens $ tupleToHs typ_} -> IO (#{toHs e})) v
|]
renderManagedCppTuple5 _ = ""

renderManagedCppTuple6 :: PT.Tuple -> Text
renderManagedCppTuple6 typ_@(PT.Tuple (_ : _ : _ : _ : _ : f : _)) =
  [st|
instance CppTuple6 (ForeignPtr #{withParens $ tupleToHs typ_}) where
  type F (ForeignPtr #{withParens $ tupleToHs typ_}) = #{toManagedHs f}
  get5 v = cast1 (get5 :: Ptr #{withParens $ tupleToHs typ_} -> IO (#{toHs f})) v
|]
renderManagedCppTuple6 _ = ""

renderManagedCppTuple7 :: PT.Tuple -> Text
renderManagedCppTuple7 typ_@(PT.Tuple (_ : _ : _ : _ : _ : _ : g : _)) =
  [st|
instance CppTuple7 (ForeignPtr #{withParens $ tupleToHs typ_}) where
  type G (ForeignPtr #{withParens $ tupleToHs typ_}) = #{toManagedHs g}
  get6 v = cast1 (get6 :: Ptr #{withParens $ tupleToHs typ_} -> IO (#{toHs g})) v
|]
renderManagedCppTuple7 _ = ""

renderManagedCppTuple8 :: PT.Tuple -> Text
renderManagedCppTuple8 typ_@(PT.Tuple (_ : _ : _ : _ : _ : _ : _ : h : _)) =
  [st|
instance CppTuple8 (ForeignPtr #{withParens $ tupleToHs typ_}) where
  type H (ForeignPtr #{withParens $ tupleToHs typ_}) = #{toManagedHs h}
  get7 v = cast1 (get7 :: Ptr #{withParens $ tupleToHs typ_} -> IO (#{toHs h})) v
|]
renderManagedCppTuple8 _ = ""

renderManagedCppTuple9 :: PT.Tuple -> Text
renderManagedCppTuple9 typ_@(PT.Tuple (_ : _ : _ : _ : _ : _ : _ : _ : i : _)) =
  [st|
instance CppTuple9 (ForeignPtr #{withParens $ tupleToHs typ_}) where
  type I (ForeignPtr #{withParens $ tupleToHs typ_}) = #{toManagedHs i}
  get8 v = cast1 (get8 :: Ptr #{withParens $ tupleToHs typ_} -> IO (#{toHs i})) v
|]
renderManagedCppTuple9 _ = ""

renderTuples :: Bool -> [PT.Tuple] -> Text
renderTuples True [] = ""
renderTuples True (x : xs) =
  renderManagedCppTuple2 x
    <> renderManagedCppTuple3 x
    <> renderManagedCppTuple4 x
    <> renderManagedCppTuple5 x
    <> renderManagedCppTuple6 x
    <> renderManagedCppTuple7 x
    <> renderManagedCppTuple8 x
    <> renderManagedCppTuple9 x
    <> renderTuples True xs
renderTuples False [] = ""
renderTuples False (x : xs) =
  renderCppObject x
    <> renderCppTuple2 x
    <> renderCppTuple3 x
    <> renderCppTuple4 x
    <> renderCppTuple5 x
    <> renderCppTuple6 x
    <> renderCppTuple7 x
    <> renderCppTuple8 x
    <> renderCppTuple9 x
    <> renderTuples False xs

decodeAndCodeGen :: String -> String -> IO ()
decodeAndCodeGen basedir fileName = do
  maybe_decls <- Y.decodeFileEither fileName :: IO (Either ParseException [D.Declaration])
  --funcs <- Y.decodeFileEither fileName :: IO (Either ParseException [PT.Tuple])
  case maybe_decls of
    Left err' -> print err'
    Right decls -> do
      let tuples = nubBy tupleHsTypeEq $ mapMaybe (getTupleType . D.returns) decls
      createDirectoryIfMissing True (basedir <> "/Torch/Internal/Unmanaged/Type")
      T.writeFile (basedir <> "/Torch/Internal/Unmanaged/Type/Tuple.hs") $
        template False "Torch.Internal.Unmanaged.Type.Tuple" tuples
      createDirectoryIfMissing True (basedir <> "/Torch/Internal/Managed/Type")
      T.writeFile (basedir <> "/Torch/Internal/Managed/Type/Tuple.hs") $
        template True "Torch.Internal.Managed.Type.Tuple" tuples
  where
    getTupleType :: [D.Type] -> Maybe PT.Tuple
    getTupleType [] = Nothing
    getTupleType [_] = Nothing
    getTupleType rets = Just $ PT.Tuple $ map D.type2type rets

    tupleHsTypeEq :: PT.Tuple -> PT.Tuple -> Bool
    tupleHsTypeEq a b = (fmap parsableToHsType (PT.types a)) == (fmap parsableToHsType (PT.types b))

template :: Bool -> Text -> [PT.Tuple] -> Text
template is_managed module_name types =
  [st|
-- generated by using spec/tuples.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module #{module_name} where

#{renderImport is_managed}

#{renderTuples is_managed types}
|]
