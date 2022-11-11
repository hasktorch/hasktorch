
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.C10Dict where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Control.Monad (forM)
import Control.Exception.Safe (bracket)

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/core/Dict.h>"
C.include "<vector>"

newC10Dict :: Ptr IValue -> Ptr IValue -> IO (Ptr (C10Dict '(IValue,IValue)))
newC10Dict key value = [C.throwBlock| c10::Dict<at::IValue,at::IValue>* {
  return new c10::impl::GenericDict($(at::IValue* key)->type(),$(at::IValue* value)->type());
}|]

c10Dict_empty :: Ptr (C10Dict '(IValue,IValue)) -> IO (CBool)
c10Dict_empty _obj = [C.throwBlock| bool { return (*$(c10::Dict<at::IValue,at::IValue>* _obj)).empty(); }|]

c10Dict_size :: Ptr (C10Dict '(IValue,IValue)) -> IO (CSize)
c10Dict_size _obj = [C.throwBlock| size_t { return (*$(c10::Dict<at::IValue,at::IValue>* _obj)).size(); }|]

c10Dict_at :: Ptr (C10Dict '(IValue,IValue)) -> Ptr IValue -> IO (Ptr IValue)
c10Dict_at _obj _s = [C.throwBlock| at::IValue* { return new at::IValue((*$(c10::Dict<at::IValue,at::IValue>* _obj)).at(*$(at::IValue* _s))); }|]

c10Dict_insert :: Ptr (C10Dict '(IValue,IValue)) -> Ptr IValue  -> Ptr IValue -> IO ()
c10Dict_insert _obj _key _value =
  [C.throwBlock| void {
    (*$(c10::Dict<at::IValue,at::IValue>* _obj)).insert(*$(at::IValue* _key),*$(at::IValue* _value));
  }|]

c10Dict_toList :: Ptr (C10Dict '(IValue,IValue)) -> IO [(Ptr IValue,Ptr IValue)]
c10Dict_toList _obj = do
  let new = [C.throwBlock| std::vector<std::array<at::IValue,2>>* {
              auto obj = *$(c10::Dict<at::IValue,at::IValue>* _obj);
              auto ret = new std::vector<std::array<at::IValue,2> >();
              for(auto i = obj.begin() ; i != obj.end() ; i++){
                ret->push_back({i->key(),i->value()});
              }
              return ret;
             }|]
      free dat = [C.throwBlock| void {
              delete $(std::vector<std::array<at::IValue,2>>* dat);
             }|]
  bracket new free $ \dat -> do
    size <- [C.throwBlock| int64_t { return (long int)$(std::vector<std::array<at::IValue,2>>* dat)->size();}|]
    ret <- forM [0..(size-1)] $ \i -> do
      key <- [C.throwBlock| at::IValue* { return new at::IValue($(std::vector<std::array<at::IValue,2>>* dat)->at($(int64_t i))[0]);}|]
      val <- [C.throwBlock| at::IValue* { return new at::IValue($(std::vector<std::array<at::IValue,2>>* dat)->at($(int64_t i))[1]);}|]
      return (key,val)
    return ret
