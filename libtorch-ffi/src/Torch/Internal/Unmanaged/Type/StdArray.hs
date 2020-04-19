{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Internal.Unmanaged.Type.StdArray where

import qualified Data.Map as Map
import Foreign hiding (newForeignPtr)
import Foreign.C.String
import Foreign.C.Types
import Foreign.Concurrent
import qualified Language.C.Inline.Context as C
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Types as C
import Torch.Internal.Class
import Torch.Internal.Type

C.context $ C.cppCtx <> mempty {C.ctxTypesTable = typeTable}

C.include "<ATen/ATen.h>"

C.include "<array>"

newStdArrayBool2 ::
  IO (Ptr (StdArray '(CBool, 2)))
newStdArrayBool2 =
  [C.throwBlock| std::array<bool,2>* { return new std::array<bool,2>(
    );
  }|]

newStdArrayBool2_bb ::
  CBool ->
  CBool ->
  IO (Ptr (StdArray '(CBool, 2)))
newStdArrayBool2_bb b0 b1 =
  [C.throwBlock| std::array<bool,2>* { return new std::array<bool,2>({$(bool b0),$(bool b1)}); }|]

instance CppTuple2 (Ptr (StdArray '(CBool, 2))) where
  type A (Ptr (StdArray '(CBool, 2))) = CBool
  type B (Ptr (StdArray '(CBool, 2))) = CBool
  get0 v = [C.throwBlock| bool { return std::get<0>(*$(std::array<bool,2>* v));}|]
  get1 v = [C.throwBlock| bool { return std::get<1>(*$(std::array<bool,2>* v));}|]

newStdArrayBool3 ::
  IO (Ptr (StdArray '(CBool, 3)))
newStdArrayBool3 =
  [C.throwBlock| std::array<bool,3>* { return new std::array<bool,3>(
    );
  }|]

newStdArrayBool3_bbb ::
  CBool ->
  CBool ->
  CBool ->
  IO (Ptr (StdArray '(CBool, 3)))
newStdArrayBool3_bbb b0 b1 b2 =
  [C.throwBlock| std::array<bool,3>* { return new std::array<bool,3>({$(bool b0),$(bool b1),$(bool b2)}); }|]

instance CppTuple2 (Ptr (StdArray '(CBool, 3))) where
  type A (Ptr (StdArray '(CBool, 3))) = CBool
  type B (Ptr (StdArray '(CBool, 3))) = CBool
  get0 v = [C.throwBlock| bool { return std::get<0>(*$(std::array<bool,3>* v));}|]
  get1 v = [C.throwBlock| bool { return std::get<1>(*$(std::array<bool,3>* v));}|]

instance CppTuple3 (Ptr (StdArray '(CBool, 3))) where
  type C (Ptr (StdArray '(CBool, 3))) = CBool
  get2 v = [C.throwBlock| bool { return std::get<2>(*$(std::array<bool,3>* v));}|]

newStdArrayBool4 ::
  IO (Ptr (StdArray '(CBool, 4)))
newStdArrayBool4 =
  [C.throwBlock| std::array<bool,4>* { return new std::array<bool,4>(
    );
  }|]

newStdArrayBool4_bbbb ::
  CBool ->
  CBool ->
  CBool ->
  CBool ->
  IO (Ptr (StdArray '(CBool, 4)))
newStdArrayBool4_bbbb b0 b1 b2 b3 =
  [C.throwBlock| std::array<bool,4>* { return new std::array<bool,4>({$(bool b0),$(bool b1),$(bool b2),$(bool b3)}); }|]

instance CppTuple2 (Ptr (StdArray '(CBool, 4))) where
  type A (Ptr (StdArray '(CBool, 4))) = CBool
  type B (Ptr (StdArray '(CBool, 4))) = CBool
  get0 v = [C.throwBlock| bool { return std::get<0>(*$(std::array<bool,4>* v));}|]
  get1 v = [C.throwBlock| bool { return std::get<1>(*$(std::array<bool,4>* v));}|]

instance CppTuple3 (Ptr (StdArray '(CBool, 4))) where
  type C (Ptr (StdArray '(CBool, 4))) = CBool
  get2 v = [C.throwBlock| bool { return std::get<2>(*$(std::array<bool,4>* v));}|]

instance CppTuple4 (Ptr (StdArray '(CBool, 4))) where
  type D (Ptr (StdArray '(CBool, 4))) = CBool
  get3 v = [C.throwBlock| bool { return std::get<3>(*$(std::array<bool,4>* v));}|]

deleteStdArrayBool2 :: Ptr (StdArray '(CBool, 2)) -> IO ()
deleteStdArrayBool2 object = [C.throwBlock| void { delete $(std::array<bool,2>* object);}|]

deleteStdArrayBool3 :: Ptr (StdArray '(CBool, 3)) -> IO ()
deleteStdArrayBool3 object = [C.throwBlock| void { delete $(std::array<bool,3>* object);}|]

deleteStdArrayBool4 :: Ptr (StdArray '(CBool, 4)) -> IO ()
deleteStdArrayBool4 object = [C.throwBlock| void { delete $(std::array<bool,4>* object);}|]

instance CppObject (StdArray '(CBool, 2)) where
  fromPtr ptr = newForeignPtr ptr (deleteStdArrayBool2 ptr)

instance CppObject (StdArray '(CBool, 3)) where
  fromPtr ptr = newForeignPtr ptr (deleteStdArrayBool3 ptr)

instance CppObject (StdArray '(CBool, 4)) where
  fromPtr ptr = newForeignPtr ptr (deleteStdArrayBool4 ptr)
