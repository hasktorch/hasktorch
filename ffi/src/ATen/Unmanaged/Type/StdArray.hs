
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module ATen.Unmanaged.Type.StdArray where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import ATen.Type
import ATen.Class

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/ATen.h>"
C.include "<array>"



newStdArrayBool2
  :: IO (Ptr (StdArray CBool 2))
newStdArrayBool2  =
  [C.throwBlock| std::array<bool,2>* { return new std::array<bool,2>(
    );
  }|]

newStdArrayBool3
  :: IO (Ptr (StdArray CBool 3))
newStdArrayBool3  =
  [C.throwBlock| std::array<bool,3>* { return new std::array<bool,3>(
    );
  }|]

newStdArrayBool4
  :: IO (Ptr (StdArray CBool 4))
newStdArrayBool4  =
  [C.throwBlock| std::array<bool,4>* { return new std::array<bool,4>(
    );
  }|]


deleteStdArrayBool2 :: Ptr (StdArray CBool 2) -> IO ()
deleteStdArrayBool2 object = [C.throwBlock| void { delete $(std::array<bool,2>* object);}|]

deleteStdArrayBool3 :: Ptr (StdArray CBool 3) -> IO ()
deleteStdArrayBool3 object = [C.throwBlock| void { delete $(std::array<bool,3>* object);}|]

deleteStdArrayBool4 :: Ptr (StdArray CBool 4) -> IO ()
deleteStdArrayBool4 object = [C.throwBlock| void { delete $(std::array<bool,4>* object);}|]


instance CppObject (StdArray CBool 2) where
  fromPtr ptr = newForeignPtr ptr (deleteStdArrayBool2 ptr)

instance CppObject (StdArray CBool 3) where
  fromPtr ptr = newForeignPtr ptr (deleteStdArrayBool3 ptr)

instance CppObject (StdArray CBool 4) where
  fromPtr ptr = newForeignPtr ptr (deleteStdArrayBool4 ptr)



