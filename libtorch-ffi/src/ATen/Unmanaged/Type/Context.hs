
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module ATen.Unmanaged.Type.Context where


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
C.include "<vector>"





deleteContext :: Ptr Context -> IO ()
deleteContext object = [C.throwBlock| void { delete $(at::Context* object);}|]

instance CppObject Context where
  fromPtr ptr = newForeignPtr ptr (deleteContext ptr)





init
  :: IO (())
init  =
  [C.throwBlock| void {  (at::init(
    ));
  }|]

hasCUDA
  :: IO (CBool)
hasCUDA  =
  [C.throwBlock| bool { return (at::hasCUDA(
    ));
  }|]

hasHIP
  :: IO (CBool)
hasHIP  =
  [C.throwBlock| bool { return (at::hasHIP(
    ));
  }|]

hasXLA
  :: IO (CBool)
hasXLA  =
  [C.throwBlock| bool { return (at::hasXLA(
    ));
  }|]

getNumGPUs
  :: IO (CSize)
getNumGPUs  =
  [C.throwBlock| size_t { return (at::getNumGPUs(
    ));
  }|]

hasOpenMP
  :: IO (CBool)
hasOpenMP  =
  [C.throwBlock| bool { return (at::hasOpenMP(
    ));
  }|]

hasMKL
  :: IO (CBool)
hasMKL  =
  [C.throwBlock| bool { return (at::hasMKL(
    ));
  }|]

hasLAPACK
  :: IO (CBool)
hasLAPACK  =
  [C.throwBlock| bool { return (at::hasLAPACK(
    ));
  }|]

hasMAGMA
  :: IO (CBool)
hasMAGMA  =
  [C.throwBlock| bool { return (at::hasMAGMA(
    ));
  }|]

hasMKLDNN
  :: IO (CBool)
hasMKLDNN  =
  [C.throwBlock| bool { return (at::hasMKLDNN(
    ));
  }|]

manual_seed_L
  :: Word64
  -> IO (())
manual_seed_L _seed =
  [C.throwBlock| void {  (at::manual_seed(
    $(uint64_t _seed)));
  }|]

