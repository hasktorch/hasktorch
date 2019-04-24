
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Managed.Type.StdArray where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Aten.Type
import Aten.Class
import Aten.Cast
import qualified Aten.Unmanaged.Type.StdArray as Unmanaged



newStdArrayBool2
  :: IO (ForeignPtr (StdArray CBool 2))
newStdArrayBool2 = cast0 Unmanaged.newStdArrayBool2

newStdArrayBool3
  :: IO (ForeignPtr (StdArray CBool 3))
newStdArrayBool3 = cast0 Unmanaged.newStdArrayBool3

newStdArrayBool4
  :: IO (ForeignPtr (StdArray CBool 4))
newStdArrayBool4 = cast0 Unmanaged.newStdArrayBool4





