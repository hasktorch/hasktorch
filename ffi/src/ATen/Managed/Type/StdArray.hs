
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module ATen.Managed.Type.StdArray where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import ATen.Type
import ATen.Class
import ATen.Cast
import qualified ATen.Unmanaged.Type.StdArray as Unmanaged



newStdArrayBool2
  :: IO (ForeignPtr (StdArray CBool 2))
newStdArrayBool2 = cast0 Unmanaged.newStdArrayBool2

newStdArrayBool3
  :: IO (ForeignPtr (StdArray CBool 3))
newStdArrayBool3 = cast0 Unmanaged.newStdArrayBool3

newStdArrayBool4
  :: IO (ForeignPtr (StdArray CBool 4))
newStdArrayBool4 = cast0 Unmanaged.newStdArrayBool4





