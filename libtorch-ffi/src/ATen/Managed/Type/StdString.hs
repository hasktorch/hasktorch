{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module ATen.Managed.Type.StdString where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import ATen.Type
import ATen.Class
import ATen.Cast
import qualified ATen.Unmanaged.Type.StdString as Unmanaged



newStdString
  :: IO (ForeignPtr StdString)
newStdString = cast0 Unmanaged.newStdString

newStdString_s
  :: String
  -> IO (ForeignPtr StdString)
newStdString_s str = cast1 Unmanaged.newStdString_s str

string_c_str
  :: ForeignPtr StdString
  -> IO String
string_c_str str = cast1 Unmanaged.string_c_str str

instance Castable (ForeignPtr StdString) String where
  cast str f = newStdString_s str >>= f
  uncast xs f = string_c_str xs >>= f
