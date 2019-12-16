{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Internal.Managed.Type.StdString where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import qualified Torch.Internal.Unmanaged.Type.StdString as Unmanaged



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

instance Castable String (ForeignPtr StdString) where
  cast str f = newStdString_s str >>= f
  uncast xs f = string_c_str xs >>= f
