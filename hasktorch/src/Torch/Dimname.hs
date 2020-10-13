{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Dimname where

import Data.String
import Foreign.ForeignPtr
import System.IO.Unsafe
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Managed.Type.Dimname as ATen
import qualified Torch.Internal.Managed.Type.StdString as ATen
import qualified Torch.Internal.Managed.Type.Symbol as ATen
import qualified Torch.Internal.Type as ATen

newtype Dimname = Dimname (ForeignPtr ATen.Dimname)

instance IsString Dimname where
  fromString str = unsafePerformIO $ do
    str' <- ATen.newStdString_s str
    symbol <- ATen.dimname_s str'
    dimname <- ATen.fromSymbol_s symbol
    return $ Dimname dimname

instance Castable Dimname (ForeignPtr ATen.Dimname) where
  cast (Dimname dname) f = f dname
  uncast dname f = f $ Dimname dname

instance Castable [Dimname] (ForeignPtr ATen.DimnameList) where
  cast xs f = do
    ptr_list <- mapM (\x -> cast x return :: IO (ForeignPtr ATen.Dimname)) xs
    cast (map Dimname ptr_list) f
  uncast xs f = uncast xs $ \ptr_list -> do
    dname_list <- mapM ((\(x :: ForeignPtr ATen.Dimname) -> uncast x return) . (\(Dimname dname) -> dname)) ptr_list
    f dname_list
