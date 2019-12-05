{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Dimname where


import Foreign.ForeignPtr
import LibTorch.ATen.Class (Castable(..))
import qualified LibTorch.ATen.Const as ATen
import qualified LibTorch.ATen.Type as ATen
import qualified LibTorch.ATen.Managed.Type.Symbol as ATen
import qualified LibTorch.ATen.Managed.Type.Dimname as ATen
import qualified LibTorch.ATen.Managed.Type.StdString as ATen
import Data.String
import System.IO.Unsafe

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
    ptr_list <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Dimname))) xs
    cast (map Dimname ptr_list) f
  uncast xs f = uncast xs $ \ptr_list -> do
    dname_list <- mapM (\(x :: ForeignPtr ATen.Dimname) -> uncast x return) $ map (\(Dimname dname) -> dname) ptr_list
    f dname_list

