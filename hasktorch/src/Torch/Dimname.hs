{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Dimname where


import Foreign.ForeignPtr
import ATen.Class (Castable(..))
import qualified ATen.Const as ATen
import qualified ATen.Type as ATen

newtype Dimname = Dimname (ForeignPtr ATen.Dimname)

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
