
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.DimnameList where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Type.DimnameList as Unmanaged





newDimnameList
  :: IO (ForeignPtr DimnameList)
newDimnameList = _cast0 Unmanaged.newDimnameList

dimnameList_empty
  :: ForeignPtr DimnameList
  -> IO (CBool)
dimnameList_empty = _cast1 Unmanaged.dimnameList_empty

dimnameList_size
  :: ForeignPtr DimnameList
  -> IO (CSize)
dimnameList_size = _cast1 Unmanaged.dimnameList_size

dimnameList_at_s
  :: ForeignPtr DimnameList
  -> CSize
  -> IO (ForeignPtr Dimname)
dimnameList_at_s = _cast2 Unmanaged.dimnameList_at_s

dimnameList_push_back_n
  :: ForeignPtr DimnameList
  -> ForeignPtr Dimname
  -> IO (())
dimnameList_push_back_n = _cast2 Unmanaged.dimnameList_push_back_n

