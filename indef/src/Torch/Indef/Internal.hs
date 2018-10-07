{-# LANGUAGE TypeFamilies #-}
module Torch.Indef.Internal
  ( throwFIXME
  , throwNE
  , throwGT4
  ) where

import Foreign
import GHC.ForeignPtr (ForeignPtr)
import Control.Exception.Safe
import qualified Foreign.Marshal.Array as FM

import Torch.Sig.Types

throwFIXME :: MonadThrow io => String -> String -> io x
throwFIXME fixme msg = throwString $ msg ++ " (FIXME: " ++ fixme ++ ")"

throwNE :: MonadThrow io => String -> io x
throwNE = throwFIXME "make this function only take a non-empty [Nat]"

throwGT4 :: MonadThrow io => String -> io x
throwGT4 fnname = throwFIXME
  ("review how TH supports `" ++ fnname ++ "` operations on > rank-4 tensors")
  (fnname ++ " with >4 rank")


