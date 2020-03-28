
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.Helper where

import qualified Language.C.Inline.Context as C
import qualified Language.C.Inline as C
import qualified Language.C.Types as C
import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)

callbackHelper :: (Ptr () -> IO (Ptr ())) -> IO (FunPtr (Ptr () -> IO (Ptr ())))
callbackHelper func = $(C.mkFunPtr [t| Ptr () -> IO (Ptr ()) |]) func

