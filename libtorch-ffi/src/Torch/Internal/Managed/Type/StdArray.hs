
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.StdArray where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Type.StdArray as Unmanaged



newStdArrayBool2
  :: IO (ForeignPtr (StdArray '(CBool,2)))
newStdArrayBool2 = _cast0 Unmanaged.newStdArrayBool2

newStdArrayBool2_bb
  :: CBool
  -> CBool
  -> IO (ForeignPtr (StdArray '(CBool,2)))
newStdArrayBool2_bb = _cast2 Unmanaged.newStdArrayBool2_bb

instance CppTuple2 (ForeignPtr (StdArray '(CBool,2))) where
  type A (ForeignPtr (StdArray '(CBool,2))) = CBool
  type B (ForeignPtr (StdArray '(CBool,2))) = CBool
  get0 v = _cast1 (get0 :: Ptr (StdArray '(CBool,2)) -> IO CBool) v
  get1 v = _cast1 (get1 :: Ptr (StdArray '(CBool,2)) -> IO CBool) v

newStdArrayBool3
  :: IO (ForeignPtr (StdArray '(CBool,3)))
newStdArrayBool3 = _cast0 Unmanaged.newStdArrayBool3

newStdArrayBool3_bbb
  :: CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdArray '(CBool,3)))
newStdArrayBool3_bbb = _cast3 Unmanaged.newStdArrayBool3_bbb

instance CppTuple2 (ForeignPtr (StdArray '(CBool,3))) where
  type A (ForeignPtr (StdArray '(CBool,3))) = CBool
  type B (ForeignPtr (StdArray '(CBool,3))) = CBool
  get0 v = _cast1 (get0 :: Ptr (StdArray '(CBool,3)) -> IO CBool) v
  get1 v = _cast1 (get1 :: Ptr (StdArray '(CBool,3)) -> IO CBool) v

instance CppTuple3 (ForeignPtr (StdArray '(CBool,3))) where
  type C (ForeignPtr (StdArray '(CBool,3))) = CBool
  get2 v = _cast1 (get2 :: Ptr (StdArray '(CBool,3)) -> IO CBool) v

newStdArrayBool4
  :: IO (ForeignPtr (StdArray '(CBool,4)))
newStdArrayBool4 = _cast0 Unmanaged.newStdArrayBool4

newStdArrayBool4_bbbb
  :: CBool
  -> CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdArray '(CBool,4)))
newStdArrayBool4_bbbb = _cast4 Unmanaged.newStdArrayBool4_bbbb

instance CppTuple2 (ForeignPtr (StdArray '(CBool,4))) where
  type A (ForeignPtr (StdArray '(CBool,4))) = CBool
  type B (ForeignPtr (StdArray '(CBool,4))) = CBool
  get0 v = _cast1 (get0 :: Ptr (StdArray '(CBool,4)) -> IO CBool) v
  get1 v = _cast1 (get1 :: Ptr (StdArray '(CBool,4)) -> IO CBool) v

instance CppTuple3 (ForeignPtr (StdArray '(CBool,4))) where
  type C (ForeignPtr (StdArray '(CBool,4))) = CBool
  get2 v = _cast1 (get2 :: Ptr (StdArray '(CBool,4)) -> IO CBool) v

instance CppTuple4 (ForeignPtr (StdArray '(CBool,4))) where
  type D (ForeignPtr (StdArray '(CBool,4))) = CBool
  get3 v = _cast1 (get3 :: Ptr (StdArray '(CBool,4)) -> IO CBool) v
