{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Cast where

import Foreign.ForeignPtr

import Torch.Internal.Managed.Type.IntArray
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast

-- define useful casts

instance CppTuple2 (ForeignPtr IntArray) where
  type A (ForeignPtr IntArray) = Int
  type B (ForeignPtr IntArray) = Int
  get0 v = cast1 (flip intArray_at_s 0) v
  get1 v = cast1 (flip intArray_at_s 1) v

instance CppTuple3 (ForeignPtr IntArray) where
  type C (ForeignPtr IntArray) = Int
  get2 v = cast1 (flip intArray_at_s 2) v

instance CppTuple4 (ForeignPtr IntArray) where
  type D (ForeignPtr IntArray) = Int
  get3 v = cast1 (flip intArray_at_s 3) v

instance CppTuple5 (ForeignPtr IntArray) where
  type E (ForeignPtr IntArray) = Int
  get4 v = cast1 (flip intArray_at_s 4) v

instance CppTuple6 (ForeignPtr IntArray) where
  type F (ForeignPtr IntArray) = Int
  get5 v = cast1 (flip intArray_at_s 5) v

