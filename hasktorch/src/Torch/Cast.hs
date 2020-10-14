{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Cast where

import Foreign.ForeignPtr
import Torch.Internal.Cast
import Torch.Internal.Class
import Torch.Internal.Managed.Type.IntArray
import Torch.Internal.Type

-- define useful casts

instance CppTuple2 (ForeignPtr IntArray) where
  type A (ForeignPtr IntArray) = Int
  type B (ForeignPtr IntArray) = Int
  get0 = cast1 (`intArray_at_s` 0)
  get1 = cast1 (`intArray_at_s` 1)

instance CppTuple3 (ForeignPtr IntArray) where
  type C (ForeignPtr IntArray) = Int
  get2 = cast1 (`intArray_at_s` 2)

instance CppTuple4 (ForeignPtr IntArray) where
  type D (ForeignPtr IntArray) = Int
  get3 = cast1 (`intArray_at_s` 3)

instance CppTuple5 (ForeignPtr IntArray) where
  type E (ForeignPtr IntArray) = Int
  get4 = cast1 (`intArray_at_s` 4)

instance CppTuple6 (ForeignPtr IntArray) where
  type F (ForeignPtr IntArray) = Int
  get5 = cast1 (`intArray_at_s` 5)
