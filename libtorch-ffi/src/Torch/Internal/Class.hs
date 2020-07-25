{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Internal.Class where

import Foreign (Ptr, ForeignPtr)

class Castable a b where
  cast   :: a -> (b -> IO r) -> IO r
  uncast :: b -> (a -> IO r) -> IO r

class CppObject a where
  fromPtr :: Ptr a -> IO (ForeignPtr a)

class CppTuple2 m where
  type A m
  type B m
  get0 :: m -> IO (A m)
  get1 :: m -> IO (B m)

class CppTuple2 m => CppTuple3 m where
  type C m
  get2 :: m -> IO (C m)

class CppTuple3 m => CppTuple4 m where
  type D m
  get3 :: m -> IO (D m)

class CppTuple4 m => CppTuple5 m where
  type E m
  get4 :: m -> IO (E m)

class CppTuple5 m => CppTuple6 m where
  type F m
  get5 :: m -> IO (F m)
