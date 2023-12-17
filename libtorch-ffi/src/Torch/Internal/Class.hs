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
  deletePtr :: Ptr a -> IO ()

class CppTuple2 m where
  type A m
  type B m
  get0 :: m -> IO (A m)
  get1 :: m -> IO (B m)
  makeTuple2 :: (A m, B m) -> IO m
  makeTuple2 = error "makeTuple2 is not implemented."
  -- | Just an idea of default-signatures
  -- default makeTuple2
  --   :: forall a b c.
  --    ( CppObject c
  --    , ForeignPtr a ~ A m
  --    , ForeignPtr b ~ B m
  --    , ForeignPtr c ~ m
  --    , Ptr a ~ A (Ptr c)
  --    , Ptr b ~ B (Ptr c)
  --    , CppTuple2 (Ptr c))
  --   => (A m, B m)
  --   -> IO m
  -- makeTuple2 (a,b) =
  --   withForeignPtr a $ \a' -> do
  --     withForeignPtr b $ \b' -> do
  --       fromPtr =<< makeTuple2 (a',b')

class CppTuple2 m => CppTuple3 m where
  type C m
  get2 :: m -> IO (C m)
  makeTuple3 :: (A m,B m,C m) -> IO m
  makeTuple3 = error "makeTuple3 is not implemented."

class CppTuple3 m => CppTuple4 m where
  type D m
  get3 :: m -> IO (D m)
  makeTuple4 :: (A m,B m,C m,D m) -> IO m
  makeTuple4 = error "makeTuple4 is not implemented."

class CppTuple4 m => CppTuple5 m where
  type E m
  get4 :: m -> IO (E m)
  makeTuple5 :: (A m,B m,C m,D m,E m) -> IO m
  makeTuple5 = error "makeTuple5 is not implemented."

class CppTuple5 m => CppTuple6 m where
  type F m
  get5 :: m -> IO (F m)
  makeTuple6 :: (A m,B m,C m,D m,E m,F m) -> IO m
  makeTuple6 = error "makeTuple6 is not implemented."

class CppTuple6 m => CppTuple7 m where
  type G m
  get6 :: m -> IO (G m)
  makeTuple7 :: (A m,B m,C m,D m,E m,F m,G m) -> IO m
  makeTuple7 = error "makeTuple7 is not implemented."

class CppTuple7 m => CppTuple8 m where
  type H m
  get7 :: m -> IO (H m)
  makeTuple8 :: (A m,B m,C m,D m,E m,F m,G m,H m) -> IO m
  makeTuple8 = error "makeTuple8 is not implemented."

class CppTuple8 m => CppTuple9 m where
  type I m
  get8 :: m -> IO (I m)
  makeTuple9 :: (A m,B m,C m,D m,E m,F m,G m,H m,I m) -> IO m
  makeTuple9 = error "makeTuple9 is not implemented."
