{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}

module Torch.Data.ActionTransitionSystem where

import GHC.Generics
import Data.Kind (Type)

-- https://stackoverflow.com/questions/17675054/deserialising-with-ghc-generics?rq=1


data Production = Production
  deriving (Eq, Ord, Show)

newtype Token = Token String
  deriving (Eq, Ord, Show)

data Action = ApplyRule Production
            | Reduce
            | GenToken Token
  deriving (Eq, Ord, Show)

class ActionTransitionSystem (a :: Type) where
  toActions :: a -> [Action]

  fromActions :: [Action] -> a

instance (Generic a, GActionTransitionSystem (Rep a)) => ActionTransitionSystem a where
  fromActions = to . gFromActions
  toActions = gToActions . from

class GActionTransitionSystem (f :: Type -> Type) where
  gFromActions :: forall a . [Action] -> f a
  gToActions :: forall a . f a -> [Action]

instance GActionTransitionSystem U1 where
  gFromActions _ = U1
  gToActions _ = []

instance 

data Stuff = Stuff { anInt :: Int, foo :: Foo, moreStuff :: [Stuff] }
  deriving (Eq, Show, Generic)

data Foo = Foo { aString :: String, stuff :: Stuff }
  deriving (Eq, Show, Generic)
