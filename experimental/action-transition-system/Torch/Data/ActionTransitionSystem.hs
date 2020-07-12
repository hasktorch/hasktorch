{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Data.ActionTransitionSystem where

import Prelude hiding ()
import GHC.Generics
import Control.Lens
import Data.Generics.Product
import Data.Generics.Sum
import Data.Kind (Type)
import Control.Foldl (Fold(..), fold, head)
import Control.Applicative (liftA2, pure, Alternative(..), empty, (<|>))
import Control.Monad (mfilter, MonadPlus(..))
import Data.Text (pack, Text)
import GHC.IO (unsafePerformIO)
import Control.Monad.Yoctoparsec (token, Parser)
import Control.Monad.Trans.Free (iterTM, runFreeT, FreeT(..), FreeF(..))
import Control.Monad.State (StateT (..), runStateT, get, put, modify)
import Control.Monad (void)
import Control.Monad.Trans (MonadTrans(lift))
import Data.Map (Map)
import Data.Set (Set)

-- https://stackoverflow.com/questions/17675054/deserialising-with-ghc-generics?rq=1
-- http://hackage.haskell.org/package/cereal-0.5.8.1/docs/Data-Serialize.html
-- https://hackage.haskell.org/package/attoparsec-0.13.2.4/docs/src/Data.Attoparsec.Text.Internal.html
-- https://hackage.haskell.org/package/yoctoparsec-0.1.0.0/docs/src/Control-Monad-Yoctoparsec.html#Parser
-- https://vaibhavsagar.com/blog/2018/02/04/revisiting-monadic-parsing-haskell/
-- https://github.com/alphaHeavy/protobuf/blob/46cda829cf1e7b6bba2ff450e267fb1a9ace4fb3/src/Data/ProtocolBuffers/Ppr.hs

data Relation = Relation
  deriving (Eq, Ord, Show, Generic)

data Env = Env
  { meta :: Maybe M
  , pos :: Pos
  , relations :: Map (Pos, Pos) Relation
  , attentionMask :: Set (Pos, Pos)
  , keyPaddingMask :: Set Pos
  , validActionsMask :: Map Pos (Set Action)
  }
  deriving (Eq, Ord, Show, Generic)

data M = D Text | C Text | S Text
  deriving (Eq, Ord, Show, Generic)

newtype Pos = Pos { unPos :: Int }
  deriving (Eq, Ord, Show, Num, Generic)

data Action = L | R | IToken Int | SToken Text
  deriving (Eq, Ord, Show)

type To t a = a -> t Action
type From b a = Parser (StateT Env b) Action a

token' :: forall b t . Monad b => Parser (StateT Env b) t t
token' = do
  t <- token
  modify (field @"pos" %~ (+1))
  pure t

is :: (MonadPlus b, Eq t) => t -> Parser (StateT Env b) t t
is t = mfilter (== t) token'

class ToActions (t :: Type -> Type) (a :: Type) where
  toActions :: To t a

  default toActions :: (Generic a, GToActions t (Rep a)) => To t a
  toActions = gToActions @t . GHC.Generics.from

class FromActions (b :: Type -> Type) (a :: Type) where
  fromActions :: From b a

  default fromActions :: (Monad b, Generic a, GFromActions b (Rep a)) => From b a
  fromActions = GHC.Generics.to <$> gFromActions @b

class GToActions (t :: Type -> Type) (f :: Type -> Type) where
  gToActions :: forall a . To t (f a)

class GFromActions (b :: Type -> Type) (f :: Type -> Type) where
  gFromActions :: forall a . From b (f a)

instance GToActions t f => GToActions t (M1 i c f) where
  gToActions = gToActions @t . unM1

instance (Monad b, GFromActions b f, Datatype d) => GFromActions b (D1 d f) where
  gFromActions = do
    modify $ field @"meta" .~ (pure . D . pack . datatypeName @d $ undefined)
    M1 <$> gFromActions @b

instance (Monad b, GFromActions b f, Constructor c) => GFromActions b (C1 c f) where
  gFromActions = do
    modify $ field @"meta" .~ (pure . C . pack . conName @c $ undefined)
    M1 <$> gFromActions @b

instance (Monad b, GFromActions b f, Selector s) => GFromActions b (S1 s f) where
  gFromActions = do
    modify $ field @"meta" .~ (pure . S . pack . selName @s $ undefined)
    M1 <$> gFromActions @b

instance ToActions t a => GToActions t (K1 i a) where
  gToActions = toActions @t . unK1

instance (Monad b, FromActions b a) => GFromActions b (K1 i a) where
  gFromActions = K1 <$> fromActions @b

instance Alternative t => GToActions t U1 where
  gToActions _ = empty

instance Monad b => GFromActions b U1 where
  gFromActions = pure U1

instance GToActions t V1 where
  gToActions v = v `seq` error "GFromActions.V1"

instance MonadFail b => GFromActions b V1 where
  gFromActions = fail "GFromActions.V1"

instance (Alternative t, GToActions t f, GToActions t g) => GToActions t (f :*: g) where
  gToActions (f :*: g) = gToActions @t f <|> gToActions @t g

instance (Monad b, GFromActions b f, GFromActions b g) => GFromActions b (f :*: g) where
  gFromActions = (:*:) <$> gFromActions @b <*> gFromActions @b

instance (Applicative t, Alternative t, GToActions t f, GToActions t g) => GToActions t (f :+: g) where
  gToActions (L1 f) = (pure L) <|> gToActions @t f
  gToActions (R1 g) = (pure R) <|> gToActions @t g

instance (MonadPlus b, GFromActions b f, GFromActions b g) => GFromActions b (f :+: g) where
  gFromActions = (is L >> L1 <$> gFromActions @b) <|> (is R >> R1 <$> gFromActions @b)

instance (Alternative t, ToActions t a, ToActions t b) => ToActions t (a, b)
instance (Monad b, FromActions b a, FromActions b b') => FromActions b (a, b')

instance (Alternative t, ToActions t a, ToActions t b, ToActions t c) => ToActions t (a, b, c)
instance (Monad b, FromActions b a, FromActions b b',  FromActions b c) => FromActions b (a, b', c)

instance (Alternative t, ToActions t a, ToActions t b, ToActions t c, ToActions t d) => ToActions t (a, b, c, d)
instance (Monad b, FromActions b a, FromActions b b',  FromActions b c, FromActions b d) => FromActions b (a, b', c, d)

instance (Alternative t, ToActions t a, ToActions t b, ToActions t c, ToActions t d, ToActions t e) => ToActions t (a, b, c, d, e)
instance (Monad b, FromActions b a, FromActions b b',  FromActions b c, FromActions b d, FromActions b e) => FromActions b (a, b', c, d, e)

instance (Applicative t, Alternative t, ToActions t a) => ToActions t [a]
instance (MonadPlus b, FromActions b a) => FromActions b [a]

instance (Applicative t, Alternative t, ToActions t a) => ToActions t (Maybe a)
instance (MonadPlus b, FromActions b a) => FromActions b (Maybe a)

instance (Applicative t, Alternative t, ToActions t a, ToActions t b) => ToActions t (Either a b)
instance (MonadPlus b, FromActions b a, FromActions b b') => FromActions b (Either a b')

instance Applicative t => ToActions t Text where
  toActions = pure . SToken

instance MonadFail b => FromActions b Text where
  fromActions = do
    a <- token'
    case a of
      SToken s -> pure s
      _ -> fail "text"

instance Applicative t => ToActions t Int where
  toActions = pure . IToken

instance MonadFail b => FromActions b Int where
  fromActions = do
    a <- token'
    case a of
      IToken i -> pure i
      _ -> fail "int"

-- FreeT ((->) i) b a ~ StateT (b i) b a ???

-- iterTM specialized to Parser b i a ~ FreeT ((->) i) b a
-- iterTM :: (Monad b, MonadTrans t, Monad (t b)) => ((i -> t b a) -> t b a) -> Parser b i a -> t b a
-- iterTM f p = do
--   val <- lift . runFreeT $ p
--   case val of
--     Pure x -> return x
--     Free y -> f $ \i -> iterTM f (y i)

-- this version of @'iterTM'@ exposes the invermediate step
iterTM' :: (MonadTrans t, Monad b, Monad (t b)) => ((i -> Parser b i a) -> (Parser b i a -> t b a) -> t b a) -> Parser b i a -> t b a
iterTM' f p = do
  val <- lift . runFreeT $ p
  case val of
    Pure x -> return x
    Free y -> f y (iterTM' f)

-- | Idea here:
-- * instead of reducing @s@, we grow it, starting, e.g., from @[]@
-- * @i -> 'Parser' b i a@ is evaluated against the vocabulary, e.g. @[i]@, and only those @i@'s for which the parser does not fail are considered for continuation. among those, the model decides which to pick
-- * @s@ is the sequence of actions, @[i]@, and, at each step, we feed all previous actions to the model to get the next one
-- * in order to support the prediction, information about the parsing step is encoded in @b@
-- I shall test this idea with a random model.
-- How do to beam search?
-- does this work for training? I guess @next@ would build up a loss term. How do handle batching?
parse
  :: forall s b i a
   . Monad b
  => ((i -> Parser b i a) -> s -> b (Parser b i a, s))
  -> Parser b i a
  -> s
  -> b (a, s)
parse next =
  -- let f ip ps = StateT $ \s -> do
  --                 ~(p, s') <- next ip s
  --                 runStateT (ps p) s'
  let f ip ps = StateT (next ip) >>= ps
  in runStateT . iterTM' f

pures :: (Foldable g, Alternative g) => g (FreeF f a (FreeT f m a)) -> g a
pures = foldr (\x xs -> case x of Pure a -> pure a <|> xs; _ -> xs) empty

frees :: (Foldable g, Alternative g) => g (FreeF f a (FreeT f m a)) -> g (f (FreeT f m a))
frees = foldr (\x xs -> case x of Free fb -> pure fb <|> xs; _ -> xs) empty

batchedIterTM
  :: forall f t b a i
   . (Traversable f, Foldable f, Alternative f, MonadTrans t, Monad b, Monad (t b))
  => (f a -> f (i -> Parser b i a) -> (f (Parser b i a) -> t b (f a)) -> t b (f a))
  -> f (Parser b i a)
  -> t b (f a)
batchedIterTM f ps = do 
  vals <- traverse (lift @t . runFreeT) ps
  f (pures vals) (frees vals) (batchedIterTM f)

batchedParse
  :: forall f s b i a
   . (Traversable f, Foldable f, Alternative f, Monad b)
  => (f (i -> Parser b i a) -> s -> b (f (Parser b i a), s))
  -> f (Parser b i a)
  -> s
  -> b (f a, s)
batchedParse next = do
  let f as ip ps = StateT (next ip) >>= ps >>= (pure . (<|> as))
  runStateT . batchedIterTM f

-- | Runs the parser on the supplied input and returns whether or not the parse succeeded.
-- Results are discarded.
-- TODO: this isn't nice yet. It would be great if there was a stronger signal for failure than just 'mzero'.
-- Parser b i a ~ FreeT ((->) i) b a
check :: forall b i a . MonadPlus b => Parser b i a -> i -> b ()
check p i = do
  val <- runFreeT p
  case val of
    Pure a -> mzero
    Free f -> void . runFreeT $ f i


data Stuff = SomeStuff { anInt :: Int, moreStuff :: [Stuff], maybeFoo :: Maybe Foo }
           | NoStuff
  deriving (Eq, Show, Generic)

instance ToActions [] Stuff
instance FromActions [] Stuff

data Foo = SomeFoo { someText :: Text, stuff :: Stuff }
         | NoFoo
  deriving (Eq, Show, Generic)

instance ToActions [] Foo
instance FromActions [] Foo

test :: ([Action], [((Foo, [Action]), Env)], [((), Env)])
test =
  let env = Env Nothing (Pos 0) mempty mempty mempty mempty
      stuff 0 = []
      stuff n = SomeStuff n [] Nothing : stuff (n - 1)
      foo 0 = SomeFoo "a" $ SomeStuff 0 [SomeStuff 2 [] Nothing] Nothing
      foo n = SomeFoo "a" $ SomeStuff n [SomeStuff 2 (stuff n) Nothing] (Just $ foo (n - 1))
      challenge = foo 2
      actions = toActions @[] challenge
      parser = fromActions @[]
      result' = let f ap [] = empty
                    f ap (a : as) = let p = ap a in do
                      env' <- get
                      pure $ unsafePerformIO $ print (a, env') >> pure (p, as)
                in runStateT (parse f parser actions) env
  in (actions, result', runStateT (check parser (IToken 1)) env)
