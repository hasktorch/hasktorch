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

import Prelude hiding (head, length)
import GHC.Generics
import Data.Kind (Type)
-- import Data.Attoparsec.Internal.Types (State, Pos(..), Parser(..))
-- import Data.Attoparsec.Combinator (eitherP)
import Control.Foldl (Fold(..), fold, head)
import Control.Applicative (liftA2, pure, Alternative(..), empty, (<|>))
import Control.Monad (MonadPlus(..))
import qualified Control.Monad.Fail (MonadFail(..))
import Data.Text (Text)

-- https://stackoverflow.com/questions/17675054/deserialising-with-ghc-generics?rq=1
-- http://hackage.haskell.org/package/cereal-0.5.8.1/docs/Data-Serialize.html
-- https://hackage.haskell.org/package/attoparsec-0.13.2.4/docs/src/Data.Attoparsec.Text.Internal.html

newtype Pos = Pos { fromPos :: Int }
  deriving (Eq, Ord, Show, Num)

data Result i r =
    Fail [String] String
  | Partial (i -> Result i r)
  | Done r

instance Show r => Show (Result i r) where
    show (Fail stack _) = "Fail " ++ show stack
    show (Partial _)  = "Partial _"
    show (Done r)  = "Done " ++ show r

instance Functor (Result i) where
    fmap _ (Fail stack msg) = Fail stack msg
    fmap f (Partial k)      = Partial (fmap f . k)
    fmap f (Done r)         = Done (f r)

newtype Parser i a = Parser {
    runParser
      :: forall r
       . Pos
      -> Failure i   r
      -> Success i a r
      -> Result  i   r
  }

type Failure i   r = Pos -> [String] -> String -> Result i r
type Success i a r = Pos -> a -> Result i r

instance Monad (Parser i) where
  return = Control.Applicative.pure
  {-# INLINE return #-}
  m >>= k = Parser $ \ !pos lose succ ->
    let succ' !pos' a = runParser (k a) pos' lose succ
    in runParser m pos lose succ'
  {-# INLINE (>>=) #-}
  (>>) = (*>)
  {-# INLINE (>>) #-}

instance MonadFail (Parser i) where
  fail err = Parser $ \pos lose _succ -> lose pos [] msg
    where msg = "Failed reading: " ++ err
  {-# INLINE fail #-}

plus :: Parser i a -> Parser i a -> Parser i a
plus f g = Parser $ \pos lose succ ->
  let lose' _pos' _ctx _msg = runParser g pos lose succ
  in runParser f pos lose' succ

instance MonadPlus (Parser i) where
  mzero = fail "mzero"
  {-# INLINE mzero #-}
  mplus = plus

instance Functor (Parser i) where
  fmap f p = Parser $ \pos lose succ ->
    let succ' pos' a = succ pos' (f a)
    in runParser p pos lose succ'
  {-# INLINE fmap #-}

apP :: Parser i (a -> b) -> Parser i a -> Parser i b
apP d e = do
  b <- d
  a <- e
  return (b a)
{-# INLINE apP #-}

instance Applicative (Parser i) where
  pure v = Parser $ \ !pos _lose succ -> succ pos v
  {-# INLINE pure #-}
  (<*>) = apP
  {-# INLINE (<*>) #-}
  m *> k = m >>= \_ -> k
  {-# INLINE (*>) #-}
  x <* y = x >>= \a -> y >> pure a
  {-# INLINE (<*) #-}

instance Semigroup (Parser i a) where
  (<>) = plus
  {-# INLINE (<>) #-}

instance Monoid (Parser i a) where
  mempty = fail "mempty"
  {-# INLINE mempty #-}
  mappend = (<>)
  {-# INLINE mappend #-}

instance Alternative (Parser i) where
  empty = fail "empty"
  {-# INLINE empty #-}
  (<|>) = plus
  {-# INLINE (<|>) #-}
  many v = many_v
    where
      many_v = some_v <|> pure []
      some_v = (:) <$> v <*> many_v
  {-# INLINE many #-}
  some v = some_v
    where
      many_v = some_v <|> pure []
      some_v = (:) <$> v <*> many_v
  {-# INLINE some #-}

failK :: Failure i a
failK _pos stack msg = Fail stack msg

successK :: Success i a a
successK _pos a = Done a

parse :: forall i a . Parser i a -> Result i a
parse p = runParser p 0 failK successK

choice :: Alternative f => [f a] -> f a
choice = foldr (<|>) empty

option :: Alternative f => a -> f a -> f a
option a p = p <|> pure a

many1 :: Alternative f => f a -> f [a]
many1 p = liftA2 (:) p (many p)
{-# INLINE many1 #-}

manyTill :: Alternative f => f a -> f b -> f [a]
manyTill p end = scan
  where scan = (end *> pure []) <|> liftA2 (:) p scan

skipMany :: Alternative f => f a -> f ()
skipMany p = scan
  where scan = (p *> scan) <|> pure ()

skipMany1 :: Alternative f => f a -> f ()
skipMany1 p = p *> skipMany p

feed :: Result i r -> i -> Result i r
feed fail@(Fail _ _) _ = fail
feed (Partial k) i = k i
feed done@(Done _) _ = done

feeds :: Parser i r -> Fold i (Result i r)
feeds p = let step = feed
              initial = parse p
              extract = id
          in Fold step initial extract

-- lookAhead :: Parser i a -> Parser i a
-- lookAhead p = Parser $ \pos lose succ ->
--   let succ' _pos' = succ pos
--   in runParser p pos lose succ'

advance :: forall i . Pos -> Parser i ()
advance n = Parser $ \pos _lose succ -> succ (pos + n) ()

demandInput :: forall i . Parser i i
demandInput = Parser $ \pos lose succ -> Partial $ \i -> succ pos i

satisfy :: forall i . (i -> Bool) -> Parser i i
satisfy p = do
  i <- demandInput
  if p i
    then advance 1 >> return i
    else fail "satisfy"
{-# INLINE satisfy #-}

-- | Match a specific input.
is :: forall i . (Eq i, Show i) => i -> Parser i i
is i = satisfy (== i) <?> show i
{-# INLINE is #-}

-- | Match any input ecept the given one.
isNot :: forall i. (Eq i, Show i) => i -> Parser i i
isNot i = satisfy (/= i) <?> "not " ++ show i
{-# INLINE isNot #-}

-- | Name the parser, in case failure occurs.
(<?>) :: Parser i i -> String -> Parser i i
p <?> name = Parser $ \pos lose succ ->
  let lose' pos' stack msg = lose pos' (name : stack) msg
  in runParser p pos lose' succ
{-# INLINE (<?>) #-}
infix 0 <?>

data Action = L | R | Grow | Reduce | IToken Int | SToken Text
  deriving (Eq, Ord, Show)

type ToActions t a = a -> t Action
type FromActions = Parser Action

class ActionTransitionSystem (t :: Type -> Type) (a :: Type) where
  toActions :: ToActions t a
  fromActions :: FromActions a

  default toActions :: (Generic a, GToActions t (Rep a)) => ToActions t a
  toActions = gToActions . from

  default fromActions :: (Generic a, GFromActions t (Rep a)) => FromActions a
  fromActions = to <$> gFromActions @t

class GToActions (t :: Type -> Type) (f :: Type -> Type) where
  gToActions :: forall a . ToActions t (f a)

class GFromActions (t :: Type -> Type) (f :: Type -> Type) where
  gFromActions :: forall a . FromActions (f a)

instance GToActions t f => GToActions t (M1 i c f) where
  gToActions = gToActions . unM1

instance GFromActions t f => GFromActions t (M1 i c f) where
  gFromActions = M1 <$> gFromActions @t

instance ActionTransitionSystem t a => GToActions t (K1 i a) where
  gToActions = toActions . unK1

instance ActionTransitionSystem t a => GFromActions t (K1 i a) where
  gFromActions = K1 <$> fromActions @t

instance Monoid (t Action) => GToActions t U1 where
  gToActions _ = mempty

instance GFromActions t U1 where
  gFromActions = pure U1

instance GToActions t V1 where
  gToActions v = v `seq` error "GFromActions.V1"

instance GFromActions t V1 where
  gFromActions = fail "GFromActions.V1"

instance (Semigroup (t Action), GToActions t f, GToActions t g) => GToActions t (f :*: g) where
  gToActions (f :*: g) = gToActions f <> gToActions g

instance (GFromActions t f, GFromActions t g) => GFromActions t (f :*: g) where
  gFromActions = (:*:) <$> gFromActions @t <*> gFromActions @t

instance (Applicative t, Semigroup (t Action), GToActions t f, GToActions t g) => GToActions t (f :+: g) where
  gToActions (L1 f) = (pure L) <> gToActions f
  gToActions (R1 g) = (pure R) <> gToActions g

instance (GFromActions t f, GFromActions t g) => GFromActions t (f :+: g) where
  gFromActions = (is L >> L1 <$> gFromActions @t) <|> (is R >> R1 <$> gFromActions @t)

-- instance (Applicative t, Monoid (t Action), ActionTransitionSystem t a) => ActionTransitionSystem t [a] where
--   toActions as = pure Grow <> foldMap toActions as <> pure Reduce
--   fromActions = is Grow >> manyTill (fromActions @t) (is Reduce)

-- instance (Applicative t, ActionTransitionSystem t a) => ActionTransitionSystem t (Maybe a) where
--   toActions Nothing = pure Reduce
--   toActions (Just a) = toActions a
--   fromActions = (is Reduce >> pure Nothing) <|> fromActions @t

instance Applicative t => ActionTransitionSystem t Text where
  toActions = pure . SToken
  fromActions = do
    a <- demandInput
    case a of
      SToken s -> pure s
      _ -> fail "text"

instance Applicative t => ActionTransitionSystem t Int where
  toActions = pure . IToken
  fromActions = do
    a <- demandInput
    case a of
      IToken i -> pure i
      _ -> fail "int"

data Stuff = Stuff { anInt :: Int, moreStuff :: [Stuff], maybeFoo :: Maybe Foo }
  deriving (Eq, Show, Generic)

data Foo = Foo { someText :: Text, stuff :: Stuff }
  deriving (Eq, Show, Generic)

instance ActionTransitionSystem [] a => ActionTransitionSystem [] [a]
instance ActionTransitionSystem [] (Maybe Foo)
instance ActionTransitionSystem [] Stuff
instance ActionTransitionSystem [] Foo

test :: ([Action], Result Action Foo)
test =
  let foo = Foo "a" $ Stuff 1 [Stuff 2 [] Nothing] Nothing
      actions = toActions foo
  in (actions, fold (feeds (fromActions @[])) actions)

test2 :: ([Action], Result Action [Int])
test2 =
  let foo = [1] :: [Int]
      actions = toActions foo
  in (actions, fold (feeds (fromActions @[])) actions)
