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

import Prelude hiding (head, length)
import GHC.Generics
import Control.Lens
import Data.Generics.Product
import Data.Generics.Sum
import Data.Kind (Type)
-- import Data.Attoparsec.Internal.Types (State, Pos(..), Parser(..))
-- import Data.Attoparsec.Combinator (eitherP)
import Control.Foldl (Fold(..), fold, head)
import Control.Applicative (liftA2, pure, Alternative(..), empty, (<|>))
import Control.Monad (mfilter, MonadPlus(..))
import qualified Control.Monad.Fail (MonadFail(..))
import Data.Text (pack, Text)
import GHC.IO (unsafePerformIO)
import Control.Monad.Yoctoparsec (parseString, token, Parser)
import Control.Monad.Trans.Free (iterTM, runFreeT, FreeT(..), FreeF(..))
import Data.List (length)
import Control.Monad.State (StateT (..), runStateT, get, put, modify)
import Data.List (uncons)
import Control.Monad (void)
import Control.Monad.Trans (MonadTrans)
import Control.Monad.Trans (MonadTrans(lift))
import Control.Monad.Cont (runContT, ContT(ContT))

-- https://stackoverflow.com/questions/17675054/deserialising-with-ghc-generics?rq=1
-- http://hackage.haskell.org/package/cereal-0.5.8.1/docs/Data-Serialize.html
-- https://hackage.haskell.org/package/attoparsec-0.13.2.4/docs/src/Data.Attoparsec.Text.Internal.html
-- https://hackage.haskell.org/package/yoctoparsec-0.1.0.0/docs/src/Control-Monad-Yoctoparsec.html#Parser
-- https://vaibhavsagar.com/blog/2018/02/04/revisiting-monadic-parsing-haskell/
-- https://github.com/alphaHeavy/protobuf/blob/46cda829cf1e7b6bba2ff450e267fb1a9ace4fb3/src/Data/ProtocolBuffers/Ppr.hs

data Env = Env { meta :: Maybe M, pos :: Pos }
  deriving (Eq, Ord, Show, Generic)

data M = D Text | C Text | S Text
  deriving (Eq, Ord, Show, Generic)

newtype Pos = Pos { fromPos :: Int }
  deriving (Eq, Ord, Show, Num, Generic)

-- data Result i r =
--     Fail [String] String
--   | Partial (i -> Result i r)
--   | Done r

-- instance Show r => Show (Result i r) where
--     show (Fail stack msg) = "Fail " ++ show stack ++ " " ++ show msg
--     show (Partial _)  = "Partial _"
--     show (Done r)  = "Done " ++ show r

-- instance Functor (Result i) where
--     fmap _ (Fail stack msg) = Fail stack msg
--     fmap f (Partial k)      = Partial (fmap f . k)
--     fmap f (Done r)         = Done (f r)

-- newtype Parser i a = Parser {
--     runParser
--       :: forall r
--        . Buffer i
--       -> Pos
--       -> Failure i   r
--       -> Success i a r
--       -> Result  i   r
--   }

-- type Buffer  i     = Maybe i
-- type Failure i   r = Buffer i -> Pos -> [String] -> String -> Result i r
-- type Success i a r = Buffer i -> Pos -> a -> Result i r

-- instance Show i => Monad (Parser i) where
--   return = Control.Applicative.pure
--   {-# INLINE return #-}
--   m >>= k = Parser $ \buf !pos lose succ ->
--     let succ' buf' !pos' a = unsafePerformIO $ do
--           -- print ((fromPos pos, buf), (fromPos pos', buf'))
--           pure $ runParser (k a) buf' pos' lose succ
--     in runParser m buf pos lose succ'
--   {-# INLINE (>>=) #-}
--   (>>) = (*>)
--   {-# INLINE (>>) #-}

-- instance Show i => MonadFail (Parser i) where
--   fail err = Parser $ \buf pos lose _succ -> lose buf pos [] msg
--     where msg = "Failed reading: " ++ err
--   {-# INLINE fail #-}

-- plus :: Parser i a -> Parser i a -> Parser i a
-- plus f g = Parser $ \buf pos lose succ ->
--   let lose' buf' _pos' _ctx _msg = runParser g buf' pos lose succ
--   in runParser f buf pos lose' succ

-- instance Show i => MonadPlus (Parser i) where
--   mzero = fail "mzero"
--   {-# INLINE mzero #-}
--   mplus = plus

-- instance Functor (Parser i) where
--   fmap f p = Parser $ \buf pos lose succ ->
--     let succ' buf' pos' a = succ buf' pos' (f a)
--     in runParser p buf pos lose succ'
--   {-# INLINE fmap #-}

-- apP :: forall i a b . Show i => Parser i (a -> b) -> Parser i a -> Parser i b
-- apP d e = do
--   b <- d
--   a <- e
--   return (b a)
-- {-# INLINE apP #-}

-- instance Show i => Applicative (Parser i) where
--   pure v = Parser $ \buf !pos _lose succ -> succ buf pos v
--   {-# INLINE pure #-}
--   (<*>) = apP
--   {-# INLINE (<*>) #-}
--   m *> k = m >>= \_ -> k
--   {-# INLINE (*>) #-}
--   x <* y = x >>= \a -> y >> pure a
--   {-# INLINE (<*) #-}

-- instance Show i => Semigroup (Parser i a) where
--   (<>) = plus
--   {-# INLINE (<>) #-}

-- instance Show i => Monoid (Parser i a) where
--   mempty = fail "mempty"
--   {-# INLINE mempty #-}
--   mappend = (<>)
--   {-# INLINE mappend #-}

-- instance Show i => Alternative (Parser i) where
--   empty = fail "empty"
--   {-# INLINE empty #-}
--   (<|>) = plus
--   {-# INLINE (<|>) #-}
--   many v = many_v
--     where
--       many_v = some_v <|> pure []
--       some_v = (:) <$> v <*> many_v
--   {-# INLINE many #-}
--   some v = some_v
--     where
--       many_v = some_v <|> pure []
--       some_v = (:) <$> v <*> many_v
--   {-# INLINE some #-}

-- failK :: Failure i a
-- failK _buf _pos stack msg = Fail stack msg

-- successK :: Success i a a
-- successK _buf _pos a = Done a

-- parse :: forall i a . Parser i a -> Result i a
-- parse p = runParser p Nothing 0 failK successK

-- choice :: Alternative f => [f a] -> f a
-- choice = foldr (<|>) empty

-- option :: Alternative f => a -> f a -> f a
-- option a p = p <|> pure a

-- many1 :: Alternative f => f a -> f [a]
-- many1 p = liftA2 (:) p (many p)
-- {-# INLINE many1 #-}

-- manyTill :: Alternative f => f a -> f b -> f [a]
-- manyTill p end = scan
--   where scan = (end *> pure []) <|> liftA2 (:) p scan

-- skipMany :: Alternative f => f a -> f ()
-- skipMany p = scan
--   where scan = (p *> scan) <|> pure ()

-- skipMany1 :: Alternative f => f a -> f ()
-- skipMany1 p = p *> skipMany p

-- feed :: Result i r -> i -> Result i r
-- feed fail@(Fail _ _) _ = fail
-- feed (Partial k) i = k i
-- feed done@(Done _) _ = done

-- feeds :: Parser i r -> Fold i (Result i r)
-- feeds p = let step = feed
--               initial = parse p
--               extract = id
--           in Fold step initial extract

-- -- lookAhead :: Parser i a -> Parser i a
-- -- lookAhead p = Parser $ \pos lose succ ->
-- --   let succ' _pos' = succ pos
-- --   in runParser p pos lose succ'

-- advance :: forall i . Pos -> Parser i ()
-- advance n = Parser $ \buf pos _lose succ -> succ buf (pos + n) ()

-- demandInput :: forall i . Parser i ()
-- demandInput = Parser $ \_buf pos _lose succ ->
--   Partial $ \i -> succ (Just i) pos ()

-- buffer :: forall i r . Buffer i -> Parser i ()
-- buffer buf = Parser $ \_buf' pos _lose succ -> succ buf pos ()

-- ensureSuspended :: forall i r . Show i => Pos -> Failure i r -> Success i i r -> Result i r
-- ensureSuspended pos lose succ =
--   runParser (demandInput >> go) Nothing pos lose succ
--  where go = Parser $ \buf' pos' lose' succ' ->
--          case buf' of
--            Just i -> succ' Nothing pos' i
--            Nothing -> runParser (demandInput >> go) Nothing pos' lose' succ'

-- ensure :: forall i . Show i => Parser i i
-- ensure = Parser $ \buf pos lose succ ->
--   case buf of
--     Just i -> succ Nothing pos i
--     Nothing -> ensureSuspended pos lose succ

-- satisfy :: forall i . Show i => (i -> Bool) -> Parser i i
-- satisfy p = do
--   i <- ensure
--   if p i
--     then advance 1 >> return i
--     else buffer (Just i) >> fail "satisfy"
-- {-# INLINE satisfy #-}

-- -- | Match a specific input.
-- is :: forall i . (Eq i, Show i) => i -> Parser i i
-- is i = satisfy (== i) <?> show i
-- {-# INLINE is #-}

-- -- | Match any input ecept the given one.
-- isNot :: forall i. (Eq i, Show i) => i -> Parser i i
-- isNot i = satisfy (/= i) <?> "not " ++ show i
-- {-# INLINE isNot #-}

-- -- | Name the parser, in case failure occurs.
-- (<?>) :: Parser i i -> String -> Parser i i
-- p <?> name = Parser $ \buf pos lose succ ->
--   let lose' buf' pos' stack msg = lose buf' pos' (name : stack) msg
--   in runParser p buf pos lose' succ
-- {-# INLINE (<?>) #-}
-- infix 0 <?>

-- -- | Runs the parser on the supplied input and returns whether or not the parse succeeded.
-- -- Results are discarded.
-- check :: forall i a . Parser i a -> i -> Bool
-- check = undefined

data Action = L | R | IToken Int | SToken Text
  deriving (Eq, Ord, Show)

type ToActions t a = a -> t Action
type FromActions b a = Parser (StateT Env b) Action a

token' :: forall b t . Monad b => Parser (StateT Env b) t t
token' = modify (field @"pos" %~ (+1)) >> token

is :: (MonadPlus b, Eq t) => t -> Parser (StateT Env b) t t
is t = mfilter (== t) token'

class ActionTransitionSystem (t :: Type -> Type) (b :: Type -> Type) (a :: Type) where
  toActions :: ToActions t a
  fromActions :: FromActions b a

  default toActions :: (Generic a, GToActions t b (Rep a)) => ToActions t a
  toActions = gToActions @t @b . GHC.Generics.from

  default fromActions :: (Monad b, Generic a, GFromActions t b (Rep a)) => FromActions b a
  fromActions = GHC.Generics.to <$> gFromActions @t @b

class GToActions (t :: Type -> Type) (b :: Type -> Type) (f :: Type -> Type) where
  gToActions :: forall a . ToActions t (f a)

class GFromActions (t :: Type -> Type) (b :: Type -> Type) (f :: Type -> Type) where
  gFromActions :: forall a . FromActions b (f a)

instance GToActions t b f => GToActions t b (M1 i c f) where
  gToActions = gToActions @t @b . unM1

-- instance (Monad b, GFromActions t b f) => GFromActions t b (M1 i c f) where
--   gFromActions = M1 <$> gFromActions @t @b

instance (Monad b, GFromActions t b f, Datatype d) => GFromActions t b (D1 d f) where
  gFromActions = do
    modify $ field @"meta" .~ (pure . D . pack . datatypeName @d $ undefined)
    M1 <$> gFromActions @t @b

instance (Monad b, GFromActions t b f, Constructor c) => GFromActions t b (C1 c f) where
  gFromActions = do
    modify $ field @"meta" .~ (pure . D . pack . conName @c $ undefined)
    M1 <$> gFromActions @t @b

instance (Monad b, GFromActions t b f, Selector s) => GFromActions t b (S1 s f) where
  gFromActions = do
    modify $ field @"meta" .~ (pure . D . pack . selName @s $ undefined)
    M1 <$> gFromActions @t @b

instance ActionTransitionSystem t b a => GToActions t b (K1 i a) where
  gToActions = toActions @t @b . unK1

instance (Monad b, ActionTransitionSystem t b a) => GFromActions t b (K1 i a) where
  gFromActions = K1 <$> fromActions @t @b

instance Monoid (t Action) => GToActions t b U1 where
  gToActions _ = mempty

instance Monad b => GFromActions t b U1 where
  gFromActions = pure U1

instance GToActions t b V1 where
  gToActions v = v `seq` error "GFromActions.V1"

instance MonadFail b => GFromActions t b V1 where
  gFromActions = fail "GFromActions.V1"

instance (Semigroup (t Action), GToActions t b f, GToActions t b g) => GToActions t b (f :*: g) where
  gToActions (f :*: g) = gToActions @t @b f <> gToActions @t @b g

instance (Monad b, GFromActions t b f, GFromActions t b g) => GFromActions t b (f :*: g) where
  gFromActions = (:*:) <$> gFromActions @t @b <*> gFromActions @t @b

instance (Applicative t, Semigroup (t Action), GToActions t b f, GToActions t b g) => GToActions t b (f :+: g) where
  gToActions (L1 f) = (pure L) <> gToActions @t @b f
  gToActions (R1 g) = (pure R) <> gToActions @t @b g

instance (MonadPlus b, GFromActions t b f, GFromActions t b g) => GFromActions t b (f :+: g) where
  gFromActions = (is L >> L1 <$> gFromActions @t @b) <|> (is R >> R1 <$> gFromActions @t @b)

instance (Semigroup (t Action), Monad b, ActionTransitionSystem t b a, ActionTransitionSystem t b b') => ActionTransitionSystem t b (a, b')
instance (Semigroup (t Action), Monad b, ActionTransitionSystem t b a, ActionTransitionSystem t b b', ActionTransitionSystem t b c) => ActionTransitionSystem t b (a, b', c)
instance (Semigroup (t Action), Monad b, ActionTransitionSystem t b a, ActionTransitionSystem t b b', ActionTransitionSystem t b c, ActionTransitionSystem t b d) => ActionTransitionSystem t b (a, b', c, d)
instance (Semigroup (t Action), Monad b, ActionTransitionSystem t b a, ActionTransitionSystem t b b', ActionTransitionSystem t b c, ActionTransitionSystem t b d, ActionTransitionSystem t b e) => ActionTransitionSystem t b (a, b', c, d, e)

instance (Applicative t, Monoid (t Action), MonadPlus b, ActionTransitionSystem t b a) => ActionTransitionSystem t b [a]
--   toActions as = pure Grow <> foldMap toActions as <> pure Reduce
--   fromActions = is Grow >> manyTill (fromActions @t) (is Reduce)

instance (Applicative t, Monoid (t Action), MonadPlus b, ActionTransitionSystem t b a) => ActionTransitionSystem t b (Maybe a)
--   toActions Nothing = pure Reduce
--   toActions (Just a) = toActions a
--   fromActions = (is Reduce >> pure Nothing) <|> fromActions @t

instance (Applicative t, Monoid (t Action), MonadPlus b, ActionTransitionSystem t b a, ActionTransitionSystem t b b') => ActionTransitionSystem t b (Either a b')

-- poop :: forall i a . Show i => (i -> Maybe a) -> Parser i a
-- poop f = do
--     i <- ensure
--     case f i of
--       Just a -> advance 1 >> pure a
--       Nothing -> buffer (Just i) >> fail "text"

-- bsss = let f a = case a of
--             SToken s -> Just s
--             _ -> Nothing
--        in poop f 

instance (Applicative t, MonadFail b) => ActionTransitionSystem t b Text where
  toActions = pure . SToken
  fromActions = do
    a <- token'
    case a of
      SToken s -> pure s
      _ -> fail "text"
  -- fromActions = do
  --   a <- ensure
  --   case a of
  --     SToken s -> advance 1 >> pure s
  --     _ -> buffer (Just a) >> fail "text"

instance (Applicative t, MonadFail b) => ActionTransitionSystem t b Int where
  toActions = pure . IToken
  fromActions = do
    a <- token'
    case a of
      IToken i -> pure i
      _ -> fail "int"
  -- fromActions = do
  --   a <- ensure
  --   case a of
  --     IToken i -> advance 1 >> pure i
  --     _ -> buffer (Just a) >> fail "int"

data Stuff = Stuff { anInt :: Int, moreStuff :: [Stuff], maybeFoo :: Maybe Foo }
  deriving (Eq, Show, Generic)

data Foo = Foo { someText :: Text, stuff :: Stuff }
  deriving (Eq, Show, Generic)

data BarBaz = Bar | Baz
  deriving (Eq, Show, Generic)

instance ActionTransitionSystem [] [] Stuff
instance ActionTransitionSystem [] [] Foo
instance ActionTransitionSystem [] Maybe BarBaz

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
parse :: Monad b => ((i -> Parser b i a) -> s -> b (Parser b i a, s)) -> Parser b i a -> s -> b (a, s)
parse next =
  -- let f ip ps = StateT $ \s -> do
  --                 ~(p, s') <- next ip s
  --                 runStateT (ps p) s'
  let f ip ps = StateT (next ip) >>= ps
  in runStateT . iterTM' f

pures :: [FreeF f a (FreeT f m a)] -> [a]
pures [] = []
pures ((Pure a) : xs) = a : pures xs
pures ((Free _) : xs) = pures xs

frees :: [FreeF f a (FreeT f m a)] -> [f (FreeT f m a)]
frees [] = []
frees ((Pure _) : xs) = frees xs
frees ((Free fb) : xs) = fb : frees xs

-- version of iterTM' for batching
batchedIterTM :: forall t b a i . (MonadTrans t, Monad b, Monad (t b)) => ([t b a] -> [i -> Parser b i a] -> ([Parser b i a] -> t b [t b a]) -> t b [t b a]) -> [Parser b i a] -> t b [t b a]
batchedIterTM f ps = do 
  vals <- traverse (lift @t . runFreeT) ps
  let pures' = return <$> pures vals
      frees' = frees vals
  f pures' frees' (batchedIterTM f)

batchedParse :: Monad b => ([i -> Parser b i a] -> [s] -> b ([Parser b i a], [s])) -> [Parser b i a] -> [s] -> b ([(a, [s])], [s])
batchedParse next ps s = do
  let f as ip ps' = StateT $ \s' -> do
                     ~(p, s'') <- next ip s'
                     (x, y) <- runStateT (ps' p) s''
                     pure (as ++ x, y)
  (z, q) <- (runStateT . batchedIterTM f) ps s
  z' <- traverse (\x -> runStateT x s) z
  pure (z', q)

-- parseStream' :: Monad b => (s -> b (i, s)) -> Parser b i a -> s -> b (a, s)
-- -- parseStream' next = runStateT . foo (StateT next >>=)
-- parseStream' next = 
--   let f k = StateT $ \s -> do
--               ~(i, s') <- next s
--               runStateT (k i) s'
--   in runStateT . foo f
--   -- where iterTM :: ((i -> StateT s b a) -> StateT s b a) -> Parser b i a -> StateT s b a

-- parseString' :: MonadPlus b => Parser b t a -> [t] -> b (a, [t])
-- parseString' = parseStream' (maybe empty pure . uncons)

-- | Runs the parser on the supplied input and returns whether or not the parse succeeded.
-- Results are discarded.
-- Parser b i a ~ FreeT ((->) i) b a
check :: forall b i a . MonadPlus b => Parser b i a -> i -> b ()
check p i = do
  val <- runFreeT p
  case val of
    Pure a -> mzero
    Free f -> void . runFreeT $ f i

test :: ([Action], [((Foo, [Action]), Env)], [((), Env)])
test =
  let env = Env Nothing (Pos 0)
      stuff 0 = []
      stuff n = Stuff n [] Nothing : stuff (n - 1)
      foo 0 = Foo "a" $ Stuff 0 [Stuff 2 [] Nothing] Nothing
      foo n = Foo "a" $ Stuff n [Stuff 2 (stuff n) Nothing] (Just $ foo (n - 1))
      challenge = foo 2
      actions = toActions @[] @[] challenge
      parser = fromActions @[] @[]
      result = runStateT (parseString parser actions) env
      -- bar = do
      --   val <- runFreeT parser
      --   case val of
      --     Pure a -> pure (Pure a)
      --     Free f -> undefined
  in (actions, result, runStateT (check parser (IToken 1)) env)

-- test2 :: ([Action], Result Action (Int, Text))
-- test2 =
--   let foo = (1 :: Int, "a" :: Text)
--       actions = toActions foo
--   in (actions, fold (feeds (fromActions @[])) actions)

-- test3 :: ([Action], Result Action [[Int]])
-- test3 =
--   let foo = [[1], []] :: [[Int]]
--       actions = toActions foo
--   in (actions, fold (feeds (fromActions @[])) actions)

-- test4 :: ([Action], Result Action Int)
-- test4 =
--   let foo = 1 :: Int
--       actions = toActions foo
--   in (actions, fold (feeds (fromActions @[])) actions)

-- test5 :: ([Action], Maybe (BarBaz, [Action]))
-- test5 =
--   let baz = Baz
--       actions = toActions @[] @Maybe baz
--   in (actions, parseString (fromActions @[] @Maybe) actions)
