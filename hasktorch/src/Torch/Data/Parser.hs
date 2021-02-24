{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeApplications #-}

module Torch.Data.Parser where

import Control.Applicative (Alternative (..), liftA2, optional)
import Control.Monad (MonadPlus, mfilter, replicateM)
import Control.Monad.Logic
import Control.Monad.State (MonadState, StateT (..), get, put)
import Control.Monad.Trans (MonadTrans (lift))
import Control.Monad.Trans.Free (FreeF (..), FreeT (..), iterTM)
import Data.Char (isAlpha, isAlphaNum, isDigit, isSpace)
import Data.Foldable (asum)
import Data.Functor (($>))
import Data.Kind (Type)
import Data.List

-- | @Parser b i a@ is a parser that consumes a stream of @i@ tokens and as a
-- result yields a value of type @a@, while operating under the @b@
-- non-determinism monad.
--
-- For most purposes, the non-determinism monad @b@ should be a 'MonadPlus'.
-- Useful examples include @[]@ or @Logic@ if you want backtracking,
-- and 'Maybe' if you want no backtracking.
--
-- Use @'StateT' s []@ if you want to maintain a state @s@ that is
-- automatically reverted when backtracking via @[]@.
--
-- 'hoistFreeT' can be used to change the backtracking monad.
--
-- 'FreeT' provides instances for 'Functor', 'Applicative', 'Monad',
-- 'Alternative' and 'MonadPlus'.
type Parser
  (b :: Type -> Type)
  (i :: Type)
  (a :: Type) =
  FreeT ((->) i) b a

instance (Applicative f, MonadLogic b) => MonadLogic (FreeT f b) where
  -- msplit :: FreeT f b a -> FreeT f b (Maybe (a, FreeT f b a))
  msplit (FreeT b) = FreeT $ go b []
    where
      go b ws = do
        r <- msplit b
        case r of
          Nothing -> pure $ case ws of
            [] -> Pure Nothing
            (w : ws) ->
              let go' fas [] = fas
                  go' fas (w : ws) = go' (liftA2 (:) w fas) ws
               in Free $ fmap (msplit . asum) (go' (fmap pure w) ws)
          Just (val, b') ->
            case val of
              Pure a -> pure . Pure $ Just (a, FreeT b')
              Free w -> go b' (w : ws)

  ifte t th el = msplit t >>= maybe el (\(a, m) -> th a <|> (m >>= th))

  once m = do
    (a, _) <- maybe empty pure =<< msplit m
    pure a

  lnot m = ifte (once m) (const empty) (pure ())

-- notFollowedBy p = msplit p >>= maybe (pure ()) (const empty)

-- | Recurse over a parser.
--
-- Tears down the free monad transformer over the '(->) i' functor using iteration:
-- @
-- recurse next parser = next parser (\parser' -> next parser' (\parser'' -> next parser'' (\parser''' -> next parser''' (...))))
-- @
recurse ::
  forall t b i a.
  (Parser b i a -> (Parser b i a -> t b a) -> t b a) ->
  Parser b i a ->
  t b a
recurse next parser =
  let cont = recurse next
   in next parser cont

parseStream :: forall s b i a. Monad b => (s -> b (i, s)) -> Parser b i a -> s -> b (a, s)
parseStream next = runStateT . iterTM (StateT next >>=)

parseString :: forall b i a. MonadPlus b => Parser b i a -> [i] -> b (a, [i])
parseString = parseStream (maybe empty pure . uncons)

-- | @token@ is trivial parser that consumes a single token @i@ and yields it.
--
-- Other parsers can be derived from this one using methods of the
-- 'Functor', 'Applicative', 'Monad', 'Alternative', and 'MonadPlus' typeclasses
-- and the parser combinators in this module.
token :: forall b i. Applicative b => Parser b i i
token = FreeT . pure . Free $ FreeT . pure . Pure

-- | @satisfy p@ is a simple parser that consumes a single token @i@ and yields it
-- if and only if @p i@ evaluates to 'True'. Otherwise, the parser fails.
satisfy :: forall b i. MonadPlus b => (i -> Bool) -> Parser b i i
satisfy p = mfilter p token

-- | @is i@ is a simple parser that consumes a single token and yields it
-- if and only if it is equal to @i@. Otherwise, the parser fails.
is :: forall b i. (MonadPlus b, Eq i) => i -> Parser b i i
is i = satisfy (== i)

-- | @isNot i@ is a simple parser that consumes a single token and yields it
-- if and only if it is not equal to @i@. If the token is equal to @i@,
-- the parser fails.
isNot :: forall b i. (MonadPlus b, Eq i) => i -> Parser b i i
isNot i = satisfy (/= i)

-- | @choice ps@ tries to apply the parsers in the list @ps@ in order,
-- until one of them succeeds. Returns the value of the succeeding
-- parser.
choice :: (Foldable t, Alternative f) => t (f a) -> f a
choice = asum

-- | @option a p@ tries to apply parser @p@. If the parser @p@ fails,
-- it returns the value @a@, otherwise the value returned by the parser @p@.
option :: Alternative f => a -> f a -> f a
option a p = p <|> pure a

-- | @many1 p@ applies the parser @p@ /one/ or more times. Returns a
-- list of the returned values of @p@.
many1 :: Alternative f => f a -> f [a]
many1 p = liftA2 (:) p (many p)
{-# INLINE many1 #-}

-- | @manyTill p end@ applies the parser @p@ /zero/ or more times until
-- the parser @end@ succeeds, and returns the list of values returned by
-- @p@. The result of @end@ is discarded.
--
-- Note that this can be inefficient if the parsers @p@ and @end@ overlap,
-- as it can lead to a lot of backtracking.
manyTill :: Alternative f => f a -> f b -> f [a]
manyTill p end = scan where scan = (end $> []) <|> liftA2 (:) p scan

-- | @manyTill p end@ applies the parser @p@ /one/ or more times until
-- the parser @end@ succeeds, and returns the list of values returned by
-- @p@. The result of @end@ is discarded.
--
-- Note that this can be inefficient if the parsers @p@ and @end@ overlap,
-- as it can lead to a lot of backtracking.
many1Till :: Alternative f => f a -> f b -> f [a]
many1Till p end = liftA2 (:) p (manyTill p end)

-- | Stateful scanner.
--
-- >>> :{
--   f s a | "ell" `isInfixOf` (s ++ [a]) = Nothing
--         | otherwise                    = Just (s ++ [a])
-- :}
-- >>> head $ parseString @[] (scan f "" token) "hello 123"
scan :: (Alternative m, Monad m) => (s -> a -> Maybe s) -> s -> m a -> m [a]
scan f s p = many_p s
  where
    many_p s = many1_p s <|> pure []
    many1_p s = p >>= \a -> maybe empty (fmap (a :) . many_p) (f s a)

-- | @repeatP n p@ applies the parser @p@ @n@ times and returns
-- every parsing result. If parsing of @p@ succeeds less the @n@ times,
-- @repeatP n p@ fails.
repeatP :: Monad m => Int -> m a -> m [a]
repeatP = replicateM

-- | @atMost n p@ applies the parser @p@ at most @n@ times and returns
-- every parsing result. If parsing of @p@ succeeds less the @n@ times,
-- @repeatP n p@ succeeds as well.
--
-- >>> head $ parseString @[] (atMost 2 (is 'a')) "aaaaaab"
-- ("aa","aaaab")
atMost :: (Alternative m, Monad m) => Int -> m a -> m [a]
atMost n =
  let f s _
        | s >= n = Nothing
        | otherwise = Just (s + 1)
   in scan f 0

-- | @skipMany p@ skips /zero/ or more instances of the parser @p@.
-- The parsing results are discarded.
skipMany :: Alternative f => f a -> f ()
skipMany p = scan where scan = (p *> scan) <|> pure ()

-- | @skipMany1 p@ skips /one/ or more instances of the parser @p@.
-- The parsing results are discarded.
skipMany1 :: Alternative f => f a -> f ()
skipMany1 p = p *> skipMany p

-- | @sepBy p sep@ applies /zero/ or more occurrences of the parser @p@,
-- separated by @sep@. Returns a list of the values returned by @p@ and
-- discards the results of @sep@.
sepBy :: Alternative f => f a -> f sep -> f [a]
sepBy p sep = (p `sepBy1` sep) <|> pure []

-- | @sepBy1 p sep@ applies /one/ or more occurrences of the parser @p@,
-- separated by @sep@. Returns a list of the values returned by @p@ and
-- discards the results of @sep@.
sepBy1 :: Alternative f => f a -> f sep -> f [a]
sepBy1 p sep = (:) <$> p <*> many (sep *> p)

-- | @maybeP p@ applies the parser @p@ optionally and returns the result
-- wrapped in @Maybe@.
maybeP :: Alternative f => f a -> f (Maybe a)
maybeP = optional

-- | @eitherP p p'@ combines the two alternatives @p@ and @p'@.
eitherP :: Alternative f => f a -> f b -> f (Either a b)
eitherP p p' = (Left <$> p) <|> (Right <$> p')

-- | @void p@ applies the parser @p@ and discards its result.
void :: Functor f => f a -> f ()
void p = p $> ()

-- | @combine p p'@ merges the results of @p@ and @p'@ using the 'Semigroup' instance.
combine :: (Applicative f, Semigroup a) => f a -> f a -> f a
combine = liftA2 (<>)

-- | @combines ps@ merges the results of the parsers @ps@ using the 'Monoid' instance.
combines :: (Applicative f, Monoid a) => [f a] -> f a
combines = foldl combine (pure mempty)

-- | @between open close p@ applies the parsers @open@, @p@, and @close@
-- in that order. Only the result of @p@ is returned, the results of @open@
-- and @close@ are discarded.
--
-- This combinator is useful for parsing expressions wrapped in parentheses,
-- for example.
between :: Applicative f => f a1 -> f a2 -> f a -> f a
between open close p = open *> p <* close

-- | @isString s@ is a simple parser that consumes 'Char' tokens and yields them
-- if and only if they assemble the 'String' @s@. Otherwise, the parser fails.
isString :: (Traversable t, MonadPlus b, Eq i) => t i -> Parser b i (t i)
isString = traverse is

-- | @string@ matches any string
--
-- >>> parseString @[] string "a string"
-- [("a string",""),("a strin","g"),("a stri","ng"),("a str","ing"),("a st","ring"),("a s","tring"),("a ","string"),("a"," string"),("","a string")]
-- >>> p = string @[] >>= \s -> (guard ("dog" `isInfixOf` s) >> pure s)
-- >>> head $ parseString p "this is a string with a dog"
-- ("this is a string with a dog","")
-- >>> p = string @[] >>= \s -> (guard (not $ "dog" `isInfixOf` s) >> pure s)
-- >>> head $ parseString p "this is also string with a dog"
-- ("this is also string with a do","g")
string :: MonadPlus b => Parser b i [i]
string = many token

string1 :: MonadPlus b => Parser b i [i]
string1 = many1 token

space :: MonadPlus b => Parser b Char String
space = many (satisfy isSpace)

space1 :: MonadPlus b => Parser b Char String
space1 = many1 (satisfy isSpace)

alpha1 :: MonadPlus b => Parser b Char String
alpha1 = many1 (satisfy isAlpha)

alphaNum1 :: MonadPlus b => Parser b Char String
alphaNum1 = many1 (satisfy isAlphaNum)

digits1 :: MonadPlus b => Parser b Char String
digits1 = many1 (satisfy isDigit)
