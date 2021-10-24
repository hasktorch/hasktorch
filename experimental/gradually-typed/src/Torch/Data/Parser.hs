{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Torch.Data.Parser where

import Control.Applicative (Alternative (..), liftA2, optional)
import Control.Monad (MonadPlus, mfilter, replicateM)
import Control.Monad.Logic
import Control.Monad.State (MonadState, StateT (..), get, put)
import Control.Monad.Trans (MonadTrans (lift))
import Control.Monad.Trans.Free (FreeF (..), FreeT (..), iterT, iterTM, wrap)
import Data.Char (isAlpha, isAlphaNum, isDigit, isSpace)
import Data.Foldable (asum)
import Data.Functor (($>))
import Data.Kind (Type)
import Data.List
import qualified Text.Parser.Char as Text
import qualified Text.Parser.Combinators as Text
import qualified Text.Parser.Token as Text
import Text.Read (readMaybe)

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

instance (Applicative f, MonadLogic b, MonadPlus b) => MonadLogic (FreeT f b) where
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

-- lnot m = msplit m >>= maybe (pure ()) (const empty)

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

parseStream :: forall s b i a. Monad b => (s -> b (i, s)) -> Parser (StateT s b) i a -> s -> b (a, s)
parseStream next = runStateT . iterT (StateT next >>=)

parseString :: forall b i a. MonadPlus b => Parser (StateT [i] b) i a -> [i] -> b (a, [i])
parseString = parseStream (maybe empty pure . uncons)

-- | @token@ is trivial parser that consumes a single token @i@ and yields it.
--
-- Other parsers can be derived from this one using methods of the
-- 'Functor', 'Applicative', 'Monad', 'Alternative', and 'MonadPlus' typeclasses
-- and the parser combinators in this module.
token :: forall b i. Monad b => Parser b i i
token = wrap $ FreeT . pure . Pure

eof :: Alternative b => Parser (StateT [i] b) i ()
eof = FreeT . StateT $ \s ->
  case s of
    [] -> pure (Pure (), s)
    _ -> empty

-- >>> parseString @[] (sequence [token <* notFollowedBy (is 'a'), token]) "ab"
-- [("ab","")]
-- >>> parseString @[] (sequence [token <* notFollowedBy (is 'a'), token]) "aa"
-- []
-- >>> parseString @[] (notFollowedBy (traverse is "ab")) "a"
-- [((),"a")]
-- >>> parseString @[] (notFollowedBy (traverse is "ab")) "ab"
-- []
notFollowedBy ::
  forall b i a.
  (Alternative b, Foldable b, MonadPlus b) =>
  Parser (StateT [i] b) i a ->
  Parser (StateT [i] b) i ()
notFollowedBy p = FreeT . StateT $ \s ->
  if null (parseString p s)
    then pure (Pure (), s)
    else empty

instance
  (Alternative b, Foldable b, MonadPlus b) =>
  Text.Parsing (FreeT ((->) i) (StateT [i] b))
  where
  try = id
  (<?>) = const
  skipMany p = scan where scan = (p *> scan) <|> pure ()
  skipSome p = p *> Text.skipMany p
  unexpected = const empty
  eof = Torch.Data.Parser.eof
  notFollowedBy = Torch.Data.Parser.notFollowedBy

instance
  (Alternative b, Foldable b, MonadPlus b) =>
  Text.CharParsing (FreeT ((->) Char) (StateT [Char] b))
  where
  satisfy = Torch.Data.Parser.satisfy
  char = isToken
  notChar = isNotToken
  anyChar = token
  string = isString

instance
  (Alternative b, Foldable b, MonadPlus b) =>
  Text.TokenParsing (FreeT ((->) Char) (StateT [Char] b))

-- | @satisfy p@ is a simple parser that consumes a single token @i@ and yields it
-- if and only if @p i@ evaluates to 'True'. Otherwise, the parser fails.
satisfy :: forall b i. MonadPlus b => (i -> Bool) -> Parser b i i
satisfy p = mfilter p token

-- | @is i@ is a simple parser that consumes a single token and yields it
-- if and only if it is equal to @i@. Otherwise, the parser fails.
isToken :: forall b i. (MonadPlus b, Eq i) => i -> Parser b i i
isToken i = Torch.Data.Parser.satisfy (== i)

-- | @isNot i@ is a simple parser that consumes a single token and yields it
-- if and only if it is not equal to @i@. If the token is equal to @i@,
-- the parser fails.
isNotToken :: forall b i. (MonadPlus b, Eq i) => i -> Parser b i i
isNotToken i = Torch.Data.Parser.satisfy (/= i)

-- | Stateful scanner.
--
-- >>> :{
--   f s a | "ell" `isInfixOf` (s ++ [a]) = Nothing
--         | otherwise                    = Just (s ++ [a])
-- :}
--
-- >>> head $ parseString @[] (scan f "" token) "hello 123"
-- ("hel","lo 123")
scan :: (Alternative m, Monad m) => (s -> a -> Maybe s) -> s -> m a -> m [a]
scan f s p = many_p s
  where
    many_p s = many1_p s <|> pure []
    many1_p s = p >>= \a -> maybe empty (fmap (a :) . many_p) (f s a)

-- | @atMost n p@ applies the parser @p@ at most @n@ times and returns
-- every parsing result. If parsing of @p@ succeeds less the @n@ times,
-- @repeatP n p@ succeeds as well.
--
-- >>> head $ parseString @[] (atMost 2 (isToken 'a')) "aaaaaab"
-- ("aa","aaaab")
atMost :: (Alternative m, Monad m) => Int -> m a -> m [a]
atMost n =
  let f s _
        | s >= n = Nothing
        | otherwise = Just (s + 1)
   in scan f 0

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

-- | @isString s@ is a simple parser that consumes 'Char' tokens and yields them
-- if and only if they assemble the 'String' @s@. Otherwise, the parser fails.
isString :: (Traversable t, MonadPlus b, Eq i) => t i -> Parser b i (t i)
isString = traverse isToken

-- | @string@ matches any string
--
-- >>> parseString @[] string "a string"
-- [("a string",""),("a strin","g"),("a stri","ng"),("a str","ing"),("a st","ring"),("a s","tring"),("a ","string"),("a"," string"),("","a string")]
--
-- -- >>> p = string @[] >>= \s -> (guard ("dog" `isInfixOf` s) >> pure s)
-- -- >>> head $ parseString p "this is a string with a dog"
-- -- ("this is a string with a dog","")
-- -- >>> p = string @[] >>= \s -> (guard (not $ "dog" `isInfixOf` s) >> pure s)
-- -- >>> head $ parseString p "this is also string with a dog"
-- -- ("this is also string with a do","g")
string :: MonadPlus b => Parser b i [i]
string = many token

-- string1 :: MonadPlus b => Parser b i [i]
-- string1 = many1 token

-- alphas1 :: MonadPlus b => Parser b Char String
-- alphas1 = many1 (satisfy isAlpha)

-- alphaNums1 :: MonadPlus b => Parser b Char String
-- alphaNums1 = many1 (satisfy isAlphaNum)

intP :: (Text.CharParsing m, Monad m) => m Int
intP = some Text.digit >>= maybe empty pure . readMaybe

doubleP :: (Text.CharParsing m, Monad m) => m Double
doubleP = some (Text.satisfy (not . isSpace)) >>= maybe empty pure . readMaybe
