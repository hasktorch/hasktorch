{-# LANGUAGE GADTs #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE LambdaCase #-}
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
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Data.ActionTransitionSystem where

import Prelude hiding (lookup)

import GHC.Generics
import GHC.IO (unsafePerformIO)
import GHC.TypeLits
import qualified GHC.Exts as Exts

import Bound (fromScope, toScope, instantiate1, abstract1, (>>>=), Scope, Var)

import Control.Lens
import Control.Foldl (Fold(..), fold, head)
import qualified Control.Foldl as L
import Control.Applicative (liftA2, pure, Alternative(..), empty, (<|>))
import Control.Monad.Fresh
import Control.Monad.Yoctoparsec (Parser)
import Control.Monad.Trans.Maybe (MaybeT(..))
import Control.Monad.Trans.RWS (RWST)
import Control.Monad.Trans.Free (wrap, iterTM, runFreeT, Free, FreeT(..), FreeF(..))
import Control.Monad.State (evalStateT, StateT (..), runStateT, get, put, modify)
import Control.Monad (guard, join, ap, void, mfilter, MonadPlus(..))
import Control.Monad.Trans (MonadTrans(lift))
import Control.Monad.Reader (ask, local, runReaderT, ReaderT)

import Data.Generics.Product
import Data.Generics.Sum
import Data.Functor.Classes (Eq1(..), Ord1(..), Show1(..), eq1, compare1, showsPrec1)
import Data.Deriving (deriveEq1, deriveOrd1, deriveShow1)
import Data.Kind (Type)
import Data.Text (pack, Text)
import Data.Maybe (isJust, fromJust, fromMaybe)
import Data.List as List (take, iterate, isInfixOf, replicate, length, filter, sort, nub)
import Data.Map as Map (delete, elems, toList, (!), adjust, update, keys, null, insertWith, singleton, fromList, unionWith, Map, insert, lookup)
import Data.Set as Set (elems, filter, cartesianProduct, unions, toList, fromList, member, singleton, union, Set, insert, findIndex)
import qualified Data.Set.Ordered as OSet
import qualified Data.Set as Set (empty)
import qualified Data.Map as Map (empty)

import Hedgehog (GenT, distributeT, discover, checkParallel, PropertyT, check, Property, (===), forAll, property, Gen, MonadGen(..))
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

import Torch.Typed
import Torch (asValue, ATenTensor, toType, toDense, asTensor', withDevice, withDType, defaultOpts, sparseCooTensor)
import qualified Torch (ones)
import Torch.Distributions.Bernoulli (fromProbs)
import Torch.Distributions.Distribution (sample)
import Pipes (enumerate, yield, (>->), each, for, ListT(Select), runEffect, Effect)
import qualified Pipes.Safe as Safe
import Pipes (MonadIO(liftIO))
import Pipes.Prelude (take, repeatM, drain)
import Pipes.Prelude (foldM)
import Control.Exception (tryJust, Exception, try)
import System.IO.Error (ioeGetErrorString)
import qualified Hedgehog.Internal.Seed as Seed
import qualified Hedgehog.Internal.Tree as Tree
import Hedgehog.Internal.Gen (evalGen)
import Torch.Internal.Class (Castable)
import Torch.Vision (grayScale10)
import Text.Printf (printf)
import Torch.Data.StreamedPipeline (makeListT', Datastream(..))
import Control.Monad.Trans.Cont (ContT(runContT))
import Control.Concurrent (threadDelay)
import System.Mem (performGC)
import Data.Text.Prettyprint.Doc (vsep, hcat, hsep, (<+>), sep, pretty, parens, Doc)
import Data.Text.Prettyprint.Doc.Render.Terminal
import Data.Word (Word64)

-- https://stackoverflow.com/questions/17675054/deserialising-with-ghc-generics?rq=1
-- http://hackage.haskell.org/package/cereal-0.5.8.1/docs/Data-Serialize.html
-- https://hackage.haskell.org/package/attoparsec-0.13.2.4/docs/src/Data.Attoparsec.Text.Internal.html
-- https://hackage.haskell.org/package/yoctoparsec-0.1.0.0/docs/src/Control-Monad-Yoctoparsec.html#Parser
-- https://vaibhavsagar.com/blog/2018/02/04/revisiting-monadic-parsing-haskell/
-- https://github.com/alphaHeavy/protobuf/blob/46cda829cf1e7b6bba2ff450e267fb1a9ace4fb3/src/Data/ProtocolBuffers/Ppr.hs

data Env t = Env
  { pos :: Pos -- state
  , tEnv :: TEnv t
  , mEnv :: MEnv
  , rEnv :: REnv
  , aEnv :: AEnv
  -- , validActionsMask :: Map Pos (Set Action)
  } deriving (Eq, Ord, Show, Generic)

defaultEnv :: Env t
defaultEnv = Env
  { pos = Pos 0
  , tEnv = TEnv
    { tokens = mempty
    }
  , mEnv = MEnv
    { meta = Nothing
    , metas = mempty
    }
  , rEnv = REnv
    { parentPos = Nothing
    , parents = mempty
    , relations = mempty
    }
  , aEnv = AEnv
    { currentScope = Nothing
    , knownScopes = mempty
    , attentionMask = mempty
    , keyPaddingMask = mempty
    }
  -- , validActionsMask = mempty
  }

newtype Pos = Pos { unPos :: Int }
  deriving (Eq, Ord, Show, Num, Enum, Generic)

data TEnv t = TEnv
  { tokens :: Map Pos t
  } deriving (Eq, Ord, Show, Generic)

data MEnv = MEnv
  { meta :: Maybe M -- state
  , metas :: Map Pos M -- writer
  } deriving (Eq, Ord, Show, Generic)

data Relation =
    ChildParentRelation
  | ParentChildRelation
  | SiblingDistRelation { siblingDist :: Int }
  deriving (Eq, Ord, Show, Generic)

data REnv = REnv
  { parentPos :: Maybe Pos -- state
  , parents :: Map Pos (Set Pos) -- writer
  , relations :: Map (Pos, Pos) (Set Relation) -- writer
  } deriving (Eq, Ord, Show, Generic)

type ScopeId = Text

data AttentionScope = AttentionScope
  { scopeKind :: AttentionKind
  , scopeConnections :: Set ScopeId
  , scopePositions :: Set Pos
  } deriving (Eq, Ord, Show, Generic)

data AttentionKind = 
    BidirectionalAttention
  | BackwardAttention
  | ForwardAttention
    deriving (Eq, Ord, Show, Generic)

data AEnv = AEnv
  { currentScope :: Maybe ScopeId -- state
  , knownScopes :: Map ScopeId AttentionScope -- writer
  , attentionMask :: Set (Pos, Pos) -- writer
  , keyPaddingMask :: Set Pos -- writer
  } deriving (Eq, Ord, Show, Generic)

data Token a = Pad | Unk | Mask | Token a
  deriving (Eq, Ord, Show, Generic, Functor)

data M = D Text | C Text | S Text
  deriving (Eq, Ord, Show, Generic)

data Action =
    L
  | R
  | Grow
  | Reduce
  | IToken Int
  | SToken Text
  | BToken Bool
  | CToken Char
    deriving (Eq, Ord, Show)

type To t a = a -> t Action
type From b a = Parser (StateT (Env Action) b) Action a

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

ancestralRelations :: forall f . Monad f => Pos -> StateT REnv f ()
ancestralRelations pos = get >>= (go . view (field @"parentPos"))
 where go Nothing          = pure ()
       go (Just parentPos) = 
         let rels' = update (Map.singleton (pos, parentPos) . Set.singleton $ ChildParentRelation)
                            (Map.singleton (parentPos, pos) . Set.singleton $ ParentChildRelation)
         in  modify (field @"relations" %~ (update rels'))
       update rels' rels = Map.unionWith Set.union rels' rels

siblingRelations :: forall f . Monad f => Pos -> StateT REnv f ()
siblingRelations pos = get >>= ap (go . view (field @"parentPos")) (view (field @"parents"))
  where go Nothing          parents = pure ()
        go (Just parentPos) parents = do
          let siblings = maybe mempty (Set.insert pos) $ lookup parentPos parents
              sibIndex = findIndex pos siblings
              step pos' (rels', idx) =
                let rels'' = update (Map.singleton (pos, pos') . Set.singleton . SiblingDistRelation $ sibIndex - idx)
                                    (Map.singleton (pos', pos) . Set.singleton . SiblingDistRelation $ idx - sibIndex)
                in  (update rels'' rels', idx + 1)
              (rels, _) = foldr step (mempty, 0) siblings
          modify (field @"relations" %~ (update rels))
          modify (field @"parents" %~ (Map.insert parentPos siblings))
        update rels' rels = Map.unionWith Set.union rels' rels

updateRelations :: forall f . Monad f => Pos -> StateT REnv f ()
updateRelations = ancestralRelations @f >> siblingRelations @f

updateAttention :: forall f. Monad f => Pos -> StateT AEnv f ()
updateAttention pos = do
  mScopeId <- (^. field @"currentScope") <$> get
  knownScopes <- (^. field @"knownScopes") <$> get
  case mScopeId of
    Just scopeId -> do
      modify (field @"attentionMask" %~ go pos scopeId knownScopes)
      modify (field @"knownScopes" %~ go' pos scopeId)
    Nothing -> pure ()
  modify (field @"keyPaddingMask" %~ Set.insert pos)
  where
    go pos thisScopeId knownScopes mask =
      let constrainAttention BidirectionalAttention = id
          constrainAttention BackwardAttention = Set.filter (uncurry (>=))
          constrainAttention ForwardAttention = Set.filter (uncurry (<=))
          mkMask kind from to = constrainAttention kind $ Set.cartesianProduct from to
          thisScope = knownScopes ! thisScopeId
          attendSelf = Set.singleton pos
          attendTo = Set.unions $ scopePositions . (knownScopes !) <$> Set.toList (scopeConnections $ thisScope)
          attendFrom = List.filter (member thisScopeId . scopeConnections) $ Map.elems knownScopes
          mask' =
            Set.unions
              [ mkMask (scopeKind thisScope) attendSelf attendTo
              , Set.unions $ (\thatScope -> mkMask (scopeKind thatScope) (scopePositions thatScope) attendSelf) <$> attendFrom
              , mkMask (scopeKind thisScope) attendSelf attendSelf
              ]
       in Set.union mask' mask
    go' pos = Map.adjust (field @"scopePositions" %~ (Set.insert pos))

updateMeta :: forall f . Monad f => Pos -> StateT MEnv f ()
updateMeta pos = do
  mMeta <- (^. field @"meta") <$> get
  case mMeta of
    Just meta -> modify (field @"metas" %~ Map.insert pos meta)
    Nothing -> pure ()

updateTokens :: forall f t . Monad f => t -> Pos -> StateT (TEnv t) f ()
updateTokens t pos =
  modify (field @"tokens" %~ Map.insert pos t)

-- TODO: move writes somewhere else
token :: forall b t . Monad b => Parser (StateT (Env t) b) t t
token = do
  t <- wrap $ FreeT . pure . Pure
  pos <- (^. field @"pos") <$> get
  zoom (field @"tEnv") . lift . updateTokens t $ pos
  zoom (field @"mEnv") . lift . updateMeta $ pos
  zoom (field @"rEnv") . lift . updateRelations $ pos
  zoom (field @"aEnv") . lift . updateAttention $ pos
  modify (field @"pos" %~ (+1))
  pure t

is :: (MonadPlus b, Eq t) => t -> Parser (StateT (Env t) b) t t
is t = mfilter (== t) token

isNot :: (MonadPlus b, Eq t) => t -> Parser (StateT (Env t) b) t t
isNot t = mfilter (/= t) token

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
    modify $ field @"mEnv" . field @"meta" .~ (pure . D . pack . datatypeName @d $ undefined)
    pos <- (^. field @"pos") <$> get
    modify $ field @"rEnv" . field @"parentPos" .~ (pure pos)
    M1 <$> gFromActions @b

instance (Monad b, GFromActions b f, Constructor c) => GFromActions b (C1 c f) where
  gFromActions = do
    modify $ field @"mEnv" . field @"meta" .~ (pure . C . pack . conName @c $ undefined)
    M1 <$> gFromActions @b

instance (Monad b, GFromActions b f, Selector s) => GFromActions b (S1 s f) where
  gFromActions = do
    modify $ field @"mEnv" . field @"meta" .~ (pure . S . pack . selName @s $ undefined)
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

instance (Applicative t, Alternative t, ToActions t a) => ToActions t [a] where
  toActions as = pure Grow <|> go as
    where go [] = pure Reduce
          go (a : as) = toActions a <|> go as
instance (MonadPlus b, FromActions b a) => FromActions b [a] where
  fromActions = is Grow >> manyTill (fromActions @b) (is Reduce)

instance (Applicative t, Alternative t, ToActions t a) => ToActions t (Maybe a) where
  toActions ma = pure Grow <|> go ma <|> pure Reduce
    where go Nothing = empty
          go (Just a) = toActions a
instance (MonadPlus b, FromActions b a) => FromActions b (Maybe a) where
  fromActions = is Grow >> option Nothing (Just <$> fromActions) >>= (is Reduce >>) . pure

instance (Applicative t, Alternative t, ToActions t a, ToActions t b) => ToActions t (Either a b)
instance (MonadPlus b, FromActions b a, FromActions b b') => FromActions b (Either a b')

instance Applicative t => ToActions t Text where
  toActions = pure . SToken

instance MonadFail b => FromActions b Text where
  fromActions = token >>= (\case SToken s -> pure s; _ -> fail "text")

instance Applicative t => ToActions t Char where
  toActions = pure . CToken

instance MonadFail b => FromActions b Char where
  fromActions = token >>= (\case CToken c -> pure c; _ -> fail "char")

instance Applicative t => ToActions t Int where
  toActions = pure . IToken

instance MonadFail b => FromActions b Int where
  fromActions = token >>= (\case IToken i -> pure i; _ -> fail "int")

instance Applicative t => ToActions t Bool where
  toActions = pure . BToken

instance MonadFail b => FromActions b Bool where
  fromActions = token >>= (\case BToken b -> pure b; _ -> fail "bool")

instance Alternative t => ToActions t () where
  toActions = const empty

instance Monad b => FromActions b () where
  fromActions = pure ()

-- | Runs the parser on the supplied input and returns whether or not the parse succeeded.
-- Results are discarded.
-- TODO: this isn't nice yet. It would be great if there was a stronger signal for failure than just 'mzero'.
-- Parser b i a ~ FreeT ((->) i) b a
checkParser :: forall b i a . MonadPlus b => Parser b i a -> i -> b ()
checkParser p i = do
  val <- runFreeT p
  case val of
    Pure a -> mzero
    Free f -> void . runFreeT $ f i

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
-- fresh values when backtracking: https://hackage.haskell.org/package/monad-gen-0.1.0.0/docs/Control-Monad-Gen.html
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


data Stuff = SomeStuff { anInt :: Int, aBool :: Bool, moreStuff :: [Stuff], maybeFoo :: Maybe Foo }
          --  | NoStuff
  deriving (Eq, Show, Generic)

instance ToActions [] Stuff
instance FromActions [] Stuff

data Foo = SomeFoo { someText :: Text, stuff :: Stuff }
        --  | NoFoo
  deriving (Eq, Show, Generic)

instance ToActions [] Foo
instance FromActions [] Foo

test :: ([Action], [((Foo, [Action]), Env Action)], [((), Env Action)])
test =
  let env = defaultEnv
      stuff 0 = []
      stuff n = SomeStuff n True [] Nothing : stuff (n - 1)
      foo 0 = SomeFoo "a" $ SomeStuff 0 False [SomeStuff 2 True [] Nothing] Nothing
      foo n = SomeFoo "a" $ SomeStuff n ((==0) . (`rem` 3) $ n) [SomeStuff 2 False (stuff n) Nothing] (Just $ foo (n - 1))
      challenge = foo 2
      actions = toActions @[] challenge
      parser = fromActions @[]
      result' = let f ap [] = empty
                    f ap (a : as) = let p = ap a in do
                      env' <- get
                      pure $ unsafePerformIO $ print
                        ( a
                        , view (field @"mEnv" . field @"meta") env'
                        , view (field @"pos") env'
                        , view (field @"rEnv" . field @"parentPos") env'
                        -- , view (field @"rEnv" . field @"relations") env'
                        ) >> pure (p, as)
                in runStateT (parse f parser actions) env
  in (actions, result', runStateT (checkParser parser (IToken 1)) env)

------------------------------------------------------------------------
-- A simply-typed lambda calculus with ints, bools, and strings
-- from https://github.com/hedgehogqa/haskell-hedgehog/blob/master/hedgehog-example/src/Test/Example/STLC.hs

-- input: randomly generated lambda calculus terms
-- output: reduced lambda calculus terms
-- accuracy based on exact match
-- variables need to be anonymized, use relations to indicate identical variables
-- constrain the production using `guard` on typecheck
-- how can I typecheck during parsing?
-- backtracking of FreeT ((->) i) [] a will solve it automatically

data Ty =
    Arr Ty Ty
  | Nat
  deriving (Eq, Ord, Show, Generic)

instance ToActions [] Ty
instance FromActions [] Ty

-- | Lambda terms.
-- TODO: use De Bruijn indices https://en.wikipedia.org/wiki/De_Bruijn_index
-- https://stackoverflow.com/questions/28931477/haskell-convert-de-bruijn-terms-to-lambda-terms-and-vice-versa
-- https://hackage.haskell.org/package/lambda-sampler-1.1/docs/Data-Lambda.html
-- https://mroman42.github.io/mikrokosmos/haddock/Stlc-Types.html
-- https://www.schoolofhaskell.com/user/edwardk/bound
-- https://github.com/ekmett/bound/blob/master/examples/Simple.hs
-- http://hackage.haskell.org/package/bound-gen
-- http://hackage.haskell.org/package/bound-extras
-- https://github.com/robrix/path/blob/master/src/Path/Term.hs
-- https://en.wikipedia.org/wiki/Programming_Computable_Functions
-- https://github.com/jozefg/pcf/blob/master/src/Language/Pcf.hs
-- https://jozefg.bitbucket.io/posts/2014-12-17-variables.html
-- http://blog.ielliott.io/topsy-turvy-reverse-state/
-- http://www.a-rx.info/pire/ref/src/Syntax.html
data Exp a =
    Var a -- ^ Variable.
  | Lam Ty (Scope () Exp a) -- ^ Lambda abstraction.
  | Exp a :@ Exp a -- ^ Term application.
  | Suc (Exp a)
  | Zero
  deriving (Functor, Foldable, Traversable, Generic)

infixl 9 :@

-- TODO: test using https://github.com/hedgehogqa/haskell-hedgehog-classes
instance Applicative Exp where
  pure = Var
  (<*>) = ap

-- TODO: test using https://github.com/hedgehogqa/haskell-hedgehog-classes
instance Monad Exp where
  return = Var
  Var a >>= f = f a
  (x :@ y) >>= f = (x >>= f) :@ (y >>= f)
  Lam ty e >>= f = Lam ty (e >>>= f)
  Suc e >>= f = Suc (e >>= f)
  Zero >>= _ = Zero

deriveEq1 ''Exp
deriveOrd1 ''Exp
deriveShow1 ''Exp

-- TODO: test using https://github.com/hedgehogqa/haskell-hedgehog-classes
instance Eq a => Eq (Exp a) where (==) = eq1
instance Ord a => Ord (Exp a) where compare = compare1
instance Show a => Show (Exp a) where showsPrec = showsPrec1

instance ToActions [] a => ToActions [] (Var () (Exp a))
instance ToActions [] a => ToActions [] (Scope () Exp a)
instance ToActions [] a => ToActions [] (Exp a)

instance FromActions [] a => FromActions [] (Var () (Exp a))
instance FromActions [] a => FromActions [] (Scope () Exp a)
instance FromActions [] a => FromActions [] (Exp a)

lam :: forall a . Eq a => Ty -> a -> Exp a -> Exp a
lam ty uname bind = Lam ty (abstract1 uname bind)

-- | Smart constructor that converts the given positive integer to a corresponding Nat.
nat :: forall a n . (Num n, Eq n) => n -> Exp a
nat 0 = Zero
nat n = Suc $ nat (n-1)

-- | Compute the normal form of an expression.
nf :: forall a . Exp a -> Exp a
nf e@Var{} = e
nf (Lam ty b) = Lam ty (toScope . nf . fromScope $ b)
nf (f :@ a) = case whnf f of
  Lam ty b -> nf (instantiate1 a b)
  f' -> nf f' :@ nf a
nf (Suc e) = Suc (nf e)
nf e@Zero = e

-- | Reduce a term to weak head normal form.
whnf :: forall a . Exp a -> Exp a
whnf e@Var{} = e
whnf e@Lam{} = e
whnf (f :@ a) = case whnf f of
  Lam _ b -> whnf (instantiate1 a b)
  f' -> f' :@ a
whnf e@Suc{} = e
whnf e@Zero = e

type TyM a = MaybeT (Fresh a)

assertTy :: Ord a => Map a Ty -> Exp a -> Ty -> TyM a ()
assertTy env e t = (== t) <$> typeCheck env e >>= guard

typeCheck :: Ord a => Map a Ty -> Exp a -> TyM a Ty
typeCheck _ Zero = return Nat
typeCheck env (Suc e) = assertTy env e Nat >> return Nat
typeCheck env (Var a) = MaybeT . return $ Map.lookup a env
typeCheck env (f :@ a) = typeCheck env f >>= \case
  Arr fTy tTy -> assertTy env a fTy >> return tTy
  _ -> mzero
typeCheck env (Lam ty bind) = do
  uname <- fresh
  Arr ty <$> typeCheck (Map.insert uname ty env) (instantiate1 (Var uname) bind)

type TyP a = Fresh a (Doc AnsiStyle)

class Pretty a p where
  pprint :: a -> p -> Doc AnsiStyle
  ppr :: Int -> p -> TyP a

  default pprint :: (Enum a) => a -> p -> Doc AnsiStyle
  pprint a p = runFreshFrom @a a $ ppr 0 p

instance Pretty Int Int where
  ppr _ = pure . pretty

instance Pretty Char Char where
  ppr _ = pure . pretty

-- instance Pretty a String where
--   ppr _ = pure . pretty

-- instance Pretty a (Doc AnsiStyle) where
--   ppr _ = pure . id

instance (Enum a, Ord a, Pretty a a) => Pretty a (Exp a) where
  ppr p Zero = parensIf (p > 0) (pure . pretty $ 'Z')
  ppr p (Suc e) = parensIf (p > 0) $ do
    ps <- pure . pretty $ 'S'
    pe <- ppr (p + 1) e
    pure $ ps <> pe
  ppr p (Var uname) = ppr p uname
  ppr p e@(_ :@ _) =
    let (f, xs) = viewApp e
    in parensIf (p > 0) $ do
        pf <- ppr (p + 1) f
        args <- sep <$> (traverse (ppr (p + 1)) xs)
        pure $ pf <+> args
  ppr p e@(Lam _ty bind) = do
    (vars, body) <- viewBody e
    pb <- ppr 0 body
    pl <- pure . pretty $ 'λ'
    pd <- pure . pretty $ '.'
    pvs <- traverse (ppr 0) $ reverse vars
    let pvs' = hcat $ (\pv -> pl <> pv <> pd) <$> pvs
    parensIf (p > 0) $ do
      pure $ pvs' <+> pb

parensIf :: Bool -> TyP a -> TyP a
parensIf True = fmap parens
parensIf False = id

viewApp :: Exp a -> (Exp a, [Exp a])
viewApp (e1 :@ e2) = go e1 [e2]
  where
    go (a :@ b) xs = go a (b : xs)
    go f xs = (f, xs)

viewBody :: Ord a => Exp a -> Fresh a ([a], Exp a)
viewBody e = go [] e
  where
    go env (Lam _ bind) = do
      uname <- fresh
      go (uname : env) (instantiate1 (Var uname) bind)
    go env x = pure (env, x)

render :: forall a . (Ord a, Enum a, Pretty a a) => Exp a -> IO ()
render = render' (toEnum 0)

render' :: forall a . (Ord a, Enum a, Pretty a a) => a -> Exp a -> IO ()
render' a e = putDoc $ pprint a e

testBound :: IO ()
testBound = do
  let term :: Exp Char = (lam Nat 'a' (Var 'a')) :@ Zero
  print term -- Lam Nat (Scope (Var (B ()))) :@ Zero
  render' 'x' term -- (λx. x) Z
  print $ runFresh . runMaybeT . typeCheck Map.empty $ term -- Just Nat
  print $ toActions @[] $ term -- [R,L,L,R,R,L,L,L,R,R,R]
  print $ whnf term -- Zero
  print $ nf term -- Zero
  print $ toActions @[] $ nf term -- [R,R,R]

-- | Monad transformer stack for term and type generation.
-- Notably, contains the @FreshT@ transformer for generating fresh variable names
-- and a @ReaderT@ for the environment of scoped typed @Var@s.
type GTyM a = ReaderT (Map Ty [Exp a]) (FreshT a Gen)

-- | Generate a type.
-- We cannot generate an expression without generating a type for it first.
genTy :: forall m . MonadGen m => m Ty
genTy =
  Gen.recursive Gen.choice [
      -- non-recursive generators
      pure Nat
    ] [
      -- recursive generators
      Arr <$> genTy <*> genTy
    ]

-- | Finalize generation by running the monad transformers for the environment
-- and the fresh variable name computation.
genWellTypedExp :: forall a . (Eq a, Enum a) => Ty -> Gen (Exp a)
genWellTypedExp ty = runFreshT $ runReaderT (genWellTypedExp' ty) mempty

-- | Main recursive mechanism for genersating expressions for a given type.
genWellTypedExp' :: forall a . Eq a => Ty -> GTyM a (Exp a)
genWellTypedExp' ty =
  Gen.shrink shrinkExp $
  genWellTypedPath ty <|> Gen.recursive Gen.choice [
      -- non-recursive generators
      genWellTypedExp'' ty
    ] [
      -- recursive generators
      genWellTypedApp ty
    , genWellTypedExp''' ty
    ]

shrinkExp :: forall a . Exp a -> [Exp a]
shrinkExp (f :@ a) = case whnf f of
  Lam _ b -> [whnf (instantiate1 a b)]
  _ -> []
shrinkExp _ = []

-- | Pattern match on a given type and produce a corresponding term.
-- @Lam@ is generated from @Arr@ by first obtaining a fresh variable name for @Var@ and
-- then calling the @lam@ smart constructor on an expression that
-- was produced for an environment to which @Var@ was added.
-- A term of type @Nat@ is generated by converting a random integer through induction.
genWellTypedExp'' :: forall a . Eq a => Ty -> GTyM a (Exp a)
genWellTypedExp'' (Arr ty ty') = do
  uname <- fresh
  lam ty uname <$> local (insertVar uname ty) (genWellTypedExp' ty')
genWellTypedExp'' Nat = nat <$> Gen.int (Range.linear 0 10)

genWellTypedExp''' :: forall a . Eq a => Ty -> GTyM a (Exp a)
genWellTypedExp''' Nat = Suc <$> genWellTypedExp' Nat
genWellTypedExp''' ty = genWellTypedExp' ty

-- | Add @Var@ of given type to the given env so that it can be used for sampling later.
insertVar :: forall a . Eq a => a -> Ty -> Map Ty [Exp a] -> Map Ty [Exp a]
insertVar uname ty =
  Map.insertWith (<>) ty [Var uname] . fmap (List.filter (/= Var uname))

-- | Generate app by first producing type and value of the argument
-- and then generating a compatible @Lam@. 
genWellTypedApp :: forall a . Eq a => Ty -> GTyM a (Exp a)
genWellTypedApp ty = do
  tg <- genKnownTypeMaybe
  eg <- genWellTypedExp' tg
  let tf = Arr tg ty
  ef <- genWellTypedExp' tf
  pure (ef :@ eg)

-- | Try to look up a known expression of the desired type from the environment.
-- This does not always succeed, throwing `empty` when unavailable.
genWellTypedPath :: forall a . Ty -> GTyM a (Exp a)
genWellTypedPath ty = do
  paths <- ask
  case fromMaybe [] (Map.lookup ty paths) of
    [] -> empty
    es -> Gen.element es

-- | Generate either known types from the environment or new types.
genKnownTypeMaybe :: forall a . GTyM a Ty
genKnownTypeMaybe = do
  known <- ask
  if Map.null known then
    genTy
  else
    Gen.frequency [
        (2, Gen.element $ Map.keys known)
      , (1, genTy)
      ]

------------------------------------------------------------------------

data Example a b = Example
  { input :: Input a
  , target :: Target b
  } deriving (Eq, Ord, Show, Generic)

instance (Alternative t, ToActions t a, ToActions t b) => ToActions t (Example a b)
instance (Monad b, FromActions b a, FromActions b b') => FromActions b (Example a b')

newtype Input  a = Input  a deriving (Eq, Ord, Show, Generic)
newtype Target a = Target a deriving (Eq, Ord, Show, Generic)

instance ToActions t a => ToActions t (Input a)
instance (Monad b, FromActions b a) => FromActions b (Input a) where
  fromActions = do
    modify $ field @"aEnv" . field @"currentScope" .~ pure "input"
    modify $ field @"aEnv" . field @"knownScopes" %~ go "input"
    Input <$> fromActions
    where go scopeId = Map.insert scopeId (AttentionScope BidirectionalAttention (Set.singleton "input") mempty)

instance ToActions t a => ToActions t (Target a)
instance (Monad b, FromActions b a) => FromActions b (Target a) where
  fromActions = do
    modify $ field @"aEnv" . field @"currentScope" .~ pure "target"
    modify $ field @"aEnv" . field @"knownScopes" %~ go "target"
    Target <$> fromActions
    where go scopeId = Map.insert scopeId (AttentionScope BackwardAttention (Set.fromList ["input", "target"]) mempty)

------------------------------------------------------------------------

prep :: PropertyT IO (Ty, Example (Exp Int) (Exp Int), [((Example (Exp Int) (Exp Int), [Action]), Env Action)])
prep = do
  ty <- forAll genTy
  input <- forAll (genWellTypedExp ty)
  let target = nf input
      ex = Example (Input input) (Target target)
      env = defaultEnv
      actions = toActions @[] ex
  guard (List.length actions <= 512)
  let parser = fromActions @[] @(Example (Exp Int) (Exp Int))
      result = let f ap [] = empty
                   f ap (a : as) = let p = ap a in pure (p, as)
               in  runStateT (parse f parser actions) env
  pure (ty, ex, result)

prop_welltyped :: Property
prop_welltyped =
  property $ do
    (ty, Example (Input input) (Target target), _) <- prep
    let (Just ty') = runFresh . runMaybeT . typeCheck Map.empty $ input
    let (Just ty'') = runFresh . runMaybeT . typeCheck Map.empty $ target
    ty === ty'
    ty === ty''

-- test that every position belongs only to at most one attention scope
prop_attentionScope :: Property
prop_attentionScope = property $ do
  (_, _, [(_, Env {..})]) <- prep
  let r = Map.elems $ scopePositions <$> knownScopes aEnv
      c = sort . join $ Set.toList <$> r
      u = Set.toList . Set.unions $ r
  c === u

-- test presence of self attention
prop_selfAttention :: Property
prop_selfAttention = property $ do
  (_, _, [(_, Env {..})]) <- prep
  let sa = foldr (\(pos, pos') -> \b -> if pos == pos' then Set.insert pos b else b) Set.empty (attentionMask aEnv)
  sa === keyPaddingMask aEnv

-- test round trip serialization-deserialization
prop_roundTrip :: Property
prop_roundTrip = property $ do
  (_, ex, [((reconstructedEx, _), _)]) <- prep
  ex === reconstructedEx

testSTLC :: IO Bool
testSTLC = checkParallel $$(discover)

--------------------------------------------------------------------------------
-- Relation-Aware Transformer Masked Language Model
--------------------------------------------------------------------------------

data
  RATransformerMLMSpec
    (numAttnLayers :: Nat)
    (numHeads :: Nat)
    (headDim :: Nat)
    (ffnDim :: Nat)
    (tokenPaddingIdx :: Nat)
    (tokenNumEmbeds :: Nat)
    (metaPaddingIdx :: Nat)
    (metaNumEmbeds :: Nat)
    (relPaddingIdx :: Nat)
    (relNumEmbeds :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat)) where
  RATransformerMLMSpec
    :: forall numAttnLayers numHeads headDim ffnDim tokenPaddingIdx tokenNumEmbeds metaPaddingIdx metaNumEmbeds relPaddingIdx relNumEmbeds dtype device
     . { ratDropoutSpec :: DropoutSpec
       , ratLayerSpec   :: TransformerLayerSpec (headDim * numHeads) numHeads ffnDim dtype device
       }
    -> RATransformerMLMSpec numAttnLayers numHeads headDim ffnDim tokenPaddingIdx tokenNumEmbeds metaPaddingIdx metaNumEmbeds relPaddingIdx relNumEmbeds dtype device
  deriving (Show)

data
  RATransformerMLM
    (numAttnLayers :: Nat)
    (numHeads :: Nat)
    (headDim :: Nat)
    (ffnDim :: Nat)
    (tokenPaddingIdx :: Nat)
    (tokenNumEmbeds :: Nat)
    (metaPaddingIdx :: Nat)
    (metaNumEmbeds :: Nat)
    (relPaddingIdx :: Nat)
    (relNumEmbeds :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat)) where
  RATransformerMLM
    :: forall numAttnLayers numHeads headDim ffnDim tokenPaddingIdx tokenNumEmbeds metaPaddingIdx metaNumEmbeds relPaddingIdx relNumEmbeds dtype device
     . { ratEmbedding     :: Embedding ('Just tokenPaddingIdx) tokenNumEmbeds (headDim * numHeads) 'Learned dtype device
       , ratMetaEmbedding :: Embedding ('Just metaPaddingIdx) metaNumEmbeds (headDim * numHeads) 'Learned dtype device
       , ratRelEmbedding  :: Embedding ('Just relPaddingIdx) relNumEmbeds headDim 'Learned dtype device
       , ratDropout       :: Dropout
       , ratLayers        :: HList (HReplicateR numAttnLayers (TransformerLayer (headDim * numHeads) numHeads ffnDim dtype device))
       , ratProj          :: Linear (headDim * numHeads) tokenNumEmbeds dtype device
       }
    -> RATransformerMLM numAttnLayers numHeads headDim ffnDim tokenPaddingIdx tokenNumEmbeds metaPaddingIdx metaNumEmbeds relPaddingIdx relNumEmbeds dtype device
  deriving (Generic)

instance
  ( All KnownNat '[headDim, numHeads, tokenPaddingIdx, tokenNumEmbeds, metaPaddingIdx, metaNumEmbeds, relPaddingIdx, relNumEmbeds]
  , tokenPaddingIdx <= tokenNumEmbeds
  , 1 <= (tokenNumEmbeds - tokenPaddingIdx)
  , metaPaddingIdx <= metaNumEmbeds
  , 1 <= (metaNumEmbeds - metaPaddingIdx)
  , relPaddingIdx <= relNumEmbeds
  , 1 <= (relNumEmbeds - relPaddingIdx)
  , HReplicate numAttnLayers (TransformerLayerSpec (headDim * numHeads) numHeads ffnDim dtype device)
  , Randomizable (HList (HReplicateR numAttnLayers (TransformerLayerSpec (headDim * numHeads) numHeads ffnDim dtype device)))
                 (HList (HReplicateR numAttnLayers (TransformerLayer     (headDim * numHeads) numHeads ffnDim dtype device)))
  , KnownDType dtype
  , RandDTypeIsValid device dtype
  , KnownDevice device
  ) => Randomizable (RATransformerMLMSpec numAttnLayers numHeads headDim ffnDim tokenPaddingIdx tokenNumEmbeds metaPaddingIdx metaNumEmbeds relPaddingIdx relNumEmbeds dtype device)
                    (RATransformerMLM numAttnLayers numHeads headDim ffnDim tokenPaddingIdx tokenNumEmbeds metaPaddingIdx metaNumEmbeds relPaddingIdx relNumEmbeds dtype device)
 where
  sample RATransformerMLMSpec {..} =
    RATransformerMLM
      <$> Torch.Typed.sample (LearnedEmbeddingWithRandomInitSpec @( 'Just tokenPaddingIdx))
      <*> Torch.Typed.sample (LearnedEmbeddingWithRandomInitSpec @( 'Just metaPaddingIdx))
      <*> Torch.Typed.sample (LearnedEmbeddingWithRandomInitSpec @( 'Just relPaddingIdx))
      <*> Torch.Typed.sample ratDropoutSpec
      <*> Torch.Typed.sample (hreplicate @numAttnLayers ratLayerSpec)
      <*> Torch.Typed.sample LinearSpec

data
  RAFoldLayers
    (batchSize :: Nat)
    (headDim :: Nat)
    (seqLen :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  = RAFoldLayers
      { raflTrain :: Bool
      , raflAttentionMask :: Tensor device dtype '[batchSize, seqLen, seqLen]
      , raflKeyPaddingMask :: Tensor device 'Bool '[batchSize, seqLen]
      , raflRelationsK :: Tensor device dtype '[batchSize, seqLen, seqLen, headDim]
      , raflRelationsV :: Tensor device dtype '[batchSize, seqLen, seqLen, headDim]
      }

instance
  ( 1 <= numHeads
  , embedDim ~ (headDim * numHeads)
  , Mod (embedDim * 3) 3 ~ 0
  , Div (embedDim * 3) 3 ~ embedDim
  , All KnownNat '[embedDim, numHeads, seqLen, batchSize, headDim]
  , IsSuffixOf '[embedDim] '[batchSize, seqLen, embedDim]
  , KnownDType dtype
  , dtype ~ SumDType dtype
  , StandardFloatingPointDTypeValidation device dtype
  , MatMulDTypeIsValid device dtype
  , BasicArithmeticDTypeIsValid device dtype
  , SumDTypeIsValid device dtype
  , KnownDevice device
  ) => Apply' (RAFoldLayers batchSize headDim seqLen dtype device)
              ( TransformerLayer embedDim numHeads ffnDim dtype device
              , IO (Tensor device dtype '[batchSize, seqLen, embedDim])
              )
              (IO (Tensor device dtype '[batchSize, seqLen, embedDim])) where
  apply' RAFoldLayers {..} (layer, mx) = mx >>= transformerLayer layer raflTrain raflAttentionMask raflKeyPaddingMask (Just raflRelationsK) (Just raflRelationsV)

raTransformerMLM
  :: forall
       numAttnLayers
       numHeads
       headDim
       ffnDim
       tokenPaddingIdx
       tokenNumEmbeds
       metaPaddingIdx
       metaNumEmbeds
       relPaddingIdx
       relNumEmbeds
       relDim
       embedDim
       seqLen
       batchSize
       dtype
       device
   . ( All KnownNat '[tokenPaddingIdx, metaPaddingIdx, relPaddingIdx, seqLen, batchSize]
     , tokenPaddingIdx + 1 <= tokenNumEmbeds
     , metaPaddingIdx + 1 <= metaNumEmbeds
     , relPaddingIdx + 1 <= relNumEmbeds
     , embedDim ~ (headDim * numHeads)
     , 1 <= seqLen
     , HFoldrM
         IO
         (RAFoldLayers batchSize headDim seqLen dtype device)
         (Tensor device dtype '[batchSize, seqLen, embedDim])
         (HReplicateR numAttnLayers (TransformerLayer embedDim numHeads ffnDim dtype device))
         (Tensor device dtype '[batchSize, seqLen, embedDim])
     , BasicArithmeticDTypeIsValid device dtype
     , ComparisonDTypeIsValid device dtype
     , ComparisonDTypeIsValid device 'Int64
     , SumDType dtype ~ dtype
     , SumDTypeIsValid device dtype
     , KnownDType dtype
     , KnownDevice device
     )
  => RATransformerMLM numAttnLayers numHeads headDim ffnDim tokenPaddingIdx tokenNumEmbeds metaPaddingIdx metaNumEmbeds relPaddingIdx relNumEmbeds dtype device
  -> Bool -- ^ training flag
  -> RATransformerMLMInput batchSize seqLen relDim dtype device
  -> IO (Tensor device dtype '[batchSize, seqLen, tokenNumEmbeds])
raTransformerMLM RATransformerMLM {..} train RATransformerMLMInput {..} = do
  let maskedTokens = embed ratEmbedding ratMaskedTokens
      meta = embed ratMetaEmbedding ratMeta
      input = maskedTokens + meta
      relations = sumDim @3 $ embed ratRelEmbedding ratRelations
  forward ratProj <$> hfoldrM (RAFoldLayers train ratAttentionMask ratKeyPaddingMask relations relations) input ratLayers

data RATransformerMLMBatch batchSize seqLen relDim dtype device = RATransformerMLMBatch
  { ratInput :: RATransformerMLMInput batchSize seqLen relDim dtype device
  , ratTarget :: RATransformerMLMTarget batchSize seqLen device
  } deriving (Show, Generic)

data RATransformerMLMInput batchSize seqLen relDim dtype device = RATransformerMLMInput
  { ratMaskedTokens :: Tensor device 'Int64 '[batchSize, seqLen] -- ^ masked tokens
  , ratMeta :: Tensor device 'Int64 '[batchSize, seqLen] -- ^ meta
  , ratRelations :: Tensor device 'Int64 '[batchSize, seqLen, seqLen, relDim] -- ^ relations
  , ratAttentionMask :: Tensor device dtype '[batchSize, seqLen, seqLen] -- ^ attention mask (0 where attention is allowed, -inf everywhere else)
  , ratKeyPaddingMask :: Tensor device 'Bool '[batchSize, seqLen] -- ^ key padding mask (True for padding, False everywhere else)
  } deriving (Show, Generic)

data RATransformerMLMTarget batchSize seqLen device = RATransformerMLMTarget
  { ratTargetTokens :: Tensor device 'Int64 '[batchSize, seqLen] -- ^ target tokens
  , ratTokenMask :: Tensor device 'Bool '[batchSize, seqLen] -- ^ token mask
  , ratInputScopeMask :: Tensor device 'Bool '[batchSize, seqLen] -- ^ input scope mask
  , ratTargetScopeMask :: Tensor device 'Bool '[batchSize, seqLen] -- ^ target scope mask
  } deriving (Show, Generic)

loss
  :: forall batchSize seqLen numEmbeds device dtype
   . ( 1 <= numEmbeds
     , StandardFloatingPointDTypeValidation device dtype
     , MeanDTypeValidation device dtype
     , KnownDType dtype
     , KnownDevice device
     )
  => Tensor device 'Bool '[batchSize, seqLen, numEmbeds]
  -> Tensor device 'Bool '[batchSize, seqLen]
  -> Tensor device dtype '[batchSize, seqLen, numEmbeds]
  -> Tensor device 'Int64 '[batchSize, seqLen]
  -> Tensor device dtype '[]
loss validActionsMask keyPaddingMask logits target =
  let logProbs = logSoftmax @2 . maskedFill (logicalNot validActionsMask) (-1 / 0 :: Double) $ logits
      logLikelihood = squeezeDim @2 $ gatherDim @2 (unsqueeze @2 target) logProbs
      meanLogLikelihood = case maskedSelect (logicalNot keyPaddingMask) logLikelihood of
        UnknownShapeTensor t -> unsafeMeanAll t
  in (-meanLogLikelihood)

------------------------------------------------------------------------

mkRATransformerMLMBatch
  :: forall batchSize seqLen relDim dtype device a m -- tensorChunks ys
   . ( All KnownNat '[batchSize, seqLen, relDim]
     , SumDTypeIsValid device 'Int64
     , ComparisonDTypeIsValid device 'Int64
    --  , tensorChunks ~ Chunk batchSize 0 '[batchSize, seqLen, seqLen] 'Int64 device
    --  , Castable (HList tensorChunks) [ATenTensor]
    --  , HMapM' IO DisplayAttentionMask tensorChunks ys
     , KnownDType dtype
     , KnownDevice device
     , Scalar a
     , MonadIO m
     , Alternative m
     )
  => a
  -> a
  -> [[Action]]
  -> m (RATransformerMLMBatch batchSize seqLen relDim dtype device)
mkRATransformerMLMBatch pMaskInput pMaskTarget actions = do
  let fromJust' = maybe empty pure
      envs =
        let step actions =
              let parser = fromActions @[] @(Example (Exp Int) (Exp Int))
                  results =
                    let f ap [] = empty
                        f ap (a : as) = let p = ap a in pure (p, as)
                    in  runStateT (parse f parser actions) defaultEnv
              in snd <$> results
        in foldMap step actions
      tokenVocab = OSet.fromList [Pad, Unk, Mask, Token L, Token R]
      metaVocab = OSet.fromList [Pad, Unk, Token (D "Exp"), Token (D "Ty"), Token (D "Var")]
      relationsVocab = OSet.fromList [Pad, Unk, Token ChildParentRelation, Token ParentChildRelation, Token (SiblingDistRelation $ -1), Token (SiblingDistRelation 0), Token (SiblingDistRelation 1)]
  tokens <- fromJust' . mkSeq' tokenVocab $ view (field @"tEnv" . field @"tokens") <$> envs
  meta <- fromJust' . mkSeq' metaVocab $ view (field @"mEnv" . field @"metas") <$> envs
  relations <- fromJust' . mkMultiGrid' relationsVocab $ view (field @"rEnv" . field @"relations") <$> envs
  let scopeMask scopeId = fromJust' $ mkSeqMask' =<< traverse ((scopePositions <$>) . Map.lookup scopeId . view (field @"aEnv" . field @"knownScopes")) envs
  inputScopeMask <- scopeMask "input"
  targetScopeMask <- scopeMask "target"
  attentionMask <- fromJust' . mkGridMask' @batchSize @seqLen @seqLen $ view (field @"aEnv" . field @"attentionMask") <$> envs
  keyPaddingMask <- logicalNot <$> (fromJust' . mkSeqMask' $ view (field @"aEnv" . field @"keyPaddingMask") <$> envs)
  let attentionMask' = maskedFill (unsqueeze @2 keyPaddingMask) (1 :: Int) attentionMask
  -- liftIO . displayAttentionMask $ attentionMask'
  guard (attentionMaskIsProper attentionMask')
  let attentionMask'' = maskedFill (logicalNot attentionMask') (-1 / 0 :: Double) $ zeros @'[batchSize, seqLen, seqLen] @dtype @device
  tokenMask <- do
    tokenMaskInput <- bernoulliMask pMaskInput (keyPaddingMask `logicalOr` (logicalNot inputScopeMask))
    tokenMaskTarget <- bernoulliMask pMaskTarget (keyPaddingMask `logicalOr` (logicalNot targetScopeMask))
    pure (tokenMaskInput `logicalOr` tokenMaskTarget)
  let maskedTokens = maskedFill tokenMask (fromJust $ OSet.findIndex Mask tokenVocab) tokens
  pure $ RATransformerMLMBatch
           (RATransformerMLMInput
             maskedTokens
             meta
             relations
             attentionMask''
             keyPaddingMask)
           (RATransformerMLMTarget
             tokens
             tokenMask
             inputScopeMask
             targetScopeMask)

attentionMaskIsProper
  :: forall batchSize seqLen device
   . ( SumDTypeIsValid device 'Int64
     , ComparisonDTypeIsValid device 'Int64
     , KnownDevice device
     )
  => Tensor device 'Bool '[batchSize, seqLen, seqLen]
  -> Bool
attentionMaskIsProper t =
  let t' = toDType @'Int64 @'Bool t
      t'' = sumDim @2 t'
      t''' = t'' `gt` (0 :: Tensor device 'Int64 '[])
      t'''' = Torch.Typed.all t'''
  in toBool t''''

data DisplayAttentionMask = DisplayAttentionMask

display2dTensor
  :: forall dim dim' device dtype shape
   . ( All KnownNat '[dim, dim']
     , shape ~ '[dim, dim']
     , AllDimsPositive shape
     , BasicArithmeticDTypeIsValid device 'Float
     , MinMaxDTypeIsValid device 'Float
     , KnownDevice device
     )
  => Tensor device dtype shape
  -> IO ()
display2dTensor t = do
  mapM
    ( \row ->
        Text.Printf.printf "%02d" row
          >> mapM
            ( \col ->
                putChar $ grayScale !! (Prelude.floor $ scaled !! row !! col)
            )
            [0, downSamp .. natValI @dim - 1]
          >> putStrLn ""
    )
    [0, downSamp .. natValI @dim - 1]
  pure ()
  where
    downSamp = 1
    grayScale = grayScale10
    paletteMax = List.length grayScale - 1
    t' = toDType @'Float @dtype t
    scaled =
      let (mn, mx) = (Torch.Typed.min t', Torch.Typed.max t')
       in Exts.toList . Just . mulScalar paletteMax $ (t' `sub` mn) `Torch.Typed.div` (mx `sub` mn)

instance
  ( BasicArithmeticDTypeIsValid device 'Float
  , MinMaxDTypeIsValid device 'Float
  , KnownNat seqLen
  , shape ~ '[1, seqLen, seqLen]
  , AllDimsPositive shape
  , KnownDevice device
  ) => Apply' DisplayAttentionMask (Tensor device 'Int64 shape) (IO ()) where
  apply' _ = display2dTensor . squeezeDim @0

displayAttentionMask
  :: forall batchSize seqLen device shape (tensorChunks :: [Type]) ys
   . ( KnownNat batchSize
     , shape ~ '[batchSize, seqLen, seqLen]
     , tensorChunks ~ Chunk batchSize 0 shape 'Int64 device
     , Castable (HList tensorChunks) [ATenTensor]
     , HMapM' IO DisplayAttentionMask tensorChunks ys
     )
  => Tensor device 'Bool shape
  -> IO ()
displayAttentionMask t =
  let t' = toDType @'Int64 @'Bool t
      ts = chunk @batchSize @0 @shape @'Int64 @device @tensorChunks t'
  in void . hmapM' DisplayAttentionMask $ ts

testMkRATransformerMLMBatch :: IO (RATransformerMLMBatch TestBatchSize TestSeqLen TestRelDim TestDType TestDevice)
testMkRATransformerMLMBatch = do
  let input :: Exp Int = (lam Nat 0 (Var 0)) :@ Zero
      target = nf input
      ex = Example (Input input) (Target target)
      actions = toActions @[] $ ex -- [R,L,L,R,R,L,L,L,R,R,R,R,R,R]
  mkRATransformerMLMBatch @TestBatchSize @TestSeqLen @TestRelDim @TestDType @TestDevice @Float @IO 0.15 0.25 [actions]

------------------------------------------------------------------------

type TestBatchSize = 64
type TestSeqLen = 256
type TestRelDim = 2

type TestNumAttnLayers = 3
type TestNumHeads = 8
type TestHeadDim = 16
type TestFFNDim = 256
type TestPaddingIdx = 0
type TestTokenNumEmbeds = 5
type TestMetaNumEmbeds = 5
type TestRelNumEmbeds = 7
type TestDType = 'Float
type TestDataDevice = '( 'CPU, 0)
type TestDevice = '( 'CUDA, 0)

type TestRATransformerMLMSpec
  = RATransformerMLMSpec
      TestNumAttnLayers
      TestNumHeads
      TestHeadDim
      TestFFNDim
      TestPaddingIdx
      TestTokenNumEmbeds
      TestPaddingIdx
      TestMetaNumEmbeds
      TestPaddingIdx
      TestRelNumEmbeds
      TestDType
      TestDevice

bernoulliMask
  :: forall shape dtype device a m
   . (Scalar a, KnownShape shape, shape ~ Broadcast shape shape, MonadIO m)
  => a
  -> Tensor device 'Bool shape
  -> m (Tensor device 'Bool shape)
bernoulliMask p keyPaddingMask = do
  let bernoulli = fromProbs . toDynamic . mulScalar p . toDType @'Float @'Bool . onesLike $ keyPaddingMask
  samples <- liftIO $ UnsafeMkTensor @device @'Bool @shape . toType Bool <$> Torch.Distributions.Distribution.sample bernoulli (shapeVal @shape)
  pure $ maskedFill keyPaddingMask (0 :: Int) samples

testBernoulliMask :: IO (Tensor TestDevice 'Bool '[2, 3])
testBernoulliMask = bernoulliMask (0.25 :: Float) $ zeros @'[2, 3] @'Bool @TestDevice

mkSeq'
  :: forall batchSize seqLen device a
   . (Ord a, All KnownNat '[batchSize, seqLen], KnownDevice device)
  => OSet.OSet (Token a) -> [Map Pos a] -> Maybe (Tensor device 'Int64 '[batchSize, seqLen])
mkSeq' vocab ms = do
  unkIndex <- OSet.findIndex Unk vocab
  guard $ List.length ms <= natValI @batchSize
  (fstKeys, sndKeys, elems) <- 
    let step (i, m) = do
          let keys = Map.keys m
              fstKeys = const i <$> keys
              sndKeys = unPos <$> keys
              elems = (fromMaybe unkIndex . flip OSet.findIndex vocab . Token) <$> Map.elems m
          guard $ Prelude.all (< natValI @seqLen) sndKeys
          pure (fstKeys, sndKeys, elems)
    in foldMap step $ zip [(0 :: Int)..] ms
  let tensorOptions = withDType Int64 defaultOpts
      i = asTensor' [fstKeys, sndKeys] tensorOptions
      v = asTensor' elems tensorOptions
      shape = [natValI @batchSize, natValI @seqLen]
  pure . toDevice @device @'( 'CPU, 0) . UnsafeMkTensor @'( 'CPU, 0) . Torch.toDense $ sparseCooTensor i v shape tensorOptions

testMkSeq' :: Maybe (Tensor TestDevice 'Int64 '[2, 3])
testMkSeq' =
  let vocab = OSet.fromList [Pad, Unk, Token "a"]
      ms = [ Map.fromList [(Pos 0, "a"), (Pos 1, "b"), (Pos 2, "a")]
           , Map.fromList [(Pos 1, "a"), (Pos 2, "b")]
           ]
  in mkSeq' vocab ms

mkSeqMask'
  :: forall batchSize seqLen device
   . (All KnownNat '[batchSize, seqLen], KnownDevice device)
  => [Set Pos] -> Maybe (Tensor device 'Bool '[batchSize, seqLen])
mkSeqMask' ss = do
  guard $ List.length ss <= natValI @batchSize
  (fstElems, sndElems) <-
    let step (i, s) = do
          let elems = Set.elems s
              fstElems = const i <$> elems
              sndElems = unPos <$> elems
          guard $ Prelude.all (< natValI @seqLen) sndElems
          pure (fstElems, sndElems)
    in foldMap step $ zip [(0 :: Int)..] ss
  let tensorOptions = withDType Int64 defaultOpts
      i = asTensor' [fstElems, sndElems] tensorOptions
      v = Torch.ones [List.length fstElems] tensorOptions
      shape = [natValI @batchSize, natValI @seqLen]
  pure . toDevice @device @'( 'CPU, 0) . UnsafeMkTensor @'( 'CPU, 0) . Torch.toType Bool . Torch.toDense $ sparseCooTensor i v shape tensorOptions

testMkSeqMask' :: Maybe (Tensor TestDevice 'Bool '[2, 3])
testMkSeqMask' =
  let ss = [ Set.fromList [Pos 1]
           , Set.fromList [Pos 0, Pos 2]
           ]
  in mkSeqMask' ss

mkGrid'
  :: forall batchSize seqLen seqLen' device a
   . (Show a, Ord a, All KnownNat '[batchSize, seqLen, seqLen'], KnownDevice device)
  => OSet.OSet (Token a) -> [Map (Pos, Pos) a] -> Maybe (Tensor device 'Int64 '[batchSize, seqLen, seqLen'])
mkGrid' vocab ms = do
  unkIndex <- OSet.findIndex Unk vocab
  guard $ List.length ms <= natValI @batchSize
  (fstKeys, sndKeys, trdKeys, elems) <-
    let step (i, m) = do
          let keys = Map.keys m
              fstKeys = const i <$> keys
              sndKeys = unPos . fst <$> keys
              trdKeys = unPos . snd <$> keys
              elems = (fromMaybe unkIndex . flip OSet.findIndex vocab . Token) <$> Map.elems m
          guard $ Prelude.all (< natValI @seqLen) sndKeys
          guard $ Prelude.all (< natValI @seqLen') trdKeys
          pure (fstKeys, sndKeys, trdKeys, elems)
    in foldMap step $ zip [(0 :: Int)..] ms
  let tensorOptions = withDType Int64 defaultOpts
      i = asTensor' [fstKeys, sndKeys, trdKeys] tensorOptions
      v = asTensor' elems tensorOptions
      shape = [natValI @batchSize, natValI @seqLen, natValI @seqLen']
  pure . toDevice @device @'( 'CPU, 0) . UnsafeMkTensor @'( 'CPU, 0) . Torch.toDense $ sparseCooTensor i v shape tensorOptions

testMkGrid' :: Maybe (Tensor TestDevice 'Int64 '[2, 2, 3])
testMkGrid' =
  let vocab = OSet.fromList [Pad, Unk, Token "a", Token "b"]
      ms = [ Map.fromList [((Pos 0, Pos 0), "a"), ((Pos 0, Pos 1), "b"), ((Pos 1, Pos 2), "a")]
           , Map.fromList [((Pos 0, Pos 1), "b"), ((Pos 0, Pos 2), "c")]
           ]
  in mkGrid' vocab ms

mkMultiGrid'
  :: forall batchSize seqLen seqLen' dim device a
   . (Ord a, All KnownNat '[batchSize, seqLen, seqLen', dim], KnownDevice device)
  => OSet.OSet (Token a) -> [Map (Pos, Pos) (Set a)] -> Maybe (Tensor device 'Int64 '[batchSize, seqLen, seqLen', dim])
mkMultiGrid' vocab ms = do
  unkIndex <- OSet.findIndex Unk vocab
  guard $ List.length ms <= natValI @batchSize
  (fstKeys, sndKeys, trdKeys, fthKeys, elems) <-
    let step (i, m) = do
          let m' = Map.fromList $ do 
                ((pos, pos'), s) <- Map.toList m
                (i, a) <- zip [(0 :: Int)..] $ Set.toList s
                pure ((pos, pos', i), a)
              keys = Map.keys m'
              fstKeys = const i <$> keys
              sndKeys = unPos . (\(pos, _, _) -> pos) <$> keys
              trdKeys = unPos . (\(_, pos, _) -> pos) <$> keys
              fthKeys = (\(_, _, i) -> i) <$> keys
              elems = (fromMaybe unkIndex . flip OSet.findIndex vocab . Token) <$> Map.elems m'
          guard $ Prelude.all (< natValI @seqLen) sndKeys
          guard $ Prelude.all (< natValI @seqLen') trdKeys
          guard $ Prelude.all (< natValI @dim) fthKeys
          pure (fstKeys, sndKeys, trdKeys, fthKeys, elems)
    in foldMap step $ zip [(0 :: Int)..] ms
  let tensorOptions = withDType Int64 defaultOpts
      i = asTensor' [fstKeys, sndKeys, trdKeys, fthKeys] tensorOptions
      v = asTensor' elems tensorOptions
      shape = [natValI @batchSize, natValI @seqLen, natValI @seqLen', natValI @dim]
  pure . toDevice @device @'( 'CPU, 0) . UnsafeMkTensor @'( 'CPU, 0) . Torch.toDense $ sparseCooTensor i v shape tensorOptions

mkGridMask'
  :: forall batchSize seqLen seqLen' device
   . (All KnownNat '[batchSize, seqLen, seqLen'], KnownDevice device)
  => [Set (Pos, Pos)] -> Maybe (Tensor device 'Bool '[batchSize, seqLen, seqLen'])
mkGridMask' ss = do
  guard $ List.length ss <= natValI @batchSize
  (fstElems, sndElems, trdElems) <-
    let step (i, s) = do
          let elems = Set.elems s
              fstElems = const i <$> elems
              sndElems = unPos . fst <$> elems
              trdElems = unPos . snd <$> elems
          guard $ Prelude.all (< natValI @seqLen) sndElems
          guard $ Prelude.all (< natValI @seqLen') trdElems
          pure (fstElems, sndElems, trdElems)
    in foldMap step $ zip [(0 :: Int)..] ss
  let tensorOptions = withDType Int64 defaultOpts
      i = asTensor' [fstElems, sndElems, trdElems] tensorOptions
      v = Torch.ones [List.length fstElems] tensorOptions
      shape = [natValI @batchSize, natValI @seqLen, natValI @seqLen']
  pure . toDevice @device @'( 'CPU, 0) . UnsafeMkTensor @'( 'CPU, 0) . Torch.toType Bool . Torch.toDense $ sparseCooTensor i v shape tensorOptions

testMkGridMask' :: Maybe (Tensor TestDevice 'Bool '[2, 2, 3])
testMkGridMask' =
  let ss = [ Set.fromList [(Pos 0, Pos 0), (Pos 0, Pos 1), (Pos 1, Pos 2)]
           , Set.fromList [(Pos 0, Pos 2)]
           ]
  in mkGridMask' ss

------------------------------------------------------------------------

mkBatch
  :: forall batchSize seqLen relDim dtype device a m -- tensorChunks ys
   . ( All KnownNat '[batchSize, seqLen, relDim]
     , SumDTypeIsValid device 'Int64
     , ComparisonDTypeIsValid device 'Int64
    --  , tensorChunks ~ Chunk batchSize 0 '[batchSize, seqLen, seqLen] 'Int64 device
    --  , Castable (HList tensorChunks) [ATenTensor]
    --  , HMapM' IO DisplayAttentionMask tensorChunks ys
     , KnownDType dtype
     , KnownDevice device
     , Scalar a
     , MonadIO m
     , Alternative m
     )
  => a
  -> a
  -> StateT Seed.Seed m (RATransformerMLMBatch batchSize seqLen relDim dtype device)
mkBatch pMaskInput pMaskTarget = do
  -- seed <- get
  -- liftIO . putStrLn $ "Start making batch for " <> show seed
  actions <- sample' . Gen.list (Range.singleton $ natValI @batchSize) $ do
    ty <- genTy
    input <- genWellTypedExp @Int ty
    let target = nf input
        ex = Example (Input input) (Target target)
        actions = toActions @[] ex
    guard (List.length actions <= natValI @seqLen)
    pure actions
  res <- lift $ mkRATransformerMLMBatch pMaskInput pMaskTarget actions
  -- liftIO . putStrLn $ "Finished making batch for " <> show seed
  pure res

testGen :: IO ()
testGen = do
  seed <- Seed.random
  ((e, actions), _) <- runStateT (sample' $ do
    ty <- genTy
    e <- genWellTypedExp @Int ty
    guard (isJust . runFresh . runMaybeT . assertTy Map.empty e $ ty)
    let actions = toActions @[] e
    guard (List.length actions <= 256)
    pure (e, actions)) seed
  print actions
  putDoc . vsep $ [mempty, pprint (0 :: Int) e, mempty, pprint (0 :: Int) (nf e), mempty]

sample' :: MonadIO m => Gen a -> StateT Seed.Seed m a
sample' gen =
  let
    go = do
      seed <- get
      let (seed', seed'') = Seed.split seed
      put seed''
      case evalGen 20 seed' gen of
        Nothing ->
          go
        Just x ->
          pure $ Tree.treeValue x
  in
    go

testMkBatch :: IO (RATransformerMLMBatch TestBatchSize TestSeqLen TestRelDim TestDType TestDevice)
testMkBatch = do
  seed <- Seed.random
  (batch, _) <- runStateT (mkBatch @TestBatchSize @TestSeqLen @TestRelDim @TestDType @TestDevice @Float 0.15 0.25) seed
  pure batch

data ClipGradValue a = ClipGradValue a

instance
  (Scalar a, Num a) => Apply' (ClipGradValue a) (Tensor device dtype shape) (Tensor device dtype shape) where
  apply' (ClipGradValue a) = clamp (-a) a

data GuardGradAgainstNaN = GuardGradAgainstNaN

instance Apply' GuardGradAgainstNaN (Tensor device dtype shape, Maybe ()) (Maybe ()) where
  apply' _ (t, acc) = acc >> (guard . not . toBool . Torch.Typed.any . Torch.Typed.isNaN $ t)

data RATransformerMLMData (batchSize :: Nat) (seqLen :: Nat) (relDim :: Nat) (dtype :: DType) (device :: (DeviceType, Nat)) a = RATransformerMLMData a a Int

instance
  ( SumDTypeIsValid device 'Int64
  , ComparisonDTypeIsValid device 'Int64
  , All KnownNat '[batchSize, seqLen, relDim]
  , KnownDType dtype
  , KnownDevice device
  , Scalar a
  ) => Datastream
    (Safe.SafeT IO)
    Seed.Seed
    (RATransformerMLMData batchSize seqLen relDim dtype device a)
    (RATransformerMLMBatch batchSize seqLen relDim dtype device) 
  where
  -- streamBatch :: dataset -> seed -> ListT m batch
  streamBatch (RATransformerMLMData pMaskInput pMaskTarget len) seed = 
    let go s = do
          (b, s') <- lift $ runStateT (mkBatch pMaskInput pMaskTarget) s
          yield b
          go s'
    in Select $ go seed >-> Pipes.Prelude.take len

testProgram
  :: LearningRate TestDevice TestDType -- ^ learning rate
  -> Int -- ^ number of epochs
  -> Int -- ^ number batches taken from training file per epoch
  -> Int -- ^ number batches taken from evaluation file per epoch
  -> FilePath -- ^^ file name
  -> IO ()
testProgram learningRate numEpochs trainingLen evaluationLen ptFile = Safe.runSafeT . runEffect $ go
  where
    go :: Effect (Safe.SafeT IO) ()
    go = do
      let
        pMaskInput = 0 :: Float
        pMaskTarget = 0.2 :: Float
        trainingSeeds = List.take 10 $ Seed.from <$> List.iterate (+ 1) (0 :: Word64)
        trainingData = makeListT' (RATransformerMLMData @TestBatchSize @TestSeqLen @TestRelDim @TestDType @TestDataDevice pMaskInput pMaskTarget trainingLen) trainingSeeds
        evaluationsSeeds = List.take 1 $ Seed.from <$> List.iterate (+ 1) (0 :: Word64)
        evaluationData = makeListT' (RATransformerMLMData @TestBatchSize @TestSeqLen @TestRelDim @TestDType @TestDataDevice pMaskInput pMaskTarget evaluationLen) evaluationsSeeds
      model <- liftIO . Torch.Typed.sample $
                  (RATransformerMLMSpec 
                    (DropoutSpec 0.2)
                    (TransformerLayerSpec
                      (MultiheadAttentionSpec
                        (DropoutSpec 0.2)
                      )
                      (DropoutSpec 0.2)
                      0.001
                      (TransformerMLPSpec
                        (DropoutSpec 0.2)
                        (DropoutSpec 0.2)
                        0.001
                      )
                    ) :: TestRATransformerMLMSpec
                  )
      let optim = mkAdam 0 0.9 0.999 (flattenParameters model)
          -- optim = mkGDM 0.9 (flattenParameters model)
          training (model', optim') =
            let step (model'', optim'') (RATransformerMLMBatch {..}, batch) = do
                  lift . putStrLn $ "Training batch " <> show batch
                  let input = toDevice @TestDevice @TestDataDevice ratInput
                  -- lift . threadDelay $ 10000000 -- delay by 10 seconds
                  prediction <- lift $ raTransformerMLM model'' True input
                  let targetTokens = toDevice @TestDevice @TestDataDevice . ratTargetTokens $ ratTarget
                      cre = loss ones (ratKeyPaddingMask input) prediction targetTokens
                      parameters = flattenParameters model''
                      gradients = grad cre parameters
                      clippedGradients = hmap' (ClipGradValue (1e1 :: Float)) gradients
                  lift performGC -- force GC cleanup after every batch
                  maybe
                    (lift (print "encountered NaN in gradients, repeating training step") >> pure (model'', optim''))
                    (const $ lift (runStep' model'' optim'' learningRate clippedGradients))
                    (hfoldrM GuardGradAgainstNaN () clippedGradients)
                begin = pure (model', optim')
                done = pure
            in runContT trainingData (foldM step begin done . enumerate)
          evaluation model' =
            let step (cre, _step) (RATransformerMLMBatch {..}, batch) = do
                  lift . putStrLn $ "Evaluation batch " <> show batch
                  let input = toDevice @TestDevice @TestDataDevice ratInput
                  prediction <- lift $ raTransformerMLM model' False input
                  let target = toDevice @TestDevice @TestDataDevice ratTarget
                      loss' mask = loss ones mask prediction (ratTargetTokens target)
                      cre' = CRE
                        { cre = toFloat . loss' $ ratKeyPaddingMask input
                        , creInput = toFloat . loss' $ ratKeyPaddingMask input `logicalOr` (logicalNot $ ratInputScopeMask target)
                        , creTarget = toFloat . loss' $ ratKeyPaddingMask input `logicalOr` (logicalNot $ ratTargetScopeMask target)
                        , creMasked = toFloat . loss' $ (logicalNot $ ratTokenMask target) `logicalOr` ratKeyPaddingMask input
                        , creMaskedInput = toFloat . loss' $ (logicalNot $ ratTokenMask target) `logicalOr` ratKeyPaddingMask input `logicalOr` (logicalNot $ ratInputScopeMask target)
                        , creMaskedTarget = toFloat . loss' $ (logicalNot $ ratTokenMask target) `logicalOr` ratKeyPaddingMask input `logicalOr` (logicalNot $ ratTargetScopeMask target)
                        , creNonMasked = toFloat . loss' $ ratTokenMask target `logicalOr` ratKeyPaddingMask input
                        , creNonMaskedInput = toFloat . loss' $ ratTokenMask target `logicalOr` ratKeyPaddingMask input `logicalOr` (logicalNot $ ratInputScopeMask target)
                        , creNonMaskedTarget = toFloat . loss' $ ratTokenMask target `logicalOr` ratKeyPaddingMask input `logicalOr` (logicalNot $ ratTargetScopeMask target)
                        }
                  -- guard (not . toBool . Torch.Typed.isNaN $ cre)
                  let res = (cre <> cre', _step + 1)
                  lift performGC -- force GC cleanup after every batch
                  pure res
                begin = pure (mempty, 0 :: Int)
                done (_, 0) = pure CRE
                  { cre = 1
                  , creInput = 1
                  , creTarget = 1
                  , creMasked = 1
                  , creMaskedInput = 1
                  , creMaskedTarget = 1
                  , creNonMasked = 1
                  , creNonMaskedInput = 1
                  , creNonMaskedTarget = 1
                  }
                done (cre, _step) =
                  let scale x = x / (fromInteger . toInteger $ _step)
                  in pure (scale <$> cre)
            in runContT evaluationData (foldM step begin done . enumerate)
          step (model', optim') epoch = do
            lift . putStrLn $ "Epoch " <> show epoch
            (model'', optim'') <- training (model', optim')
            cre <- evaluation model''
            lift . putStrLn $ "Average evaluation loss " <> show cre
            lift . save (hmap' ToDependent . flattenParameters $ model'') $ ptFile
            pure (model'', optim'')
          begin = do
            cre <- evaluation model
            lift . putStrLn $ "Average evaluation loss " <> show cre
            pure (model, optim)
          done = pure
      _ <- lift $ foldM step begin done (Pipes.each [1 .. numEpochs])
      pure ()

data CRE a = CRE
  { cre :: a
  , creInput :: a
  , creTarget :: a
  , creMasked :: a
  , creMaskedInput :: a
  , creMaskedTarget :: a
  , creNonMasked :: a
  , creNonMaskedInput :: a
  , creNonMaskedTarget :: a
  } deriving (Show, Functor)

instance Num a => Semigroup (CRE a) where
  a <> b = CRE
    { cre = cre a + cre b
    , creInput = creInput a + creInput b
    , creTarget = creTarget a + creTarget b
    , creMasked = creMasked a + creMasked b
    , creMaskedInput = creMaskedInput a + creMaskedInput b
    , creMaskedTarget = creMaskedTarget a + creMaskedTarget b
    , creNonMasked = creNonMasked a + creNonMasked b
    , creNonMaskedInput = creNonMaskedInput a + creNonMaskedInput b
    , creNonMaskedTarget = creNonMaskedTarget a + creNonMaskedTarget b
    }

instance Num a => Monoid (CRE a) where
  mempty = CRE
    { cre = 0
    , creInput = 0
    , creTarget = 0
    , creMasked = 0
    , creMaskedInput = 0
    , creMaskedTarget = 0
    , creNonMasked = 0
    , creNonMaskedInput = 0
    , creNonMaskedTarget = 0
    }

-- implement hash function for expressions?
-- access variability of generated lambda expressions, calculate mean number of samples until repetition
-- calculate sample statistics: expression depth, type depth, number of lambda abstractions, applications, Nat tower depth
-- how many expressions are part of other expressions?
-- implement inference
-- add type checking to inference
-- implement valid actions mask
-- add simple ^-shaped learning rate schedule
-- add optimizer checkpointing, save both weights and iteration number
-- make it possible to resume training from a model and an optimizer checkpoint
-- add rolling model checkpointing based on best observed performance
-- make an existentially quantified model checkpoint that carries all constraints
-- test that there are no empty sequences
-- test that there are no unks
-- split training and evaluation data in non-overlapping sets (based on predefined expression and/or type properties or hash parity)
-- generalize multi-headed attention: use query, key, value as inputs, use seqLen and seqLen', make attentionMask and keyPaddingMask both optional
-- improve lambda calculus pretty printing: parentheses are not always good
-- ~make it possible to use different pMask values for input and target sections~
-- ~distinguish between loss on input and target reconstruction~
-- ~compute and the report the loss on the masked tokens and the unmasked tokens individually~
-- ~add lambda calculus pretty printing~
-- ~add simple model checkpointing~
-- ~use Andre's new streaming data pipeline~
-- ~why does it segfault on cuda?~
-- ~what is the cause of NaNs in the evaluation?~
-- ~ > the loss is NaN if the length of at least one item is the seqLen, i.e. if there is no padding in one row~
-- ~when there are NaNs in the loss gradients, is the loss NaN, too?~
-- ~ > it is likely that the loss gradients are NaN iff the loss is NaN~
-- ~test that there is no position that attends to nothing~
