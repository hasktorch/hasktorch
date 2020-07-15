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

module Torch.Data.ActionTransitionSystem where

import Prelude hiding (lookup)
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
import Control.Monad.Yoctoparsec (Parser)
import Control.Monad.Trans.Free (wrap, iterTM, runFreeT, Free, FreeT(..), FreeF(..))
import Control.Monad.State (StateT (..), runStateT, get, put, modify)
import Control.Monad (ap, void)
import Control.Monad.Trans (MonadTrans(lift))
import Data.Map as Map (elems, toList, (!), adjust, update, keys, null, insertWith, singleton, fromList, unionWith, Map, insert, lookup)
import Data.Set as Set (filter, cartesianProduct, unions, toList, fromList, member, singleton, union, Set, insert, findIndex)
import qualified Data.Set as Set (empty)
import Data.List as List (filter, sort, nub)
import Control.Monad.Reader (ask, local, runReaderT, ReaderT)
import Hedgehog (PropertyT, check, Property, (===), forAll, property, Gen, MonadGen)
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import Control.Monad (guard)
import Data.Maybe (fromMaybe)
import Control.Monad (join)

-- https://stackoverflow.com/questions/17675054/deserialising-with-ghc-generics?rq=1
-- http://hackage.haskell.org/package/cereal-0.5.8.1/docs/Data-Serialize.html
-- https://hackage.haskell.org/package/attoparsec-0.13.2.4/docs/src/Data.Attoparsec.Text.Internal.html
-- https://hackage.haskell.org/package/yoctoparsec-0.1.0.0/docs/src/Control-Monad-Yoctoparsec.html#Parser
-- https://vaibhavsagar.com/blog/2018/02/04/revisiting-monadic-parsing-haskell/
-- https://github.com/alphaHeavy/protobuf/blob/46cda829cf1e7b6bba2ff450e267fb1a9ace4fb3/src/Data/ProtocolBuffers/Ppr.hs

data Env = Env
  { pos :: Pos
  , meta :: Maybe M
  , rEnv :: REnv
  , aEnv :: AEnv
  -- , validActionsMask :: Map Pos (Set Action)
  } deriving (Eq, Ord, Show, Generic)

defaultEnv :: Env
defaultEnv = Env
  { pos = Pos 0
  , meta = Nothing
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
  deriving (Eq, Ord, Show, Num, Generic)

data Relation =
    ChildParentRelation
  | ParentChildRelation
  | SiblingDistRelation { siblingDist :: Int }
  deriving (Eq, Ord, Show, Generic)

data REnv = REnv
  { parentPos :: Maybe Pos
  , parents :: Map Pos (Set Pos)
  , relations :: Map (Pos, Pos) (Set Relation)
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
  { currentScope :: Maybe ScopeId
  , knownScopes :: Map ScopeId AttentionScope
  , attentionMask :: Set (Pos, Pos)
  , keyPaddingMask :: Set Pos
  } deriving (Eq, Ord, Show, Generic)

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
    deriving (Eq, Ord, Show)

type To t a = a -> t Action
type From b a = Parser (StateT Env b) Action a

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

-- TODO: move state updates somewhere else?
token :: forall b t . Monad b => Parser (StateT Env b) t t
token = do
  t <- wrap $ FreeT . pure . Pure
  pos <- (^. field @"pos") <$> get
  zoom (field @"rEnv") . lift . updateRelations $ pos
  zoom (field @"aEnv") . lift . updateAttention $ pos
  modify (field @"pos" %~ (+1))
  pure t

is :: (MonadPlus b, Eq t) => t -> Parser (StateT Env b) t t
is t = mfilter (== t) token

isNot :: (MonadPlus b, Eq t) => t -> Parser (StateT Env b) t t
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
    modify $ field @"meta" .~ (pure . D . pack . datatypeName @d $ undefined)
    pos <- (^. field @"pos") <$> get
    modify $ field @"rEnv" . field @"parentPos" .~ (pure pos)
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

instance Applicative t => ToActions t Int where
  toActions = pure . IToken

instance MonadFail b => FromActions b Int where
  fromActions = token >>= (\case IToken i -> pure i; _ -> fail "int")

instance Applicative t => ToActions t Bool where
  toActions = pure . BToken

instance MonadFail b => FromActions b Bool where
  fromActions = token >>= (\case BToken b -> pure b; _ -> fail "bool")

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

test :: ([Action], [((Foo, [Action]), Env)], [((), Env)])
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
                        , view (field @"meta") env'
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

data TType =
    TBool
  | TInt
  | TString
  | TArrow TType TType
    deriving (Eq, Ord, Show, Generic)

instance ToActions [] TType
instance FromActions [] TType

data Expr =
    EBool Bool
  | EInt Int
  | EString Text
  | EVar Text
  | ELam Text TType Expr
  | EApp Expr Expr
    deriving (Eq, Ord, Show, Generic)

instance ToActions [] Expr
instance FromActions [] Expr

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

-- | Evaluate to weak head normal form.
evaluate :: Expr -> Expr
evaluate expr =
  case expr of
    EBool _ ->
      expr
    EInt _ ->
      expr
    EString _ ->
      expr
    EVar _ ->
      expr
    ELam _ _ _ ->
      expr
    EApp f g ->
      case evaluate f of
        ELam x _t e ->
          evaluate (subst x g e)
        h ->
          EApp h g

subst :: Text -> Expr -> Expr -> Expr
subst x y expr =
  case expr of
    EBool _ ->
      expr
    EInt _ ->
      expr
    EString _ ->
      expr
    EVar z ->
      if x == z then
        y
      else
        expr
    ELam n t g ->
      if n == x then
        ELam n t g
      else
        ELam n t (subst x y g)
    EApp f g ->
      EApp (subst x y f) (subst x y g)

-- | Collect all the free variables in an 'Expr'.
free :: Expr -> Set Text
free =
  free' mempty mempty

free' :: Set Text -> Set Text -> Expr -> Set Text
free' binds frees expr =
  case expr of
    EBool _ ->
      frees
    EInt _ ->
      frees
    EString _ ->
      frees
    EVar x ->
      if Set.member x binds then
        frees
      else
        Set.insert x frees
    ELam x _t y ->
      free' (Set.insert x binds) frees y
    EApp f g ->
      free' binds frees f <> free' binds frees g

------------------------------------------------------------------------

data TypeError =
    Mismatch TType TType
  | FreeVariable Text
  | ExpectedArrow TType
  deriving (Eq, Ord, Show)

-- | Typecheck some expression.
typecheck :: Expr -> Either TypeError TType
typecheck =
  typecheck' mempty

typecheck' :: Map Text TType -> Expr -> Either TypeError TType
typecheck' env expr =
  case expr of
    EBool _ ->
      pure TBool

    EInt _ ->
      pure TInt

    EString _ ->
      pure TString

    EVar x ->
      maybe (Left (FreeVariable x)) pure (Map.lookup x env)

    ELam x t y ->
      TArrow t <$> typecheck' (Map.insert x t env) y

    EApp f g -> do
      tf <- typecheck' env f
      tg <- typecheck' env g
      case tf of
        TArrow ta tb ->
          if ta == tg then
            pure tb
          else
            Left (Mismatch ta tg)
        _ ->
          Left (ExpectedArrow tf)

------------------------------------------------------------------------

genType :: MonadGen m => m TType
genType =
  Gen.recursive Gen.choice [
      pure TBool
    , pure TInt
    , pure TString
    ] [
      TArrow <$> genType <*> genType
    ]

------------------------------------------------------------------------

genWellTypedExpr :: TType -> Gen Expr
genWellTypedExpr =
  flip runReaderT mempty . genWellTypedExpr'

genWellTypedExpr' :: TType -> ReaderT (Map TType [Expr]) Gen Expr
genWellTypedExpr' want =
  Gen.shrink shrinkExpr $
  Gen.recursive Gen.choice [
      genWellTypedExpr'' want
    ] [
      genWellTypedPath want <|> genWellTypedApp want
    , genWellTypedApp want
    ]

shrinkExpr :: Expr -> [Expr]
shrinkExpr expr =
  case expr of
    EApp f g ->
      case evaluate f of
        ELam x _ e ->
          [evaluate (subst x g e)]
        _ ->
          []
    _ ->
      []

genWellTypedExpr'' :: TType -> ReaderT (Map TType [Expr]) Gen Expr
genWellTypedExpr'' want =
  case want of
    TBool ->
      EBool <$> Gen.element [True, False]
    TInt ->
      EInt <$> Gen.int (Range.linear 0 10000)
    TString ->
      EString <$> Gen.text (Range.linear 0 25) Gen.lower
    TArrow t1 t2 -> do
      x <- Gen.text (Range.linear 1 25) Gen.lower
      ELam x t1 <$> local (insertVar x t1) (genWellTypedExpr' t2)

insertVar :: Text -> TType -> Map TType [Expr] -> Map TType [Expr]
insertVar n typ =
  Map.insertWith (<>) typ [EVar n] .
  fmap (List.filter (/= EVar n))

genWellTypedApp :: TType -> ReaderT (Map TType [Expr]) Gen Expr
genWellTypedApp want = do
  tg <- genKnownTypeMaybe
  eg <- genWellTypedExpr' tg
  let tf = TArrow tg want
  ef <- genWellTypedExpr' tf
  pure (EApp ef eg)

-- | This tries to look up a known expression of the desired type from the env.
-- It does not always succeed, throwing `empty` when unavailable.
genWellTypedPath :: TType -> ReaderT (Map TType [Expr]) Gen Expr
genWellTypedPath want = do
  paths <- ask
  case fromMaybe [] (Map.lookup want paths) of
    [] ->
      empty
    es ->
      Gen.element es

genKnownTypeMaybe :: ReaderT (Map TType [Expr]) Gen TType
genKnownTypeMaybe = do
  known <- ask
  if Map.null known then
    genType
  else
    Gen.frequency [
        (2, Gen.element $ Map.keys known)
      , (1, genType)
      ]

------------------------------------------------------------------------

-- Generates a term that is ill-typed at some point.
genIllTypedExpr :: Gen Expr
genIllTypedExpr = do
  be <- genIllTypedApp
  Gen.recursive Gen.choice [
      -- Don't grow - just dish up the broken expr
      pure be
    ] [
      -- Grow a reasonable app expression around the error
      do tg <- genType
         tf <- genType
         let ta = TArrow tg tf
         ea <- genWellTypedExpr ta
         pure (EApp ea be)
    ]

-- Generates a term that is ill-typed at the very top.
genIllTypedApp :: Gen Expr
genIllTypedApp = do
  t1 <- genType
  t2 <- genType
  t3 <- genType
  guard (t1 /= t2)
  f <- genWellTypedExpr t3
  g <- genWellTypedExpr t2
  x <- Gen.text (Range.linear 1 25) Gen.lower
  pure $ EApp (ELam x t1 f) g

------------------------------------------------------------------------

prep :: PropertyT IO (Example Expr Expr, [((Example Expr Expr, [Action]), Env)])
prep = do
  ty <- forAll genType
  input <- forAll (genWellTypedExpr ty)
  let target = evaluate input
      ex = Example (Input input) (Target target)
      env = defaultEnv
      actions = toActions @[] ex
  guard (length actions <= 512)
  let parser = fromActions @[] @(Example Expr Expr)
      result = let f ap [] = empty
                   f ap (a : as) = let p = ap a in pure (p, as)
               in  runStateT (parse f parser actions) env
  pure (ex, result)

-- test that every position belongs only to at most one attention scope
propAEnv :: Property
propAEnv = property $ do
  (_, [(_, Env {..})]) <- prep
  let r = Map.elems $ scopePositions <$> knownScopes aEnv
      c = sort . join $ Set.toList <$> r
      u = Set.toList . Set.unions $ r
  c === u

-- test presence of self attention
propSelfAttention :: Property
propSelfAttention = property $ do
  (_, [(_, Env {..})]) <- prep
  let sa = foldr (\(pos, pos') -> \b -> if pos == pos' then Set.insert pos b else b) Set.empty (attentionMask aEnv)
  sa === keyPaddingMask aEnv

propRoundTrip :: Property
propRoundTrip = property $ do
  (ex, result) <- prep
  [ex] === ((fst . fst) <$> result)

testSTLC :: IO Bool
testSTLC = check propRoundTrip
