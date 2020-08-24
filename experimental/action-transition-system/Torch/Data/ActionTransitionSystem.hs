{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Data.ActionTransitionSystem where

import Bound (Scope, Var, abstract1, fromScope, instantiate1, toScope, (>>>=))
import Control.Applicative (Alternative (..), empty, liftA2, pure, (<|>))
import Control.Concurrent (threadDelay)
import Control.Exception (Exception, try, tryJust)
import Control.Foldl (Fold (..), fold, head)
import qualified Control.Foldl as L
import Control.Lens hiding (cons, uncons)
import Control.Monad (MonadPlus (..), ap, guard, join, mfilter, replicateM, void)
import Control.Monad.Fresh
import Control.Monad.Trans (MonadTrans (lift))
import Control.Monad.Trans.Cont (ContT (..), evalContT)
import Control.Monad.Trans.Free (Free, FreeF (..), FreeT (..), iterTM, runFreeT, wrap)
import Control.Monad.Trans.Maybe (MaybeT (..))
import Control.Monad.Trans.RWS (RWST)
import Control.Monad.Trans.Reader (ReaderT, ask, local, runReaderT)
import Control.Monad.Trans.State.Strict (State, StateT (..), evalStateT, get, modify, put, runStateT)
import Control.Monad.Yoctoparsec (Parser)
import Data.Deriving (deriveEq1, deriveOrd1, deriveShow1)
import Data.Functor.Classes (Eq1 (..), Ord1 (..), Show1 (..), compare1, eq1, showsPrec1)
import Data.Generics.Product
import Data.Generics.Sum
import Data.Kind (Type)
import Data.List as List (elem, filter, intercalate, intersperse, isInfixOf, iterate, length, nub, replicate, sort, take)
import Data.Map as Map (Map, (!))
import qualified Data.Map as Map (adjust, delete, elems, empty, fromList, insert, insertWith, keys, lookup, mapMaybe, null, singleton, toList, unionWith, update)
import Data.Maybe (catMaybes, fromJust, fromMaybe, isJust, listToMaybe)
import Data.Proxy (Proxy (..))
import Data.Set (Set)
import qualified Data.Set as Set (cartesianProduct, elems, empty, filter, findIndex, fromList, insert, member, notMember, singleton, toList, union, unions)
import qualified Data.Set.Ordered as OSet
import Data.Text (Text, pack, unpack)
import Data.Text.Prettyprint.Doc (Doc, hcat, hsep, parens, pretty, sep, vsep, (<+>))
import Data.Text.Prettyprint.Doc.Render.Terminal
import Data.Word (Word64)
import qualified GHC.Exts as Exts
import GHC.Float (float2Double)
import GHC.Generics
import GHC.IO (unsafePerformIO)
import GHC.TypeLits
import qualified Graphics.Vega.VegaLite as Vega
import Hedgehog (Gen, GenT, MonadGen (..), Property, PropertyT, check, checkParallel, discover, distributeT, forAll, property, (===))
import qualified Hedgehog.Gen as Gen hiding (shuffle)
import Hedgehog.Internal.Gen (evalGen)
import qualified Hedgehog.Internal.Seed as Seed
import qualified Hedgehog.Internal.Tree as Tree
import qualified Hedgehog.Range as Range
import Pipes (Effect, ListT (Select), MonadIO (liftIO), each, enumerate, for, runEffect, yield, (>->))
import qualified Pipes as P
import qualified Pipes.Concurrent as P
import qualified Pipes.Group as P
import Pipes.Prelude (drain, foldM, repeatM, take)
import qualified Pipes.Prelude as P
import qualified Pipes.Safe as Safe
import System.IO.Error (ioeGetErrorString)
import System.Mem (performGC)
import System.Random (getStdGen)
import Test.Hspec (hspec, it, shouldBe)
import Text.Printf (printf)
import Torch (ATenTensor, asTensor', asValue, defaultOpts, sparseCooTensor, toDense, toType, withDType, withDevice)
import qualified Torch (ones)
import Torch.Data.Internal (fromInput', toOutput', withBufferLifted)
import Torch.Data.Pipeline (Dataset (..), MapStyleOptions (..), Sample (..), makeListT, mapStyleOpts)
import Torch.Data.StreamedPipeline (MonadBaseControl)
import Torch.Distributions.Bernoulli (fromProbs)
import Torch.Distributions.Distribution (sample)
import qualified Torch.Internal.Cast as ATen
import Torch.Internal.Class (Castable)
import qualified Torch.Internal.Managed.Native as ATen.Managed
import Torch.Typed
import Torch.Vision (grayScale10, grayScale70)
import Prelude hiding (lookup)

-- https://stackoverflow.com/questions/17675054/deserialising-with-ghc-generics?rq=1
-- http://hackage.haskell.org/package/cereal-0.5.8.1/docs/Data-Serialize.html
-- https://hackage.haskell.org/package/attoparsec-0.13.2.4/docs/src/Data.Attoparsec.Text.Internal.html
-- https://hackage.haskell.org/package/yoctoparsec-0.1.0.0/docs/src/Control-Monad-Yoctoparsec.html#Parser
-- https://vaibhavsagar.com/blog/2018/02/04/revisiting-monadic-parsing-haskell/
-- https://github.com/alphaHeavy/protobuf/blob/46cda829cf1e7b6bba2ff450e267fb1a9ace4fb3/src/Data/ProtocolBuffers/Ppr.hs

data Env action a = Env
  { pos :: Pos, -- state
    tEnv :: TEnv action a,
    mEnv :: MEnv,
    rEnv :: REnv,
    aEnv :: AEnv
  }
  deriving (Eq, Ord, Show, Generic)

defaultEnv :: Ord a => Env action a
defaultEnv =
  Env
    { pos = Pos 0,
      tEnv =
        TEnv
          { tokens = mempty,
            validActionMask = mempty,
            types = mempty
          },
      mEnv =
        MEnv
          { meta = mempty,
            metas = mempty
          },
      rEnv =
        REnv
          { parentPos = mempty,
            parents = mempty,
            otherPositions = mempty,
            relations = mempty
          },
      aEnv =
        AEnv
          { currentScope = Nothing,
            knownScopes = mempty,
            attentionMask = mempty,
            keyPaddingMask = mempty
          }
    }

newtype Pos = Pos {unPos :: Int}
  deriving (Eq, Ord, Show, Num, Enum, Generic)

data TEnv action a = TEnv
  { tokens :: Map Pos action, -- writer
    validActionMask :: Map Pos (Set action), -- writer
    types :: Map a Ty -- writer
  }
  deriving (Eq, Ord, Show, Generic)

data MEnv = MEnv
  { meta :: M (Maybe Text), -- state
    metas :: Map Pos (M (Maybe Text)) -- writer
  }
  deriving (Eq, Ord, Show, Generic)

data Relation
  = ChildParentRelation
  | ParentChildRelation
  | SiblingDistRelation {siblingDist :: Int}
  | DistRelation {dist :: Int}
  deriving (Eq, Ord, Show, Generic)

data REnv = REnv
  { parentPos :: [Pos], -- state
    parents :: Map Pos (Set Pos), -- writer
    otherPositions :: Set Pos, -- writer
    relations :: Map (Pos, Pos) (Set Relation) -- writer
  }
  deriving (Eq, Ord, Show, Generic)

type ScopeId = Text

data AttentionScope = AttentionScope
  { scopeKind :: AttentionKind,
    scopeConnections :: Set ScopeId,
    scopePositions :: Set Pos
  }
  deriving (Eq, Ord, Show, Generic)

data AttentionKind
  = BidirectionalAttention
  | BackwardAttention
  | ForwardAttention
  deriving (Eq, Ord, Show, Generic)

data AEnv = AEnv
  { currentScope :: Maybe ScopeId, -- state
    knownScopes :: Map ScopeId AttentionScope, -- writer
    attentionMask :: Set (Pos, Pos), -- writer
    keyPaddingMask :: Set Pos -- writer
  }
  deriving (Eq, Ord, Show, Generic)

data Token action = Pad | Unk | Token action
  deriving (Eq, Ord, Show, Generic, Functor)

data M a = M
  { dataType :: a,
    constructor :: a,
    selector :: a
  }
  deriving (Eq, Ord, Show, Generic, Functor)

instance Semigroup a => Semigroup (M a) where
  x <> y =
    M
      { dataType = dataType x <> dataType y,
        constructor = constructor x <> constructor y,
        selector = selector x <> selector y
      }

instance Monoid a => Monoid (M a) where
  mempty =
    M
      { dataType = mempty,
        constructor = mempty,
        selector = mempty
      }

-- add Universe instance?
-- https://hackage.haskell.org/package/universe-base-1.1.1/docs/Data-Universe-Class.html#t:Finite
data BaseA
  = MaskA
  | -- | left choice for sum types
    L
  | -- | right choice for sum types
    R
  | -- | for growing lists and maybes, e.g.
    Grow
  | -- | for reducing lists and maybes, e.g.
    Reduce
  | -- | integers
    IntA Int
  | -- | texts
    TextA Text
  | -- | booleans
    BoolA Bool
  | -- | characters
    CharA Char
  deriving (Eq, Ord, Show, Generic)

type To t action a = a -> t action

type From env b action a = Parser (StateT env b) action a

choice :: Alternative f => [f a] -> f a
choice = foldr (<|>) empty

option :: Alternative f => a -> f a -> f a
option a p = p <|> pure a

many1 :: Alternative f => f a -> f [a]
many1 p = liftA2 (:) p (many p)
{-# INLINE many1 #-}

manyTill :: Alternative f => f a -> f b -> f [a]
manyTill p end = scan
  where
    scan = (end *> pure []) <|> liftA2 (:) p scan

skipMany :: Alternative f => f a -> f ()
skipMany p = scan
  where
    scan = (p *> scan) <|> pure ()

skipMany1 :: Alternative f => f a -> f ()
skipMany1 p = p *> skipMany p

ancestralRelations :: forall f. Monad f => Pos -> StateT REnv f ()
ancestralRelations pos = get >>= (go . view (field @"parentPos"))
  where
    go [] = pure ()
    go (parentPos : _) =
      let rels' =
            update
              (Map.singleton (pos, parentPos) . Set.singleton $ ChildParentRelation)
              (Map.singleton (parentPos, pos) . Set.singleton $ ParentChildRelation)
       in modify (field @"relations" %~ (update rels'))
    update rels' rels = Map.unionWith Set.union rels' rels

testAncestralRelations =
  let rEnv =
        REnv
          { parentPos = [Pos 0],
            parents = mempty,
            otherPositions = Set.fromList [Pos 0],
            relations = mempty
          }
   in runStateT (ancestralRelations @Identity (Pos 1)) rEnv

siblingRelations :: forall f. Monad f => Int -> Pos -> StateT REnv f ()
siblingRelations maxDist pos = get >>= ap (go . view (field @"parentPos")) (view (field @"parents"))
  where
    go [] parents = pure ()
    go (parentPos : _) parents = do
      let siblings = Set.insert pos $ maybe mempty id $ Map.lookup parentPos parents
          idx = Set.findIndex pos siblings
          step pos' rels' =
            let idx' = Set.findIndex pos' siblings
                dist = idx' - idx
             in if (Prelude.abs dist <= maxDist)
                  then
                    let rels'' =
                          update
                            (Map.singleton (pos, pos') . Set.singleton . SiblingDistRelation $ dist)
                            (Map.singleton (pos', pos) . Set.singleton . SiblingDistRelation $ - dist)
                     in (update rels'' rels')
                  else rels'
          rels = foldr step mempty siblings
      modify (field @"relations" %~ (update rels))
      modify (field @"parents" %~ (Map.insert parentPos siblings))
    update rels' rels = Map.unionWith Set.union rels' rels

testSiblingRelations :: Identity ((), REnv)
testSiblingRelations =
  let rEnv =
        REnv
          { parentPos = [Pos 0],
            parents = mempty,
            otherPositions = Set.fromList [Pos 0],
            relations = mempty
          }
   in runStateT (siblingRelations @Identity 2 (Pos 1)) rEnv

distRelations :: forall f. Monad f => Int -> Pos -> StateT REnv f ()
distRelations maxDist pos = get >>= (go . view (field @"otherPositions"))
  where
    go otherPositions = do
      let otherPositions' = Set.insert pos otherPositions
          idx = Set.findIndex pos otherPositions'
          step pos' rels' =
            let idx' = Set.findIndex pos' otherPositions'
                dist = idx' - idx
             in if (Prelude.abs dist <= maxDist)
                  then
                    let rels'' =
                          update
                            (Map.singleton (pos, pos') . Set.singleton . DistRelation $ dist)
                            (Map.singleton (pos', pos) . Set.singleton . DistRelation $ - dist)
                     in (update rels'' rels')
                  else rels'
          rels = foldr step mempty otherPositions'
      modify (field @"relations" %~ (update rels))
      modify (field @"otherPositions" %~ (const otherPositions'))
    update rels' rels = Map.unionWith Set.union rels' rels

testDistRelations =
  let rEnv =
        REnv
          { parentPos = [Pos 0],
            parents = mempty,
            otherPositions = Set.fromList [Pos 0],
            relations = mempty
          }
   in runStateT (distRelations @Identity 2 (Pos 1)) rEnv

updateRelations :: forall f. Monad f => Int -> Pos -> StateT REnv f ()
updateRelations maxDist pos = do
  ancestralRelations pos
  siblingRelations maxDist pos
  distRelations maxDist pos

testUpdateRelations =
  let rEnv =
        REnv
          { parentPos = [Pos 0],
            parents = mempty,
            otherPositions = Set.fromList [Pos 0],
            relations = mempty
          }
   in runStateT (updateRelations @Identity 2 (Pos 1)) rEnv

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
          attendFrom = List.filter (Set.member thisScopeId . scopeConnections) $ Map.elems knownScopes
          mask' =
            Set.unions
              [ mkMask (scopeKind thisScope) attendSelf attendTo,
                Set.unions $ (\thatScope -> mkMask (scopeKind thatScope) (scopePositions thatScope) attendSelf) <$> attendFrom,
                mkMask (scopeKind thisScope) attendSelf attendSelf
              ]
       in Set.union mask' mask
    go' pos = Map.adjust (field @"scopePositions" %~ (Set.insert pos))

updateMeta :: forall f. Monad f => Pos -> StateT MEnv f ()
updateMeta pos = do
  meta <- (^. field @"meta") <$> get
  modify (field @"metas" %~ Map.insert pos meta)

updateTokens :: forall f action a. (Monad f, Ord action) => action -> [action] -> Pos -> StateT (TEnv action a) f ()
updateTokens t validNextActions pos = do
  modify (field @"validActionMask" %~ Map.insert pos (Set.fromList validNextActions))
  modify (field @"tokens" %~ Map.insert pos t)

updateEnv ::
  forall m env action.
  ( Monad m,
    HasType Pos env,
    HasType AEnv env,
    HasType MEnv env,
    HasType REnv env,
    HasType (TEnv action Int) env,
    Ord action
  ) =>
  [action] ->
  action ->
  StateT env m ()
updateEnv validNextActions action = do
  pos <- (^. typed @Pos) <$> get
  zoom (typed @(TEnv action Int)) . updateTokens action validNextActions $ pos
  zoom (typed @MEnv) . updateMeta $ pos
  zoom (typed @REnv) . updateRelations 3 $ pos
  zoom (typed @AEnv) . updateAttention $ pos

-- | pull an action from the state, feed the parser, and update the environment
next ::
  forall env action b a.
  ( b ~ [],
    HasType Pos env,
    HasType AEnv env,
    HasType MEnv env,
    HasType REnv env,
    HasType (TEnv action Int) env,
    Ord action
  ) =>
  -- | available next actions
  [action] ->
  -- | parser
  Parser (StateT env b) action a ->
  -- | continuation
  (Parser (StateT env b) action a -> StateT [action] (StateT env b) a) ->
  -- | returned stateful computation
  StateT [action] (StateT env b) a
next availableActions parser cont = do
  validNextActions <- lift $ do
    env <- get
    let vals = runStateT (runFreeT parser) env
        f (Pure _, _) = []
        f (Free feed, env') = validateActions availableActions feed env'
    pure $ vals >>= f
  val <- lift . runFreeT $ parser
  case val of
    Pure a -> pure a
    Free feed -> do
      s <- get
      case s of
        [] -> empty
        action : actions -> do
          guard (List.elem action validNextActions)
          lift $ updateEnv validNextActions action
          lift $ modify (typed @Pos %~ (+ 1))
          put actions
          cont (feed action)

next' ::
  forall env action b a.
  ( b ~ []
  ) =>
  -- | available next actions
  [action] ->
  -- | parser
  Parser (StateT env b) action a ->
  -- | continuation
  (Parser (StateT env b) action a -> StateT [action] (StateT env b) a) ->
  -- | returned stateful computation
  StateT () (StateT env b) a
next' availableActions parser cont = do
  validNextActions <- lift $ do
    env <- get
    let vals = runStateT (runFreeT parser) env
        f (Pure _, _) = []
        f (Free feed, env') = validateActions availableActions feed env'
    pure $ vals >>= f
  val <- lift . runFreeT $ parser
  case val of
    Pure a -> pure a
    Free feed -> undefined

-- | given a list of available next actions, return only those that do not end in an empty parsing result
validateActions ::
  forall s action a.
  -- | available next actions
  [action] ->
  -- | parsing feed
  (action -> Parser (StateT s []) action a) ->
  -- | input state
  s ->
  -- | valid next actions
  [action]
validateActions availableActions feed s =
  let f action =
        let val = runFreeT . feed $ action
         in case runStateT val s of
              [] -> Nothing
              _ -> Just action
   in catMaybes $ f <$> availableActions

token :: forall b action. Monad b => Parser b action action
token = wrap $ FreeT . pure . Pure

token' :: forall b t action. (MonadPlus b, AsType t action) => Parser b action t
token' = do
  action <- token
  case projectTyped action of
    Nothing -> empty
    Just t -> pure t

is :: forall b action. (MonadPlus b, Eq action) => action -> Parser b action action
is action = mfilter (== action) token

is' :: forall b t action. (MonadPlus b, Eq t, AsType t action) => t -> Parser b action t
is' t = mfilter (== t) token'

isNot :: forall b action. (MonadPlus b, Eq action) => action -> Parser b action action
isNot action = mfilter (/= action) token

isNot' :: forall b t action. (MonadPlus b, Eq t, AsType t action) => t -> Parser b action t
isNot' t = mfilter (/= t) token'

class ToActions (t :: Type -> Type) (action :: Type) (a :: Type) where
  toActions :: To t action a
  default toActions :: (Generic a, GToActions t action (Rep a)) => To t action a
  toActions = gToActions @t @action . GHC.Generics.from

class FromActions (env :: Type) (b :: Type -> Type) (action :: Type) (a :: Type) where
  fromActions :: From env b action a
  default fromActions :: (Monad b, Generic a, GFromActions env b action (Rep a)) => From env b action a
  fromActions = GHC.Generics.to <$> gFromActions @env @b @action

class GToActions (t :: Type -> Type) (action :: Type) (f :: Type -> Type) where
  gToActions :: forall a. To t action (f a)

class GFromActions (env :: Type) (b :: Type -> Type) (action :: Type) (f :: Type -> Type) where
  gFromActions :: forall a. From env b action (f a)

instance GToActions t action f => GToActions t action (M1 i c f) where
  gToActions = gToActions @t @action . unM1

instance
  ( Monad b,
    HasType MEnv env,
    GFromActions env b action f,
    Datatype d
  ) =>
  GFromActions env b action (D1 d f)
  where
  gFromActions = do
    lift . modify $ typed @MEnv . field @"meta" . field @"dataType" .~ (pure . pack . datatypeName @d $ undefined)
    M1 <$> gFromActions @env @b @action

instance
  ( Monad b,
    HasType Pos env,
    HasType MEnv env,
    HasType REnv env,
    GFromActions env b action f,
    Constructor c
  ) =>
  GFromActions env b action (C1 c f)
  where
  gFromActions = do
    lift . modify $ typed @MEnv . field @"meta" . field @"constructor" .~ (pure . pack . conName @c $ undefined)
    pos <- (^. typed @Pos) <$> lift get
    lift . modify $ typed @REnv . field @"parentPos" %~ (pos :)
    res <- M1 <$> gFromActions @env @b @action
    lift . modify $ typed @REnv . field @"parentPos" %~ tail
    pure res

instance
  ( Monad b,
    HasType MEnv env,
    GFromActions env b action f,
    Selector s
  ) =>
  GFromActions env b action (S1 s f)
  where
  gFromActions = do
    lift . modify $ typed @MEnv . field @"meta" . field @"selector" .~ (pure . pack . selName @s $ undefined)
    M1 <$> gFromActions @env @b @action

instance ToActions t action a => GToActions t action (K1 i a) where
  gToActions = toActions @t @action . unK1

instance
  (Monad b, FromActions env b action a) =>
  GFromActions env b action (K1 i a)
  where
  gFromActions = K1 <$> fromActions @env @b @action

instance Alternative t => GToActions t action U1 where
  gToActions _ = empty

instance Monad b => GFromActions env b action U1 where
  gFromActions = pure U1

instance GToActions t action V1 where
  gToActions v = v `seq` error "GToActions.V1"

instance MonadFail b => GFromActions env b action V1 where
  gFromActions = fail "GFromActions.V1"

instance
  ( Alternative t,
    GToActions t action f,
    GToActions t action g
  ) =>
  GToActions t action (f :*: g)
  where
  gToActions (f :*: g) = gToActions @t @action f <|> gToActions @t @action g

instance
  ( Monad b,
    GFromActions env b action f,
    GFromActions env b action g
  ) =>
  GFromActions env b action (f :*: g)
  where
  gFromActions = (:*:) <$> gFromActions @env @b @action <*> gFromActions @env @b @action

instance
  ( Applicative t,
    Alternative t,
    AsType BaseA action,
    GToActions t action f,
    GToActions t action g
  ) =>
  GToActions t action (f :+: g)
  where
  gToActions (L1 f) = (pure . injectTyped $ L) <|> gToActions @t @action f
  gToActions (R1 g) = (pure . injectTyped $ R) <|> gToActions @t @action g

instance
  ( MonadPlus b,
    AsType BaseA action,
    GFromActions env b action f,
    GFromActions env b action g
  ) =>
  GFromActions env b action (f :+: g)
  where
  gFromActions =
    (is' L >> L1 <$> gFromActions @env @b @action)
      <|> (is' R >> R1 <$> gFromActions @env @b @action)

instance
  ( Alternative t,
    ToActions t action a,
    ToActions t action b
  ) =>
  ToActions t action (a, b)

instance
  ( Monad b,
    HasType Pos env,
    HasType MEnv env,
    HasType REnv env,
    FromActions env b action a,
    FromActions env b action b'
  ) =>
  FromActions env b action (a, b')

instance
  ( Alternative t,
    ToActions t action a,
    ToActions t action b,
    ToActions t action c
  ) =>
  ToActions t action (a, b, c)

instance
  ( Monad b,
    HasType Pos env,
    HasType MEnv env,
    HasType REnv env,
    FromActions env b action a,
    FromActions env b action b',
    FromActions env b action c
  ) =>
  FromActions env b action (a, b', c)

instance
  ( Alternative t,
    ToActions t action a,
    ToActions t action b,
    ToActions t action c,
    ToActions t action d
  ) =>
  ToActions t action (a, b, c, d)

instance
  ( Monad b,
    HasType Pos env,
    HasType MEnv env,
    HasType REnv env,
    FromActions env b action a,
    FromActions env b action b',
    FromActions env b action c,
    FromActions env b action d
  ) =>
  FromActions env b action (a, b', c, d)

instance
  ( Alternative t,
    ToActions t action a,
    ToActions t action b,
    ToActions t action c,
    ToActions t action d,
    ToActions t action e
  ) =>
  ToActions t action (a, b, c, d, e)

instance
  ( Monad b,
    HasType Pos env,
    HasType MEnv env,
    HasType REnv env,
    FromActions env b action a,
    FromActions env b action b',
    FromActions env b action c,
    FromActions env b action d,
    FromActions env b action e
  ) =>
  FromActions env b action (a, b', c, d, e)

instance
  ( Applicative t,
    Alternative t,
    AsType BaseA action,
    ToActions t action a
  ) =>
  ToActions t action [a]
  where
  toActions as = (pure . injectTyped $ Grow) <|> go as
    where
      go [] = pure . injectTyped $ Reduce
      go (a : as) = toActions a <|> go as

instance
  ( MonadPlus b,
    AsType BaseA action,
    FromActions env b action a
  ) =>
  FromActions env b action [a]
  where
  fromActions = is' Grow >> manyTill (fromActions @env @b @action) (is' Reduce)

instance
  ( Applicative t,
    Alternative t,
    AsType BaseA action,
    ToActions t action a
  ) =>
  ToActions t action (Maybe a)
  where
  toActions ma = (pure . injectTyped $ Grow) <|> go ma <|> (pure . injectTyped $ Reduce)
    where
      go Nothing = empty
      go (Just a) = toActions a

instance
  ( MonadPlus b,
    AsType BaseA action,
    FromActions env b action a
  ) =>
  FromActions env b action (Maybe a)
  where
  fromActions = is' Grow >> option Nothing (Just <$> fromActions @env @b @action) >>= (is' Reduce >>) . pure

instance
  ( Applicative t,
    Alternative t,
    AsType BaseA action,
    ToActions t action a,
    ToActions t action b
  ) =>
  ToActions t action (Either a b)

instance
  ( MonadPlus b,
    HasType Pos env,
    HasType MEnv env,
    HasType REnv env,
    AsType BaseA action,
    FromActions env b action a,
    FromActions env b action b'
  ) =>
  FromActions env b action (Either a b')

instance
  ( Applicative t,
    AsType BaseA action
  ) =>
  ToActions t action Text
  where
  toActions = pure . injectTyped . TextA

instance
  ( MonadFail b,
    MonadPlus b,
    AsType BaseA action
  ) =>
  FromActions env b action Text
  where
  fromActions = token' >>= (\case TextA s -> pure s; _ -> fail "text")

instance
  ( Applicative t,
    AsType BaseA action
  ) =>
  ToActions t action Char
  where
  toActions = pure . injectTyped . CharA

instance
  ( MonadFail b,
    MonadPlus b,
    AsType BaseA action
  ) =>
  FromActions env b action Char
  where
  fromActions = token' >>= (\case CharA c -> pure c; _ -> fail "char")

instance
  ( Applicative t,
    AsType BaseA action
  ) =>
  ToActions t action Int
  where
  toActions = pure . injectTyped . IntA

instance
  ( MonadFail b,
    MonadPlus b,
    AsType BaseA action
  ) =>
  FromActions env b action Int
  where
  fromActions = token' >>= (\case IntA i -> pure i; _ -> fail "int")

instance
  ( Applicative t,
    AsType BaseA action
  ) =>
  ToActions t action Bool
  where
  toActions = pure . injectTyped . BoolA

instance
  ( MonadFail b,
    MonadPlus b,
    AsType BaseA action
  ) =>
  FromActions env b action Bool
  where
  fromActions = token' >>= (\case BoolA b -> pure b; _ -> fail "bool")

instance Alternative t => ToActions t action () where
  toActions = const empty

instance Monad b => FromActions env b action () where
  fromActions = pure ()

-- | Runs the parser on the supplied input and returns whether or not the parse succeeded.
-- Results are discarded.
-- TODO: this isn't nice yet. It would be great if there was a stronger signal for failure than just 'mzero'.
-- Parser b i a ~ FreeT ((->) i) b a
checkParser :: forall b i a. MonadPlus b => Parser b i a -> i -> b ()
checkParser p i = do
  val <- runFreeT p
  case val of
    Pure a -> mzero
    Free feed -> void . runFreeT $ feed i

-- FreeT ((->) i) b a ~ StateT (b i) b a ???

-- iterTM specialized to Parser b i a ~ FreeT ((->) i) b a
-- iterTM :: (Monad b, MonadTrans t, Monad (t b)) => ((i -> t b a) -> t b a) -> Parser b i a -> t b a
-- iterTM f p = do
--   val <- lift . runFreeT $ p
--   case val of
--     Pure x -> return x
--     Free y -> f $ \i -> iterTM f (y i)

-- this version of @'iterTM'@ exposes the intermediate step
iterTM' ::
  (MonadTrans t, Monad b, Monad (t b)) =>
  ((i -> Parser b i a) -> (Parser b i a -> t b a) -> t b a) ->
  Parser b i a ->
  t b a
iterTM' f p = do
  val <- lift . runFreeT $ p
  case val of
    Pure x -> return x
    Free y -> f y (iterTM' f)

-- just recursing...
iterTM''' ::
  (Parser b i a -> (Parser b i a -> t b a) -> t b a) ->
  Parser b i a ->
  t b a
iterTM''' f p = f p (iterTM''' f)

-- | Idea here:
-- * instead of reducing @s@, we grow it, starting, e.g., from @[]@
-- * @i -> 'Parser' b i a@ is evaluated against the vocabulary, e.g. @[i]@, and only those @i@'s for which the parser does not fail are considered for continuation. among those, the model decides which to pick
-- * @s@ is the sequence of actions, @[i]@, and, at each step, we feed all previous actions to the model to get the next one
-- * in order to support the prediction, information about the parsing step is encoded in @b@
-- I shall test this idea with a random model.
-- How do to beam search?
-- does this work for training? I guess @next@ would build up a loss term. How do handle batching?
-- fresh values when backtracking: https://hackage.haskell.org/package/monad-gen-0.1.0.0/docs/Control-Monad-Gen.html
-- parse
--   :: forall s b i a
--    . Monad b
--   => ((i -> Parser b i a) -> s -> b (Parser b i a, s))
--   -> Parser b i a
--   -> s
--   -> b (a, s)
-- parse next =
--   -- let f ip ps = StateT $ \s -> do
--   --                 ~(p, s') <- next ip s
--   --                 runStateT (ps p) s'
--   let f ip ps = StateT (next ip) >>= ps
--   in runStateT . iterTM' f
parse ::
  Monad b =>
  (Parser b i a -> (Parser b i a -> StateT s b a) -> StateT s b a) ->
  Parser b i a ->
  s ->
  b (a, s)
parse next = runStateT . iterTM''' next

-- pures :: (Foldable g, Alternative g) => g (FreeF f a (FreeT f m a)) -> g a
-- pures = foldr (\x xs -> case x of Pure a -> pure a <|> xs; _ -> xs) empty

-- frees :: (Foldable g, Alternative g) => g (FreeF f a (FreeT f m a)) -> g (f (FreeT f m a))
-- frees = foldr (\x xs -> case x of Free fb -> pure fb <|> xs; _ -> xs) empty

-- batchedIterTM
--   :: forall f t b a i
--    . (Traversable f, Foldable f, Alternative f, MonadTrans t, Monad b, Monad (t b))
--   => (f a -> f (i -> Parser b i a) -> (f (Parser b i a) -> t b (f a)) -> t b (f a))
--   -> f (Parser b i a)
--   -> t b (f a)
-- batchedIterTM f ps = do
--   vals <- traverse (lift @t . runFreeT) ps
--   f (pures vals) (frees vals) (batchedIterTM f)

-- batchedParse
--   :: forall f s b i a
--    . (Traversable f, Foldable f, Alternative f, Monad b)
--   => (f (i -> Parser b i a) -> s -> b (f (Parser b i a), s))
--   -> f (Parser b i a)
--   -> s
--   -> b (f a, s)
-- batchedParse next = do
--   let f as ip ps = StateT (next ip) >>= ps >>= (pure . (<|> as))
--   runStateT . batchedIterTM f

-- data Stuff = SomeStuff { anInt :: Int, aBool :: Bool, moreStuff :: [Stuff], maybeFoo :: Maybe Foo }
--           --  | NoStuff
--   deriving (Eq, Show, Generic)

-- instance (Alternative t, AsType BaseA action) => ToActions t action Stuff
-- instance (MonadFail b, MonadPlus b, HasType Pos env, HasType MEnv env, HasType REnv env, AsType BaseA action) => FromActions env b action Stuff

-- data Foo = SomeFoo { someText :: Text, stuff :: Stuff }
--         --  | NoFoo
--   deriving (Eq, Show, Generic)

-- instance (Alternative t, AsType BaseA action) => ToActions t action Foo
-- instance (MonadFail b, MonadPlus b, HasType Pos env, HasType MEnv env, HasType REnv env, AsType BaseA action) => FromActions env b action Foo

-- test :: ([Action], [((Foo, [Action]), Env Action)], [((), Env Action)])
-- test =
--   let env = defaultEnv
--       stuff 0 = []
--       stuff n = SomeStuff n True [] Nothing : stuff (n - 1)
--       foo 0 = SomeFoo "a" $ SomeStuff 0 False [SomeStuff 2 True [] Nothing] Nothing
--       foo n = SomeFoo "a" $ SomeStuff n ((==0) . (`rem` 3) $ n) [SomeStuff 2 False (stuff n) Nothing] (Just $ foo (n - 1))
--       challenge = foo 2
--       actions = toActions @[] challenge
--       parser = fromActions @(Env Action) @[]
--       result' = -- let f ap [] = empty
--                 --     f ap (a : as) = let p = ap a in do
--                 --       env' <- get
--                 --       pure $ unsafePerformIO $ print
--                 --         ( a
--                 --         , view (field @"mEnv" . field @"meta") env'
--                 --         , view (field @"pos") env'
--                 --         , view (field @"rEnv" . field @"parentPos") env'
--                 --         -- , view (field @"rEnv" . field @"relations") env'
--                 --         ) >> pure (p, as)
--                 -- in runStateT (parse f parser actions) env
--                 runStateT (parse (next []) parser actions) env
--   in (actions, result', runStateT (checkParser parser (IntA 1)) env)

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

data Ty
  = Arr Ty Ty
  | Nat
  deriving (Eq, Ord, Show, Generic)

data TyA
  = ArrA
  | NatA
  deriving (Eq, Ord, Show, Generic)

instance (Alternative t, AsType TyA action) => ToActions t action Ty where
  toActions (Arr ty ty') = (pure . injectTyped $ ArrA) <|> toActions ty <|> toActions ty'
  toActions Nat = pure . injectTyped $ NatA

instance (MonadFail b, MonadPlus b, HasType Pos env, HasType MEnv env, HasType REnv env, AsType TyA action) => FromActions env b action Ty where
  fromActions =
    (is' ArrA >> Arr <$> fromActions <*> fromActions)
      <|> (is' NatA >> pure Nat)

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
data Exp a
  = -- | Variable.
    Var a
  | -- | Lambda abstraction.
    Lam {ty :: Ty, lambdaTerm :: (Scope () Exp a)}
  | -- | Term application.
    (:@) {function :: Exp a, argument :: Exp a}
  | Succ (Exp a)
  | Zero
  deriving (Functor, Foldable, Traversable, Generic)

infixl 9 :@

data ExpA
  = VarA
  | LamA
  | AppA
  | SuccA
  | ZeroA
  deriving (Eq, Ord, Show, Generic)

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
  Succ e >>= f = Succ (e >>= f)
  Zero >>= _ = Zero

deriveEq1 ''Exp
deriveOrd1 ''Exp
deriveShow1 ''Exp

-- TODO: test using https://github.com/hedgehogqa/haskell-hedgehog-classes
instance Eq a => Eq (Exp a) where (==) = eq1

instance Ord a => Ord (Exp a) where compare = compare1

instance Show a => Show (Exp a) where showsPrec = showsPrec1

instance (Alternative t, AsType BaseA action, AsType ExpA action, AsType TyA action, ToActions t action a) => ToActions t action (Var () (Exp a))

instance (Alternative t, AsType BaseA action, AsType ExpA action, AsType TyA action, ToActions t action a) => ToActions t action (Scope () Exp a)

instance (Alternative t, AsType BaseA action, AsType ExpA action, AsType TyA action, ToActions t action a) => ToActions t action (Exp a) where
  toActions (Var a) = (pure . injectTyped $ VarA) <|> toActions a
  toActions (Lam ty e) = (pure . injectTyped $ LamA) <|> toActions ty <|> toActions e
  toActions ((:@) f e) = (pure . injectTyped $ AppA) <|> toActions f <|> toActions e
  toActions (Succ e) = (pure . injectTyped $ SuccA) <|> toActions e
  toActions Zero = pure . injectTyped $ ZeroA

instance (Ord a, MonadFail b, MonadPlus b, HasType Pos env, HasType MEnv env, HasType REnv env, HasType (TEnv action Int) env, AsType BaseA action, AsType ExpA action, AsType TyA action, FromActions env b action a) => FromActions env b action (Var () (Exp a))

instance (Ord a, MonadFail b, MonadPlus b, HasType Pos env, HasType MEnv env, HasType REnv env, HasType (TEnv action Int) env, AsType BaseA action, AsType ExpA action, AsType TyA action, FromActions env b action a) => FromActions env b action (Scope () Exp a)

instance (Ord a, MonadFail b, MonadPlus b, HasType Pos env, HasType MEnv env, HasType REnv env, HasType (TEnv action Int) env, AsType BaseA action, AsType ExpA action, AsType TyA action, FromActions env b action a) => FromActions env b action (Exp a) where
  fromActions = do
    e <-
      (is' VarA >> Var <$> fromActions)
        <|> (is' LamA >> Lam <$> fromActions <*> fromActions)
        <|> (is' AppA >> (:@) <$> fromActions <*> fromActions)
        <|> (is' SuccA >> Succ <$> fromActions)
        <|> (is' ZeroA >> pure Zero)
    -- types <- (^. typed @(TEnv action Int) . field @"types") <$> (lift get)
    -- let _ = typeCheck types e
    pure e

-- fromActions = do
--   s <- lift get
--   GHC.Generics.to <$> gFromActions @[]

lam :: forall a. Eq a => Ty -> a -> Exp a -> Exp a
lam ty uname bind = Lam ty (abstract1 uname bind)

-- | Smart constructor that converts the given positive integer to a corresponding Nat.
nat :: forall a n. (Num n, Eq n) => n -> Exp a
nat 0 = Zero
nat n = Succ $ nat (n -1)

-- | Compute the normal form of an expression.
nf :: forall a. Exp a -> Exp a
nf e@Var {} = e
nf (Lam ty b) = Lam ty (toScope . nf . fromScope $ b)
nf (f :@ a) = case whnf f of
  Lam ty b -> nf (instantiate1 a b)
  f' -> nf f' :@ nf a
nf (Succ e) = Succ (nf e)
nf e@Zero = e

-- | Reduce a term to weak head normal form.
whnf :: forall a. Exp a -> Exp a
whnf e@Var {} = e
whnf e@Lam {} = e
whnf (f :@ a) = case whnf f of
  Lam _ b -> whnf (instantiate1 a b)
  f' -> f' :@ a
whnf e@Succ {} = e
whnf e@Zero = e

type TyM a = MaybeT (Fresh a)

assertTy :: Ord a => Map a Ty -> Exp a -> Ty -> TyM a ()
assertTy env e t = (== t) <$> typeCheck env e >>= guard

typeCheck :: forall a. Ord a => Map a Ty -> Exp a -> TyM a Ty
typeCheck _ Zero = return Nat
typeCheck env (Succ e) = assertTy env e Nat >> return Nat
typeCheck env (Var a) = MaybeT . return $ Map.lookup a env
typeCheck env (f :@ a) =
  typeCheck env f >>= \case
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
  ppr p (Succ e) = parensIf (p > 0) $ do
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

render :: forall a. (Ord a, Enum a, Pretty a a) => Exp a -> IO ()
render = render' (toEnum 0)

render' :: forall a. (Ord a, Enum a, Pretty a a) => a -> Exp a -> IO ()
render' a e = putDoc $ pprint a e

testBound :: IO ()
testBound = do
  let term :: Exp Char = (lam Nat 'a' (Succ $ Var 'a')) :@ Zero
      term' :: Exp Char = (lam Nat 'b' (Succ $ Var 'b')) :@ Zero
  print term -- (:@) {function = Lam {ty = Nat, lambdaTerm = Scope (Succ (Var (B ())))}, argument = Zero}
  print term' -- (:@) {function = Lam {ty = Nat, lambdaTerm = Scope (Succ (Var (B ())))}, argument = Zero}
  putDoc . vsep $ ((pprint ('x' :: Char)) <$> [term, term']) <> [mempty] -- (λx. Sx) Z, (λx. Sx) Z
  print $ term == term' -- True
  print $ runFresh . runMaybeT . typeCheck Map.empty $ term -- Just Nat
  print $ toActions @[] @Action $ term -- [R,L,L,R,R,R,R,L,L,L,L,R,R,R]
  print $ whnf term -- Succ Zero
  print $ nf term -- Succ Zero
  print $ toActions @[] @Action $ nf term -- [R,R,L,R,R,R]

-- | Monad transformer stack for term and type generation.
-- Notably, contains the @FreshT@ transformer for generating fresh variable names
-- and a @ReaderT@ for the environment of scoped typed @Var@s.
type GTyM a = ReaderT (Map Ty [Exp a]) (FreshT a Gen)

-- | Generate a type.
-- We cannot generate an expression without generating a type for it first.
genTy :: forall m. MonadGen m => m Ty
genTy =
  Gen.recursive
    Gen.choice
    [ -- non-recursive generators
      pure Nat
    ]
    [ -- recursive generators
      Arr <$> genTy <*> genTy
    ]

-- | Finalize generation by running the monad transformers for the environment
-- and the fresh variable name computation.
genWellTypedExp :: forall a. (Eq a, Enum a) => Ty -> Gen (Exp a)
genWellTypedExp ty = runFreshT $ runReaderT (genWellTypedExp' ty) mempty

-- | Main recursive mechanism for genersating expressions for a given type.
genWellTypedExp' :: forall a. Eq a => Ty -> GTyM a (Exp a)
genWellTypedExp' ty =
  Gen.shrink shrinkExp $
    genWellTypedPath ty
      <|> Gen.recursive
        Gen.choice
        [ -- non-recursive generators
          genWellTypedExp'' ty
        ]
        [ -- recursive generators
          genWellTypedApp ty,
          genWellTypedExp''' ty
        ]

shrinkExp :: forall a. Exp a -> [Exp a]
shrinkExp (f :@ a) = case whnf f of
  Lam _ b -> [whnf (instantiate1 a b)]
  _ -> []
shrinkExp _ = []

-- | Pattern match on a given type and produce a corresponding term.
-- @Lam@ is generated from @Arr@ by first obtaining a fresh variable name for @Var@ and
-- then calling the @lam@ smart constructor on an expression that
-- was produced for an environment to which @Var@ was added.
-- A term of type @Nat@ is generated by converting a random integer through induction.
genWellTypedExp'' :: forall a. Eq a => Ty -> GTyM a (Exp a)
genWellTypedExp'' (Arr ty ty') = do
  uname <- fresh
  lam ty uname <$> local (insertVar uname ty) (genWellTypedExp' ty')
genWellTypedExp'' Nat = nat <$> Gen.int (Range.linear 0 10)

genWellTypedExp''' :: forall a. Eq a => Ty -> GTyM a (Exp a)
genWellTypedExp''' Nat = Succ <$> genWellTypedExp' Nat
genWellTypedExp''' ty = genWellTypedExp' ty

-- | Add @Var@ of given type to the given env so that it can be used for sampling later.
insertVar :: forall a. Eq a => a -> Ty -> Map Ty [Exp a] -> Map Ty [Exp a]
insertVar uname ty =
  Map.insertWith (<>) ty [Var uname] . fmap (List.filter (/= Var uname))

-- | Generate app by first producing type and value of the argument
-- and then generating a compatible @Lam@.
genWellTypedApp :: forall a. Eq a => Ty -> GTyM a (Exp a)
genWellTypedApp ty = do
  tg <- genKnownTypeMaybe
  eg <- genWellTypedExp' tg
  let tf = Arr tg ty
  ef <- genWellTypedExp' tf
  pure (ef :@ eg)

-- | Try to look up a known expression of the desired type from the environment.
-- This does not always succceed, throwing `empty` when unavailable.
genWellTypedPath :: forall a. Ty -> GTyM a (Exp a)
genWellTypedPath ty = do
  paths <- ask
  case fromMaybe [] (Map.lookup ty paths) of
    [] -> empty
    es -> Gen.element es

-- | Generate either known types from the environment or new types.
genKnownTypeMaybe :: forall a. GTyM a Ty
genKnownTypeMaybe = do
  known <- ask
  if Map.null known
    then genTy
    else
      Gen.frequency
        [ (2, Gen.element $ Map.keys known),
          (1, genTy)
        ]

------------------------------------------------------------------------

data Example a b = Example
  { input :: Input a,
    target :: Target b
  }
  deriving (Eq, Ord, Show, Generic)

instance (Alternative t, ToActions t action a, ToActions t action b) => ToActions t action (Example a b)

instance (Monad b, HasType Pos env, HasType MEnv env, HasType REnv env, HasType AEnv env, FromActions env b action a, FromActions env b action b') => FromActions env b action (Example a b')

newtype Input a = Input a deriving (Eq, Ord, Show, Generic)

newtype Target a = Target a deriving (Eq, Ord, Show, Generic)

instance ToActions t action a => ToActions t action (Input a)

instance (Monad b, HasType AEnv env, FromActions env b action a) => FromActions env b action (Input a) where
  fromActions = do
    lift . modify $ typed @AEnv . field @"currentScope" .~ pure "input"
    lift . modify $ typed @AEnv . field @"knownScopes" %~ go "input"
    Input <$> fromActions
    where
      go scopeId = Map.insert scopeId (AttentionScope BidirectionalAttention (Set.singleton "input") mempty)

instance ToActions t action a => ToActions t action (Target a)

instance (Monad b, HasType AEnv env, FromActions env b action a) => FromActions env b action (Target a) where
  fromActions = do
    lift . modify $ typed @AEnv . field @"currentScope" .~ pure "target"
    lift . modify $ typed @AEnv . field @"knownScopes" %~ go "target"
    Target <$> fromActions
    where
      go scopeId = Map.insert scopeId (AttentionScope BackwardAttention (Set.fromList ["input", "target"]) mempty)

------------------------------------------------------------------------

data Action
  = BaseAction BaseA
  | ExpAction ExpA
  | TyAction TyA
  deriving (Eq, Ord, Show, Generic)

prep :: PropertyT IO (Ty, Example (Exp Int) (Exp Int), [((Example (Exp Int) (Exp Int), [Action]), Env Action Int)])
prep = do
  ty <- forAll genTy
  input <- forAll (genWellTypedExp ty)
  let target = nf input
      ex = Example (Input input) (Target target)
      env = defaultEnv
      actions = toActions @[] ex
  guard (List.length actions <= 512)
  let parser = fromActions @(Env Action Int) @[] @Action @(Example (Exp Int) (Exp Int))
      result = runStateT (parse (next []) parser actions) env
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
    (dataTypePaddingIdx :: Nat)
    (dataTypeNumEmbeds :: Nat)
    (constructorPaddingIdx :: Nat)
    (constructorNumEmbeds :: Nat)
    (selectorPaddingIdx :: Nat)
    (selectorNumEmbeds :: Nat)
    (relationPaddingIdx :: Nat)
    (relationNumEmbeds :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  where
  RATransformerMLMSpec ::
    forall numAttnLayers numHeads headDim ffnDim tokenPaddingIdx tokenNumEmbeds dataTypePaddingIdx dataTypeNumEmbeds constructorPaddingIdx constructorNumEmbeds selectorPaddingIdx selectorNumEmbeds relationPaddingIdx relationNumEmbeds dtype device.
    { ratDropoutSpec :: DropoutSpec,
      ratLayerSpec :: TransformerLayerSpec (headDim * numHeads) (headDim * numHeads) (headDim * numHeads) numHeads ffnDim dtype device
    } ->
    RATransformerMLMSpec numAttnLayers numHeads headDim ffnDim tokenPaddingIdx tokenNumEmbeds dataTypePaddingIdx dataTypeNumEmbeds constructorPaddingIdx constructorNumEmbeds selectorPaddingIdx selectorNumEmbeds relationPaddingIdx relationNumEmbeds dtype device
  deriving (Show)

data
  RATransformerMLM
    (numAttnLayers :: Nat)
    (numHeads :: Nat)
    (headDim :: Nat)
    (ffnDim :: Nat)
    (tokenPaddingIdx :: Nat)
    (tokenNumEmbeds :: Nat)
    (dataTypePaddingIdx :: Nat)
    (dataTypeNumEmbeds :: Nat)
    (constructorPaddingIdx :: Nat)
    (constructorNumEmbeds :: Nat)
    (selectorPaddingIdx :: Nat)
    (selectorNumEmbeds :: Nat)
    (relationPaddingIdx :: Nat)
    (relationNumEmbeds :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  where
  RATransformerMLM ::
    forall numAttnLayers numHeads headDim ffnDim tokenPaddingIdx tokenNumEmbeds dataTypePaddingIdx dataTypeNumEmbeds constructorPaddingIdx constructorNumEmbeds selectorPaddingIdx selectorNumEmbeds relationPaddingIdx relationNumEmbeds dtype device.
    { ratTokenEmbedding :: Embedding ('Just tokenPaddingIdx) tokenNumEmbeds (Div (headDim * numHeads) 4) 'Learned dtype device,
      ratDataTypeEmbedding :: Embedding ('Just dataTypePaddingIdx) dataTypeNumEmbeds (Div (headDim * numHeads) 4) 'Learned dtype device,
      ratConstructorEmbedding :: Embedding ('Just constructorPaddingIdx) constructorNumEmbeds (Div (headDim * numHeads) 4) 'Learned dtype device,
      ratSelectorEmbedding :: Embedding ('Just selectorPaddingIdx) selectorNumEmbeds (Div (headDim * numHeads) 4) 'Learned dtype device,
      ratRelationEmbedding :: Embedding ('Just relationPaddingIdx) relationNumEmbeds headDim 'Learned dtype device,
      ratDropout :: Dropout,
      ratLayers :: HList (HReplicateR numAttnLayers (TransformerLayer (headDim * numHeads) (headDim * numHeads) (headDim * numHeads) numHeads ffnDim dtype device)),
      ratProj :: Linear (headDim * numHeads) tokenNumEmbeds dtype device
    } ->
    RATransformerMLM numAttnLayers numHeads headDim ffnDim tokenPaddingIdx tokenNumEmbeds dataTypePaddingIdx dataTypeNumEmbeds constructorPaddingIdx constructorNumEmbeds selectorPaddingIdx selectorNumEmbeds relationPaddingIdx relationNumEmbeds dtype device
  deriving (Generic)

instance
  ( All KnownNat '[headDim, numHeads, tokenPaddingIdx, tokenNumEmbeds, dataTypePaddingIdx, dataTypeNumEmbeds, constructorPaddingIdx, constructorNumEmbeds, selectorPaddingIdx, selectorNumEmbeds, relationPaddingIdx, relationNumEmbeds],
    tokenPaddingIdx <= tokenNumEmbeds,
    1 <= (tokenNumEmbeds - tokenPaddingIdx),
    dataTypePaddingIdx <= dataTypeNumEmbeds,
    1 <= (dataTypeNumEmbeds - dataTypePaddingIdx),
    constructorPaddingIdx <= constructorNumEmbeds,
    1 <= (constructorNumEmbeds - constructorPaddingIdx),
    selectorPaddingIdx <= selectorNumEmbeds,
    1 <= (selectorNumEmbeds - selectorPaddingIdx),
    relationPaddingIdx <= relationNumEmbeds,
    1 <= (relationNumEmbeds - relationPaddingIdx),
    HReplicate numAttnLayers (TransformerLayerSpec (headDim * numHeads) (headDim * numHeads) (headDim * numHeads) numHeads ffnDim dtype device),
    Randomizable
      (HList (HReplicateR numAttnLayers (TransformerLayerSpec (headDim * numHeads) (headDim * numHeads) (headDim * numHeads) numHeads ffnDim dtype device)))
      (HList (HReplicateR numAttnLayers (TransformerLayer (headDim * numHeads) (headDim * numHeads) (headDim * numHeads) numHeads ffnDim dtype device))),
    KnownDType dtype,
    RandDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  Randomizable
    ( RATransformerMLMSpec
        numAttnLayers
        numHeads
        headDim
        ffnDim
        tokenPaddingIdx
        tokenNumEmbeds
        dataTypePaddingIdx
        dataTypeNumEmbeds
        constructorPaddingIdx
        constructorNumEmbeds
        selectorPaddingIdx
        selectorNumEmbeds
        relationPaddingIdx
        relationNumEmbeds
        dtype
        device
    )
    ( RATransformerMLM
        numAttnLayers
        numHeads
        headDim
        ffnDim
        tokenPaddingIdx
        tokenNumEmbeds
        dataTypePaddingIdx
        dataTypeNumEmbeds
        constructorPaddingIdx
        constructorNumEmbeds
        selectorPaddingIdx
        selectorNumEmbeds
        relationPaddingIdx
        relationNumEmbeds
        dtype
        device
    )
  where
  sample RATransformerMLMSpec {..} =
    RATransformerMLM
      <$> Torch.Typed.sample (LearnedEmbeddingWithRandomInitSpec @('Just tokenPaddingIdx))
      <*> Torch.Typed.sample (LearnedEmbeddingWithRandomInitSpec @('Just dataTypePaddingIdx))
      <*> Torch.Typed.sample (LearnedEmbeddingWithRandomInitSpec @('Just constructorPaddingIdx))
      <*> Torch.Typed.sample (LearnedEmbeddingWithRandomInitSpec @('Just selectorPaddingIdx))
      <*> Torch.Typed.sample (LearnedEmbeddingWithRandomInitSpec @('Just relationPaddingIdx))
      <*> Torch.Typed.sample ratDropoutSpec
      <*> Torch.Typed.sample (hreplicate @numAttnLayers ratLayerSpec)
      <*> Torch.Typed.sample LinearSpec

data
  RAFoldLayers
    (batchSize :: Nat)
    (headDim :: Nat)
    (seqLen :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat)) = RAFoldLayers
  { raflTrain :: Bool,
    raflAttentionMask :: Tensor device dtype '[batchSize, seqLen, seqLen],
    raflKeyPaddingMask :: Tensor device 'Bool '[batchSize, seqLen],
    raflKeyRelations :: Tensor device dtype '[batchSize, seqLen, seqLen, headDim],
    raflValueRelations :: Tensor device dtype '[batchSize, seqLen, seqLen, headDim]
  }

instance
  ( 1 <= numHeads,
    embedDim ~ (headDim * numHeads),
    All KnownNat '[embedDim, numHeads, seqLen, batchSize, headDim],
    IsSuffixOf '[embedDim] '[batchSize, seqLen, embedDim],
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    dtype ~ SumDType dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  Apply'
    (RAFoldLayers batchSize headDim seqLen dtype device)
    ( TransformerLayer embedDim embedDim embedDim numHeads ffnDim dtype device,
      IO (Tensor device dtype '[batchSize, seqLen, embedDim])
    )
    (IO (Tensor device dtype '[batchSize, seqLen, embedDim]))
  where
  apply' RAFoldLayers {..} (layer, mx) =
    mx >>= \x ->
      transformerLayer
        layer
        raflTrain
        (Just raflAttentionMask)
        (Just raflKeyPaddingMask)
        (Just raflKeyRelations)
        (Just raflValueRelations)
        x
        x
        x

raTransformerMLM ::
  forall
    numAttnLayers
    numHeads
    headDim
    ffnDim
    tokenPaddingIdx
    tokenNumEmbeds
    dataTypePaddingIdx
    dataTypeNumEmbeds
    constructorPaddingIdx
    constructorNumEmbeds
    selectorPaddingIdx
    selectorNumEmbeds
    relationPaddingIdx
    relationNumEmbeds
    relDim
    embedDim
    seqLen
    batchSize
    dtype
    device.
  ( All
      KnownNat
      '[ tokenPaddingIdx,
         dataTypePaddingIdx,
         constructorPaddingIdx,
         selectorPaddingIdx,
         relationPaddingIdx,
         seqLen,
         batchSize
       ],
    tokenPaddingIdx + 1 <= tokenNumEmbeds,
    dataTypePaddingIdx + 1 <= dataTypeNumEmbeds,
    constructorPaddingIdx + 1 <= constructorNumEmbeds,
    selectorPaddingIdx + 1 <= selectorNumEmbeds,
    relationPaddingIdx + 1 <= relationNumEmbeds,
    embedDim ~ (headDim * numHeads),
    embedDim ~ (Div embedDim 4 + (Div embedDim 4 + (Div embedDim 4 + Div embedDim 4))),
    1 <= seqLen,
    HFoldrM
      IO
      (RAFoldLayers batchSize headDim seqLen dtype device)
      (Tensor device dtype '[batchSize, seqLen, embedDim])
      (HReplicateR numAttnLayers (TransformerLayer embedDim embedDim embedDim numHeads ffnDim dtype device))
      (Tensor device dtype '[batchSize, seqLen, embedDim]),
    BasicArithmeticDTypeIsValid device dtype,
    ComparisonDTypeIsValid device dtype,
    ComparisonDTypeIsValid device 'Int64,
    SumDType dtype ~ dtype,
    SumDTypeIsValid device dtype,
    KnownDType dtype,
    KnownDevice device
  ) =>
  RATransformerMLM
    numAttnLayers
    numHeads
    headDim
    ffnDim
    tokenPaddingIdx
    tokenNumEmbeds
    dataTypePaddingIdx
    dataTypeNumEmbeds
    constructorPaddingIdx
    constructorNumEmbeds
    selectorPaddingIdx
    selectorNumEmbeds
    relationPaddingIdx
    relationNumEmbeds
    dtype
    device ->
  -- | training flag
  Bool ->
  RATransformerMLMInput batchSize seqLen relDim dtype device ->
  IO (Tensor device dtype '[batchSize, seqLen, tokenNumEmbeds])
raTransformerMLM RATransformerMLM {..} train RATransformerMLMInput {..} = do
  let maskedTokens = embed ratTokenEmbedding ratMaskedTokens
      dataTypes = embed ratDataTypeEmbedding . dataType $ ratMeta
      constructors = embed ratConstructorEmbedding . constructor $ ratMeta
      selectors = embed ratSelectorEmbedding . selector $ ratMeta
      input = cat @2 $ maskedTokens :. dataTypes :. constructors :. selectors :. HNil
      relations = sumDim @3 $ embed ratRelationEmbedding ratRelations
  forward ratProj
    <$> hfoldrM
      (RAFoldLayers train ratAttentionMask ratKeyPaddingMask relations relations)
      input
      ratLayers

instance
  ( All
      KnownNat
      '[ tokenPaddingIdx,
         dataTypePaddingIdx,
         constructorPaddingIdx,
         selectorPaddingIdx,
         relationPaddingIdx,
         seqLen,
         batchSize
       ],
    tokenPaddingIdx + 1 <= tokenNumEmbeds,
    dataTypePaddingIdx + 1 <= dataTypeNumEmbeds,
    constructorPaddingIdx + 1 <= constructorNumEmbeds,
    selectorPaddingIdx + 1 <= selectorNumEmbeds,
    relationPaddingIdx + 1 <= relationNumEmbeds,
    embedDim ~ (headDim * numHeads),
    embedDim ~ (Div embedDim 4 + (Div embedDim 4 + (Div embedDim 4 + Div embedDim 4))),
    1 <= seqLen,
    HFoldrM
      IO
      (RAFoldLayers batchSize headDim seqLen dtype device)
      (Tensor device dtype '[batchSize, seqLen, embedDim])
      (HReplicateR numAttnLayers (TransformerLayer embedDim embedDim embedDim numHeads ffnDim dtype device))
      (Tensor device dtype '[batchSize, seqLen, embedDim]),
    BasicArithmeticDTypeIsValid device dtype,
    ComparisonDTypeIsValid device dtype,
    ComparisonDTypeIsValid device 'Int64,
    SumDType dtype ~ dtype,
    SumDTypeIsValid device dtype,
    KnownDType dtype,
    KnownDevice device
  ) =>
  HasForward
    ( RATransformerMLM
        numAttnLayers
        numHeads
        headDim
        ffnDim
        tokenPaddingIdx
        tokenNumEmbeds
        dataTypePaddingIdx
        dataTypeNumEmbeds
        constructorPaddingIdx
        constructorNumEmbeds
        selectorPaddingIdx
        selectorNumEmbeds
        relationPaddingIdx
        relationNumEmbeds
        dtype
        device
    )
    (RATransformerMLMInput batchSize seqLen relDim dtype device)
    (Tensor device dtype '[batchSize, seqLen, tokenNumEmbeds])
  where
  forward model input = unsafePerformIO $ raTransformerMLM model False input
  forwardStoch model input = raTransformerMLM model True input

data RATransformerMLMBatch batchSize seqLen numEmbeds relDim dtype device = RATransformerMLMBatch
  { ratInput :: RATransformerMLMInput batchSize seqLen relDim dtype device,
    ratTarget :: RATransformerMLMTarget batchSize seqLen numEmbeds device
  }
  deriving (Show, Generic)

data RATransformerMLMInput batchSize seqLen relDim dtype device = RATransformerMLMInput
  { -- | masked tokens
    ratMaskedTokens :: Tensor device 'Int64 '[batchSize, seqLen],
    -- | meta
    ratMeta :: M (Tensor device 'Int64 '[batchSize, seqLen]),
    -- | relations
    ratRelations :: Tensor device 'Int64 '[batchSize, seqLen, seqLen, relDim],
    -- | attention mask (0 where attention is allowed, -inf everywhere else)
    ratAttentionMask :: Tensor device dtype '[batchSize, seqLen, seqLen],
    -- | key padding mask (True for padding, False everywhere else)
    ratKeyPaddingMask :: Tensor device 'Bool '[batchSize, seqLen]
  }
  deriving (Show, Generic)

data RATransformerMLMTarget batchSize seqLen numEmbeds device = RATransformerMLMTarget
  { -- | target tokens
    ratTargetTokens :: Tensor device 'Int64 '[batchSize, seqLen],
    -- | token mask
    ratTokenMask :: Tensor device 'Bool '[batchSize, seqLen],
    -- | input scope mask
    ratInputScopeMask :: Tensor device 'Bool '[batchSize, seqLen],
    -- | target scope mask
    ratTargetScopeMask :: Tensor device 'Bool '[batchSize, seqLen],
    -- | valid next action mask
    ratInvalidTokenMask :: Tensor device 'Bool '[batchSize, seqLen, numEmbeds]
  }
  deriving (Show, Generic)

loss ::
  forall batchSize seqLen numEmbeds device dtype.
  ( StandardFloatingPointDTypeValidation device dtype,
    MeanDTypeValidation device dtype,
    KnownDType dtype,
    KnownDevice device,
    KnownShape '[batchSize, seqLen, numEmbeds]
  ) =>
  Tensor device 'Bool '[batchSize, seqLen, numEmbeds] ->
  Tensor device 'Bool '[batchSize, seqLen] ->
  Tensor device dtype '[batchSize, seqLen, numEmbeds] ->
  Tensor device 'Int64 '[batchSize, seqLen] ->
  Tensor device dtype '[]
loss invalidTokenMask selectionMask logits target =
  let logProbs = logSoftmax @2 . maskedFill invalidTokenMask (-1 / 0 :: Double) $ logits
      logLikelihood = squeezeDim @2 $ gatherDim @2 (unsqueeze @2 target) logProbs
      selected = maskedSelect selectionMask logLikelihood
      meanLogLikelihood = case selected of
        UnknownShapeTensor t -> unsafeMeanAll t
   in (- meanLogLikelihood)

------------------------------------------------------------------------

mkActions ::
  forall seqLen m.
  (KnownNat seqLen, Monad m) =>
  Seed.Seed ->
  m [Action]
mkActions seed = (flip evalStateT) seed . sample' $ do
  ty <- genTy
  input <- genWellTypedExp @Int ty
  let target = nf input
      ex = Example (Input input) (Target target)
      actions = toActions @[] ex
  guard (List.length actions <= natValI @seqLen)
  pure actions

mkRATransformerMLMItem ::
  forall seqLen numEmbeds relDim dtype device pMask m.
  ( All KnownNat '[seqLen, numEmbeds, relDim],
    SumDTypeIsValid device 'Int64,
    ComparisonDTypeIsValid device 'Int64,
    KnownDType dtype,
    KnownDevice device,
    Scalar pMask,
    MonadIO m,
    MonadFail m,
    Alternative m
  ) =>
  pMask ->
  pMask ->
  [Action] ->
  m (RATransformerMLMBatch 1 seqLen numEmbeds relDim dtype device)
mkRATransformerMLMItem pMaskInput pMaskTarget actions = do
  let fromJust' = maybe (fail "invalid list shapes") pure
      availableActions =
        [ injectTyped MaskA,
          injectTyped L,
          injectTyped R,
          injectTyped VarA,
          injectTyped LamA,
          injectTyped AppA,
          injectTyped SuccA,
          injectTyped ZeroA,
          injectTyped ArrA,
          injectTyped NatA
        ]
      parser = fromActions @(Env Action Int) @[] @Action @(Example (Exp Int) (Exp Int))
  env <- case runStateT (parse (next availableActions) parser actions) defaultEnv of
    [(_, env)] -> pure env
    _ -> fail "invalid parsing state"
  let tokenVocab =
        OSet.fromList
          ( [Pad, Unk]
              <> (Token <$> availableActions)
          )
      dataTypeVocab :: OSet.OSet (Token Text) =
        OSet.fromList
          [ Pad,
            Unk,
            Token "Example",
            Token "Exp",
            Token "Ty",
            Token "Scope",
            Token "Var"
          ]
      constructorVocab :: OSet.OSet (Token Text) =
        OSet.fromList
          [ Pad,
            Unk,
            Token "Example",
            Token ":@",
            Token "Lam",
            Token "Arr",
            Token "Nat",
            Token "Scope",
            Token "Var",
            Token "B",
            Token "Zero",
            Token "Succ",
            Token "F"
          ]
      selectorVocab :: OSet.OSet (Token Text) =
        OSet.fromList
          [ Pad,
            Unk,
            Token "",
            Token "input",
            Token "target",
            Token "function",
            Token "ty",
            Token "unscope",
            Token "argument"
          ]
      relationsVocab =
        OSet.fromList
          [ Pad,
            Unk,
            Token ChildParentRelation,
            Token ParentChildRelation,
            Token (SiblingDistRelation $ -3),
            Token (SiblingDistRelation $ -2),
            Token (SiblingDistRelation $ -1),
            Token (SiblingDistRelation 0),
            Token (SiblingDistRelation 1),
            Token (SiblingDistRelation 2),
            Token (SiblingDistRelation 3),
            Token (DistRelation $ -3),
            Token (DistRelation $ -2),
            Token (DistRelation $ -1),
            Token (DistRelation 0),
            Token (DistRelation 1),
            Token (DistRelation 2),
            Token (DistRelation 3)
          ]
  tokens <-
    fromJust'
      . mkSeq tokenVocab
      . pure
      . view (field @"tEnv" . field @"tokens")
      $ env
  -- liftIO . print $ view (field @"tEnv" . field @"validActionMask") <$> envs
  invalidTokenMask <-
    fromJust'
      . mkInvalidTokenMask tokenVocab
      . pure
      . view (field @"tEnv" . field @"validActionMask")
      $ env
  meta <- do
    let f vocab g =
          fromJust'
            . mkSeq vocab
            . pure
            . (Map.mapMaybe g . view (field @"mEnv" . field @"metas"))
            $ env
    dataTypes <- f dataTypeVocab dataType
    constructors <- f constructorVocab constructor
    selectors <- f selectorVocab selector
    pure $
      M
        { dataType = dataTypes,
          constructor = constructors,
          selector = selectors
        }
  relations <-
    fromJust'
      . mkMultiGrid relationsVocab
      . pure
      . view (field @"rEnv" . field @"relations")
      $ env
  let scopeMask scopeId =
        fromJust' $
          mkSeqMask
            . pure
            . scopePositions
            =<< ( Map.lookup scopeId
                    . view (field @"aEnv" . field @"knownScopes")
                    $ env
                )
  inputScopeMask <- scopeMask "input"
  targetScopeMask <- scopeMask "target"
  attentionMask <-
    fromJust'
      . mkGridMask @1 @seqLen @seqLen
      . pure
      . view (field @"aEnv" . field @"attentionMask")
      $ env
  keyPaddingMask <-
    logicalNot
      <$> ( fromJust'
              . mkSeqMask
              . pure
              . view (field @"aEnv" . field @"keyPaddingMask")
              $ env
          )
  let attentionMask' = maskedFill (unsqueeze @2 keyPaddingMask) (1 :: Int) attentionMask
  guard (attentionMaskIsProper attentionMask')
  let attentionMask'' =
        maskedFill
          (logicalNot attentionMask')
          (-1 / 0 :: Double)
          $ zeros @'[1, seqLen, seqLen] @dtype @device
  tokenMask <- do
    tokenMaskInput <-
      bernoulliMask
        pMaskInput
        (keyPaddingMask `logicalOr` (logicalNot inputScopeMask))
    tokenMaskTarget <-
      bernoulliMask
        pMaskTarget
        (keyPaddingMask `logicalOr` (logicalNot targetScopeMask))
    pure (tokenMaskInput `logicalOr` tokenMaskTarget)
  let maskedTokens =
        maskedFill
          tokenMask
          (fromJust $ OSet.findIndex (Token . injectTyped $ MaskA) tokenVocab)
          tokens
  pure $
    RATransformerMLMBatch
      { ratInput =
          RATransformerMLMInput
            { ratMaskedTokens = maskedTokens,
              ratMeta = meta,
              ratRelations = relations,
              ratAttentionMask = attentionMask'',
              ratKeyPaddingMask = keyPaddingMask
            },
        ratTarget =
          RATransformerMLMTarget
            { ratTargetTokens = tokens,
              ratTokenMask = tokenMask,
              ratInputScopeMask = inputScopeMask,
              ratTargetScopeMask = targetScopeMask,
              ratInvalidTokenMask = invalidTokenMask
            }
      }

instance
  ( All
      KnownNat
      '[ seqLen,
         numEmbeds,
         relDim
       ],
    KnownDType dtype,
    Scalar pMask,
    KnownDevice device,
    SumDTypeIsValid device 'Int64,
    ComparisonDTypeIsValid device 'Int64,
    MonadIO m,
    MonadFail m,
    Alternative m
  ) =>
  Dataset
    m
    (RATransformerMLMData seqLen numEmbeds relDim dtype device pMask)
    Seed.Seed
    (RATransformerMLMBatch 1 seqLen numEmbeds relDim dtype device)
  where
  getItem RATransformerMLMData {..} seed = do
    guard $ Set.member seed seeds
    actions <- mkActions @seqLen seed
    mkRATransformerMLMItem pMaskInput pMaskTarget actions
  keys RATransformerMLMData {..} = seeds

testDataset ::
  Float ->
  Float ->
  IO
    ( Maybe
        ( RATransformerMLMBatch
            TestBatchSize
            TestSeqLen
            TestTokenNumEmbeds
            TestRelDim
            TestDType
            TestDataDevice
        )
    )
testDataset pMaskInput pMaskTarget = do
  seeds <- replicateM (natValI @TestBatchSize) Seed.random
  let dataset =
        RATransformerMLMData
          @TestSeqLen
          @TestTokenNumEmbeds
          @TestRelDim
          @TestDType
          @TestDataDevice
          @Float
          "test"
          pMaskInput
          pMaskTarget
          (Set.fromList seeds)
  items <- traverse (getItem dataset) seeds
  let batch = mkBatch @TestBatchSize items
  hspec $ do
    it "keys should be complete" $ do
      keys @IO dataset `shouldBe` Set.fromList seeds
    it "batch should be computable" $ do
      isJust batch `shouldBe` True
  -- case batch of
  --   Just batch' -> display3dTensorBatch . ratRelations . ratInput $ batch'
  --   Nothing -> pure ()
  pure batch

type MkBatchR batchSize shape dtype device tensors =
  ( '(batchSize ': shape, dtype, device) ~ Cat 0 tensors,
    Castable (HList tensors) [ATenTensor],
    tensors ~ (HReplicateR batchSize (Tensor device dtype (1 ': shape))),
    Exts.IsList (Maybe (HList tensors)),
    Exts.Item (Maybe (HList tensors)) ~ Tensor device dtype (1 ': shape)
  )

data Vect (n :: Nat) a where
  VNil :: Vect 0 a
  VCons :: a -> Vect n a -> Vect (n + 1) a

instance Functor (Vect n) where
  fmap _ VNil = VNil
  fmap f (VCons a as) = VCons (f a) (fmap f as)

cons :: forall n a. a -> Vect n a -> Vect (n + 1) a
cons a as = VCons a as

uncons :: forall n a. Vect (n + 1) a -> (a, Vect n a)
uncons VNil = error "cannot conjure up a value from nothing"
uncons (VCons a as) = (a, as)

class VectIsList (flag :: Bool) n a where
  vectFromList :: Proxy flag -> [a] -> Maybe (Vect n a)
  vectToList :: Proxy flag -> Maybe (Vect n a) -> [a]

instance VectIsList 'False 0 a where
  vectFromList _ [] = Just VNil
  vectFromList _ _ = Nothing
  vectToList _ (Just VNil) = []
  vectToList _ _ = []

instance
  Exts.IsList (Maybe (Vect (n - 1) a)) =>
  VectIsList 'True n a
  where
  vectFromList _ (a : as) = liftA2 (cons @(n - 1)) (Just a) (Exts.fromList as)
  vectFromList _ _ = Nothing
  vectToList _ (Just (VCons a as)) = a : Exts.toList @(Maybe (Vect (n - 1) a)) (Just as)
  vectToList _ _ = []

instance
  ( (1 <=? n) ~ flag,
    VectIsList flag n a
  ) =>
  Exts.IsList (Maybe (Vect n a))
  where
  type Item (Maybe (Vect n a)) = a
  fromList = vectFromList (Proxy :: Proxy flag)
  toList = vectToList (Proxy :: Proxy flag)

class HasBatch a b | a -> b where
  batch :: a -> b

instance
  (Exts.IsList (Maybe (Vect batchSize (Tensor device dtype (1 ': shape))))) =>
  HasBatch
    (Vect batchSize (Tensor device dtype (1 ': shape)))
    (Tensor device dtype (batchSize ': shape))
  where
  batch v =
    let tensors = Exts.toList . Just $ v
     in unsafePerformIO $ ATen.cast2 ATen.Managed.cat_ll tensors (0 :: Int)

mkBatch ::
  forall batchSize seqLen numEmbeds relDim dtype device tensors tensors' tensors'' tensors''' tensors''''.
  -- ( MkBatchR batchSize '[seqLen] 'Int64 device tensors,
  --   MkBatchR batchSize '[seqLen, seqLen, relDim] 'Int64 device tensors',
  --   MkBatchR batchSize '[seqLen, seqLen] dtype device tensors'',
  --   MkBatchR batchSize '[seqLen] 'Bool device tensors''',
  --   MkBatchR batchSize '[seqLen, numEmbeds] 'Bool device tensors''''
  -- ) =>
  ( Exts.IsList (Maybe (Vect batchSize (Tensor device 'Int64 [1, seqLen]))),
    Exts.IsList (Maybe (Vect batchSize (Tensor device 'Int64 [1, seqLen, seqLen, relDim]))),
    Exts.IsList (Maybe (Vect batchSize (Tensor device dtype [1, seqLen, seqLen]))),
    Exts.IsList (Maybe (Vect batchSize (Tensor device 'Bool [1, seqLen]))),
    Exts.IsList (Maybe (Vect batchSize (Tensor device 'Bool [1, seqLen, numEmbeds])))
  ) =>
  [RATransformerMLMBatch 1 seqLen numEmbeds relDim dtype device] ->
  Maybe
    ( RATransformerMLMBatch
        batchSize
        seqLen
        numEmbeds
        relDim
        dtype
        device
    )
mkBatch items = RATransformerMLMBatch <$> inputs <*> targets
  where
    inputs =
      RATransformerMLMInput
        <$> (batch <$> (Exts.fromList $ ratMaskedTokens . ratInput <$> items :: Maybe (Vect batchSize (Tensor device 'Int64 [1, seqLen]))))
        <*> ( M
                <$> (batch <$> (Exts.fromList $ dataType . ratMeta . ratInput <$> items :: Maybe (Vect batchSize (Tensor device 'Int64 [1, seqLen]))))
                <*> (batch <$> (Exts.fromList $ constructor . ratMeta . ratInput <$> items :: Maybe (Vect batchSize (Tensor device 'Int64 [1, seqLen]))))
                <*> (batch <$> (Exts.fromList $ selector . ratMeta . ratInput <$> items :: Maybe (Vect batchSize (Tensor device 'Int64 [1, seqLen]))))
            )
        <*> (batch <$> (Exts.fromList $ ratRelations . ratInput <$> items :: Maybe (Vect batchSize (Tensor device 'Int64 [1, seqLen, seqLen, relDim]))))
        <*> (batch <$> (Exts.fromList $ ratAttentionMask . ratInput <$> items :: Maybe (Vect batchSize (Tensor device dtype [1, seqLen, seqLen]))))
        <*> (batch <$> (Exts.fromList $ ratKeyPaddingMask . ratInput <$> items :: Maybe (Vect batchSize (Tensor device 'Bool [1, seqLen]))))
    -- RATransformerMLMInput
    --   <$> (cat @0 <$> (Exts.fromList $ ratMaskedTokens . ratInput <$> items :: Maybe (HList tensors)))
    --   <*> ( M
    --           <$> (cat @0 <$> (Exts.fromList $ dataType . ratMeta . ratInput <$> items :: Maybe (HList tensors)))
    --           <*> (cat @0 <$> (Exts.fromList $ constructor . ratMeta . ratInput <$> items :: Maybe (HList tensors)))
    --           <*> (cat @0 <$> (Exts.fromList $ selector . ratMeta . ratInput <$> items :: Maybe (HList tensors)))
    --       )
    --   <*> (cat @0 <$> (Exts.fromList $ ratRelations . ratInput <$> items :: Maybe (HList tensors')))
    --   <*> (cat @0 <$> (Exts.fromList $ ratAttentionMask . ratInput <$> items :: Maybe (HList tensors'')))
    --   <*> (cat @0 <$> (Exts.fromList $ ratKeyPaddingMask . ratInput <$> items :: Maybe (HList tensors''')))
    targets =
      RATransformerMLMTarget
        <$> (batch <$> (Exts.fromList $ ratTargetTokens . ratTarget <$> items :: Maybe (Vect batchSize (Tensor device 'Int64 [1, seqLen]))))
        <*> (batch <$> (Exts.fromList $ ratTokenMask . ratTarget <$> items :: Maybe (Vect batchSize (Tensor device 'Bool [1, seqLen]))))
        <*> (batch <$> (Exts.fromList $ ratInputScopeMask . ratTarget <$> items :: Maybe (Vect batchSize (Tensor device 'Bool [1, seqLen]))))
        <*> (batch <$> (Exts.fromList $ ratTargetScopeMask . ratTarget <$> items :: Maybe (Vect batchSize (Tensor device 'Bool [1, seqLen]))))
        <*> (batch <$> (Exts.fromList $ ratInvalidTokenMask . ratTarget <$> items :: Maybe (Vect batchSize (Tensor device 'Bool [1, seqLen, numEmbeds]))))

-- RATransformerMLMTarget
--   <$> (cat @0 <$> (Exts.fromList $ ratTargetTokens . ratTarget <$> items :: Maybe (HList tensors)))
--   <*> (cat @0 <$> (Exts.fromList $ ratTokenMask . ratTarget <$> items :: Maybe (HList tensors''')))
--   <*> (cat @0 <$> (Exts.fromList $ ratInputScopeMask . ratTarget <$> items :: Maybe (HList tensors''')))
--   <*> (cat @0 <$> (Exts.fromList $ ratTargetScopeMask . ratTarget <$> items :: Maybe (HList tensors''')))
--   <*> (cat @0 <$> (Exts.fromList $ ratInvalidTokenMask . ratTarget <$> items :: Maybe (HList tensors'''')))

collate ::
  forall batchSize seqLen numEmbeds relDim dtype device r m.
  _ =>
  Int ->
  P.ListT m (RATransformerMLMBatch 1 seqLen numEmbeds relDim dtype device, Int) ->
  ContT r m (P.ListT m (RATransformerMLMBatch batchSize seqLen numEmbeds relDim dtype device, Int))
collate n = bufferedCollate (P.bounded n) (natValI @batchSize) f
  where
    f exs =
      let (items, iters) = unzip exs
       in (,) <$> mkBatch items <*> listToMaybe iters

bufferedCollate ::
  forall a b r m.
  MonadBaseControl IO m =>
  P.Buffer b ->
  Int ->
  ([a] -> Maybe b) ->
  P.ListT m a ->
  ContT r m (P.ListT m b)
bufferedCollate buffer batchSize f as = ContT $ \g ->
  snd
    <$> withBufferLifted
      buffer
      fOutput
      (g . P.Select . fromInput')
  where
    fOutput output = P.runEffect $ (P.>-> (toOutput' output)) . (P.>-> P.mapMaybe f) . L.purely P.folds L.list . view (P.chunksOf batchSize) . P.enumerate $ as

attentionMaskIsProper ::
  forall batchSize seqLen device.
  ( SumDTypeIsValid device 'Int64,
    ComparisonDTypeIsValid device 'Int64,
    KnownDevice device
  ) =>
  Tensor device 'Bool '[batchSize, seqLen, seqLen] ->
  Bool
attentionMaskIsProper t =
  let t' = toDType @'Int64 @'Bool t
      t'' = sumDim @2 t'
      t''' = t'' `gt` (0 :: Tensor device 'Int64 '[])
      t'''' = Torch.Typed.all t'''
   in toBool t''''

display2dTensor ::
  forall dim dim' device dtype shape.
  ( All KnownNat '[dim, dim'],
    shape ~ '[dim, dim'],
    AllDimsPositive shape,
    BasicArithmeticDTypeIsValid device 'Float,
    MinMaxDTypeIsValid device 'Float,
    KnownDevice device
  ) =>
  Tensor device dtype shape ->
  IO ()
display2dTensor t = do
  mapM
    ( \row ->
        Text.Printf.printf "%02d" row
          >> mapM
            ( \col ->
                putChar $ grayScale !! (Prelude.floor $ scaled !! row !! col)
            )
            [0, downSamp .. natValI @dim' - 1]
          >> putStrLn ""
    )
    [0, downSamp .. natValI @dim - 1]
  pure ()
  where
    downSamp = 1
    grayScale = grayScale70
    paletteMax = List.length grayScale - 1
    t' = toDType @'Float @dtype t
    scaled =
      let (mn, mx) = (Torch.Typed.min t', Torch.Typed.max t')
       in Exts.toList . Just . mulScalar paletteMax $ (t' `sub` mn) `Torch.Typed.div` (mx `sub` mn)

display3dTensor ::
  forall dim dim' dim'' device dtype shape.
  ( All KnownNat '[dim, dim', dim''],
    shape ~ '[dim, dim', dim''],
    AllDimsPositive shape,
    BasicArithmeticDTypeIsValid device 'Float,
    MinMaxDTypeIsValid device 'Float,
    KnownDevice device
  ) =>
  Tensor device dtype shape ->
  IO ()
display3dTensor t = do
  mapM
    ( \row ->
        Text.Printf.printf "%02d" row
          >> mapM
            ( \col ->
                mapM
                  ( \col' ->
                      putChar $ grayScale !! (Prelude.floor $ scaled !! row !! col !! col')
                  )
                  [0, downSamp .. natValI @dim'' - 1]
            )
            [0, downSamp .. natValI @dim' - 1]
          >> putStrLn ""
    )
    [0, downSamp .. natValI @dim - 1]
  pure ()
  where
    downSamp = 1
    grayScale = grayScale70
    paletteMax = List.length grayScale - 1
    t' = toDType @'Float @dtype t
    scaled =
      let (mn, mx) = (Torch.Typed.min t', Torch.Typed.max t')
       in Exts.toList . Just . mulScalar paletteMax $ (t' `sub` mn) `Torch.Typed.div` (mx `sub` mn)

data Display2dTensorBatch = Display2dTensorBatch

instance
  ( BasicArithmeticDTypeIsValid device 'Float,
    MinMaxDTypeIsValid device 'Float,
    All KnownNat '[dim, dim'],
    shape ~ '[1, dim, dim'],
    AllDimsPositive shape,
    KnownDevice device
  ) =>
  Apply' Display2dTensorBatch (Tensor device 'Int64 shape) (IO ())
  where
  apply' _ = display2dTensor . squeezeDim @0

display2dTensorBatch ::
  forall batchSize dim dim' device dtype shape (tensorChunks :: [Type]) ys.
  ( KnownNat batchSize,
    shape ~ '[batchSize, dim, dim'],
    tensorChunks ~ Chunk batchSize 0 shape 'Int64 device,
    Castable (HList tensorChunks) [ATenTensor],
    HMapM' IO Display2dTensorBatch tensorChunks ys
  ) =>
  Tensor device dtype shape ->
  IO ()
display2dTensorBatch t =
  let t' = toDType @'Int64 @dtype t
      ts = chunk @batchSize @0 @shape @'Int64 @device @tensorChunks t'
   in void . hmapM' Display2dTensorBatch $ ts

data Display3dTensorBatch = Display3dTensorBatch

instance
  ( BasicArithmeticDTypeIsValid device 'Float,
    MinMaxDTypeIsValid device 'Float,
    All KnownNat '[dim, dim', dim''],
    shape ~ '[1, dim, dim', dim''],
    AllDimsPositive shape,
    KnownDevice device
  ) =>
  Apply' Display3dTensorBatch (Tensor device 'Int64 shape) (IO ())
  where
  apply' _ = display3dTensor . squeezeDim @0

display3dTensorBatch ::
  forall batchSize dim dim' dim'' device dtype shape (tensorChunks :: [Type]) ys.
  ( KnownNat batchSize,
    shape ~ '[batchSize, dim, dim', dim''],
    tensorChunks ~ Chunk batchSize 0 shape 'Int64 device,
    Castable (HList tensorChunks) [ATenTensor],
    HMapM' IO Display3dTensorBatch tensorChunks ys
  ) =>
  Tensor device dtype shape ->
  IO ()
display3dTensorBatch t =
  let t' = toDType @'Int64 @dtype t
      ts = chunk @batchSize @0 @shape @'Int64 @device @tensorChunks t'
   in void . hmapM' Display3dTensorBatch $ ts

------------------------------------------------------------------------

type TestBatchSize = 64

type TestSeqLen = 256

type TestRelDim = 4

type TestNumAttnLayers = 6

type TestNumHeads = 8

type TestHeadDim = 16

type TestFFNDim = 256

type TestPaddingIdx = 0

type TestTokenNumEmbeds = 12

type TestDataTypeNumEmbeds = 7

type TestConstructorNumEmbeds = 13

type TestSelectorNumEmbeds = 9

type TestRelationNumEmbeds = 18

type TestDType = 'Float

type TestDataDevice = '( 'CPU, 0)

type TestDevice = '( 'CUDA, 0)

type TestRATransformerMLMSpec =
  RATransformerMLMSpec
    TestNumAttnLayers
    TestNumHeads
    TestHeadDim
    TestFFNDim
    TestPaddingIdx
    TestTokenNumEmbeds
    TestPaddingIdx
    TestDataTypeNumEmbeds
    TestPaddingIdx
    TestConstructorNumEmbeds
    TestPaddingIdx
    TestSelectorNumEmbeds
    TestPaddingIdx
    TestRelationNumEmbeds
    TestDType
    TestDevice

bernoulliMask ::
  forall shape dtype device a m.
  (Scalar a, KnownShape shape, shape ~ Broadcast shape shape, MonadIO m) =>
  a ->
  Tensor device 'Bool shape ->
  m (Tensor device 'Bool shape)
bernoulliMask p keyPaddingMask = do
  let bernoulli =
        fromProbs
          . toDynamic
          . mulScalar p
          . toDType @'Float @'Bool
          . onesLike
          $ keyPaddingMask
  samples <-
    liftIO $
      UnsafeMkTensor @device @'Bool @shape
        . toType Bool
        <$> Torch.Distributions.Distribution.sample bernoulli (shapeVal @shape)
  pure $ maskedFill keyPaddingMask (0 :: Int) samples

testBernoulliMask :: IO (Tensor TestDevice 'Bool '[2, 3])
testBernoulliMask = bernoulliMask (0.25 :: Float) $ zeros @'[2, 3] @'Bool @TestDevice

mkSeq ::
  forall batchSize seqLen device a.
  (Show a, Ord a, All KnownNat '[batchSize, seqLen], KnownDevice device) =>
  OSet.OSet (Token a) ->
  [Map Pos a] ->
  Maybe (Tensor device 'Int64 '[batchSize, seqLen])
mkSeq vocab ms = do
  unkIndex <- OSet.findIndex Unk vocab
  guard $ List.length ms <= natValI @batchSize
  (fstKeys, sndKeys, elems) <-
    let step (i, m) = do
          let keys = Map.keys m
              fstKeys = const i <$> keys
              sndKeys = unPos <$> keys
              elems = (fromMaybe unkIndex . flip OSet.findIndex vocab . Token) <$> Map.elems m
          if List.elem unkIndex elems then error $ show $ (vocab, m) else pure ()
          guard $ Prelude.all (< natValI @seqLen) sndKeys
          pure (fstKeys, sndKeys, elems)
     in foldMap step $ zip [(0 :: Int) ..] ms
  let tensorOptions = withDType Int64 defaultOpts
      i = asTensor' [fstKeys, sndKeys] tensorOptions
      v = asTensor' elems tensorOptions
      shape = [natValI @batchSize, natValI @seqLen]
  pure
    . toDevice @device @'( 'CPU, 0)
    . UnsafeMkTensor @'( 'CPU, 0)
    . Torch.toDense
    $ sparseCooTensor i v shape tensorOptions

testMkSeq :: Maybe (Tensor TestDevice 'Int64 '[2, 3])
testMkSeq =
  let vocab = OSet.fromList [Pad, Unk, Token "a"]
      ms =
        [ Map.fromList [(Pos 0, "a"), (Pos 1, "b"), (Pos 2, "a")],
          Map.fromList [(Pos 1, "a"), (Pos 2, "b")]
        ]
   in mkSeq vocab ms

mkSeqMask ::
  forall batchSize seqLen device.
  (All KnownNat '[batchSize, seqLen], KnownDevice device) =>
  [Set Pos] ->
  Maybe (Tensor device 'Bool '[batchSize, seqLen])
mkSeqMask ss = do
  guard $ List.length ss <= natValI @batchSize
  (fstElems, sndElems) <-
    let step (i, s) = do
          let elems = Set.elems s
              fstElems = const i <$> elems
              sndElems = unPos <$> elems
          guard $ Prelude.all (< natValI @seqLen) sndElems
          pure (fstElems, sndElems)
     in foldMap step $ zip [(0 :: Int) ..] ss
  let tensorOptions = withDType Int64 defaultOpts
      i = asTensor' [fstElems, sndElems] tensorOptions
      v = Torch.ones [List.length fstElems] tensorOptions
      shape = [natValI @batchSize, natValI @seqLen]
  pure
    . toDevice @device @'( 'CPU, 0)
    . UnsafeMkTensor @'( 'CPU, 0)
    . Torch.toType Bool
    . Torch.toDense
    $ sparseCooTensor i v shape tensorOptions

testMkSeqMask :: Maybe (Tensor TestDevice 'Bool '[2, 3])
testMkSeqMask =
  let ss =
        [ Set.fromList [Pos 1],
          Set.fromList [Pos 0, Pos 2]
        ]
   in mkSeqMask ss

mkGrid ::
  forall batchSize seqLen seqLen' device a.
  (Show a, Ord a, All KnownNat '[batchSize, seqLen, seqLen'], KnownDevice device) =>
  OSet.OSet (Token a) ->
  [Map (Pos, Pos) a] ->
  Maybe (Tensor device 'Int64 '[batchSize, seqLen, seqLen'])
mkGrid vocab ms = do
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
     in foldMap step $ zip [(0 :: Int) ..] ms
  let tensorOptions = withDType Int64 defaultOpts
      i = asTensor' [fstKeys, sndKeys, trdKeys] tensorOptions
      v = asTensor' elems tensorOptions
      shape = [natValI @batchSize, natValI @seqLen, natValI @seqLen']
  pure
    . toDevice @device @'( 'CPU, 0)
    . UnsafeMkTensor @'( 'CPU, 0)
    . Torch.toDense
    $ sparseCooTensor i v shape tensorOptions

testMkGrid :: Maybe (Tensor TestDevice 'Int64 '[2, 2, 3])
testMkGrid =
  let vocab = OSet.fromList [Pad, Unk, Token "a", Token "b"]
      ms =
        [ Map.fromList [((Pos 0, Pos 0), "a"), ((Pos 0, Pos 1), "b"), ((Pos 1, Pos 2), "a")],
          Map.fromList [((Pos 0, Pos 1), "b"), ((Pos 0, Pos 2), "c")]
        ]
   in mkGrid vocab ms

mkMultiGrid ::
  forall batchSize seqLen seqLen' dim device a.
  (Ord a, All KnownNat '[batchSize, seqLen, seqLen', dim], KnownDevice device) =>
  OSet.OSet (Token a) ->
  [Map (Pos, Pos) (Set a)] ->
  Maybe (Tensor device 'Int64 '[batchSize, seqLen, seqLen', dim])
mkMultiGrid vocab ms = do
  unkIndex <- OSet.findIndex Unk vocab
  guard $ List.length ms <= natValI @batchSize
  (fstKeys, sndKeys, trdKeys, fthKeys, elems) <-
    let step (i, m) = do
          let m' = Map.fromList $ do
                ((pos, pos'), s) <- Map.toList m
                (i, a) <- zip [(0 :: Int) ..] $ Set.toList s
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
     in foldMap step $ zip [(0 :: Int) ..] ms
  let tensorOptions = withDType Int64 defaultOpts
      i = asTensor' [fstKeys, sndKeys, trdKeys, fthKeys] tensorOptions
      v = asTensor' elems tensorOptions
      shape = [natValI @batchSize, natValI @seqLen, natValI @seqLen', natValI @dim]
  pure
    . toDevice @device @'( 'CPU, 0)
    . UnsafeMkTensor @'( 'CPU, 0)
    . Torch.toDense
    $ sparseCooTensor i v shape tensorOptions

mkGridMask ::
  forall batchSize seqLen seqLen' device.
  ( All KnownNat '[batchSize, seqLen, seqLen'],
    KnownDevice device
  ) =>
  [Set (Pos, Pos)] ->
  Maybe (Tensor device 'Bool '[batchSize, seqLen, seqLen'])
mkGridMask ss = do
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
     in foldMap step $ zip [(0 :: Int) ..] ss
  let tensorOptions = withDType Int64 defaultOpts
      i = asTensor' [fstElems, sndElems, trdElems] tensorOptions
      v = Torch.ones [List.length fstElems] tensorOptions
      shape = [natValI @batchSize, natValI @seqLen, natValI @seqLen']
  pure
    . toDevice @device @'( 'CPU, 0)
    . UnsafeMkTensor @'( 'CPU, 0)
    . Torch.toType Bool
    . Torch.toDense
    $ sparseCooTensor i v shape tensorOptions

testMkGridMask :: Maybe (Tensor TestDevice 'Bool '[2, 2, 3])
testMkGridMask =
  let ss =
        [ Set.fromList [(Pos 0, Pos 0), (Pos 0, Pos 1), (Pos 1, Pos 2)],
          Set.fromList [(Pos 0, Pos 2)]
        ]
   in mkGridMask ss

mkInvalidTokenMask ::
  forall batchSize seqLen numEmbeds device action.
  ( All KnownNat '[batchSize, seqLen, numEmbeds],
    KnownDevice device,
    Ord action
  ) =>
  OSet.OSet (Token action) ->
  [Map Pos (Set action)] ->
  Maybe (Tensor device 'Bool '[batchSize, seqLen, numEmbeds])
mkInvalidTokenMask vocab ms = do
  guard $ List.length ms <= natValI @batchSize
  (fstElems, sndElems, trdElems) <-
    let step (i, m) = do
          let s :: Set (Pos, Int) = Set.fromList $ do
                (pos, validActions) <- Map.toList m
                (i, token) <- zip [(0 :: Int) ..] $ OSet.toAscList vocab
                case token of
                  Token action -> guard $ Set.notMember action validActions
                  _ -> pure () -- Pad, Unk, Mask, etc. are invalid tokens
                pure (pos, i)
              elems = Set.elems s
              fstElems = const i <$> elems
              sndElems = unPos . fst <$> elems
              trdElems = snd <$> elems
          guard $ Prelude.all (< natValI @seqLen) sndElems
          guard $ Prelude.all (< natValI @numEmbeds) trdElems
          pure (fstElems, sndElems, trdElems)
     in foldMap step $ zip [(0 :: Int) ..] ms
  let tensorOptions = withDType Int64 defaultOpts
      i = asTensor' [fstElems, sndElems, trdElems] tensorOptions
      v = Torch.ones [List.length fstElems] tensorOptions
      shape = [natValI @batchSize, natValI @seqLen, natValI @numEmbeds]
  pure
    . toDevice @device @'( 'CPU, 0)
    . UnsafeMkTensor @'( 'CPU, 0)
    . Torch.toType Bool
    . Torch.toDense
    $ sparseCooTensor i v shape tensorOptions

testMkInvalidTokenMask :: Maybe (Tensor TestDevice 'Bool '[2, 3, 5])
testMkInvalidTokenMask =
  let vocab = OSet.fromList [Pad, Unk, Token MaskA, Token L, Token R]
      ms =
        [ Map.fromList
            [ (Pos 0, Set.fromList [L, R]),
              (Pos 1, Set.fromList [R]),
              (Pos 1, Set.fromList [R])
            ],
          Map.fromList
            [ (Pos 0, Set.fromList [L]),
              (Pos 1, Set.empty)
            ]
        ]
   in mkInvalidTokenMask vocab ms

------------------------------------------------------------------------

testGen :: IO ()
testGen = do
  seed <- Seed.random
  ((e, actions), _) <-
    runStateT
      ( sample' $ do
          ty <- genTy
          e <- genWellTypedExp @Int ty
          guard (isJust . runFresh . runMaybeT . assertTy Map.empty e $ ty)
          let actions = toActions @[] @Action e
          guard (List.length actions <= 256)
          pure (e, actions)
      )
      seed
  print actions
  putDoc . vsep $
    [ mempty,
      pprint (0 :: Int) e,
      mempty,
      pprint (0 :: Int) (nf e),
      mempty
    ]

sample' :: Monad m => Gen a -> StateT Seed.Seed m a
sample' gen =
  let go = do
        seed <- get
        let (seed', seed'') = Seed.split seed
        put seed''
        case evalGen 20 seed' gen of
          Nothing ->
            go
          Just x ->
            pure $ Tree.treeValue x
   in go

data ClipGradValue a = ClipGradValue a

instance
  (Scalar a, Num a) =>
  Apply'
    (ClipGradValue a)
    (Tensor device dtype shape)
    (Tensor device dtype shape)
  where
  apply' (ClipGradValue a) = clamp (- a) a

data GuardGradAgainstNaN = GuardGradAgainstNaN

instance Apply' GuardGradAgainstNaN (Tensor device dtype shape, Maybe ()) (Maybe ()) where
  apply' _ (t, acc) = acc >> (guard . not . toBool . Torch.Typed.any . Torch.Typed.isNaN $ t)

data
  RATransformerMLMData
    (seqLen :: Nat)
    (numEmbeds :: Nat)
    (relDim :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
    (pMask :: Type) = RATransformerMLMData
  { name :: Text,
    pMaskInput :: pMask,
    pMaskTarget :: pMask,
    seeds :: (Set Seed.Seed)
  }
  deriving (Eq, Ord, Show, Functor, Generic)

data Config = Config
  { -- | number training examples
    trainingLen :: Int,
    -- | number evaluation examples
    evaluationLen :: Int,
    -- | input masking probability
    probMaskInput :: Float,
    -- | target masking probability
    probMaskTarget :: Float,
    -- | maximum learning rate
    maxLearningRate :: LearningRate TestDevice TestDType,
    -- | final learning rate
    finalLearningRate :: LearningRate TestDevice TestDType,
    -- | total number of epochs
    numEpochs :: Int,
    -- | number of warm-up epochs
    numWarmupEpochs :: Int,
    -- | number of cool-down epochs
    numCooldownEpochs :: Int,
    -- | model checkpoint file name
    ptFile :: FilePath,
    -- | plot file name
    plotFile :: FilePath
  }

testProgram :: Config -> IO ()
testProgram Config {..} = do
  model <-
    -- initialize the model
    liftIO . Torch.Typed.sample $
      let dropoutSpec = DropoutSpec 0.2
          eps = 0.001
       in ( RATransformerMLMSpec
              dropoutSpec
              ( TransformerLayerSpec
                  ( MultiheadAttentionSpec
                      dropoutSpec
                  )
                  dropoutSpec
                  eps
                  ( TransformerMLPSpec
                      dropoutSpec
                      dropoutSpec
                      eps
                  )
              ) ::
              TestRATransformerMLMSpec
          )

  let -- use the Adam optimization algorithm, see https://arxiv.org/abs/1412.6980
      optim = mkAdam 0 0.9 0.999 (flattenParameters model)

  let -- learning rate schedule
      learningRateSchedule epoch
        | epoch <= 0 = 0.0
        | 0 < epoch && epoch <= numWarmupEpochs =
          let a :: Float = fromIntegral epoch / fromIntegral numWarmupEpochs
           in mulScalar a maxLearningRate
        | numWarmupEpochs < epoch && epoch < numEpochs - numCooldownEpochs =
          let a :: Float =
                fromIntegral (numEpochs - numCooldownEpochs - epoch)
                  / fromIntegral (numEpochs - numCooldownEpochs - numWarmupEpochs)
           in mulScalar a maxLearningRate + mulScalar (1 - a) finalLearningRate
        | otherwise = finalLearningRate

  options <- getStdGen >>= (\g -> pure (mapStyleOpts 1) {shuffle = Shuffle g})

  let -- create a dataset unique training examples
      trainingSeeds =
        Set.fromList
          . List.take trainingLen
          $ Seed.from <$> List.iterate (+ 1) (0 :: Word64)
      trainingData =
        RATransformerMLMData
          @TestSeqLen
          @TestTokenNumEmbeds
          @TestRelDim
          @TestDType
          @TestDataDevice
          "training"
          probMaskInput
          probMaskTarget
          trainingSeeds

      -- create a dataset of unique evaluation examples
      evaluationsSeeds =
        Set.fromList
          . List.take evaluationLen
          $ Seed.from <$> List.iterate (+ 1) (fromInteger . toInteger $ trainingLen :: Word64)
      evaluationData =
        RATransformerMLMData
          @TestSeqLen
          @TestTokenNumEmbeds
          @TestRelDim
          @TestDType
          @TestDataDevice
          "evaluation"
          probMaskInput
          probMaskTarget
          evaluationsSeeds

  let -- generate statistics and plots for each epoch
      stats model learningRate trainingCREs evaluationCREs learningRates options = do
        learningRates <- pure $ toFloat learningRate : learningRates
        (trainingCRE, options) <- evaluate model options trainingData
        trainingCREs <- pure $ trainingCRE : trainingCREs
        (evaluationCRE, options) <- evaluate model options evaluationData
        evaluationCREs <- pure $ evaluationCRE : evaluationCREs
        plot "plot.html" trainingCREs evaluationCREs learningRates
        pure (trainingCREs, evaluationCREs, learningRates, options)
      -- program step function
      step (model, optim, trainingCREs, evaluationCREs, learningRates, options) epoch = do
        let learningRate = learningRateSchedule epoch
        (model, optim, options) <-
          -- train the model
          train model optim learningRate options trainingData
        (trainingCREs, evaluationCREs, learningRates, options) <-
          -- calculate epoch statistics
          stats model learningRate trainingCREs evaluationCREs learningRates options
        -- save the model weights to a file and end program
        lift $ save (hmap' ToDependent . flattenParameters $ model) ptFile
        pure (model, optim, trainingCREs, evaluationCREs, learningRates, options)
      -- initial program state
      init = do
        let learningRate = learningRateSchedule 0
        (trainingCREs, evaluationCREs, learningRates, options) <-
          -- calculate initial statistics
          stats model learningRate [] [] [] options
        pure (model, optim, trainingCREs, evaluationCREs, learningRates, options)
      -- just keep the final model
      done (model, _, _, _, _, _) = pure model
  -- the whole program is a fold over epochs
  void . evalContT . P.foldM step init done . P.each $ [1 .. numEpochs]

-- | Train the model for one epoch
train ::
  _ =>
  -- | initial model datatype holding the weights
  model ->
  -- | initial optimizer, e.g. Adam
  optim ->
  -- | learning rate, 'LearningRate device dtype' is a type alias for 'Tensor device dtype '[]'
  LearningRate device dtype ->
  -- | pipeline options
  MapStyleOptions ->
  -- | dataset of training examples
  RATransformerMLMData
    TestSeqLen
    TestTokenNumEmbeds
    TestRelDim
    TestDType
    TestDataDevice
    pMask ->
  -- | final model and optimizer and updated pipeline options
  ContT r IO (model, optim, MapStyleOptions)
train model optim learningRate options dataset =
  let -- training step function
      step (model, optim) (RATransformerMLMBatch {..}, _iter) = do
        let input = toDevice @TestDevice @TestDataDevice ratInput
        prediction <- forwardStoch model input
        let targetTokens = toDevice @TestDevice @TestDataDevice . ratTargetTokens $ ratTarget
            tokenMask = toDevice @TestDevice @TestDataDevice . ratTokenMask $ ratTarget
            invalidTokenMask = toDevice @TestDevice @TestDataDevice . ratInvalidTokenMask $ ratTarget
            cre =
              loss
                invalidTokenMask
                (tokenMask `logicalAnd` (logicalNot . ratKeyPaddingMask $ input))
                prediction
                targetTokens
            parameters = flattenParameters model
            gradients = grad cre parameters
            clippedGradients = hmap' (ClipGradValue (1e1 :: Float)) gradients
        -- force GC cleanup after every batch
        performGC
        maybe
          (print "encountered NaN in gradients, repeating training step" >> pure (model, optim))
          (const $ (runStep' model optim learningRate clippedGradients))
          (hfoldrM GuardGradAgainstNaN () clippedGradients)
      -- initial training state
      init = pure (model, optim)
      -- trivial extraction function
      done = pure
   in do
        (examples, shuffle) <- makeListT options dataset
        (model, optim) <-
          lift . P.foldM step init done . enumerate
            =<< collate @TestBatchSize 1 examples
        pure (model, optim, options {shuffle = shuffle})

data CRE a = CRE
  { cre :: a,
    creInput :: a,
    creTarget :: a,
    creMasked :: a,
    creMaskedInput :: a,
    creMaskedTarget :: a,
    creNonMasked :: a,
    creNonMaskedInput :: a,
    creNonMaskedTarget :: a
  }
  deriving (Show, Functor)

instance Num a => Semigroup (CRE a) where
  a <> b =
    CRE
      { cre = cre a + cre b,
        creInput = creInput a + creInput b,
        creTarget = creTarget a + creTarget b,
        creMasked = creMasked a + creMasked b,
        creMaskedInput = creMaskedInput a + creMaskedInput b,
        creMaskedTarget = creMaskedTarget a + creMaskedTarget b,
        creNonMasked = creNonMasked a + creNonMasked b,
        creNonMaskedInput = creNonMaskedInput a + creNonMaskedInput b,
        creNonMaskedTarget = creNonMaskedTarget a + creNonMaskedTarget b
      }

instance Num a => Monoid (CRE a) where
  mempty =
    CRE
      { cre = 0,
        creInput = 0,
        creTarget = 0,
        creMasked = 0,
        creMaskedInput = 0,
        creMaskedTarget = 0,
        creNonMasked = 0,
        creNonMaskedInput = 0,
        creNonMaskedTarget = 0
      }

evaluate ::
  _ =>
  model ->
  MapStyleOptions ->
  RATransformerMLMData seqLen numEmbeds relDim dtype device pMask ->
  ContT r IO (CRE Float, MapStyleOptions)
evaluate model options dataset =
  let step (cre, _) (RATransformerMLMBatch {..}, iter) = do
        let input = toDevice @TestDevice @TestDataDevice ratInput
            prediction = forward model input
        let target = toDevice @TestDevice @TestDataDevice ratTarget
            loss' mask =
              toFloat $
                loss
                  (ratInvalidTokenMask target)
                  (mask `logicalAnd` (logicalNot . ratKeyPaddingMask $ input))
                  prediction
                  (ratTargetTokens target)
            cre' =
              CRE
                { cre = loss' ones,
                  creInput = loss' $ ratInputScopeMask target,
                  creTarget = loss' $ ratTargetScopeMask target,
                  creMasked = loss' $ ratTokenMask target,
                  creMaskedInput = loss' $ ratTokenMask target `logicalAnd` ratInputScopeMask target,
                  creMaskedTarget = loss' $ ratTokenMask target `logicalAnd` ratTargetScopeMask target,
                  creNonMasked = loss' $ (logicalNot . ratTokenMask $ target),
                  creNonMaskedInput = loss' $ (logicalNot . ratTokenMask $ target) `logicalAnd` (ratInputScopeMask target),
                  creNonMaskedTarget = loss' $ (logicalNot . ratTokenMask $ target) `logicalAnd` (ratTargetScopeMask target)
                }
        -- guard (not . toBool . Torch.Typed.isNaN $ cre)
        let res = (cre <> cre', iter)
        -- force GC cleanup after every batch
        performGC
        pure res
      init = pure (mempty, 0)
      done (_, 0) = pure $ const 0 <$> mempty
      done (cre, iter) =
        let scale x = x / (fromInteger . toInteger $ iter + natValI @TestBatchSize)
         in pure (scale <$> cre)
   in do
        (examples, shuffle) <- makeListT options dataset
        cre <-
          lift . P.foldM step init done . enumerate
            =<< collate @TestBatchSize 1 examples
        lift . putStrLn $ "Average " <> unpack (name dataset) <> " losses: " <> show cre
        pure (cre, options {shuffle = shuffle})

plot ::
  _ =>
  -- | output file path
  FilePath ->
  -- | training losses
  [CRE Float] ->
  -- | evaluation losses
  [CRE Float] ->
  -- | learning rates
  [Float] ->
  -- | returned continuation
  ContT r IO ()
plot file trainingCREs evaluationCREs learningRates = do
  let epochs = [0, 1 ..]
      lossData =
        let mkDataRows name cres =
              join
                . zipWith
                  ( \epoch CRE {..} ->
                      let f loss title =
                            Vega.dataRow
                              [ ("epoch", Vega.Number . fromInteger $ epoch),
                                ("title", Vega.Str $ name <> " " <> title),
                                ("loss", Vega.Number . float2Double $ loss)
                              ]
                       in [ f cre "loss",
                            f creInput "loss input",
                            f creTarget "loss target",
                            f creMasked "loss masked",
                            f creMaskedInput "loss masked input",
                            f creMaskedTarget "loss masked target",
                            f creNonMasked "loss non-masked",
                            f creNonMaskedInput "loss non-masked input",
                            f creNonMaskedTarget "loss non-masked target"
                          ]
                  )
                  epochs
                $ reverse cres
         in Vega.dataFromRows [] . foldr id [] $
              mkDataRows "training" trainingCREs
                <> mkDataRows "evaluation" evaluationCREs
      learningRateData =
        let dataRows =
              zipWith
                ( \epoch learningRate ->
                    Vega.dataRow
                      [ ("epoch", Vega.Number . fromInteger $ epoch),
                        ("learning_rate", Vega.Number . float2Double $ learningRate)
                      ]
                )
                epochs
                (reverse learningRates)
         in Vega.dataFromRows [] . foldr id [] $ dataRows
  lift . Vega.toHtmlFile file . mkVegaLite . Vega.datasets $
    [ ("losses", lossData),
      ("learning_rates", learningRateData)
    ]

mkVegaLite :: Vega.Data -> Vega.VegaLite
mkVegaLite dataset =
  let -- width and height of the individual plots (in pixels)
      w = Vega.width 1400
      h = Vega.height 700

      encLosses =
        Vega.encoding
          . Vega.position
            Vega.X
            [ Vega.PName "epoch",
              Vega.PmType Vega.Quantitative,
              Vega.PScale scaleOptsEpoch
            ]
          . Vega.position
            Vega.Y
            [ Vega.PName "loss",
              Vega.PmType Vega.Quantitative,
              Vega.PScale [Vega.SType Vega.ScLog]
            ]
          . Vega.color
            [ Vega.MName "title",
              Vega.MmType Vega.Nominal,
              Vega.MLegend [Vega.LTitle "", Vega.LOrient Vega.LOBottom]
            ]
      scaleOptsEpoch =
        [ Vega.SDomain (Vega.DNumbers [0, 1000]),
          Vega.SNice (Vega.IsNice False)
        ]
      losses =
        Vega.asSpec
          [ Vega.dataFromSource "losses" [],
            encLosses [],
            Vega.mark Vega.Line [],
            w,
            h
          ]

      encLearningRate =
        Vega.encoding
          . Vega.position
            Vega.X
            [ Vega.PName "epoch",
              Vega.PmType Vega.Quantitative,
              Vega.PScale scaleOptsEpoch
            ]
          . Vega.position
            Vega.Y
            [ Vega.PName "learning_rate",
              Vega.PTitle "learning rate",
              Vega.PmType Vega.Quantitative
            ]
      learningRates =
        Vega.asSpec
          [ Vega.dataFromSource "learning_rates" [],
            encLearningRate [],
            Vega.mark Vega.Line [],
            w,
            h
          ]
   in Vega.toVegaLite
        [ dataset,
          Vega.vConcat [losses, learningRates]
        ]

-- | checks
-- * for a random model, changing a token at a random position should change the outputs of all positions as permitted by the attention mask
-- * for a random model, changing a relation at a random position should change the outputs of all positions as permitted by the attention mask
testTheModel :: IO ()
testTheModel = do
  model <-
    liftIO . Torch.Typed.sample $
      let dropoutSpec = DropoutSpec 0.2
          eps = 0.001
       in ( RATransformerMLMSpec
              dropoutSpec
              ( TransformerLayerSpec
                  ( MultiheadAttentionSpec
                      dropoutSpec
                  )
                  dropoutSpec
                  eps
                  ( TransformerMLPSpec
                      dropoutSpec
                      dropoutSpec
                      eps
                  )
              ) ::
              TestRATransformerMLMSpec
          )
  let pMaskInput = 0 :: Float
      pMaskTarget = 0 :: Float
  Just batch <- testDataset pMaskInput pMaskTarget
  let batch' = toDevice @TestDevice @TestDataDevice batch
  let prediction = forward model . ratInput $ batch'
  randomMask <- bernoulliMask (0.25 :: Float) (zerosLike . ratKeyPaddingMask . ratInput $ batch')
  let maskOnlyPadding = maskedFill (logicalNot . ratKeyPaddingMask . ratInput $ batch') (0 :: Int) randomMask
      maskOnlyInput = maskedFill (logicalNot . ratInputScopeMask . ratTarget $ batch') (0 :: Int) randomMask
      maskOnlyTarget = maskedFill (logicalNot . ratTargetScopeMask . ratTarget $ batch') (0 :: Int) randomMask
      flippedAt m (maskedTokens :: Tensor TestDevice Int64 '[TestBatchSize, TestSeqLen]) =
        let test v = maskedTokens `eq` (v :: Tensor TestDevice Int64 '[])
            m' v = m `logicalAnd` test v
         in maskedFill (m' 0) (1 :: Int) . maskedFill (m' 3) (4 :: Int) . maskedFill (m' 4) (3 :: Int) $ maskedTokens
  let intervene flipMask comparisonMask = do
        let batch'' = field @"ratInput" . field @"ratMaskedTokens" %~ (flippedAt flipMask) $ batch'
        prediction' <- raTransformerMLM model False . ratInput $ batch''
        let comparison = isclose 1e-03 1e-04 False prediction prediction'
        case maskedSelect (unsqueeze @2 comparisonMask) comparison of
          UnknownShapeTensor t -> pure . toBool . Torch.Typed.all $ t
  intervention0 <- intervene maskOnlyPadding (logicalNot . ratKeyPaddingMask . ratInput $ batch')
  intervention1 <- intervene maskOnlyPadding (ratKeyPaddingMask . ratInput $ batch')
  intervention2 <- intervene maskOnlyInput maskOnlyInput
  intervention3 <- intervene maskOnlyInput (logicalNot maskOnlyInput)
  intervention4 <- intervene maskOnlyTarget (ratTargetScopeMask . ratTarget $ batch')
  intervention5 <- intervene maskOnlyTarget (ratInputScopeMask . ratTarget $ batch')
  intervention6 <- do
    let batch'' = field @"ratInput" . field @"ratRelations" %~ (\x -> zerosLike x) $ batch'
    prediction' <- raTransformerMLM model False . ratInput $ batch''
    pure . toBool . Torch.Typed.all $ isclose 1e-03 1e-04 False prediction prediction'
  hspec $ do
    it "flipped padding tokens should not affect non-padding outputs" $ do
      intervention0 `shouldBe` True
    it "flipped padding tokens should affect padding outputs" $ do
      intervention1 `shouldBe` False
    it "flipped input tokens should affect the outputs for those input tokens" $ do
      intervention2 `shouldBe` False
    it "flipped input tokens should affect the outputs for other non-flipped tokens" $ do
      intervention3 `shouldBe` False
    it "flipped target tokens should affect the outputs for target tokens" $ do
      intervention4 `shouldBe` False
    it "flipped target tokens should not affect the outputs for input tokens" $ do
      intervention5 `shouldBe` True
    it "setting all relations to zero should change outputs" $ do
      intervention6 `shouldBe` False

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
