{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

-- | Loading and saving models by /name/, in PyTorch's state-dict format.
--
-- A checkpoint key like @transformer.h.0.attn.weight@ is a path through a
-- record structure: field names joined by dots, list positions as numbers.
-- Haskell models here /are/ such record structures, so both directions are
-- derivable from 'GHC.Generics.Generic': 'fromStateDict' walks the record and
-- looks each leaf up by its path, 'toStateDict' walks it and writes each leaf
-- down.  No per-model plumbing, and unknown or missing keys fail with the
-- full path in the error.
--
-- The typed counterparts ("Torch.Typed.StateDict") add shape, dtype, and
-- device checking at each leaf, so a mismatched checkpoint fails at load
-- time with the offending path and both shapes.
module Torch.StateDict
  ( StateDict,
    FromStateDict (..),
    ToStateDict (..),
    loadStateDict,
    saveStateDict,
    stateDictKeys,
    -- * Path helpers (for writing manual instances)
    childPath,
    lookupPath,
  )
where

import Control.Monad (forM)
import Data.List (sortOn)
import qualified Data.Map.Strict as Map
import Data.Proxy (Proxy (..))
import GHC.Generics
import GHC.TypeLits (KnownSymbol, symbolVal)
import Torch.Autograd (IndependentTensor (..), makeIndependent)
import Torch.NN (Linear, Parameter)
import Torch.Script (IValue (..))
import qualified Torch.Serialize as S
import Torch.Tensor (Tensor)

-- | A checkpoint: tensors addressed by dotted paths.
type StateDict = Map.Map String Tensor

-- | Read a value out of a state dict at (and below) a path prefix.  For the
-- whole checkpoint, the prefix is @\"\"@.
class FromStateDict a where
  fromStateDict :: StateDict -> String -> IO a
  default fromStateDict :: (Generic a, GFromStateDict (Rep a)) => StateDict -> String -> IO a
  fromStateDict sd prefix = to <$> gFromStateDict sd prefix

-- | Write a value into a state dict at (and below) a path prefix.
class ToStateDict a where
  toStateDict :: a -> String -> StateDict
  default toStateDict :: (Generic a, GToStateDict (Rep a)) => a -> String -> StateDict
  toStateDict a prefix = gToStateDict (from a) prefix

-- | Extend a path with a child segment.
childPath :: String -> String -> String
childPath "" seg = seg
childPath prefix seg = prefix <> "." <> seg

-- | Look a path up, failing with the path (and the nearby keys) on a miss.
lookupPath :: StateDict -> String -> IO Tensor
lookupPath sd path = case Map.lookup path sd of
  Just t -> pure t
  Nothing ->
    fail $
      "state dict has no key "
        <> show path
        <> "; nearby keys: "
        <> show (take 10 (filter (samePrefix path) (Map.keys sd)))
  where
    samePrefix p k = takeWhile (/= '.') p == takeWhile (/= '.') k

instance FromStateDict Tensor where
  fromStateDict sd prefix = lookupPath sd prefix

instance ToStateDict Tensor where
  toStateDict t prefix = Map.singleton prefix t

instance FromStateDict Parameter where
  fromStateDict sd prefix = makeIndependent =<< lookupPath sd prefix

instance ToStateDict Parameter where
  toStateDict p prefix = Map.singleton prefix (toDependent p)

-- | Lists are numbered children @prefix.0@, @prefix.1@, …; loading probes
-- indices upward until the first one with no keys below it.
instance FromStateDict a => FromStateDict [a] where
  fromStateDict sd prefix = go 0
    where
      go :: Int -> IO [a]
      go i =
        let p = childPath prefix (show i)
         in if keyBelow p
              then (:) <$> fromStateDict sd p <*> go (i + 1)
              else pure []
      keyBelow p = p `Map.member` sd || not (Map.null (submap p))
      submap p = Map.filterWithKey (\k _ -> (p <> ".") `isPrefixOfKey` k) sd
      isPrefixOfKey pre k = take (length pre) k == pre

instance ToStateDict a => ToStateDict [a] where
  toStateDict xs prefix =
    Map.unions [toStateDict x (childPath prefix (show i)) | (i, x) <- zip [(0 :: Int) ..] xs]

-- | Load a PyTorch checkpoint saved as a pickled dict of tensors — the
-- @torch.save(dict(model.state_dict()), path)@ format.
loadStateDict :: FilePath -> IO StateDict
loadStateDict path = do
  iv <- S.pickleLoad path
  case iv of
    IVGenericDict kvs ->
      fmap (Map.fromList . concat) . forM kvs $ \kv -> case kv of
        (IVString k, IVTensor t) -> pure [(k, t)]
        _ -> pure []
    _ -> fail $ "loadStateDict: " <> path <> " is not a pickled dict"

-- | Save in the same format 'loadStateDict' (and Python's @torch.load@)
-- reads.
saveStateDict :: StateDict -> FilePath -> IO ()
saveStateDict sd path =
  S.pickleSave
    (IVGenericDict [(IVString k, IVTensor v) | (k, v) <- Map.toList sd])
    path

-- | The keys, sorted; handy when a load fails and you want to see the
-- checkpoint's actual layout.
stateDictKeys :: StateDict -> [String]
stateDictKeys = sortOn id . Map.keys

--------------------------------------------------------------------------------
-- Generic machinery: field names become path segments
--------------------------------------------------------------------------------

class GFromStateDict f where
  gFromStateDict :: StateDict -> String -> IO (f a)

class GToStateDict f where
  gToStateDict :: f a -> String -> StateDict

instance GFromStateDict f => GFromStateDict (D1 c f) where
  gFromStateDict sd p = M1 <$> gFromStateDict sd p

instance GToStateDict f => GToStateDict (D1 c f) where
  gToStateDict (M1 x) p = gToStateDict x p

instance GFromStateDict f => GFromStateDict (C1 c f) where
  gFromStateDict sd p = M1 <$> gFromStateDict sd p

instance GToStateDict f => GToStateDict (C1 c f) where
  gToStateDict (M1 x) p = gToStateDict x p

instance
  (KnownSymbol name, FromStateDict a) =>
  GFromStateDict (S1 ('MetaSel ('Just name) su ss ds) (K1 i a))
  where
  gFromStateDict sd p =
    M1 . K1 <$> fromStateDict sd (childPath p (symbolVal (Proxy @name)))

instance
  (KnownSymbol name, ToStateDict a) =>
  GToStateDict (S1 ('MetaSel ('Just name) su ss ds) (K1 i a))
  where
  gToStateDict (M1 (K1 x)) p =
    toStateDict x (childPath p (symbolVal (Proxy @name)))

instance (GFromStateDict f, GFromStateDict g) => GFromStateDict (f :*: g) where
  gFromStateDict sd p = (:*:) <$> gFromStateDict sd p <*> gFromStateDict sd p

instance (GToStateDict f, GToStateDict g) => GToStateDict (f :*: g) where
  gToStateDict (x :*: y) p = gToStateDict x p `Map.union` gToStateDict y p

instance GFromStateDict U1 where
  gFromStateDict _ _ = pure U1

instance GToStateDict U1 where
  gToStateDict _ _ = Map.empty

-- Instances for the common built-in modules; their field names (weight,
-- bias) match the checkpoint conventions PyTorch itself uses.

instance FromStateDict Linear

instance ToStateDict Linear
