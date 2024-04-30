{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.HList where

import Control.Applicative (Applicative (liftA2))
import Data.Kind
  ( Type,
  )
import GHC.Exts (IsList (..))
import GHC.TypeLits (Nat, type (+), type (-), type (<=))
import Prelude hiding (id, (.))

type family ListLength (xs :: [k]) :: Nat where
  ListLength '[] = 0
  ListLength (_h ': t) = 1 + ListLength t

data family HList (xs :: [k])

data instance HList '[] = HNil

newtype instance HList ((x :: Type) ': xs) = HCons (x, HList xs)

pattern (:.) :: forall x (xs :: [Type]). x -> HList xs -> HList (x : xs)
pattern (:.) x xs = HCons (x, xs)

infixr 2 :.

instance Show (HList '[]) where
  show _ = "H[]"

instance (Show e, Show (HList l)) => Show (HList (e ': l)) where
  show (x :. l) =
    let 'H' : '[' : s = show l
     in "H[" ++ show x ++ (if s == "]" then s else "," ++ s)

instance Eq (HList '[]) where
  HNil == HNil = True

instance (Eq x, Eq (HList xs)) => Eq (HList (x ': xs)) where
  (x :. xs) == (y :. ys) = x == y && xs == ys

instance Semigroup (HList '[]) where
  _ <> _ = HNil

instance (Semigroup a, Semigroup (HList as)) => Semigroup (HList (a ': as)) where
  (x :. xs) <> (y :. ys) = (x <> y) :. (xs <> ys)

instance Monoid (HList '[]) where
  mempty = HNil

instance (Monoid a, Monoid (HList as)) => Monoid (HList (a ': as)) where
  mempty = mempty :. mempty


{- HLINT ignore "Redundant bracket" -}
instance IsList (Maybe (HList '[(a :: Type)])) where
  type Item (Maybe (HList '[(a :: Type)])) = a
  fromList [x] = liftA2 (:.) (Just x) (Just HNil)
  fromList _ = Nothing
  toList Nothing = []
  toList (Just (x :. HNil)) = [x]

instance
  ( IsList (Maybe (HList (a ': as))),
    a ~ Item (Maybe (HList (a ': as)))
  ) =>
  IsList (Maybe (HList ((a :: Type) ': a ': as)))
  where
  type Item (Maybe (HList (a ': a ': as))) = a
  fromList (x : xs) = liftA2 (:.) (Just x) (fromList xs)
  fromList _ = Nothing
  toList Nothing = []
  toList (Just (x :. xs)) = x : toList (Just xs)

class Apply f a b where
  apply :: f -> a -> b

-- | Stronger version of `Apply` that allows for better inference of the return type
class Apply' f a b | f a -> b where
  apply' :: f -> a -> b

data AFst = AFst

instance Apply' AFst (a, b) a where
  apply' _ (a, _) = a

data ASnd = ASnd

instance Apply' ASnd (a, b) b where
  apply' _ (_, b) = b

class HMap f (xs :: [k]) (ys :: [k]) where
  hmap :: f -> HList xs -> HList ys

instance HMap f '[] '[] where
  hmap _ _ = HNil

instance (Apply f x y, HMap f xs ys) => HMap f (x ': xs) (y ': ys) where
  hmap f (x :. xs) = apply f x :. hmap f xs

-- | Alternative version of `HMap` with better type inference based on `Apply'`
class HMap' f (xs :: [k]) (ys :: [k]) | f xs -> ys where
  hmap' :: f -> HList xs -> HList ys

instance HMap' f '[] '[] where
  hmap' _ _ = HNil

instance (Apply' f x y, HMap' f xs ys) => HMap' f (x ': xs) (y ': ys) where
  hmap' f (x :. xs) = apply' f x :. hmap' f xs

class HMapM m f (xs :: [k]) (ys :: [k]) where
  hmapM :: f -> HList xs -> m (HList ys)

instance (Monad m) => HMapM m f '[] '[] where
  hmapM _ _ = pure HNil

instance
  ( Monad m,
    Apply f x (m y),
    HMapM m f xs ys
  ) =>
  HMapM m f (x ': xs) (y ': ys)
  where
  hmapM f (x :. xs) = (:.) <$> apply f x <*> hmapM f xs

class HMapM' m f (xs :: [k]) (ys :: [k]) | f xs -> ys where
  hmapM' :: f -> HList xs -> m (HList ys)

instance (Applicative m) => HMapM' m f '[] '[] where
  hmapM' _ _ = pure HNil

instance
  ( Applicative m,
    Apply' f x (m y),
    HMapM' m f xs ys
  ) =>
  HMapM' m f (x ': xs) (y ': ys)
  where
  hmapM' f (x :. xs) = (:.) <$> apply' f x <*> hmapM' f xs

class
  Applicative f =>
  HSequence f (xs :: [k]) (ys :: [k])
    | xs -> ys,
      ys f -> xs
  where
  hsequence :: HList xs -> f (HList ys)

instance Applicative f => HSequence f '[] '[] where
  hsequence = pure

instance
  ( Applicative g,
    HSequence f xs ys,
    y ~ x,
    f ~ g
  ) =>
  HSequence g (f x ': xs) (y ': ys)
  where
  hsequence (fx :. fxs) = (:.) <$> fx <*> hsequence fxs

class HFoldr f acc xs res | f acc xs -> res where
  hfoldr :: f -> acc -> HList xs -> res

instance (acc ~ res) => HFoldr f acc '[] res where
  hfoldr _ acc _ = acc

instance
  ( Apply' f (x, res) res',
    HFoldr f acc xs res
  ) =>
  HFoldr f acc (x ': xs) res'
  where
  hfoldr f acc (x :. xs) = apply' f (x, hfoldr f acc xs)

class HFoldrM m f acc xs res | m f acc xs -> res where
  hfoldrM :: f -> acc -> HList xs -> m res

instance
  ( Monad m,
    acc ~ res
  ) =>
  HFoldrM m f acc '[] res
  where
  hfoldrM _ acc _ = pure acc

instance
  ( Monad m,
    Apply' f (x, m res) (m res'),
    HFoldrM m f acc xs res
  ) =>
  HFoldrM m f acc (x ': xs) res'
  where
  hfoldrM f acc (x :. xs) = apply' f (x, hfoldrM f acc xs :: (m res))

data HNothing = HNothing

newtype HJust x = HJust x

class HUnfold f res xs where
  hunfoldr' :: f -> res -> HList xs

type family HUnfoldRes s xs where
  HUnfoldRes _ '[] = HNothing
  HUnfoldRes s (x ': _) = HJust (x, s)

instance HUnfold f HNothing '[] where
  hunfoldr' _ _ = HNil

instance
  ( Apply f s res,
    HUnfold f res xs,
    res ~ HUnfoldRes s xs
  ) =>
  HUnfold f (HJust (x, s)) (x ': xs)
  where
  hunfoldr' f (HJust (x, s)) = x :. hunfoldr' f (apply f s :: res)

hunfoldr ::
  forall f res (xs :: [Type]) a.
  (Apply f a res, HUnfold f res xs, res ~ HUnfoldRes a xs) =>
  f ->
  a ->
  HList xs
hunfoldr f s = hunfoldr' f (apply f s :: res)

class HUnfoldM m f res xs where
  hunfoldrM' :: f -> res -> m (HList xs)

type family HUnfoldMRes m s xs where
  HUnfoldMRes m _ '[] = m HNothing
  HUnfoldMRes m s (x ': _) = m (HJust (x, s))

instance (Monad m) => HUnfoldM m f (m HNothing) '[] where
  hunfoldrM' _ _ = pure HNil

instance
  ( Monad m,
    HUnfoldM m f res xs,
    Apply f s res,
    res ~ HUnfoldMRes m s xs
  ) =>
  HUnfoldM m f (m (HJust (x, s))) (x ': xs)
  where
  hunfoldrM' f just = do
    HJust (x, s) <- just
    xs <- hunfoldrM' f (apply f s :: res)
    return (x :. xs)

hunfoldrM ::
  forall (m :: Type -> Type) f res (xs :: [Type]) a.
  (HUnfoldM m f res xs, Apply f a res, res ~ HUnfoldMRes m a xs) =>
  f ->
  a ->
  m (HList xs)
hunfoldrM f s = hunfoldrM' f (apply f s :: res)

type HReplicate n e = HReplicateFD n e (HReplicateR n e)

hreplicate :: forall n e. HReplicate n e => e -> HList (HReplicateR n e)
hreplicate = hreplicateFD @n

class
  HReplicateFD
    (n :: Nat)
    (e :: Type)
    (es :: [Type])
    | n e -> es
  where
  hreplicateFD :: e -> HList es

instance {-# OVERLAPS #-} HReplicateFD 0 e '[] where
  hreplicateFD _ = HNil

instance
  {-# OVERLAPPABLE #-}
  ( HReplicateFD (n - 1) e es,
    es' ~ (e ': es),
    1 <= n
  ) =>
  HReplicateFD n e es'
  where
  hreplicateFD e = e :. hreplicateFD @(n - 1) e

type family HReplicateR (n :: Nat) (e :: a) :: [a] where
  HReplicateR 0 _ = '[]
  HReplicateR n e = e ': HReplicateR (n - 1) e

type HConcat xs = HConcatFD xs (HConcatR xs)

hconcat :: HConcat xs => HList xs -> HList (HConcatR xs)
hconcat = hconcatFD

type family HConcatR (a :: [Type]) :: [Type]

type instance HConcatR '[] = '[]

type instance HConcatR (x ': xs) = UnHList x ++ HConcatR xs

type family UnHList a :: [Type]

type instance UnHList (HList a) = a

class HConcatFD (xxs :: [k]) (xs :: [k]) | xxs -> xs where
  hconcatFD :: HList xxs -> HList xs

instance HConcatFD '[] '[] where
  hconcatFD _ = HNil

instance (HConcatFD as bs, HAppendFD a bs cs) => HConcatFD (HList a ': as) cs where
  hconcatFD (x :. xs) = x `happendFD` hconcatFD xs

type HAppend as bs = HAppendFD as bs (as ++ bs)

happend :: HAppend as bs => HList as -> HList bs -> HList (as ++ bs)
happend = happendFD

hunappend ::
  ( cs ~ (as ++ bs),
    HAppend as bs
  ) =>
  HList cs ->
  (HList as, HList bs)
hunappend = hunappendFD

class HAppendFD (a :: [k]) (b :: [k]) (ab :: [k]) | a b -> ab, a ab -> b where
  happendFD :: HList a -> HList b -> HList ab
  hunappendFD :: HList ab -> (HList a, HList b)

type family (as :: [k]) ++ (bs :: [k]) :: [k] where
  '[] ++ bs = bs
  (a ': as) ++ bs = a ': as ++ bs

instance HAppendFD '[] b b where
  happendFD _ b = b
  hunappendFD b = (HNil, b)

instance
  ( HAppendFD as bs cs
  ) =>
  HAppendFD (a ': as :: [Type]) bs (a ': cs :: [Type])
  where
  happendFD (a :. as) bs = a :. happendFD as bs
  hunappendFD (a :. cs) = let (as, bs) = hunappendFD cs in (a :. as, bs)

class HZip (xs :: [k]) (ys :: [k]) (zs :: [k]) | xs ys -> zs, zs -> xs ys where
  hzip :: HList xs -> HList ys -> HList zs
  hunzip :: HList zs -> (HList xs, HList ys)

instance HZip '[] '[] '[] where
  hzip _ _ = HNil
  hunzip _ = (HNil, HNil)

instance ((x, y) ~ z, HZip xs ys zs) => HZip (x ': xs) (y ': ys) (z ': zs) where
  hzip (x :. xs) (y :. ys) = (x, y) :. hzip xs ys
  hunzip (~(x, y) :. zs) = let ~(xs, ys) = hunzip zs in (x :. xs, y :. ys)

class HZip' (xs :: [k]) (ys :: [k]) (zs :: [k]) | xs ys -> zs where
  hzip' :: HList xs -> HList ys -> HList zs

instance HZip' '[] '[] '[] where
  hzip' _ _ = HNil

instance
  ( HList (x ': y) ~ z,
    HZip' xs ys zs
  ) =>
  HZip' (x ': xs) (HList y ': ys) (z ': zs)
  where
  hzip' (x :. xs) (y :. ys) = (x :. y) :. hzip' xs ys

data HZipF = HZipF

instance
  ( HZip' a b c,
    x ~ (HList a, HList b),
    y ~ HList c
  ) =>
  Apply' HZipF x y
  where
  apply' _ (x, y) = hzip' x y

htranspose ::
  forall (acc :: [Type]) (xs :: [Type]) (xxs :: [Type]) (res :: Type).
  ( HReplicateFD (ListLength xs) (HList ('[] :: [Type])) acc,
    HFoldr HZipF (HList acc) (HList xs : xxs) res
  ) =>
  HList (HList xs : xxs) ->
  res
htranspose (xs :. xxs) =
  hfoldr
    HZipF
    (hreplicateFD @(ListLength xs) (HNil :: HList ('[] :: [Type])))
    (xs :. xxs)

class HZipWith f (xs :: [k]) (ys :: [k]) (zs :: [k]) | f xs ys -> zs where
  hzipWith :: f -> HList xs -> HList ys -> HList zs

instance HZipWith f '[] '[] '[] where
  hzipWith _ _ _ = HNil

instance
  ( Apply' f (x, y) z,
    HZipWith f xs ys zs
  ) =>
  HZipWith f (x ': xs) (y ': ys) (z ': zs)
  where
  hzipWith f (x :. xs) (y :. ys) = apply' f (x, y) :. hzipWith f xs ys

class HZipWithM m f (xs :: [k]) (ys :: [k]) (zs :: [k]) | f xs ys -> zs where
  hzipWithM :: f -> HList xs -> HList ys -> m (HList zs)

instance (Applicative m) => HZipWithM m f '[] '[] '[] where
  hzipWithM _ _ _ = pure HNil

instance
  ( Applicative m,
    Apply' f (x, y) (m z),
    HZipWithM m f xs ys zs
  ) =>
  HZipWithM m f (x ': xs) (y ': ys) (z ': zs)
  where
  hzipWithM f (x :. xs) (y :. ys) = (:.) <$> apply' f (x, y) <*> hzipWithM f xs ys

class
  HZip3
    (as :: [k])
    (bs :: [k])
    (cs :: [k])
    (ds :: [k])
    | as bs cs -> ds,
      ds -> as bs cs
  where
  hzip3 :: HList as -> HList bs -> HList cs -> HList ds
  hunzip3 :: HList ds -> (HList as, HList bs, HList cs)

instance HZip3 '[] '[] '[] '[] where
  hzip3 _ _ _ = HNil
  hunzip3 _ = (HNil, HNil, HNil)

instance
  ( (a, b, c) ~ d,
    HZip3 as bs cs ds
  ) =>
  HZip3 (a ': as) (b ': bs) (c ': cs) (d ': ds)
  where
  hzip3 (a :. as) (b :. bs) (c :. cs) = (a, b, c) :. hzip3 as bs cs
  hunzip3 (~(a, b, c) :. ds) =
    let ~(as, bs, cs) = hunzip3 ds in (a :. as, b :. bs, c :. cs)

class
  HZipWith3
    f
    (as :: [k])
    (bs :: [k])
    (cs :: [k])
    (ds :: [k])
    | f as bs cs -> ds
  where
  hzipWith3 :: f -> HList as -> HList bs -> HList cs -> HList ds

instance HZipWith3 f '[] '[] '[] '[] where
  hzipWith3 _ _ _ _ = HNil

instance
  ( Apply' f (a, b, c) d,
    HZipWith3 f as bs cs ds
  ) =>
  HZipWith3 f (a ': as) (b ': bs) (c ': cs) (d ': ds)
  where
  hzipWith3 f (a :. as) (b :. bs) (c :. cs) = apply' f (a, b, c) :. hzipWith3 f as bs cs

class HCartesianProduct (xs :: [k]) (ys :: [k]) (zs :: [k]) | xs ys -> zs where
  hproduct :: HList xs -> HList ys -> HList zs

instance HCartesianProduct '[] ys '[] where
  hproduct _ _ = HNil

class HAttach x (ys :: [k]) (zs :: [k]) | x ys -> zs where
  hattach :: x -> HList ys -> HList zs

instance HAttach x '[] '[] where
  hattach _ _ = HNil

instance (HAttach x ys xys) => HAttach x (y ': ys) ((x, y) ': xys) where
  hattach x (y :. ys) = (x, y) :. hattach x ys

instance
  ( HCartesianProduct xs ys zs,
    HAttach x ys xys,
    HAppendFD xys zs zs'
  ) =>
  HCartesianProduct (x ': xs) ys zs'
  where
  hproduct (x :. xs) ys = hattach x ys `happendFD` hproduct xs ys
