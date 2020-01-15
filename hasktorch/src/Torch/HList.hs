{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE FunctionalDependencies #-}

module Torch.HList where

import           Prelude                 hiding ( (.), id )
import           Control.Arrow
import           Control.Category
import           Data.Kind                      ( Constraint
                                                , Type
                                                )
import           Data.Proxy
import           GHC.TypeLits

type family ListLength (l :: [a]) :: Nat where
  ListLength '[]      = 0
  ListLength (_ ': t) = 1 + ListLength t

data family HList (l :: [Type])
data instance HList '[] = HNil
newtype instance HList (x ': xs) = HCons (x, HList xs)
pattern (:.) x xs = HCons (x, xs)

infixr 2 :.

instance Show (HList '[]) where
  show _ = "H[]"

instance (Show e, Show (HList l)) => Show (HList (e ': l)) where
  show (x :. l) =
    let 'H' : '[' : s = show l
    in  "H[" ++ show x ++ (if s == "]" then s else "," ++ s)

instance Eq (HList '[]) where
  HNil == HNil = True

instance (Eq x, Eq (HList xs)) => Eq (HList (x ': xs)) where
  (x :. xs) == (y :. ys) = x == y && xs == ys

instance Semigroup (HList '[]) where
  _ <> _ = HNil

instance (Semigroup a, Semigroup (HList as)) => Semigroup (HList (a ': as)) where
  (x :. xs) <> (y :. ys) = (x <> y) :. (xs <> ys)

instance Monoid (HList '[]) where
  mempty  = HNil
  mappend _ _ = HNil

instance (Monoid a, Monoid (HList as)) => Monoid (HList (a ': as)) where
  mempty = mempty :. mempty
  mappend (x :. xs) (y :. ys) = mappend x y :. mappend xs ys

class Apply f a b where
  apply :: f -> a -> b

-- | Stronger version of `Apply` that allows for better inference of the return type
class Apply' f a b | f a -> b where
  apply' :: f -> a -> b

data Fst = Fst

instance Apply' Fst (a, b) a
  where
    apply' _ (a, _) = a

data Snd = Snd

instance Apply' Snd (a, b) b
  where
    apply' _ (_, b) = b

class HMap f xs ys where
  hmap :: f -> HList xs -> HList ys

instance HMap f '[] '[] where
  hmap _ _ = HNil

instance (Apply f x y, HMap f xs ys) => HMap f (x ': xs) (y ': ys) where
  hmap f (x :. xs) = apply f x :. hmap f xs

-- | Alternative version of `HMap` with better type inference based on `Apply'`
class HMap' f xs ys | f xs -> ys where
  hmap' :: f -> HList xs -> HList ys

instance HMap' f '[] '[] where
  hmap' _ _ = HNil

instance (Apply' f x y, HMap' f xs ys) => HMap' f (x ': xs) (y ': ys) where
  hmap' f (x :. xs) = apply' f x :. hmap' f xs

class HMapM m f xs ys where
  hmapM :: f -> HList xs -> m (HList ys)

instance (Monad m) => HMapM m f '[] '[] where
  hmapM _ _ = pure HNil

instance (Monad m, Apply f x (m y), HMapM m f xs ys) => HMapM m f (x ': xs) (y ': ys) where
  hmapM f (x :. xs) = (:.) <$> apply f x <*> hmapM f xs

class HMapM' m f xs ys | f xs -> ys where
  hmapM' :: f -> HList xs -> m (HList ys)

instance (Monad m) => HMapM' m f '[] '[] where
  hmapM' _ _ = pure HNil

instance (Monad m, Apply' f x (m y), HMapM' m f xs ys) => HMapM' m f (x ': xs) (y ': ys) where
  hmapM' f (x :. xs) = (:.) <$> apply' f x <*> hmapM' f xs

class Applicative f => HSequence f xs ys | xs -> ys, ys f -> xs where
  hsequence :: HList xs -> f (HList ys)

instance Applicative f => HSequence f '[] '[] where
  hsequence = pure

instance ( Applicative g
         , HSequence f xs ys
         , y ~ x
         , f ~ g
         )
  => HSequence g (f x ': xs) (y ': ys)
 where
  hsequence (fx :. fxs) = (:.) <$> fx <*> hsequence fxs

class HFoldr f acc xs where
  hfoldr :: f -> acc -> HList xs -> acc

instance HFoldr f acc '[] where
  hfoldr _ acc _ = acc

instance (Apply f x (acc -> acc), HFoldr f acc xs) => HFoldr f acc (x ': xs) where
  hfoldr f acc (x :. xs) = apply f x $ hfoldr f acc xs

class HFoldrM m f acc xs where
  hfoldrM :: f -> acc -> HList xs -> m acc

instance (Monad m) => HFoldrM m f acc '[] where
  hfoldrM _ acc _ = pure acc

instance (Monad m, Apply f x (acc -> m acc), HFoldrM m f acc xs) => HFoldrM m f acc (x ': xs) where
  hfoldrM f acc (x :. xs) = apply f x =<< hfoldrM f acc xs

class HFoldrM' m f acc xs where
  hfoldrM' :: f -> HList xs -> Kleisli m acc acc

instance (Monad m) => HFoldrM' m f acc '[] where
  hfoldrM' _ _ = arr id

instance (Monad m, Apply f x (Kleisli m acc acc), HFoldrM' m f acc xs) => HFoldrM' m f acc (x ': xs) where
  hfoldrM' f (x :. xs) = hfoldrM' f xs >>> apply f x

data HNothing  = HNothing
data HJust x   = HJust x

class HUnfold f res xs where
  hunfoldr' :: f -> res -> HList xs

type family HUnfoldRes s xs where
  HUnfoldRes _ '[] = HNothing
  HUnfoldRes s (x ': _) = HJust (x, s)

instance HUnfold f HNothing '[] where
  hunfoldr' _ _ = HNil

instance (Apply f s res, HUnfold f res xs, res ~ HUnfoldRes s xs) => HUnfold f (HJust (x, s)) (x ': xs) where
  hunfoldr' f (HJust (x, s)) = x :. hunfoldr' f (apply f s :: res)

hunfoldr
  :: forall f res (xs :: [Type]) a
   . (Apply f a res, HUnfold f res xs)
  => f
  -> a
  -> HList xs
hunfoldr f s = hunfoldr' f (apply f s :: res)

class HUnfoldM m f res xs where
  hunfoldrM' :: f -> res -> m (HList xs)

type family HUnfoldMRes m s xs where
  HUnfoldMRes m _ '[] = m HNothing
  HUnfoldMRes m s (x ': _) = m (HJust (x, s))

instance (Monad m) => HUnfoldM m f (m HNothing) '[] where
  hunfoldrM' _ _ = pure HNil

instance (Monad m, HUnfoldM m f res xs, Apply f s res, res ~ HUnfoldMRes m s xs) => HUnfoldM m f (m (HJust (x, s))) (x ': xs) where
  hunfoldrM' f just = do
    HJust (x, s) <- just
    xs <- hunfoldrM' f (apply f s :: res)
    return (x :. xs)

hunfoldrM
  :: forall (m :: Type -> Type) f res (xs :: [Type]) a
   . (HUnfoldM m f res xs, Apply f a res, res ~ HUnfoldMRes m a xs)
  => f
  -> a
  -> m (HList xs)
hunfoldrM f s = hunfoldrM' f (apply f s :: res)

type HReplicate n e = HReplicateFD n e (HReplicateR n e)

hreplicate :: forall n e . HReplicate n e => e -> HList (HReplicateR n e)
hreplicate = hreplicateFD @n

class HReplicateFD (n :: Nat) e es | n e -> es where
  hreplicateFD :: e -> HList es

instance {-# OVERLAPS #-} HReplicateFD 0 e '[] where
  hreplicateFD _ = HNil

instance {-# OVERLAPPABLE #-} (HReplicateFD (n - 1) e es, es' ~ (e ': es), 1 <= n) => HReplicateFD n e es' where
  hreplicateFD e = e :. hreplicateFD @(n - 1) e

type family HReplicateR (n :: Nat) (e :: a) :: [a] where
  HReplicateR 0 e = '[]
  HReplicateR n e = e ': HReplicateR (n - 1) e

type HConcat xs = HConcatFD xs (HConcatR xs)

hconcat :: HConcat xs => HList xs -> HList (HConcatR xs)
hconcat = hconcatFD

type family HConcatR (a :: [Type]) :: [Type]
type instance HConcatR '[] = '[]
type instance HConcatR (x ': xs) = UnHList x ++ HConcatR xs

type family UnHList a :: [Type]
type instance UnHList (HList a) = a

-- for the benefit of ghc-7.10.1
class HConcatFD xxs xs | xxs -> xs
  where hconcatFD :: HList xxs -> HList xs

instance HConcatFD '[] '[] where
  hconcatFD _ = HNil

instance (HConcatFD as bs, HAppendFD a bs cs) => HConcatFD (HList a ': as) cs where
  hconcatFD (x :. xs) = x `happendFD` hconcatFD xs

type HAppend as bs = HAppendFD as bs (as ++ bs)

happend :: HAppend as bs => HList as -> HList bs -> HList (as ++ bs)
happend = happendFD

hunappend :: (cs ~ (as ++ bs), HAppend as bs) => HList cs -> (HList as, HList bs)
hunappend = hunappendFD

class HAppendFD a b ab | a b -> ab, a ab -> b where
  happendFD :: HList a -> HList b -> HList ab
  hunappendFD :: HList ab -> (HList a, HList b)

type family ((as :: [k]) ++ (bs :: [k])) :: [k] where
  '[]       ++ bs = bs
  (a ': as) ++ bs = a ': as ++ bs

instance HAppendFD '[] b b where
  happendFD _ b = b
  hunappendFD b = (HNil, b)

instance HAppendFD as bs cs => HAppendFD (a ': as) bs (a ': cs) where
  happendFD (a :. as) bs = a :. happendFD as bs
  hunappendFD (a :. cs) = let (as, bs) = hunappendFD cs in (a :. as, bs)

class HZip xs ys zs | xs ys -> zs, zs -> xs ys where
  hzip   :: HList xs -> HList ys -> HList zs
  hunzip :: HList zs -> (HList xs, HList ys)

instance HZip '[] '[] '[] where
  hzip _ _ = HNil
  hunzip _ = (HNil, HNil)

instance ((x, y) ~ z, HZip xs ys zs) => HZip (x ': xs) (y ': ys) (z ': zs) where
  hzip (x :. xs) (y :. ys) = (x, y) :. hzip xs ys
  hunzip (~(x, y) :. zs) = let ~(xs, ys) = hunzip zs in (x :. xs, y :. ys)

class HZipWith f xs ys zs | f xs ys -> zs where
  hZipWith :: f -> HList xs -> HList ys -> HList zs

instance HZipWith f '[] '[] '[] where
  hZipWith _ _ _ = HNil

instance (Apply' f (x, y) z, HZipWith f xs ys zs) => HZipWith f (x ': xs) (y ': ys) (z ': zs) where
  hZipWith f (x :. xs) (y :. ys) = apply' f (x, y) :. hZipWith f xs ys

class HZipList3 as bs cs ds | as bs cs -> ds, ds -> as bs cs where
  hZipList3   :: HList as -> HList bs -> HList cs -> HList ds
  hUnzipList3 :: HList ds -> (HList as, HList bs, HList cs)

instance HZipList3 '[] '[] '[] '[] where
  hZipList3 _ _ _ = HNil
  hUnzipList3 _ = (HNil, HNil, HNil)

instance ((a, b, c) ~ d, HZipList3 as bs cs ds) => HZipList3 (a ': as) (b ': bs) (c ': cs) (d ': ds) where
  hZipList3 (a :. as) (b :. bs) (c :. cs) = (a, b, c) :. hZipList3 as bs cs
  hUnzipList3 (~(a, b, c) :. ds) =
    let ~(as, bs, cs) = hUnzipList3 ds in (a :. as, b :. bs, c :. cs)

class HZipWith3 f as bs cs ds | f as bs cs -> ds where
  hZipWith3 :: f -> HList as -> HList bs -> HList cs -> HList ds

instance HZipWith3 f '[] '[] '[] '[] where
  hZipWith3 _ _ _ _ = HNil

instance (Apply' f (a, b, c) d, HZipWith3 f as bs cs ds) => HZipWith3 f (a ': as) (b ': bs) (c ': cs) (d ': ds) where
  hZipWith3 f (a :. as) (b :. bs) (c :. cs) = apply' f (a, b, c) :. hZipWith3 f as bs cs

class HCartesianProduct xs ys zs | xs ys -> zs where
  hproduct :: HList xs -> HList ys -> HList zs

instance HCartesianProduct '[] ys '[] where
  hproduct _ _ = HNil

class HAttach x ys zs | x ys -> zs where
  hattach :: x -> HList ys -> HList zs

instance HAttach x '[] '[] where
  hattach _ _ = HNil

instance (HAttach x ys xys) => HAttach x (y ': ys) ((x, y) ': xys) where
  hattach x (y :. ys) = (x, y) :. hattach x ys

instance ( HCartesianProduct xs ys zs
         , HAttach x ys xys
         , HAppendFD xys zs zs'
         )
  => HCartesianProduct (x ': xs) ys zs'
 where
  hproduct (x :. xs) ys = hattach x ys `happendFD` hproduct xs ys
