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

module Data.HList where

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

class HMap f xs ys where
  hmap :: f -> HList xs -> HList ys

instance HMap f '[] '[] where
  hmap _ _ = HNil

instance (Apply f x y, HMap f xs ys) => HMap f (x ': xs) (y ': ys) where
  hmap f (x :. xs) = apply f x :. hmap f xs

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

class (ListLength es ~ n) => HReplicate' (n :: Nat) e es where
  hReplicate :: Proxy n -> e -> HList es

instance HReplicate' 0 e '[] where
  hReplicate _ _ = HNil

instance (HReplicate' (n - 1) e es, e ~ e', 1 <= n) => HReplicate' n e (e' ': es) where
  hReplicate n e = e :. hReplicate (Proxy @(n - 1)) e

type HReplicate n e = HReplicate' n e (HReplicateR n e)

type family HReplicateR (n :: Nat) (e :: a) :: [a] where
  HReplicateR 0 e = '[]
  HReplicateR n e = e ': HReplicateR (n - 1) e

type HConcat xs = HConcatFD xs (HConcatR xs)

hConcat :: HConcat xs => HList xs -> HList (HConcatR xs)
hConcat = hConcatFD

type family HConcatR (a :: [Type]) :: [Type]
type instance HConcatR '[] = '[]
type instance HConcatR (x ': xs) = (UnHList x) ++ (HConcatR xs)

type family UnHList a :: [Type]
type instance UnHList (HList a) = a

-- for the benefit of ghc-7.10.1
class HConcatFD xxs xs | xxs -> xs
  where hConcatFD :: HList xxs -> HList xs

instance HConcatFD '[] '[] where
  hConcatFD _ = HNil

instance (HConcatFD as bs, HAppendFD a bs cs) => HConcatFD (HList a ': as) cs where
  hConcatFD (x :. xs) = x `hAppendFD` hConcatFD xs

class HAppendFD a b ab | a b -> ab, a ab -> b where
  hAppendFD :: HList a -> HList b -> HList ab
  hUnappendFD :: HList ab -> (HList a, HList b)

type family ((as :: [k]) ++ (bs :: [k])) :: [k] where
  '[]       ++ bs = bs
  (a ': as) ++ bs = a ': as ++ bs

instance HAppendFD '[] b b where
  hAppendFD _ b = b
  hUnappendFD b = (HNil, b)

instance HAppendFD as bs cs => HAppendFD (a ': as) bs (a ': cs) where
  hAppendFD (a :. as) bs = a :. hAppendFD as bs
  hUnappendFD (a :. cs) = let (as, bs) = hUnappendFD cs in (a :. as, bs)

class HZipList x y l | x y -> l, l -> x y where
  hZipList   :: HList x -> HList y -> HList l
  hUnzipList :: HList l -> (HList x, HList y)

instance HZipList '[] '[] '[] where
  hZipList _ _ = HNil
  hUnzipList _ = (HNil, HNil)

instance ((x, y) ~ z, HZipList xs ys zs) => HZipList (x ': xs) (y ': ys) (z ': zs) where
  hZipList (x :. xs) (y :. ys) = (x, y) :. hZipList xs ys
  hUnzipList (~(x, y) :. zs) =
    let ~(xs, ys) = hUnzipList zs in (x :. xs, y :. ys)

class HCartesianProduct xs ys zs | xs ys -> zs where
  hCartesianProduct :: HList xs -> HList ys -> HList zs

instance HCartesianProduct '[] ys '[] where
  hCartesianProduct _ _ = HNil

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
  hCartesianProduct (x :. xs) ys = hattach x ys `hAppendFD` hCartesianProduct xs ys
