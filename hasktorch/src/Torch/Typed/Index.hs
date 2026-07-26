{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TemplateHaskellQuotes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | Typed indexing and slicing, in the style of PR #613's
-- @Torch.GraduallyTyped.Tensor.Indexing@ but for the 'Torch.Typed' API: the
-- index is a type-level list, the result shape is computed by a type family,
-- and out-of-bounds indices are compile-time errors.
--
-- Given @t :: Tensor device dtype '[2, 3, 4]@:
--
-- > getSlice @'[SliceAt 1] t                    :: Tensor device dtype '[3, 4]
-- > getSlice @'[SliceAll, SliceAt 0] t          :: Tensor device dtype '[2, 4]
-- > getSlice @'[NewAxis, SliceFromUpTo 1 3] t   :: Tensor device dtype '[1, 2, 3, 4]
-- > getSlice @'[SliceFromUpToWithStep 0 3 2] t  :: Tensor device dtype '[2, 3, 4]
--
-- @getSlice \@'[SliceAt 5] t@ on a dimension of size 2 does not compile.
module Torch.Typed.Index
  ( -- * The index language
    IndexType (..),

    -- * PyTorch-style syntax
    --
    -- | The 'Torch.Index.slice' quasiquoter works in type position and
    -- produces the promoted @'[IndexType]@ list; 'parseIndices' is its
    -- implementation.
    parseIndices,

    -- * Result-shape computation
    IndexedShape,

    -- * Indexing
    getSlice,
    setSlice,
    KnownIndices (..),
  )
where

import Data.Char (isDigit, isSpace)
import Data.Type.Bool (If)
import GHC.TypeLits
import qualified Language.Haskell.TH as TH
import System.IO.Unsafe (unsafePerformIO)
import qualified Torch.Internal.Managed.Type.TensorIndex as ATen
import qualified Torch.Tensor as T
import Torch.Typed.Auxiliary (natValI)
import Torch.Typed.Tensor

-- | One index per dimension, at the type level.  The names follow PR #613.
data IndexType
  = -- | select a position, dropping the dimension
    SliceAt Nat
  | -- | keep the dimension as is
    SliceAll
  | -- | insert a dimension of size 1
    NewAxis
  | -- | @[from:]@
    SliceFrom Nat
  | -- | @[:to]@
    SliceUpTo Nat
  | -- | @[from:to]@
    SliceFromUpTo Nat Nat
  | -- | @[from::step]@
    SliceFromWithStep Nat Nat
  | -- | @[:to:step]@
    SliceUpToWithStep Nat Nat
  | -- | @[::step]@
    SliceWithStep Nat
  | -- | @[from:to:step]@
    SliceFromUpToWithStep Nat Nat Nat

-- | The number of elements a slice @[from:to:step]@ of a dimension of size
-- @n@ keeps, with all bounds checked.
type family SliceLen (from :: Nat) (to :: Nat) (step :: Nat) (n :: Nat) :: Nat where
  SliceLen _ _ 0 _ = TypeError ('Text "Slice step must be positive.")
  SliceLen from to step n =
    If
      (from <=? to)
      ( If
          (to <=? n)
          (Div (to - from + step - 1) step)
          ( TypeError
              ( 'Text "Slice end "
                  ':<>: 'ShowType to
                  ':<>: 'Text " is out of bounds for a dimension of size "
                  ':<>: 'ShowType n
                  ':<>: 'Text "."
              )
          )
      )
      ( TypeError
          ( 'Text "Slice start "
              ':<>: 'ShowType from
              ':<>: 'Text " is greater than its end "
              ':<>: 'ShowType to
              ':<>: 'Text "."
          )
      )

-- | The shape produced by applying a list of indices to a shape.  Dimensions
-- beyond the indices are kept unchanged, as in PyTorch.
type family IndexedShape (ixs :: [IndexType]) (shape :: [Nat]) :: [Nat] where
  IndexedShape '[] shape = shape
  IndexedShape ('NewAxis ': ixs) shape = 1 ': IndexedShape ixs shape
  IndexedShape ('SliceAt i ': ixs) (n ': sh) =
    If
      (i + 1 <=? n)
      (IndexedShape ixs sh)
      ( TypeError
          ( 'Text "Index "
              ':<>: 'ShowType i
              ':<>: 'Text " is out of bounds for a dimension of size "
              ':<>: 'ShowType n
              ':<>: 'Text "."
          )
      )
  IndexedShape ('SliceAll ': ixs) (n ': sh) = n ': IndexedShape ixs sh
  IndexedShape ('SliceFrom f ': ixs) (n ': sh) = SliceLen f n 1 n ': IndexedShape ixs sh
  IndexedShape ('SliceUpTo t ': ixs) (n ': sh) = SliceLen 0 t 1 n ': IndexedShape ixs sh
  IndexedShape ('SliceFromUpTo f t ': ixs) (n ': sh) = SliceLen f t 1 n ': IndexedShape ixs sh
  IndexedShape ('SliceFromWithStep f s ': ixs) (n ': sh) = SliceLen f n s n ': IndexedShape ixs sh
  IndexedShape ('SliceUpToWithStep t s ': ixs) (n ': sh) = SliceLen 0 t s n ': IndexedShape ixs sh
  IndexedShape ('SliceWithStep s ': ixs) (n ': sh) = SliceLen 0 n s n ': IndexedShape ixs sh
  IndexedShape ('SliceFromUpToWithStep f t s ': ixs) (n ': sh) = SliceLen f t s n ': IndexedShape ixs sh
  IndexedShape (ix ': ixs) '[] =
    TypeError ('Text "Too many indices for the shape of the tensor.")

-- | Reify a type-level index list to ATen tensor indices.
class KnownIndices (ixs :: [IndexType]) where
  rawIndices :: IO [T.RawTensorIndex]

instance KnownIndices '[] where
  rawIndices = pure []

instance (KnownNat i, KnownIndices ixs) => KnownIndices ('SliceAt i ': ixs) where
  rawIndices =
    (:)
      <$> (T.RawTensorIndex <$> ATen.newTensorIndexWithInt (fromIntegral (natValI @i)))
      <*> rawIndices @ixs

instance KnownIndices ixs => KnownIndices ('SliceAll ': ixs) where
  rawIndices =
    (:)
      <$> (T.RawTensorIndex <$> ATen.newTensorIndexWithSlice 0 maxBound 1)
      <*> rawIndices @ixs

instance KnownIndices ixs => KnownIndices ('NewAxis ': ixs) where
  rawIndices =
    (:)
      <$> (T.RawTensorIndex <$> ATen.newTensorIndexWithNone)
      <*> rawIndices @ixs

instance (KnownNat f, KnownIndices ixs) => KnownIndices ('SliceFrom f ': ixs) where
  rawIndices =
    (:)
      <$> (T.RawTensorIndex <$> ATen.newTensorIndexWithSlice (fromIntegral (natValI @f)) maxBound 1)
      <*> rawIndices @ixs

instance (KnownNat t, KnownIndices ixs) => KnownIndices ('SliceUpTo t ': ixs) where
  rawIndices =
    (:)
      <$> (T.RawTensorIndex <$> ATen.newTensorIndexWithSlice 0 (fromIntegral (natValI @t)) 1)
      <*> rawIndices @ixs

instance (KnownNat f, KnownNat t, KnownIndices ixs) => KnownIndices ('SliceFromUpTo f t ': ixs) where
  rawIndices =
    (:)
      <$> (T.RawTensorIndex <$> ATen.newTensorIndexWithSlice (fromIntegral (natValI @f)) (fromIntegral (natValI @t)) 1)
      <*> rawIndices @ixs

instance (KnownNat f, KnownNat s, KnownIndices ixs) => KnownIndices ('SliceFromWithStep f s ': ixs) where
  rawIndices =
    (:)
      <$> (T.RawTensorIndex <$> ATen.newTensorIndexWithSlice (fromIntegral (natValI @f)) maxBound (fromIntegral (natValI @s)))
      <*> rawIndices @ixs

instance (KnownNat t, KnownNat s, KnownIndices ixs) => KnownIndices ('SliceUpToWithStep t s ': ixs) where
  rawIndices =
    (:)
      <$> (T.RawTensorIndex <$> ATen.newTensorIndexWithSlice 0 (fromIntegral (natValI @t)) (fromIntegral (natValI @s)))
      <*> rawIndices @ixs

instance (KnownNat s, KnownIndices ixs) => KnownIndices ('SliceWithStep s ': ixs) where
  rawIndices =
    (:)
      <$> (T.RawTensorIndex <$> ATen.newTensorIndexWithSlice 0 maxBound (fromIntegral (natValI @s)))
      <*> rawIndices @ixs

instance (KnownNat f, KnownNat t, KnownNat s, KnownIndices ixs) => KnownIndices ('SliceFromUpToWithStep f t s ': ixs) where
  rawIndices =
    (:)
      <$> (T.RawTensorIndex <$> ATen.newTensorIndexWithSlice (fromIntegral (natValI @f)) (fromIntegral (natValI @t)) (fromIntegral (natValI @s)))
      <*> rawIndices @ixs

-- | The reified indices, routed through the untyped 'T.TensorIndex'
-- machinery.
newtype RawIndices = RawIndices [T.RawTensorIndex]

instance T.TensorIndex RawIndices where
  pushIndex vec (RawIndices l) = l ++ vec

-- | Index a tensor with a type-level list of indices; the result shape is
-- computed (and bounds are checked) at compile time.
--
-- > getSlice @'[SliceAt 1, SliceFromUpTo 0 2] t
-- > getSlice @[slice| 1, 0:2 |] t              -- Torch.Index's quasiquoter, in type position
getSlice ::
  forall ixs device dtype shape.
  KnownIndices ixs =>
  Tensor device dtype shape ->
  Tensor device dtype (IndexedShape ixs shape)
getSlice t = unsafePerformIO $ do
  raws <- rawIndices @ixs
  pure . UnsafeMkTensor $ toDynamic t T.! RawIndices raws

-- | Replace the indexed part of a tensor.  The value must have exactly the
-- shape that 'slice' with the same indices would produce.
setSlice ::
  forall ixs device dtype shape.
  KnownIndices ixs =>
  Tensor device dtype shape ->
  Tensor device dtype (IndexedShape ixs shape) ->
  Tensor device dtype shape
setSlice t v = unsafePerformIO $ do
  raws <- rawIndices @ixs
  pure . UnsafeMkTensor $ T.maskedFill (toDynamic t) (RawIndices raws) (toDynamic v)

--------------------------------------------------------------------------------
-- PyTorch-style syntax
--------------------------------------------------------------------------------

parseIndices :: String -> TH.Q TH.Type
parseIndices str = do
  items <-
    if all isSpace str
      then pure []
      else mapM (parseItem . filter (not . isSpace)) (splitOn ',' str)
  pure (foldr (\x xs -> TH.PromotedConsT `TH.AppT` x `TH.AppT` xs) TH.PromotedNilT items)
  where
    nat :: String -> TH.Q TH.Type
    nat s
      | not (null s) && all isDigit s = pure (TH.LitT (TH.NumTyLit (read s)))
      | otherwise = fail ("slice: expected a natural number, got " <> show s)
    con :: TH.Name -> [TH.Q TH.Type] -> TH.Q TH.Type
    con name args = foldl (\acc a -> TH.AppT <$> acc <*> a) (pure (TH.PromotedT name)) args
    parseItem :: String -> TH.Q TH.Type
    parseItem "None" = con 'NewAxis []
    parseItem s = case splitOn ':' s of
      [i] -> con 'SliceAt [nat i]
      ["", ""] -> con 'SliceAll []
      [f, ""] -> con 'SliceFrom [nat f]
      ["", t] -> con 'SliceUpTo [nat t]
      [f, t] -> con 'SliceFromUpTo [nat f, nat t]
      ["", "", ""] -> con 'SliceAll []
      [f, "", ""] -> con 'SliceFrom [nat f]
      ["", t, ""] -> con 'SliceUpTo [nat t]
      [f, t, ""] -> con 'SliceFromUpTo [nat f, nat t]
      ["", "", s'] -> con 'SliceWithStep [nat s']
      [f, "", s'] -> con 'SliceFromWithStep [nat f, nat s']
      ["", t, s'] -> con 'SliceUpToWithStep [nat t, nat s']
      [f, t, s'] -> con 'SliceFromUpToWithStep [nat f, nat t, nat s']
      _ -> fail ("slice: cannot parse index " <> show s)

splitOn :: Char -> String -> [String]
splitOn c s = case break (== c) s of
  (chunk, []) -> [chunk]
  (chunk, _ : rest) -> chunk : splitOn c rest
