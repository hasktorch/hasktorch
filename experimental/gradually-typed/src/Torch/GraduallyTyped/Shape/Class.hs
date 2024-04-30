{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Shape.Class where

import Control.Exception (Exception (..))
import Control.Monad.Catch (MonadThrow (throwM))
import Data.Singletons (Sing, SingKind (..))
import Data.Typeable (Typeable)
import GHC.TypeLits (Symbol, TypeError, type (+), type (-))
import GHC.TypeNats (Nat)
import Torch.GraduallyTyped.Prelude (Fst, LiftTimesMaybe, MapMaybe, PrependMaybe, Reverse, Snd, forgetIsChecked)
import Torch.GraduallyTyped.Prelude.List (SList (..))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SBy (..), SDim (..), SName (..), SSelectDim (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Unify (type (<+>))
import Type.Errors.Pretty (type (%), type (<>))
import Unsafe.Coerce (unsafeCoerce)

-- $setup
-- >>> import Torch.GraduallyTyped.Prelude.List (SList (..))
-- >>> import Torch.GraduallyTyped

type family AddSizeF (size :: Size Nat) (size' :: Size Nat) :: Size Nat where
  AddSizeF ('Size size) ('Size size') = 'Size (size + size')
  AddSizeF size size' = size <+> size'

type family AddDimF (dim :: Dim (Name Symbol) (Size Nat)) (dim' :: Dim (Name Symbol) (Size Nat)) :: Dim (Name Symbol) (Size Nat) where
  AddDimF ('Dim name size) ('Dim name' size') = 'Dim (name <+> name') (AddSizeF size size')

type family BroadcastSizeF (size :: Size Nat) (size' :: Size Nat) :: Maybe (Size Nat) where
  BroadcastSizeF 'UncheckedSize _ = 'Just 'UncheckedSize
  BroadcastSizeF _ 'UncheckedSize = 'Just 'UncheckedSize
  BroadcastSizeF ('Size size) ('Size size) = 'Just ('Size size)
  BroadcastSizeF ('Size size) ('Size 1) = 'Just ('Size size)
  BroadcastSizeF ('Size 1) ('Size size) = 'Just ('Size size)
  BroadcastSizeF ('Size _) ('Size _) = 'Nothing

type family BroadcastDimF (dim :: Dim (Name Symbol) (Size Nat)) (dim' :: Dim (Name Symbol) (Size Nat)) :: Maybe (Dim (Name Symbol) (Size Nat)) where
  BroadcastDimF ('Dim name size) ('Dim name' size') = MapMaybe ('Dim (name <+> name')) (BroadcastSizeF size size')

type family NumelDimF (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe Nat where
  NumelDimF ('Dim _ 'UncheckedSize) = 'Nothing
  NumelDimF ('Dim _ ('Size size)) = 'Just size

type family BroadcastDimsCheckF (dims :: [Dim (Name Symbol) (Size Nat)]) (dims' :: [Dim (Name Symbol) (Size Nat)]) (result :: Maybe [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  BroadcastDimsCheckF dims dims' 'Nothing =
    TypeError
      ( "Cannot broadcast the dimensions"
          % ""
          % "    '" <> dims <> "' and '" <> dims' <> "'."
          % ""
          % "You may need to extend, squeeze, or unsqueeze the dimensions manually."
      )
  BroadcastDimsCheckF _ _ ('Just dims) = Reverse dims

type family BroadcastDimsImplF (reversedDims :: [Dim (Name Symbol) (Size Nat)]) (reversedDims' :: [Dim (Name Symbol) (Size Nat)]) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  BroadcastDimsImplF '[] reversedDims = 'Just reversedDims
  BroadcastDimsImplF reversedDims '[] = 'Just reversedDims
  BroadcastDimsImplF (dim ': reversedDims) (dim' ': reversedDims') = PrependMaybe (BroadcastDimF dim dim') (BroadcastDimsImplF reversedDims reversedDims')

type BroadcastDimsF dims dims' = BroadcastDimsCheckF dims dims' (BroadcastDimsImplF (Reverse dims) (Reverse dims'))

type family BroadcastShapesF (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (shape' :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  BroadcastShapesF shape shape = shape
  BroadcastShapesF ('Shape dims) ('Shape dims') = 'Shape (BroadcastDimsF dims dims')
  BroadcastShapesF shape shape' = shape <+> shape'

type family NumelDimsF (dims :: [Dim (Name Symbol) (Size Nat)]) :: Maybe Nat where
  NumelDimsF '[] = 'Just 1
  NumelDimsF (dim ': dims) = LiftTimesMaybe (NumelDimF dim) (NumelDimsF dims)

type family NumelF (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Maybe Nat where
  NumelF 'UncheckedShape = 'Nothing
  NumelF ('Shape dims) = NumelDimsF dims

type family GetDimAndIndexByNameF (index :: Nat) (result :: (Maybe (Dim (Name Symbol) (Size Nat)), Maybe Nat)) (name :: Symbol) (dims :: [Dim (Name Symbol) (Size Nat)]) :: (Maybe (Dim (Name Symbol) (Size Nat)), Maybe Nat) where
  GetDimAndIndexByNameF _ result _ '[] = result
  GetDimAndIndexByNameF index _ name ('Dim 'UncheckedName _ ': dims) = GetDimAndIndexByNameF (index + 1) '( 'Just ('Dim 'UncheckedName 'UncheckedSize), 'Nothing) name dims
  GetDimAndIndexByNameF index _ name ('Dim ('Name name) size ': _) = '( 'Just ('Dim ('Name name) size), 'Just index)
  GetDimAndIndexByNameF index result name ('Dim ('Name _) _ ': dims) = GetDimAndIndexByNameF (index + 1) result name dims

type family GetDimByNameF (name :: Symbol) (dims :: [Dim (Name Symbol) (Size Nat)]) :: Maybe (Dim (Name Symbol) (Size Nat)) where
  GetDimByNameF name dims = Fst (GetDimAndIndexByNameF 0 '( 'Nothing, 'Nothing) name dims)

type family GetIndexByNameF (name :: Symbol) (dims :: [Dim (Name Symbol) (Size Nat)]) :: Maybe Nat where
  GetIndexByNameF name dims = Snd (GetDimAndIndexByNameF 0 '( 'Nothing, 'Nothing) name dims)

type family GetDimByIndexF (index :: Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) :: Maybe (Dim (Name Symbol) (Size Nat)) where
  GetDimByIndexF 0 (h ': _) = 'Just h
  GetDimByIndexF index (_ ': t) = GetDimByIndexF (index - 1) t
  GetDimByIndexF _ _ = 'Nothing

type family GetDimImplF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) :: Maybe (Dim (Name Symbol) (Size Nat)) where
  GetDimImplF ('ByName name) dims = GetDimByNameF name dims
  GetDimImplF ('ByIndex index) dims = GetDimByIndexF index dims

type GetDimErrorMessage (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) =
  "Cannot return the first dimension matching"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "in the shape"
    % ""
    % "    '" <> dims <> "'."
    % ""

type family GetDimCheckF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (result :: Maybe (Dim (Name Symbol) (Size Nat))) :: Dim (Name Symbol) (Size Nat) where
  GetDimCheckF by dims 'Nothing = TypeError (GetDimErrorMessage by dims)
  GetDimCheckF _ _ ('Just dim) = dim

type family GetDimF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Dim (Name Symbol) (Size Nat) where
  GetDimF 'UncheckedSelectDim _ = 'Dim 'UncheckedName 'UncheckedSize
  GetDimF _ 'UncheckedShape = 'Dim 'UncheckedName 'UncheckedSize
  GetDimF ('SelectDim by) ('Shape dims) = GetDimCheckF by dims (GetDimImplF by dims)

type family (!) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (_k :: k) :: Dim (Name Symbol) (Size Nat) where
  (!) shape (index :: Nat) = GetDimF ('SelectDim ('ByIndex index)) shape
  (!) shape (name :: Symbol) = GetDimF ('SelectDim ('ByName name)) shape

-- | Get dimension by index or by name from a shape.
--
-- >>> shape = SShape $ SName @"batch" :&: SSize @8 :|: SUncheckedName "feature" :&: SSize @2 :|: SNil
-- >>> dim = sGetDimFromShape (SSelectDim $ SByName @"batch") shape
-- >>> :type dim
-- dim :: MonadThrow m => m (SDim ('Dim ('Name "batch") ('Size 8)))
-- >>> fromSing <$> dim
-- Dim {dimName = Checked "batch", dimSize = Checked 8}
--
-- >>> dim = sGetDimFromShape (SSelectDim $ SByName @"feature") shape
-- >>> :type dim
-- dim :: MonadThrow m => m (SDim ('Dim UncheckedName UncheckedSize))
-- >>> fromSing <$> dim
-- Dim {dimName = Unchecked "feature", dimSize = Checked 2}
--
-- >>> dim = sGetDimFromShape (SSelectDim $ SByName @"sequence") shape
-- >>> :type dim
-- dim :: MonadThrow m => m (SDim ('Dim UncheckedName UncheckedSize))
-- >>> fromSing <$> dim
-- *** Exception: GetDimError {gdeBy = ByName "sequence"}
--
-- >>> dim = sGetDimFromShape (SSelectDim $ SByIndex @0) shape
-- >>> :type dim
-- dim :: MonadThrow m => m (SDim ('Dim ('Name "batch") ('Size 8)))
-- >>> fromSing <$> dim
-- Dim {dimName = Checked "batch", dimSize = Checked 8}
--
-- >>> :type sGetDimFromShape (SSelectDim $ SByIndex @2) shape
-- sGetDimFromShape (SSelectDim $ SByIndex @2) shape
--   :: MonadThrow m => m (SDim (TypeError ...))
sGetDimFromShape ::
  forall selectDim shape dim m.
  (dim ~ GetDimF selectDim shape, MonadThrow m) =>
  SSelectDim selectDim ->
  SShape shape ->
  m (SDim dim)
sGetDimFromShape (SUncheckedSelectDim by) (SUncheckedShape dims) = go 0 dims
  where
    go _ [] = throwM $ GetDimErrorWithDims by dims
    go index (Dim name size : dims') =
      let dim' = SDim (SUncheckedName name) (SUncheckedSize size)
       in case by of
            ByName name' | name == name' -> pure dim'
            ByIndex index' | index == index' -> pure dim'
            _ -> go (index + 1) dims'
sGetDimFromShape (SSelectDim by) (SUncheckedShape dims) = sGetDimFromShape (SUncheckedSelectDim $ fromSing by) (SUncheckedShape dims)
sGetDimFromShape (SUncheckedSelectDim by) (SShape dims) =
  let dims' = (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size)) <$> fromSing dims
   in sGetDimFromShape (SUncheckedSelectDim by) (SUncheckedShape dims')
sGetDimFromShape (SSelectDim by@SByName) (SShape SNil) =
  let by' = fromSing by
   in throwM $ GetDimError by'
sGetDimFromShape (SSelectDim by@SByName) (SShape (SCons dim@(SDim (SUncheckedName name) _) dims)) =
  let ByName name' = fromSing by
   in if name == name' then pure (unsafeCoerce @(SDim _) @(SDim dim) dim) else unsafeCoerce @(SDim _) @(SDim dim) <$> sGetDimFromShape (SSelectDim by) (SShape dims)
sGetDimFromShape (SSelectDim by@SByName) (SShape (SCons dim@(SDim SName _) dims)) =
  let ByName name' = fromSing by
      Dim name _size = (\(Dim name'' size) -> Dim (forgetIsChecked name'') (forgetIsChecked size)) $ fromSing dim
   in if name == name' then pure (unsafeCoerce @(SDim _) @(SDim dim) dim) else unsafeCoerce @(SDim _) @(SDim dim) <$> sGetDimFromShape (SSelectDim by) (SShape dims)
sGetDimFromShape (SSelectDim by@SByIndex) (SShape dims) =
  go 0 dims
  where
    by'@(ByIndex index') = fromSing by
    dims' = (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size)) <$> fromSing dims
    go :: forall dims. Integer -> SList dims -> m (SDim dim)
    go _ SNil = throwM $ GetDimErrorWithDims by' dims'
    go index (SCons dim dims'') =
      if index' == index then pure (unsafeCoerce @(Sing _) @(SDim dim) dim) else go (index + 1) dims''

data GetDimError
  = GetDimError {gdeBy :: By String Integer}
  | GetDimErrorWithDims {gdewdBy :: By String Integer, gdewdDims :: [Dim String Integer]}
  deriving stock (Show, Typeable)

instance Exception GetDimError where
  displayException GetDimError {..} =
    "Cannot return the first dimension matching `"
      <> show gdeBy
      <> "`."
  displayException GetDimErrorWithDims {..} =
    "Cannot return the first dimension matching `"
      <> show gdewdBy
      <> "` in the shape `"
      <> show gdewdDims
      <> "`."

type family ReplaceDimByIndexF (index :: Maybe Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimByIndexF ('Just 0) (_ ': t) dim = 'Just (dim ': t)
  ReplaceDimByIndexF ('Just index) (h ': t) dim = PrependMaybe ('Just h) (ReplaceDimByIndexF ('Just (index - 1)) t dim)
  ReplaceDimByIndexF _ _ _ = 'Nothing

type family ReplaceDimImplF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimImplF ('ByName name) dims dim = ReplaceDimByIndexF (GetIndexByNameF name dims) dims dim
  ReplaceDimImplF ('ByIndex index) dims dim = ReplaceDimByIndexF ('Just index) dims dim

type family ReplaceDimNameByIndexF (index :: Maybe Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (name :: Name Symbol) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimNameByIndexF ('Just 0) ('Dim _ size ': t) name' = 'Just ('Dim name' size ': t)
  ReplaceDimNameByIndexF ('Just index) (h ': t) name' = PrependMaybe ('Just h) (ReplaceDimNameByIndexF ('Just (index - 1)) t name')
  ReplaceDimNameByIndexF _ _ _ = 'Nothing

type family ReplaceDimNameImplF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (name' :: Name Symbol) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimNameImplF ('ByName name) dims name' = ReplaceDimNameByIndexF (GetIndexByNameF name dims) dims name'
  ReplaceDimNameImplF ('ByIndex index) dims name' = ReplaceDimNameByIndexF ('Just index) dims name'

type family ReplaceDimSizeByIndexF (index :: Maybe Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (size' :: Size Nat) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimSizeByIndexF ('Just 0) ('Dim name _ ': t) size' = 'Just ('Dim name size' ': t)
  ReplaceDimSizeByIndexF ('Just index) (h ': t) size' = PrependMaybe ('Just h) (ReplaceDimSizeByIndexF ('Just (index - 1)) t size')
  ReplaceDimSizeByIndexF _ _ _ = 'Nothing

type family ReplaceDimSizeImplF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (size' :: Size Nat) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimSizeImplF ('ByName name) dims size' = ReplaceDimSizeByIndexF (GetIndexByNameF name dims) dims size'
  ReplaceDimSizeImplF ('ByIndex index) dims size' = ReplaceDimSizeByIndexF ('Just index) dims size'

type ReplaceDimErrorMessage (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) =
  "Cannot replace the first dimension matching"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "in the shape"
    % ""
    % "    '" <> dims <> "'"
    % ""
    % "with"
    % ""
    % "    '" <> dim <> "'."
    % ""

type family ReplaceDimCheckF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) (result :: Maybe [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimCheckF by dims dim 'Nothing = TypeError (ReplaceDimErrorMessage by dims dim)
  ReplaceDimCheckF _ _ _ ('Just dims) = dims

type family ReplaceDimF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Shape [Dim (Name Symbol) (Size Nat)] where
  ReplaceDimF 'UncheckedSelectDim _ _ = 'UncheckedShape
  ReplaceDimF _ 'UncheckedShape _ = 'UncheckedShape
  ReplaceDimF ('SelectDim by) ('Shape dims) dim = 'Shape (ReplaceDimCheckF by dims dim (ReplaceDimImplF by dims dim))

type family InsertDimByIndexF (index :: Maybe Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  InsertDimByIndexF ('Just 0) dims dim = 'Just (dim ': dims)
  InsertDimByIndexF ('Just index) (h ': t) dim = PrependMaybe ('Just h) (InsertDimByIndexF ('Just (index - 1)) t dim)
  InsertDimByIndexF _ _ _ = 'Nothing

type family InsertDimImplF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  InsertDimImplF ('ByName name) dims dim = InsertDimByIndexF (GetIndexByNameF name dims) dims dim
  InsertDimImplF ('ByIndex index) dims dim = InsertDimByIndexF ('Just index) dims dim

type InsertDimErrorMessage (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) =
  "Cannot insert the dimension"
    % ""
    % "    '" <> dim <> "'"
    % ""
    % "before the first dimension matching"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "in the shape"
    % ""
    % "    '" <> dims <> "'."
    % ""

type family InsertDimCheckF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) (result :: Maybe [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  InsertDimCheckF by dims dim 'Nothing = TypeError (InsertDimErrorMessage by dims dim)
  InsertDimCheckF _ _ _ ('Just dims) = dims

type family InsertDimF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (dim :: Dim (Name Symbol) (Size Nat)) :: Shape [Dim (Name Symbol) (Size Nat)] where
  InsertDimF 'UncheckedSelectDim _ _ = 'UncheckedShape
  InsertDimF _ 'UncheckedShape _ = 'UncheckedShape
  InsertDimF ('SelectDim by) ('Shape dims) dim = 'Shape (InsertDimCheckF by dims dim (InsertDimImplF by dims dim))

type family PrependDimF (dim :: Dim (Name Symbol) (Size Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  PrependDimF dim shape = InsertDimF ('SelectDim ('ByIndex 0)) shape dim

type family RemoveDimByIndexF (index :: Maybe Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  RemoveDimByIndexF ('Just 0) (dim ': dims) = 'Just dims
  RemoveDimByIndexF ('Just index) (h ': t) = PrependMaybe ('Just h) (RemoveDimByIndexF ('Just (index - 1)) t)
  RemoveDimByIndexF _ _ = 'Nothing

type family RemoveDimImplF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  RemoveDimImplF ('ByName name) dims = RemoveDimByIndexF (GetIndexByNameF name dims) dims
  RemoveDimImplF ('ByIndex index) dims = RemoveDimByIndexF ('Just index) dims

type RemoveDimErrorMessage (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) =
  "Cannot remove the dimension by"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "in the shape"
    % ""
    % "    '" <> dims <> "'."
    % ""

type family RemoveDimCheckF (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) (result :: Maybe [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  RemoveDimCheckF by dims 'Nothing = TypeError (RemoveDimErrorMessage by dims)
  RemoveDimCheckF _ _ ('Just dims) = dims

-- >>> type SelectBatch = 'SelectDim ('ByName "batch")
-- >>> type Dims = '[ 'Dim ('Name "batch") ('Size 10), 'Dim ('Name "feature") ('Size 8)]
-- >>> :kind! RemoveDimF SelectBatch ('Shape Dims)
-- RemoveDimF SelectBatch ('Shape Dims) :: Shape
--                                           [Dim (Name Symbol) (Size Nat)]
-- = 'Shape
--     '[ 'Dim ('Name "feature") ('Size 8),
--        'Dim ('Name "anotherFeature") ('Size 12)]
type family RemoveDimF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  RemoveDimF 'UncheckedSelectDim _ = 'UncheckedShape
  RemoveDimF _ 'UncheckedShape = 'UncheckedShape
  RemoveDimF ('SelectDim by) ('Shape dims) = 'Shape (RemoveDimCheckF by dims (RemoveDimImplF by dims))

data UnifyNameError = UnifyNameError {uneExpect :: String, uneActual :: String}
  deriving stock (Show)

instance Exception UnifyNameError where
  displayException UnifyNameError {..} =
    "The supplied dimensions must be the same, "
      <> "but dimensions with different names were found: "
      <> show uneExpect
      <> " and "
      <> show uneActual
      <> "."

sUnifyName ::
  forall m name name'.
  MonadThrow m =>
  SName name ->
  SName name' ->
  m (SName (name <+> name'))
sUnifyName (SUncheckedName name) (SUncheckedName name') | name == name' = pure (SUncheckedName name)
sUnifyName (SUncheckedName "*") (SUncheckedName name') = pure (SUncheckedName name')
sUnifyName (SUncheckedName name) (SUncheckedName "*") = pure (SUncheckedName name)
sUnifyName name@SName (SUncheckedName name') = sUnifyName (SUncheckedName . forgetIsChecked $ fromSing name) (SUncheckedName name')
sUnifyName (SUncheckedName name) name'@SName = sUnifyName (SUncheckedName name) (SUncheckedName . forgetIsChecked $ fromSing name')
sUnifyName name@SName name'@SName | forgetIsChecked (fromSing name) == forgetIsChecked (fromSing name') = pure (unsafeCoerce @(SName name) @(SName (name <+> name')) name)
sUnifyName name@SName name'@SName | forgetIsChecked (fromSing name) == "*" = pure (unsafeCoerce @(SName name') @(SName (name <+> name')) name')
sUnifyName name@SName name'@SName | forgetIsChecked (fromSing name') == "*" = pure (unsafeCoerce @(SName name) @(SName (name <+> name')) name)
sUnifyName name name' = throwM $ UnifyNameError (forgetIsChecked (fromSing name)) (forgetIsChecked (fromSing name'))

data UnifySizeError = UnifySizeError {useExpect :: Integer, useActual :: Integer}
  deriving stock (Show)

instance Exception UnifySizeError where
  displayException UnifySizeError {..} =
    "The supplied dimensions must be the same, "
      <> "but dimensions with different sizes were found: "
      <> show useExpect
      <> " and "
      <> show useActual
      <> "."

sUnifySize ::
  forall m size size'.
  MonadThrow m =>
  SSize size ->
  SSize size' ->
  m (SSize (size <+> size'))
sUnifySize (SUncheckedSize size) (SUncheckedSize size') | size == size' = pure (SUncheckedSize size)
sUnifySize size@SSize (SUncheckedSize size') = sUnifySize (SUncheckedSize . forgetIsChecked $ fromSing size) (SUncheckedSize size')
sUnifySize (SUncheckedSize size) size'@SSize = sUnifySize (SUncheckedSize size) (SUncheckedSize . forgetIsChecked $ fromSing size')
sUnifySize size@SSize size'@SSize | forgetIsChecked (fromSing size) == forgetIsChecked (fromSing size') = pure (unsafeCoerce @(SSize size) @(SSize (size <+> size')) size)
sUnifySize size size' = throwM $ UnifySizeError (forgetIsChecked (fromSing size)) (forgetIsChecked (fromSing size'))

-- | Unify two dimensions.
--
-- >>> dimA = SName @"*" :&: SSize @0
-- >>> dimB = SName @"batch" :&: SSize @0
-- >>> dim = sUnifyDim dimA dimB
-- >>> :type dim
-- dim :: MonadThrow m => m (SDim ('Dim ('Name "batch") ('Size 0)))
-- >>> fromSing <$> dim
-- Dim {dimName = Checked "batch", dimSize = Checked 0}
--
-- >>> dimC = SName @"feature" :&: SSize @0
-- >>> :type sUnifyDim dimB dimC
-- sUnifyDim dimB dimC
--   :: MonadThrow m => m (SDim ('Dim (TypeError ...) ('Size 0)))
--
-- >>> dimD = SUncheckedName "batch" :&: SSize @0
-- >>> dim = sUnifyDim dimA dimD
-- >>> :type dim
-- dim :: MonadThrow m => m (SDim ('Dim UncheckedName ('Size 0)))
-- >>> fromSing <$> dim
-- Dim {dimName = Unchecked "batch", dimSize = Checked 0}
--
-- >>> dimE = SUncheckedName "feature" :&: SSize @0
-- >>> dim = sUnifyDim dimB dimE
-- >>> :type dim
-- dim :: MonadThrow m => m (SDim ('Dim UncheckedName ('Size 0)))
-- >>> fromSing <$> dim
-- *** Exception: UnifyNameError {uneExpect = "batch", uneActual = "feature"}
sUnifyDim ::
  forall m dim dim'.
  MonadThrow m =>
  SDim dim ->
  SDim dim' ->
  m (SDim (dim <+> dim'))
sUnifyDim (SDim name size) (SDim name' size') = do
  dim <- SDim <$> sUnifyName name name' <*> sUnifySize size size'
  pure $ unsafeCoerce @(SDim _) @(SDim (dim <+> dim')) dim
