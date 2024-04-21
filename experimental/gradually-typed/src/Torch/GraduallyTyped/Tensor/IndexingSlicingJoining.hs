{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Tensor.IndexingSlicingJoining where

import Control.Exception (Exception (..))
import Control.Monad.Catch (MonadThrow (throwM))
import Data.Bifunctor (bimap)
import Data.Coerce (coerce)
import Data.Kind (Type)
import Data.Singletons (SingI (..), SingKind (..), fromSing)
import Data.Type.Bool (If, type (&&))
import Data.Typeable (Typeable)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (ErrorMessage, Nat, Symbol, TypeError, type (-))
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.DType (DType (..), DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Index.Class (InRangeF)
import Torch.GraduallyTyped.Index.Type (DemotedIndex (..), SIndex)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.Prelude (Catch, FromMaybe, MapMaybe, MaybeF, PrependMaybe, When, forgetIsChecked)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (AddDimF, BroadcastShapesF, GetDimF, GetDimImplF, GetIndexByNameF, InsertDimImplF, NumelF, RemoveDimF, ReplaceDimF, ReplaceDimImplF, sGetDimFromShape)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SSelectDim, SShape, SelectDim (..), Shape (..), Size (..), dimSize)
import Torch.GraduallyTyped.Tensor.Type (SGetShape (sGetShape), Tensor)
import Torch.GraduallyTyped.Unify (UnifyCheck, type (<+>), type (<|>))
import Torch.HList (HList)
import qualified Torch.Internal.Cast as ATen (cast1, cast2, cast3, cast4)
import qualified Torch.Internal.Class as ATen (Castable)
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen
import Type.Errors.Pretty (ToErrorMessage, type (%), type (<>))

-- $setup
-- >>> import Torch.GraduallyTyped.Prelude.List (SList (..))
-- >>> import Torch.GraduallyTyped

class HasCat (selectDim :: SelectDim (By Symbol Nat)) k (c :: k -> Type) (a :: k) where
  type CatF selectDim a c :: Type

  -- | Concatenates the given sequence of seq tensors in the given dimension.
  -- All tensors must either have the same shape (except in the concatenating dimension) or be empty.
  --
  -- >>> t <- ones @('Gradient 'WithGradient) @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
  -- >>> :type cat @('SelectDim ('ByName "feature")) [t]
  -- cat @('SelectDim ('ByName "feature")) [t]
  --   :: MonadThrow m =>
  --      m (Tensor
  --           ('Gradient WithGradient)
  --           ('Layout Dense)
  --           ('Device CPU)
  --           ('DataType 'Float)
  --           ('Shape
  --              ['Dim ('Name "batch") ('Size 32),
  --               'Dim UncheckedName UncheckedSize]))
  -- >>> :type cat @('SelectDim ( 'ByIndex 0)) [t]
  -- cat @('SelectDim ( 'ByIndex 0)) [t]
  --   :: MonadThrow m =>
  --      m (Tensor
  --           ('Gradient WithGradient)
  --           ('Layout Dense)
  --           ('Device CPU)
  --           ('DataType 'Float)
  --           ('Shape
  --              ['Dim UncheckedName UncheckedSize,
  --               'Dim ('Name "feature") ('Size 8)]))
  -- >>> :type sCat (SUncheckedSelectDim (ByIndex 0)) [t]
  -- sCat (SUncheckedSelectDim (ByIndex 0)) [t]
  --   :: MonadThrow m =>
  --      m (Tensor
  --           ('Gradient WithGradient)
  --           ('Layout Dense)
  --           ('Device CPU)
  --           ('DataType 'Float)
  --           UncheckedShape)
  sCat :: forall m. MonadThrow m => SSelectDim selectDim -> c a -> m (CatF selectDim a c)

  cat :: forall m. (SingI selectDim, MonadThrow m) => c a -> m (CatF selectDim a c)
  cat = sCat (sing @selectDim)

type family CatListImplF (selectDim :: SelectDim (By Symbol Nat)) (tensor :: Type) :: Maybe Type where
  CatListImplF 'UncheckedSelectDim (Tensor gradient layout device dataType _) = 'Just (Tensor gradient layout device dataType 'UncheckedShape)
  CatListImplF ('SelectDim _) (Tensor gradient layout device dataType 'UncheckedShape) = 'Just (Tensor gradient layout device dataType 'UncheckedShape)
  CatListImplF ('SelectDim by) (Tensor gradient layout device dataType ('Shape dims)) = MapMaybe (Tensor gradient layout device dataType) (MapMaybe 'Shape (ReplaceDimImplF by dims ('Dim 'UncheckedName 'UncheckedSize)))

type CheckSpellingMessage = "Check the spelling of named dimensions, and make sure the number of dimensions is correct."

type family CatListCheckF (selectDim :: SelectDim (By Symbol Nat)) (tensor :: Type) (result :: Maybe Type) :: Type where
  CatListCheckF selectDim (Tensor _ _ _ _ shape) 'Nothing =
    TypeError
      ( "Cannot concatenate the dimension"
          % ""
          % "    " <> selectDim
          % ""
          % "for tensors of shape"
          % ""
          % "    " <> shape <> "."
          % ""
          % CheckSpellingMessage
      )
  CatListCheckF _ _ ('Just result) = result

type CatListF selectDim tensor = CatListCheckF selectDim tensor (CatListImplF selectDim tensor)

instance
  ATen.Castable (CatListF selectDim (Tensor gradient layout device dataType shape)) (ForeignPtr ATen.Tensor) =>
  HasCat selectDim Type [] (Tensor gradient layout device dataType shape)
  where
  type CatF selectDim (Tensor gradient layout device dataType shape) [] = CatListF selectDim (Tensor gradient layout device dataType shape)
  sCat selectDim tensors = do
    let by = forgetIsChecked . fromSing $ selectDim
    unsafeThrowableIO $ case by of
      ByName name -> ATen.cast2 ATen.cat_ln tensors name
      ByIndex index -> ATen.cast2 ATen.cat_ll tensors (fromInteger index :: Int)

type CatHListImplF ::
  SelectDim (By Symbol Nat) ->
  [Type] ->
  Maybe (Gradient RequiresGradient, Layout LayoutType, Device (DeviceType Nat), DataType DType, Shape [Dim (Name Symbol) (Size Nat)]) ->
  Type
type family CatHListImplF selectDim tensors acc where
  CatHListImplF _ '[] 'Nothing = TypeError (ToErrorMessage "Cannot concatenate an empty list of tensors.")
  CatHListImplF _ '[] ('Just '(gradient, layout, device, dataType, shape)) = Tensor gradient layout device dataType shape
  CatHListImplF selectDim (Tensor gradient layout device dataType shape ': tensors) 'Nothing =
    CatHListImplF selectDim tensors ('Just '(gradient, layout, device, dataType, shape))
  CatHListImplF selectDim (Tensor gradient layout device dataType shape ': tensors) ('Just '(gradient', layout', device', dataType', shape')) =
    CatHListImplF
      selectDim
      tensors
      ( 'Just
          '( gradient <|> gradient',
             layout <+> layout',
             device <+> device',
             dataType <+> dataType',
             ReplaceDimF
               selectDim
               (shape <+> ReplaceDimF selectDim shape' (GetDimF selectDim shape))
               (AddDimF (GetDimF selectDim shape) (GetDimF selectDim shape'))
           )
      )
  CatHListImplF _ (x ': _) _ =
    TypeError
      ( "Cannot concatenate because"
          % ""
          % "    '" <> x <> "'"
          % ""
          % "is not a tensor type."
      )

type CatHListF selectDim tensors = CatHListImplF selectDim tensors 'Nothing

instance
  ( ATen.Castable (CatHListF selectDim tensors) (ForeignPtr ATen.Tensor),
    ATen.Castable (HList tensors) (ForeignPtr ATen.TensorList)
  ) =>
  HasCat selectDim [Type] HList tensors
  where
  type CatF selectDim tensors HList = CatHListF selectDim tensors
  sCat selectDim tensors = do
    let by = forgetIsChecked . fromSing $ selectDim
    unsafeThrowableIO $ case by of
      ByName name -> ATen.cast2 ATen.cat_ln tensors name
      ByIndex index -> ATen.cast2 ATen.cat_ll tensors (fromInteger index :: Int)

type ReshapeNumelMismatchMessage ::
  Nat ->
  Nat ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  ErrorMessage

type ReshapeNumelMismatchMessage numel numel' shape shape' =
  "Cannot reshape the tensor. The original shape,"
    % ""
    % "    '" <> shape <> "',"
    % ""
    % "and the new shape,"
    % ""
    % "    '" <> shape' <> "',"
    % ""
    % "have different total numbers of elements,"
    % ""
    % "    '" <> numel <> "' versus '" <> numel' <> "',"
    % ""
    % "respectively."

type ReshapeImplF ::
  Maybe Nat ->
  Maybe Nat ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)]
type family ReshapeImplF numel numel' shape shape' where
  ReshapeImplF ('Just numel) ('Just numel) _ shape' = shape'
  ReshapeImplF ('Just numel) ('Just numel') shape shape' = TypeError (ReshapeNumelMismatchMessage numel numel' shape shape')
  ReshapeImplF 'Nothing _ _ _ = 'UncheckedShape
  ReshapeImplF _ 'Nothing _ _ = 'UncheckedShape
  ReshapeImplF _ _ 'UncheckedShape _ = 'UncheckedShape
  ReshapeImplF _ _ _ 'UncheckedShape = 'UncheckedShape

type ReshapeF :: Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)]
type family ReshapeF shape shape' where
  ReshapeF shape shape' = ReshapeImplF (NumelF shape) (NumelF shape') shape shape'

-- | Returns a tensor with the same data and number of elements as the input tensor,
-- but with the specified shape:
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> (input, _) <- sRandn (TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"*" :&: SSize @4 :|: SNil)) g
-- >>> output <- sReshape (SShape $ SName @"*" :&: SSize @2 :|: SName @"*" :&: SSize @2 :|: SNil) input
-- >>> :type output
-- output
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape ['Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 2)])
--
-- At the value level, a single dimension may be '-1',
-- in which case it is inferred from the remaining dimensions and the number of elements in the input:
--
-- >>> output' <- sReshape (SShape $ SUncheckedName "*" :&: SUncheckedSize (-1) :|: SNil) output
-- >>> :type output'
-- output'
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        UncheckedShape
-- >>> getDims output'
-- [Dim {dimName = "*", dimSize = 4}]
sReshape,
  sSetShape ::
    forall m shape' gradient layout device dataType shape shape''.
    MonadThrow m =>
    (shape'' ~ ReshapeF shape shape', Catch shape'') =>
    SShape shape' ->
    Tensor gradient layout device dataType shape ->
    m (Tensor gradient layout device dataType shape'')
sReshape shape' input = unsafeThrowableIO $ do
  let dims = forgetIsChecked . fromSing $ shape'
  t :: ForeignPtr ATen.Tensor <- ATen.cast2 ATen.reshape_tl input (forgetIsChecked . dimSize <$> dims)
  ATen.cast2 ATen.tensor_refine_names_N t (forgetIsChecked . dimName <$> dims)
sSetShape = sReshape

type AllDimSizesChecked :: Shape [Dim (Name Symbol) (Size Nat)] -> Bool
type family AllDimSizesChecked shape where
  AllDimSizesChecked 'UncheckedShape = 'False
  AllDimSizesChecked ('Shape '[]) = 'True
  AllDimSizesChecked ('Shape ('Dim name ('Size size) ': xs)) = AllDimSizesChecked ('Shape xs)

reshape ::
  forall m shape' gradient layout device dataType shape shape''.
  ( shape'' ~ ReshapeF shape shape',
    Catch shape'',
    When (AllDimSizesChecked shape) (shape' ~ shape''),
    SingI shape',
    MonadThrow m
  ) =>
  Tensor gradient layout device dataType shape ->
  m (Tensor gradient layout device dataType shape'')
reshape = sReshape (sing @shape')

type TransposeBy0Message :: By Symbol Nat -> [Dim (Name Symbol) (Size Nat)] -> ErrorMessage

type TransposeBy0Message by0 dims =
  "Cannot transpose the tensor with the dimensions"
    % ""
    % "    '" <> dims <> "'"
    % ""
    % "because the specified source dimension"
    % ""
    % "    '" <> by0 <> "'"
    % ""
    % "could not be found."

type TransposeBy1Message :: By Symbol Nat -> [Dim (Name Symbol) (Size Nat)] -> ErrorMessage

type TransposeBy1Message by1 dims =
  "Cannot transpose the tensor with the dimensions"
    % ""
    % "    '" <> dims <> "'"
    % ""
    % "because the specified target dimension"
    % ""
    % "    '" <> by1 <> "'"
    % ""
    % "could not be found."

-- | Compute transposed shapes.
--
-- >>> type SelectBatch = 'SelectDim ('ByName "batch" :: By Symbol Nat)
-- >>> type SelectFeature = 'SelectDim ('ByName "feature" :: By Symbol Nat)
-- >>> type Dims = '[ 'Dim ('Name "batch") ('Size 10), 'Dim ('Name "feature") ('Size 8), 'Dim ('Name "anotherFeature") ('Size 12)]
-- >>> :kind! TransposeF SelectBatch SelectFeature ('Shape Dims)
-- TransposeF SelectBatch SelectFeature ('Shape Dims) :: Shape
--                                                         [Dim (Name Symbol) (Size Natural)]
-- = 'Shape
--     ['Dim ('Name "feature") ('Size 8), 'Dim ('Name "batch") ('Size 10),
--      'Dim ('Name "anotherFeature") ('Size 12)]
-- >>> :kind! TransposeF SelectFeature SelectBatch ('Shape Dims)
-- TransposeF SelectFeature SelectBatch ('Shape Dims) :: Shape
--                                                         [Dim (Name Symbol) (Size Natural)]
-- = 'Shape
--     ['Dim ('Name "feature") ('Size 8), 'Dim ('Name "batch") ('Size 10),
--      'Dim ('Name "anotherFeature") ('Size 12)]
type TransposeF ::
  SelectDim (By Symbol Nat) ->
  SelectDim (By Symbol Nat) ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)]
type family TransposeF selectDim0 selectDim1 shape where
  TransposeF _ _ 'UncheckedShape = 'UncheckedShape
  TransposeF _ 'UncheckedSelectDim _ = 'UncheckedShape
  TransposeF 'UncheckedSelectDim _ _ = 'UncheckedShape
  TransposeF ('SelectDim ('ByName name0)) ('SelectDim ('ByName name1)) ('Shape dims) =
    'Shape
      ( TransposeIndexIndexDimsF
          ( FromMaybe
              (TypeError (TransposeBy0Message ('ByName name0) dims))
              (GetIndexByNameF name0 dims)
          )
          ( FromMaybe
              (TypeError (TransposeBy1Message ('ByName name1) dims))
              (GetIndexByNameF name1 dims)
          )
          dims
      )
  TransposeF ('SelectDim ('ByIndex index0)) ('SelectDim ('ByIndex index1)) ('Shape dims) = 'Shape (TransposeIndexIndexDimsF index0 index1 dims)
  TransposeF ('SelectDim by0) ('SelectDim by1) _ =
    TypeError
      ( "Cannot transpose the tensor. "
          % ""
          % "The source and target dimensions must be selected either both by name or both by index, "
          % "but mixed selectors were found: "
          % ""
          % "    '" <> 'SelectDim by0 <> "' and '" <> 'SelectDim by1 <> "'."
          % ""
      )

type TransposeIndexIndexDimsF :: Nat -> Nat -> [Dim (Name Symbol) (Size Nat)] -> [Dim (Name Symbol) (Size Nat)]
type family TransposeIndexIndexDimsF index0 index1 dims where
  TransposeIndexIndexDimsF index0 index1 dims =
    FromMaybe
      (TypeError (TransposeBy1Message ('ByIndex index1) dims))
      ( ReplaceDimImplF
          ('ByIndex index1)
          ( FromMaybe
              (TypeError (TransposeBy0Message ('ByIndex index0) dims))
              ( ReplaceDimImplF
                  ('ByIndex index0)
                  dims
                  ( FromMaybe
                      (TypeError (TransposeBy1Message ('ByIndex index1) dims))
                      (GetDimImplF ('ByIndex index1) dims)
                  )
              )
          )
          ( FromMaybe
              (TypeError (TransposeBy0Message ('ByIndex index0) dims))
              (GetDimImplF ('ByIndex index0) dims)
          )
      )

-- | Returns a tensor that is a transposed version of @input@.
-- The selected dimensions @selectDim0@ and @selectDim1@ are swapped.
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> (input, _) <- sRandn (TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"batch" :&: SSize @10 :|: SName @"feature" :&: SSize @5 :|: SNil)) g
-- >>> output <- sTranspose (SSelectDim (SByName @"batch")) (SSelectDim (SByName @"feature")) input
-- >>> :type output
-- output
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "feature") ('Size 5),
--            'Dim ('Name "batch") ('Size 10)])
-- >>> output <- sTranspose (SUncheckedSelectDim (ByIndex 0)) (SSelectDim (SByIndex @1)) input
-- >>> :type output
-- output
--   :: Tensor
--        ('Gradient WithGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        UncheckedShape
-- >>> getDims output
-- [Dim {dimName = "feature", dimSize = 5},Dim {dimName = "batch", dimSize = 10}]
sTranspose ::
  forall selectDim0 selectDim1 gradient layout device dataType shape shape' m.
  ( shape' ~ TransposeF selectDim0 selectDim1 shape,
    Catch shape',
    MonadThrow m
  ) =>
  SSelectDim selectDim0 ->
  SSelectDim selectDim1 ->
  Tensor gradient layout device dataType shape ->
  m (Tensor gradient layout device dataType shape')
sTranspose selectDim0 selectDim1 input = do
  let by0 = forgetIsChecked . fromSing $ selectDim0
      by1 = forgetIsChecked . fromSing $ selectDim1
  unsafeThrowableIO $ case (by0, by1) of
    (ByName name0, ByName name1) -> ATen.cast3 ATen.transpose_tnn input name0 name1
    (ByIndex index0, ByIndex index1) -> ATen.cast3 ATen.transpose_tll input (fromIntegral index0 :: Int) (fromIntegral index1 :: Int)
    _ -> throwM $ TransposeMixedSelectorsError by0 by1

data TransposeError = TransposeMixedSelectorsError
  { teBy0 :: By String Integer,
    teBy1 :: By String Integer
  }
  deriving stock (Show, Typeable)

instance Exception TransposeError where
  displayException TransposeMixedSelectorsError {..} =
    "Cannot transpose the tensor. "
      <> "The source and target dimensions must be selected either both by name or both by index, "
      <> "but mixed selectors were found: '"
      <> show teBy0
      <> "' and '"
      <> show teBy1
      <> "'."

transpose ::
  forall selectDim0 selectDim1 gradient layout device dataType shape shape' m.
  ( shape' ~ TransposeF selectDim0 selectDim1 shape,
    Catch shape',
    SingI selectDim0,
    SingI selectDim1,
    MonadThrow m
  ) =>
  Tensor gradient layout device dataType shape ->
  m (Tensor gradient layout device dataType shape')
transpose = sTranspose (sing @selectDim0) (sing @selectDim1)

type UnsqueezeByMessage (by :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) =
  "Cannot unsqueeze the tensor with the dimensions"
    % ""
    % "    '" <> dims <> "'"
    % ""
    % "because the specified source dimension"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "could not be found."

type family UnsqueezeF (selectDim :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  UnsqueezeF _ 'UncheckedShape = 'UncheckedShape
  UnsqueezeF 'UncheckedSelectDim _ = 'UncheckedShape
  UnsqueezeF ('SelectDim ('ByName name)) ('Shape dims) =
    'Shape
      ( UnsqueezeIndexDimsF
          ( FromMaybe
              (TypeError (UnsqueezeByMessage ('ByName name) dims))
              (GetIndexByNameF name dims)
          )
          dims
      )
  UnsqueezeF ('SelectDim ('ByIndex index)) ('Shape dims) = 'Shape (UnsqueezeIndexDimsF index dims)

type family UnsqueezeIndexDimsF (index :: Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  UnsqueezeIndexDimsF index dims =
    FromMaybe
      (TypeError (UnsqueezeByMessage ('ByIndex index) dims))
      ( InsertDimImplF
          ('ByIndex index)
          dims
          ('Dim ('Name "*") ('Size 1))
      )

-- | Unsqueezes a tensor with the specified dimension.
sUnsqueeze ::
  forall selectDim gradient layout device dataType shape shape' m.
  ( shape' ~ UnsqueezeF selectDim shape,
    Catch shape',
    MonadThrow m
  ) =>
  SSelectDim selectDim ->
  Tensor gradient layout device dataType shape ->
  m (Tensor gradient layout device dataType shape')
sUnsqueeze selectDim input =
  let by = forgetIsChecked . fromSing $ selectDim
   in case by of
        ByName _name -> undefined
        ByIndex index -> unsafeThrowableIO $ ATen.cast2 ATen.unsqueeze_tl input (fromIntegral index :: Int)

-- | Unsqueezes a tensor with the specified dimension.
unsqueeze ::
  forall selectDim gradient layout device dataType shape shape' m.
  ( shape' ~ UnsqueezeF selectDim shape,
    Catch shape',
    SingI selectDim,
    MonadThrow m
  ) =>
  Tensor gradient layout device dataType shape ->
  m (Tensor gradient layout device dataType shape')
unsqueeze = sUnsqueeze (sing @selectDim)

type SqueezeAllShapeF :: Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)]
type family SqueezeAllShapeF shape where
  SqueezeAllShapeF 'UncheckedShape = 'UncheckedShape
  SqueezeAllShapeF ('Shape dims) = MaybeF 'UncheckedShape 'Shape (SqueezeAllDimsF dims)

type SqueezeAllDimsF :: [Dim (Name Symbol) (Size Nat)] -> Maybe [Dim (Name Symbol) (Size Nat)]
type family SqueezeAllDimsF dims where
  SqueezeAllDimsF '[] = 'Just '[]
  SqueezeAllDimsF ('Dim _ 'UncheckedSize ': dims) = 'Nothing
  SqueezeAllDimsF ('Dim _ ('Size 1) ': dims) = SqueezeAllDimsF dims
  SqueezeAllDimsF (dim ': dims) = PrependMaybe ('Just dim) (SqueezeAllDimsF dims)

squeezeAll ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  Tensor gradient layout device dataType (SqueezeAllShapeF shape)
squeezeAll input = unsafePerformIO $ ATen.cast1 ATen.squeeze_t input

type SqueezeDimByIndexF :: Nat -> [Dim (Name Symbol) (Size Nat)] -> Maybe [Dim (Name Symbol) (Size Nat)]
type family SqueezeDimByIndexF dimIndex dims where
  SqueezeDimByIndexF 0 (x ': xs) =
    If
      (UnifyCheck (Dim (Name Symbol) (Size Nat)) x ('Dim ('Name "*") ('Size 1)))
      ('Just xs)
      'Nothing
  SqueezeDimByIndexF dimIndex (x ': xs) = PrependMaybe ('Just x) (SqueezeDimByIndexF (dimIndex - 1) xs)
  SqueezeDimByIndexF _ _ = 'Nothing

type SqueezeDimImplF :: By Symbol Nat -> [Dim (Name Symbol) (Size Nat)] -> Maybe [Dim (Name Symbol) (Size Nat)]
type family SqueezeDimImplF by dims where
  SqueezeDimImplF ('ByName dimName) dims = SqueezeDimByNameF (GetIndexByNameF dimName dims) dims
  SqueezeDimImplF ('ByIndex dimIndex) dims = SqueezeDimByIndexF dimIndex dims

type SqueezeDimByNameF :: Maybe Nat -> [Dim (Name Symbol) (Size Nat)] -> Maybe [Dim (Name Symbol) (Size Nat)]
type family SqueezeDimByNameF dimIndex dims where
  SqueezeDimByNameF 'Nothing dims = 'Nothing
  SqueezeDimByNameF ('Just dimIndex) dims = SqueezeDimByIndexF dimIndex dims

type SqueezeDimMessage :: By Symbol Nat -> [Dim (Name Symbol) (Size Nat)] -> ErrorMessage

type SqueezeDimMessage by dims =
  "Cannot squeeze the tensor with the dimensions"
    % ""
    % "    '" <> dims <> "'"
    % ""
    % "at the dimension"
    % ""
    % "    '" <> by <> "'."
    % ""

type SqueezeDimCheckF :: By Symbol Nat -> [Dim (Name Symbol) (Size Nat)] -> Maybe [Dim (Name Symbol) (Size Nat)] -> [Dim (Name Symbol) (Size Nat)]
type family SqueezeDimCheckF by dims result where
  SqueezeDimCheckF by dims 'Nothing = TypeError (SqueezeDimMessage by dims)
  SqueezeDimCheckF _ _ ('Just dims) = dims

-- | Calculate the output shape of a squeeze along a given dimension
--
-- >>> :kind! SqueezeDimF ('SelectDim ('ByIndex 1)) ('Shape '[ 'Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 2)])
-- ...
-- = 'Shape ['Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 2)]
type SqueezeDimF :: SelectDim (By Symbol Nat) -> Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)]
type family SqueezeDimF selectDim shape where
  SqueezeDimF 'UncheckedSelectDim _ = 'UncheckedShape
  SqueezeDimF _ 'UncheckedShape = 'UncheckedShape
  SqueezeDimF ('SelectDim by) ('Shape dims) = 'Shape (SqueezeDimCheckF by dims (SqueezeDimImplF by dims))

-- | Squeeze a particular dimension.
--
-- >>> t <- sOnes $ TensorSpec (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SShape $ SNoName :&: SSize @2 :|: SNoName :&: SSize @1 :|: SNoName :&: SSize @2 :|: SNoName :&: SSize @1 :|: SNoName :&: SSize @2 :|: SNil)
-- >>> result <- sSqueezeDim (SSelectDim $ SByIndex @1) t
-- >>> :t result
-- result
--   :: Tensor
--        ('Gradient WithoutGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 2),
--            'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 2)])
-- >>> result
-- Tensor Float [2,2,1,2] [[[[ 1.0000   ,  1.0000   ]],
--                          [[ 1.0000   ,  1.0000   ]]],
--                         [[[ 1.0000   ,  1.0000   ]],
--                          [[ 1.0000   ,  1.0000   ]]]]
sSqueezeDim ::
  forall selectDim gradient layout device dataType shape shape' m.
  (MonadThrow m, shape' ~ SqueezeDimF selectDim shape, Catch shape') =>
  SSelectDim selectDim ->
  Tensor gradient layout device dataType shape ->
  m (Tensor gradient layout device dataType shape')
sSqueezeDim selectDim input =
  let by = forgetIsChecked . fromSing $ selectDim
   in case by of
        ByName dimName -> unsafeThrowableIO $ ATen.cast2 ATen.squeeze_tn input dimName
        ByIndex dimIndex -> unsafeThrowableIO $ ATen.cast2 ATen.squeeze_tl input (fromIntegral dimIndex :: Int)

-- | Expands a tensor to the specified shape.
sExpand ::
  forall shape' shape'' gradient layout device dataType shape.
  (shape'' ~ BroadcastShapesF shape shape', Catch shape'') =>
  -- | new shape
  SShape shape' ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | output tensor
  Tensor gradient layout device dataType shape''
sExpand shape' input =
  let sizes' = fmap (\(Dim _ size) -> forgetIsChecked size) . forgetIsChecked $ fromSing shape'
   in unsafePerformIO $ ATen.cast3 ATen.tensor_expand_lb input sizes' True

-- | Expands a tensor to the specified shape.
expand ::
  forall shape' shape'' gradient layout device dataType shape.
  (SingI shape', shape'' ~ BroadcastShapesF shape shape', Catch shape'') =>
  Tensor gradient layout device dataType shape ->
  Tensor gradient layout device dataType shape''
expand = sExpand (sing @shape')

-- | Slices the self tensor along the selected dimension at the given index. This function returns a view of the original tensor with the given dimension removed.
--
-- >>> nats <- sArangeNaturals (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU) (SDataType SInt32) (SSize @8)
-- >>> input <- sReshape (SShape $ SName @"*" :&: SSize @4 :|: SName @"*" :&: SSize @2 :|: SNil) nats
-- >>> input
-- Tensor Int32 [4,2] [[ 0,  1],
--                     [ 2,  3],
--                     [ 4,  5],
--                     [ 6,  7]]
--
-- 'index' can be provided at compile-time:
--
-- >>> sSelect (SSelectDim (SByIndex @0)) (SIndex @1) input
-- Tensor Int32 [2] [ 2,  3]
--
-- 'index' can also be provided at runtime:
--
-- >>> sSelect (SSelectDim (SByIndex @0)) (SUncheckedIndex 1) input
-- Tensor Int32 [2] [ 2,  3]
--
-- It produces a runtime error if the 'index' is too large:
--
-- >>> sSelect (SSelectDim (SByIndex @0)) (SUncheckedIndex 10) input
-- *** Exception: IndexOutOfBoundError {ioobeIndex = 10, ioobeDim = Dim {dimName = "*", dimSize = 4}}
sSelect ::
  forall selectDim index gradient layout device dataType shapeIn shapeOut m.
  ( index `InRangeF` GetDimF selectDim shapeIn,
    shapeOut ~ RemoveDimF selectDim shapeIn,
    Catch shapeOut,
    SGetShape shapeIn,
    MonadThrow m
  ) =>
  SSelectDim selectDim ->
  SIndex index ->
  Tensor gradient layout device dataType shapeIn ->
  m (Tensor gradient layout device dataType shapeOut)
sSelect sSelectDim sIndex input = do
  sDim <- let inputShape = sGetShape input in sGetDimFromShape sSelectDim inputShape
  let dim = bimap forgetIsChecked forgetIsChecked . fromSing $ sDim
      index = coerce . forgetIsChecked . fromSing $ sIndex
      selectDim = forgetIsChecked . fromSing $ sSelectDim
  if index < dimSize dim
    then unsafeThrowableIO $ case selectDim of
      ByName dimName -> ATen.cast3 ATen.tensor_select_nl input dimName (fromIntegral index :: Int)
      ByIndex dimIndex -> ATen.cast3 ATen.tensor_select_ll input (fromIntegral dimIndex :: Int) (fromIntegral index :: Int)
    else throwM $ IndexOutOfBoundError index dim

data IndexOutOfBoundError = IndexOutOfBoundError {ioobeIndex :: Integer, ioobeDim :: Dim String Integer}
  deriving stock (Show, Typeable)

instance Exception IndexOutOfBoundError where
  displayException IndexOutOfBoundError {..} =
    "Index `"
      <> show ioobeIndex
      <> "` is out of bounds for dimension `"
      <> show ioobeDim
      <> "`."

select ::
  forall selectDim index gradient layout device dataType shapeIn shapeOut m.
  ( SingI selectDim,
    SingI index,
    index `InRangeF` GetDimF selectDim shapeIn,
    shapeOut ~ RemoveDimF selectDim shapeIn,
    Catch shapeOut,
    SGetShape shapeIn,
    MonadThrow m
  ) =>
  Tensor gradient layout device dataType shapeIn ->
  m (Tensor gradient layout device dataType shapeOut)
select = sSelect (sing @selectDim) (sing @index)

-- | 'GatherDimImplF' is a type-level helper function for 'sGatherDim'.
--
-- >>> :kind! GatherDimImplF ('ByIndex 1) '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 4), 'Dim ('Name "feature") ('Size 1)] '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 1), 'Dim ('Name "feature") ('Size 1)]
-- ...
-- = Just
--     ['Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 4),
--      'Dim ('Name "feature") ('Size 1)]
-- >>> :kind! GatherDimImplF ('ByIndex 1) '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 4), 'Dim ('Name "feature") ('Size 1)] '[ 'Dim ('Name "*") ('Size 2), 'Dim ('Name "sequence") ('Size 1), 'Dim ('Name "*") ('Size 1)]
-- ...
-- = Just
--     ['Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 4),
--      'Dim ('Name "feature") ('Size 1)]
-- >>> :kind! GatherDimImplF ('ByIndex 1) '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 4), 'Dim ('Name "feature") ('Size 2)] '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 1), 'Dim ('Name "feature") ('Size 1)]
-- ...
-- = Nothing
-- >>> :kind! GatherDimImplF ('ByIndex 1) '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 4), 'Dim ('Name "feature") ('Size 1)] '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "boo") ('Size 1), 'Dim ('Name "feature") ('Size 1)]
-- ...
-- = Nothing
-- >>> :kind! GatherDimImplF ('ByIndex 1) '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 0), 'Dim ('Name "feature") ('Size 1)] '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 2), 'Dim ('Name "feature") ('Size 1)]
-- ...
-- = Nothing
-- >>> :kind! GatherDimImplF ('ByIndex 1) '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 1)] '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 1), 'Dim ('Name "feature") ('Size 1)]
-- ...
-- = Nothing
-- >>> :kind! GatherDimImplF ('ByIndex 2) '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 1), 'Dim ('Name "feature") ('Size 3)] '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 1), 'Dim ('Name "feature") ('Size 1)]
-- ...
-- = Just
--     ['Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 1),
--      'Dim ('Name "feature") ('Size 3)]
-- >>> :kind! GatherDimImplF ('ByName "feature") '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 1), 'Dim ('Name "feature") ('Size 3)] '[ 'Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 1), 'Dim ('Name "*") ('Size 1)]
-- ...
-- = Just
--     ['Dim ('Name "batch") ('Size 2), 'Dim ('Name "sequence") ('Size 1),
--      'Dim ('Name "feature") ('Size 3)]
type GatherDimImplF :: By Symbol Nat -> [Dim (Name Symbol) (Size Nat)] -> [Dim (Name Symbol) (Size Nat)] -> Maybe [Dim (Name Symbol) (Size Nat)]
type family GatherDimImplF by indexDims inputDims where
  GatherDimImplF ('ByName dimName) indexDims inputDims = GatherDimByNameF (GetIndexByNameF dimName indexDims) (GetIndexByNameF dimName inputDims) indexDims inputDims
  GatherDimImplF ('ByIndex dimIndex) indexDims inputDims = GatherDimByIndexF dimIndex indexDims inputDims

type GatherDimByIndexF :: Nat -> [Dim (Name Symbol) (Size Nat)] -> [Dim (Name Symbol) (Size Nat)] -> Maybe [Dim (Name Symbol) (Size Nat)]
type family GatherDimByIndexF dimIndex indexDims inputDims where
  GatherDimByIndexF 0 ('Dim _ ('Size 0) ': _) _ = 'Nothing
  GatherDimByIndexF 0 ('Dim indexDimName indexDimSize ': indexDims) ('Dim inputDimName _ ': inputDims) =
    If
      (UnifyCheck (Name Symbol) indexDimName inputDimName && UnifyCheck [Dim (Name Symbol) (Size Nat)] indexDims inputDims)
      ('Just ('Dim (indexDimName <+> inputDimName) indexDimSize ': (indexDims <+> inputDims)))
      'Nothing
  GatherDimByIndexF dimIndex (indexDim ': indexDims) (inputDim ': inputDims) =
    If
      (UnifyCheck (Dim (Name Symbol) (Size Nat)) indexDim inputDim)
      (PrependMaybe ('Just (indexDim <+> inputDim)) (GatherDimByIndexF (dimIndex - 1) indexDims inputDims))
      'Nothing
  GatherDimByIndexF _ _ _ = 'Nothing

type GatherDimByNameF :: Maybe Nat -> Maybe Nat -> [Dim (Name Symbol) (Size Nat)] -> [Dim (Name Symbol) (Size Nat)] -> Maybe [Dim (Name Symbol) (Size Nat)]
type family GatherDimByNameF dimIndex dimIndex' indexDims inputDims where
  GatherDimByNameF 'Nothing ('Just dimIndex') indexDims inputDims = GatherDimByIndexF dimIndex' indexDims inputDims
  GatherDimByNameF ('Just dimIndex) 'Nothing indexDims inputDims = GatherDimByIndexF dimIndex indexDims inputDims
  GatherDimByNameF ('Just dimIndex) ('Just dimIndex) indexDims inputDims = GatherDimByIndexF dimIndex indexDims inputDims
  GatherDimByNameF _ _ _ _ = 'Nothing

type GatherDimMessage :: By Symbol Nat -> [Dim (Name Symbol) (Size Nat)] -> [Dim (Name Symbol) (Size Nat)] -> ErrorMessage

type GatherDimMessage by indexDims inputDims =
  "Cannot gather the tensor with the dimensions"
    % ""
    % "    '" <> inputDims <> "'"
    % ""
    % "at the dimension"
    % ""
    % "    '" <> by <> "'"
    % ""
    % "using an index of shape"
    % ""
    % "    '" <> indexDims <> "'."
    % ""

type GatherDimCheckF ::
  By Symbol Nat ->
  [Dim (Name Symbol) (Size Nat)] ->
  [Dim (Name Symbol) (Size Nat)] ->
  Maybe [Dim (Name Symbol) (Size Nat)] ->
  [Dim (Name Symbol) (Size Nat)]
type family GatherDimCheckF by indexDims inputDims result where
  GatherDimCheckF by indexDims inputDims 'Nothing = TypeError (GatherDimMessage by indexDims inputDims)
  GatherDimCheckF _ _ _ ('Just dims) = dims

-- | Calculate the output shape of a gather operation for a given index shape along a given axis.
--
-- >>> :kind! GatherDimF ('SelectDim ('ByIndex 2)) ('Shape '[ 'Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 3)]) ('Shape '[ 'Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 1)])
-- ...
-- = 'Shape
--     ['Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 1),
--      'Dim ('Name "*") ('Size 3)]
type GatherDimF :: SelectDim (By Symbol Nat) -> Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)]
type family GatherDimF selectDim indexShape inputShape where
  GatherDimF 'UncheckedSelectDim _ _ = 'UncheckedShape
  GatherDimF _ 'UncheckedShape _ = 'UncheckedShape
  GatherDimF _ _ 'UncheckedShape = 'UncheckedShape
  GatherDimF ('SelectDim by) ('Shape indexDims) ('Shape inputDims) = 'Shape (GatherDimCheckF by indexDims inputDims (GatherDimImplF by indexDims inputDims))

-- | Gather values along an axis for a specified dimension.
--
-- >>> sToTensor' = sToTensor (SGradient SWithoutGradient) (SLayout SDense) (SDevice SCPU)
-- >>> t <- sToTensor' [[1 :: Float, 2], [3, 4]]
-- >>> idx <- sToTensor' [[0 :: Int, 0], [1, 0]]
-- >>> result <- sGatherDim (SSelectDim $ SByIndex @1) idx t
-- >>> :t result
-- result
--   :: Tensor
--        ('Gradient WithoutGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape
--           ['Dim ('Name "*") UncheckedSize, 'Dim ('Name "*") UncheckedSize])
-- >>> result
-- Tensor Float [2,2] [[ 1.0000   ,  1.0000   ],
--                     [ 4.0000   ,  3.0000   ]]
-- >>> shape = SShape $ SNoName :&: SSize @2 :|: SNoName :&: SSize @2 :|: SNil
-- >>> t' <- sCheckedShape shape t
-- >>> idx' <- sCheckedShape shape idx
-- >>> result <- sGatherDim (SSelectDim $ SByIndex @1) idx' t'
-- >>> :t result
-- result
--   :: Tensor
--        ('Gradient WithoutGradient)
--        ('Layout Dense)
--        ('Device CPU)
--        ('DataType 'Float)
--        ('Shape ['Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 2)])
-- >>> result
-- Tensor Float [2,2] [[ 1.0000   ,  1.0000   ],
--                     [ 4.0000   ,  3.0000   ]]
sGatherDim ::
  forall selectDim indexGradient inputGradient indexLayout inputLayout indexDevice inputDevice indexDataType inputDataType indexShape inputShape outputShape m.
  (MonadThrow m, outputShape ~ GatherDimF selectDim indexShape inputShape, Catch outputShape, Catch (indexDataType <+> 'DataType 'Int64)) =>
  SSelectDim selectDim ->
  -- | the indices of elements to gather
  Tensor indexGradient indexLayout indexDevice indexDataType indexShape ->
  -- | input
  Tensor inputGradient inputLayout inputDevice inputDataType inputShape ->
  -- | output
  m
    ( Tensor
        (indexGradient <|> inputGradient)
        (indexLayout <+> inputLayout)
        (indexDevice <+> inputDevice)
        inputDataType
        outputShape
    )
sGatherDim selectDim index input =
  let by = forgetIsChecked . fromSing $ selectDim
   in case by of
        ByName dimName -> unsafeThrowableIO $ ATen.cast4 ATen.gather_tntb input dimName index False
        ByIndex dimIndex -> unsafeThrowableIO $ ATen.cast4 ATen.gather_tltb input (fromIntegral dimIndex :: Int) index False
