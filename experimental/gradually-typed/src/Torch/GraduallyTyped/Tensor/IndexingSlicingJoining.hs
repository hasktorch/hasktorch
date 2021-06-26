{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.Tensor.IndexingSlicingJoining where

import Data.Kind (Type)
import Data.Singletons (SingI (..), SingKind (..), fromSing)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (Nat, Symbol, TypeError)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Index.Class (InRangeF)
import Torch.GraduallyTyped.Index.Type (SIndex)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.Prelude (FromMaybe, MapMaybe, forgetIsChecked)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (AddDimF, BroadcastShapesF, GetDimF, GetDimImplF, GetIndexByNameF, InsertDimImplF, NumelF, RemoveDimF, ReplaceDimF, ReplaceDimImplF, sGetDim)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SSelectDim, SShape, SelectDim (..), Shape (..), Size (..), dimSize)
import Torch.GraduallyTyped.Tensor.Type (SGetShape (sShape), Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Torch.HList (HList)
import Torch.Internal.Cast (cast2, cast3)
import Torch.Internal.Class (Castable)
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen
import Type.Errors.Pretty (ToErrorMessage, type (%), type (<>))

-- $setup
-- >>> import Data.Singletons.Prelude.List (SList (..))
-- >>> import Torch.GraduallyTyped

class HasCat (selectDim :: SelectDim (By Symbol Nat)) k (c :: k -> Type) (a :: k) where
  type CatF selectDim a c :: Type

  -- | Concatenates the given sequence of seq tensors in the given dimension.
  -- All tensors must either have the same shape (except in the concatenating dimension) or be empty.
  --
  -- >>> t = ones @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 32), 'Dim ('Name "feature") ('Size 8)])
  -- >>> :type cat @('SelectDim ('ByName "feature")) [t]
  -- cat @('SelectDim ('ByName "feature")) [t]
  --   :: Tensor
  --        'WithGradient
  --        ('Layout 'Dense)
  --        ('Device 'CPU)
  --        ('DataType 'Float)
  --        ('Shape
  --           '[ 'Dim ('Name "batch") ('Size 32),
  --              'Dim 'UncheckedName 'UncheckedSize])
  -- >>> :type cat @('SelectDim ( 'ByIndex 0)) [t]
  -- cat @('SelectDim ( 'ByIndex 0)) [t]
  --   :: Tensor
  --        'WithGradient
  --        ('Layout 'Dense)
  --        ('Device 'CPU)
  --        ('DataType 'Float)
  --        ('Shape
  --           '[ 'Dim 'UncheckedName 'UncheckedSize,
  --              'Dim ('Name "feature") ('Size 8)])
  -- >>> :type cat @'UncheckedSelectDim (ByIndex 0) [t]
  -- cat @'UncheckedSelectDim (ByIndex 0) [t]
  --   :: Tensor
  --        'WithGradient
  --        ('Layout 'Dense)
  --        ('Device 'CPU)
  --        ('DataType 'Float)
  --        'UncheckedShape
  sCat :: SSelectDim selectDim -> c a -> CatF selectDim a c

  cat :: SingI selectDim => c a -> CatF selectDim a c
  cat = sCat (sing @selectDim)

type family CatListImplF (selectDim :: SelectDim (By Symbol Nat)) (tensor :: Type) :: Maybe Type where
  CatListImplF 'UncheckedSelectDim (Tensor requiresGradient layout device dataType _) = 'Just (Tensor requiresGradient layout device dataType 'UncheckedShape)
  CatListImplF ('SelectDim _) (Tensor requiresGradient layout device dataType 'UncheckedShape) = 'Just (Tensor requiresGradient layout device dataType 'UncheckedShape)
  CatListImplF ('SelectDim by) (Tensor requiresGradient layout device dataType ('Shape dims)) = MapMaybe (Tensor requiresGradient layout device dataType) (MapMaybe 'Shape (ReplaceDimImplF by dims ('Dim 'UncheckedName 'UncheckedSize)))

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
  Castable (CatListF selectDim (Tensor requiresGradient layout device dataType shape)) (ForeignPtr ATen.Tensor) =>
  HasCat selectDim Type [] (Tensor requiresGradient layout device dataType shape)
  where
  type CatF selectDim (Tensor requiresGradient layout device dataType shape) [] = CatListF selectDim (Tensor requiresGradient layout device dataType shape)
  sCat selectDim tensors =
    let by = forgetIsChecked . fromSing $ selectDim
     in case by of
          ByName name -> unsafePerformIO $ cast2 ATen.cat_ln tensors name
          ByIndex index -> unsafePerformIO $ cast2 ATen.cat_ll tensors (fromInteger index :: Int)

type family
  CatHListImplF
    (selectDim :: SelectDim (By Symbol Nat))
    (tensors :: [Type])
    (acc :: Maybe (RequiresGradient, Layout LayoutType, Device (DeviceType Nat), DataType DType, Shape [Dim (Name Symbol) (Size Nat)])) ::
    Type
  where
  CatHListImplF _ '[] 'Nothing = TypeError (ToErrorMessage "Cannot concatenate an empty list of tensors.")
  CatHListImplF _ '[] ('Just '(requiresGradient, layout, device, dataType, shape)) = Tensor requiresGradient layout device dataType shape
  CatHListImplF selectDim (Tensor requiresGradient layout device dataType shape ': tensors) 'Nothing =
    CatHListImplF selectDim tensors ('Just '(requiresGradient, layout, device, dataType, shape))
  CatHListImplF selectDim (Tensor requiresGradient layout device dataType shape ': tensors) ('Just '(requiresGradient', layout', device', dataType', shape')) =
    CatHListImplF
      selectDim
      tensors
      ( 'Just
          '( requiresGradient <|> requiresGradient',
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
  ( Castable (CatHListF selectDim tensors) (ForeignPtr ATen.Tensor),
    Castable (HList tensors) (ForeignPtr ATen.TensorList)
  ) =>
  HasCat selectDim [Type] HList tensors
  where
  type CatF selectDim tensors HList = CatHListF selectDim tensors
  sCat selectDim tensors =
    let by = forgetIsChecked . fromSing $ selectDim
     in case by of
          ByName name -> unsafePerformIO $ cast2 ATen.cat_ln tensors name
          ByIndex index -> unsafePerformIO $ cast2 ATen.cat_ll tensors (fromInteger index :: Int)

type ReshapeNumelMismatchMessage (numel :: Nat) (numel' :: Nat) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (shape' :: Shape [Dim (Name Symbol) (Size Nat)]) =
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

type family ReshapeImplF (numel :: Maybe Nat) (numel' :: Maybe Nat) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (shape' :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  ReshapeImplF ('Just numel) ('Just numel) _ shape' = shape'
  ReshapeImplF ('Just numel) ('Just numel') shape shape' = TypeError (ReshapeNumelMismatchMessage numel numel' shape shape')
  ReshapeImplF 'Nothing _ _ _ = 'UncheckedShape
  ReshapeImplF _ 'Nothing _ _ = 'UncheckedShape
  ReshapeImplF _ _ 'UncheckedShape _ = 'UncheckedShape
  ReshapeImplF _ _ _ 'UncheckedShape = 'UncheckedShape

type family ReshapeF (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (shape' :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  ReshapeF shape shape' = ReshapeImplF (NumelF shape) (NumelF shape') shape shape'

-- | Returns a tensor with the same data and number of elements as the input tensor,
-- but with the specified shape:
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> (input, _) = sRandn SWithGradient (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SShape $ SName @"*" :&: SSize @4 :|: SNil) g
-- >>> output = sReshape (SShape $ SName @"*" :&: SSize @2 :|: SName @"*" :&: SSize @2 :|: SNil) input
-- >>> :type output
-- output
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape '[ 'Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 2)])
--
-- At the value level, a single dimension may be '-1',
-- in which case it is inferred from the remaining dimensions and the number of elements in the input:
--
-- >>> output' = sReshape (SShape $ SUncheckedName "*" :&: SUncheckedSize (-1) :|: SNil) output
-- >>> :type output'
-- output'
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        'UncheckedShape
-- >>> shape output'
-- [Dim {dimName = "*", dimSize = 4}]
sReshape ::
  forall shape' requiresGradient layout device dataType shape shape''.
  (shape'' ~ ReshapeF shape shape') =>
  SShape shape' ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape''
sReshape shape' input =
  let dimSizes =
        fmap (\(Dim _ size) -> forgetIsChecked size)
          . forgetIsChecked
          . fromSing
          $ shape'
   in unsafePerformIO $ cast2 ATen.reshape_tl input dimSizes

reshape ::
  forall shape' requiresGradient layout device dataType shape shape''.
  ( shape'' ~ ReshapeF shape shape',
    SingI shape'
  ) =>
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape''
reshape = sReshape (sing @shape')

type TransposeBy0Message (by0 :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) =
  "Cannot transpose the tensor with the dimensions"
    % ""
    % "    '" <> dims <> "'"
    % ""
    % "because the specified source dimension"
    % ""
    % "    '" <> by0 <> "'"
    % ""
    % "could not be found."

type TransposeBy1Message (by1 :: By Symbol Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) =
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
--                                                         [Dim (Name Symbol) (Size Nat)]
-- = 'Shape
--     '[ 'Dim ('Name "feature") ('Size 8),
--        'Dim ('Name "batch") ('Size 10),
--        'Dim ('Name "anotherFeature") ('Size 12)]
-- >>> :kind! TransposeF SelectFeature SelectBatch ('Shape Dims)
-- TransposeF SelectFeature SelectBatch ('Shape Dims) :: Shape
--                                                         [Dim (Name Symbol) (Size Nat)]
-- = 'Shape
--     '[ 'Dim ('Name "feature") ('Size 8),
--        'Dim ('Name "batch") ('Size 10),
--        'Dim ('Name "anotherFeature") ('Size 12)]
type family TransposeF (selectDim0 :: SelectDim (By Symbol Nat)) (selectDim1 :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
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

type family TransposeIndexIndexDimsF (index0 :: Nat) (index1 :: Nat) (dims :: [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
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

-- | Returns a tensor that is a transposed version of 'input'.
-- The selected dimensions 'selectDim0' and 'selectDim1' are swapped.
--
-- >>> g <- mkGenerator @('Device 'CPU) 0
-- >>> (input, _) = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "batch") ('Size 10), 'Dim ('Name "feature") ('Size 5)]) g
-- >>> output = transpose @('SelectDim ('ByName "batch")) @('SelectDim ('ByName "feature")) input
-- >>> :type output
-- output
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('Name "feature") ('Size 5),
--              'Dim ('Name "batch") ('Size 10)])
-- >>> output = transpose @'UncheckedSelectDim @('SelectDim ('ByIndex 1)) (ByIndex 0) input
-- >>> :type output
-- output
--   :: Tensor
--        'WithGradient
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        'UncheckedShape
-- >>> shape output
-- [W TensorImpl.h:934] Warning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (function operator())
-- [Dim {dimName = "feature", dimSize = 5},Dim {dimName = "batch", dimSize = 10}]
sTranspose ::
  forall selectDim0 selectDim1 requiresGradient layout device dataType shape shape'.
  ( shape' ~ TransposeF selectDim0 selectDim1 shape
  ) =>
  SSelectDim selectDim0 ->
  SSelectDim selectDim1 ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape'
sTranspose selectDim0 selectDim1 input =
  let by0 = forgetIsChecked . fromSing $ selectDim0
      by1 = forgetIsChecked . fromSing $ selectDim1
   in case (by0, by1) of
        (ByName name0, ByName name1) -> unsafePerformIO $ cast3 ATen.transpose_tnn input name0 name1
        (ByIndex index0, ByIndex index1) -> unsafePerformIO $ cast3 ATen.transpose_tll input (fromIntegral index0 :: Int) (fromIntegral index1 :: Int)
        _ ->
          error $
            "Cannot transpose the tensor. "
              <> "The source and target dimensions must be selected either both by name or both by index, "
              <> "but mixed selectors were found: '"
              <> show by0
              <> "' and '"
              <> show by1
              <> "'."

transpose ::
  forall selectDim0 selectDim1 requiresGradient layout device dataType shape shape'.
  ( shape' ~ TransposeF selectDim0 selectDim1 shape,
    SingI selectDim0,
    SingI selectDim1
  ) =>
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape'
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

sUnsqueeze ::
  forall selectDim requiresGradient layout device dataType shape shape'.
  ( shape' ~ UnsqueezeF selectDim shape
  ) =>
  SSelectDim selectDim ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape'
sUnsqueeze selectDim input =
  let by = forgetIsChecked . fromSing $ selectDim
   in case by of
        ByName _name -> undefined
        ByIndex index -> unsafePerformIO $ cast2 ATen.unsqueeze_tl input (fromIntegral index :: Int)

unsqueeze ::
  forall selectDim requiresGradient layout device dataType shape shape'.
  ( shape' ~ UnsqueezeF selectDim shape,
    SingI selectDim
  ) =>
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape'
unsqueeze = sUnsqueeze (sing @selectDim)

sExpand ::
  forall shape' requiresGradient layout device dataType shape input output.
  ( input ~ Tensor requiresGradient layout device dataType shape,
    output ~ Tensor requiresGradient layout device dataType (BroadcastShapesF shape shape')
  ) =>
  SShape shape' ->
  input ->
  output
sExpand shape' input =
  let sizes' = fmap (\(Dim _ size) -> forgetIsChecked size) . forgetIsChecked $ fromSing shape'
   in unsafePerformIO $ cast3 ATen.tensor_expand_lb input sizes' True

expand ::
  forall shape' requiresGradient layout device dataType shape input output.
  ( SingI shape',
    input ~ Tensor requiresGradient layout device dataType shape,
    output ~ Tensor requiresGradient layout device dataType (BroadcastShapesF shape shape')
  ) =>
  input ->
  output
expand = sExpand (sing @shape')

-- | Slices the self tensor along the selected dimension at the given index. This function returns a view of the original tensor with the given dimension removed.
--
-- >>> nats = sArangeNaturals SWithoutGradient (SLayout SDense) (SDevice SCPU) (SDataType SInt32) (SSize @8)
-- >>> input = sReshape (SShape $ SName @"*" :&: SSize @4 :|: SName @"*" :&: SSize @2 :|: SNil) nats
-- >>> input
-- Tensor Int32 [4,2] [[ 0,  1],
--                     [ 2,  3],
--                     [ 4,  5],
--                     [ 6,  7]]
--
-- `index` can be provided at compile-time:
-- >>> sSelect (SSelectDim (SByIndex @0)) (SIndex @1) input
-- Tensor Int32 [2] [ 2,  3]
--
-- `index` can also be provided at runtime:
-- >>> sSelect (SSelectDim (SByIndex @0)) (SUncheckedIndex 1) input
-- Tensor Int32 [2] [ 2,  3]
--
-- It produces a runtime error if the `index` is too large:
-- >>> sSelect (SSelectDim (SByIndex @0)) (SUncheckedIndex 10) input
-- *** Exception: Out of bound index 10 for dimension Dim {dimName = "*", dimSize = 4}
-- CallStack (from HasCallStack):
--   error, called at ...
sSelect ::
  forall selectDim index requiresGradient layout device dataType shapeIn shapeOut.
  ( index `InRangeF` GetDimF selectDim shapeIn,
    shapeOut ~ RemoveDimF selectDim shapeIn,
    SGetShape shapeIn
  ) =>
  SSelectDim selectDim ->
  SIndex index ->
  Tensor requiresGradient layout device dataType shapeIn ->
  Tensor requiresGradient layout device dataType shapeOut
sSelect sSelectDim sIndex input =
  let sDim = unsafePerformIO $ do
        inputShape <- sShape input
        sGetDim sSelectDim inputShape
      dim = (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size)) . fromSing $ sDim
      index = forgetIsChecked . fromSing $ sIndex
      selectDim = forgetIsChecked . fromSing $ sSelectDim
   in if index < (fromInteger . dimSize $ dim)
        then case selectDim of
          ByName name -> unsafePerformIO $ cast3 ATen.tensor_select_nl input name (fromIntegral index :: Int)
          ByIndex dimIndex -> unsafePerformIO $ cast3 ATen.tensor_select_ll input (fromIntegral dimIndex :: Int) (fromIntegral index :: Int)
        else error $ "Out of bound index " <> show index <> " for dimension " <> show dim

select ::
  forall selectDim index requiresGradient layout device dataType shapeIn shapeOut.
  ( SingI selectDim,
    SingI index,
    index `InRangeF` GetDimF selectDim shapeIn,
    shapeOut ~ RemoveDimF selectDim shapeIn,
    SGetShape shapeIn
  ) =>
  Tensor requiresGradient layout device dataType shapeIn ->
  Tensor requiresGradient layout device dataType shapeOut
select = sSelect (sing @selectDim) (sing @index)
