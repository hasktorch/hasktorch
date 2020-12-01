{-# LANGUAGE AllowAmbiguousTypes #-}
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

module Torch.GraduallyTyped.Tensor.IndexingSlicingJoining where

import Data.Kind (Type)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (Nat, Symbol, TypeError)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType)
import Torch.GraduallyTyped.DType (DataType (..), UnifyDataTypeF)
import Torch.GraduallyTyped.Device (Device (..), DeviceType, UnifyDeviceF)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType, UnifyLayoutF)
import Torch.GraduallyTyped.Prelude (FromMaybe, MapMaybe)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), UnifyRequiresGradientF)
import Torch.GraduallyTyped.Shape (dimSize, GetDimImplF, GetIndexByNameF, Size(..), Name(..), AddDimF, By (..), Dim (..), GetDimF, NumelF, ReplaceDimF, ReplaceDimImplF, SelectDim (..), Shape (..), UnifyShapeF, WithSelectDimC (..), WithShapeC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorF)
import Torch.HList (HList)
import Torch.Internal.Cast (cast2, cast3)
import Torch.Internal.Class (Castable)
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Type as ATen
import Type.Errors.Pretty (ToErrorMessage, type (%), type (<>))

-- $setup
-- >>> import Torch.GraduallyTyped.Tensor.Creation (ones)
-- >>> import Torch.DType (DType (..))
-- >>> import Torch.GraduallyTyped.DType (DataType (..))
-- >>> import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
-- >>> import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
-- >>> import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
-- >>> import Torch.GraduallyTyped.Shape (Dim (..), Shape (..))

class HasCat (selectDim :: SelectDim (By Symbol Nat)) k (c :: k -> Type) (a :: k) where
  type CatF selectDim a c :: Type

  -- | Concatenates the given sequence of seq tensors in the given dimension.
  -- All tensors must either have the same shape (except in the concatenating dimension) or be empty.
  --
  -- >>> t <- ones @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'NamedSized "batch" 32, 'NamedSized "feature" 8])
  -- [W TensorImpl.h:840] Warning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (function operator())
  -- >>> :type cat @('SelectDim ('ByName "feature")) [t]
  -- cat @('SelectDim ('ByName "feature")) [t]
  -- :: Tensor
  --      'Dependent
  --      ('Layout 'Dense)
  --      ('Device 'CPU)
  --      ('DataType 'Float)
  --      ('Shape '[ 'Dim ( 'NamedSized "batch" 32), 'UncheckedDim])
  -- >>> :type cat @('SelectDim ( 'ByIndex 0)) [t]
  -- cat @('SelectDim ( 'ByIndex 0)) [t]
  --   :: Tensor
  --        'Dependent
  --        ('Layout 'Dense)
  --        ('Device 'CPU)
  --        ('DataType 'Float)
  --        ('Shape '[ 'UncheckedDim, 'Dim ( 'NamedSized "feature" 8)])
  -- >>> :type cat @'UncheckedSelectDim (SelectDim (ByIndex 0)) [t]
  -- cat @'UncheckedSelectDim (SelectDim (ByIndex 0)) [t]
  --   :: Tensor
  --        'Dependent
  --        ('Layout 'Dense)
  --        ('Device 'CPU)
  --        ('DataType 'Float)
  --        'AnyShape
  cat :: WithSelectDimF selectDim (c a -> CatF selectDim a c)

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
  CatListCheckF _ _ ( 'Just result) = result

type CatListF selectDim tensor = CatListCheckF selectDim tensor (CatListImplF selectDim tensor)

instance
  ( WithSelectDimC selectDim ([Tensor requiresGradient layout device dataType shape] -> CatListF selectDim (Tensor requiresGradient layout device dataType shape)),
    Castable (CatListF selectDim (Tensor requiresGradient layout device dataType shape)) (ForeignPtr ATen.Tensor)
  ) =>
  HasCat selectDim Type [] (Tensor requiresGradient layout device dataType shape)
  where
  type CatF selectDim (Tensor requiresGradient layout device dataType shape) [] = CatListF selectDim (Tensor requiresGradient layout device dataType shape)
  cat =
    withSelectDim @selectDim @([Tensor requiresGradient layout device dataType shape] -> CatF selectDim (Tensor requiresGradient layout device dataType shape) []) $
      \by tensors -> case by of
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
  CatHListImplF _ '[] ( 'Just '(requiresGradient, layout, device, dataType, shape)) = TensorF '(requiresGradient, layout, device, dataType, shape)
  CatHListImplF selectDim (Tensor requiresGradient layout device dataType shape ': tensors) 'Nothing =
    CatHListImplF selectDim tensors ( 'Just '(requiresGradient, layout, device, dataType, shape))
  CatHListImplF selectDim (Tensor requiresGradient layout device dataType shape ': tensors) ( 'Just '(requiresGradient', layout', device', dataType', shape')) =
    CatHListImplF
      selectDim
      tensors
      ( 'Just
          '( UnifyRequiresGradientF requiresGradient requiresGradient',
             UnifyLayoutF layout layout',
             UnifyDeviceF device device',
             UnifyDataTypeF dataType dataType',
             ReplaceDimF
               selectDim
               (UnifyShapeF (ReplaceDimF selectDim shape ('Dim 'UncheckedName 'UncheckedSize)) (ReplaceDimF selectDim shape' ('Dim 'UncheckedName 'UncheckedSize)))
               (AddDimF (GetDimF selectDim shape) (GetDimF selectDim shape))
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
  ( WithSelectDimC selectDim (HList tensors -> CatHListF selectDim tensors),
    Castable (CatHListF selectDim tensors) (ForeignPtr ATen.Tensor),
    Castable (HList tensors) (ForeignPtr ATen.TensorList)
  ) =>
  HasCat selectDim [Type] HList tensors
  where
  type CatF selectDim tensors HList = CatHListF selectDim tensors
  cat =
    withSelectDim @selectDim @(HList tensors -> CatF selectDim tensors HList) $
      \by tensors -> case by of
        ByName name -> unsafePerformIO $ cast2 ATen.cat_ln tensors name
        ByIndex index -> unsafePerformIO $ cast2 ATen.cat_ll tensors (fromInteger index :: Int)

uncheckedCat ::
  By String Integer ->
  [Tensor 'Dependent 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape] ->
  Tensor
    'Dependent
    'UncheckedLayout
    'UncheckedDevice
    'UncheckedDataType
    'UncheckedShape
uncheckedCat = cat @ 'UncheckedSelectDim

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
  ReshapeImplF ( 'Just numel) ( 'Just numel) _ shape' = shape'
  ReshapeImplF ( 'Just numel) ( 'Just numel') shape shape' = TypeError (ReshapeNumelMismatchMessage numel numel' shape shape')
  ReshapeImplF 'Nothing _ _ _ = 'UncheckedShape
  ReshapeImplF _ 'Nothing _ _ = 'UncheckedShape
  ReshapeImplF _ _ 'UncheckedShape _ = 'UncheckedShape
  ReshapeImplF _ _ _ 'UncheckedShape = 'UncheckedShape

type family ReshapeF (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (shape' :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  ReshapeF shape shape' = ReshapeImplF (NumelF shape) (NumelF shape') shape shape'

-- | Returns a tensor with the same data and number of elements as the input tensor,
-- but with the specified shape:
--
-- >>> g <- generator @('Device 'CPU) 0
-- >>> (input, _) = randn @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Sized 4)]) g
-- >>> output = reshape @('Shape '[ 'Dim ('Sized 2), 'Dim ('Sized 2)]) input
-- >>> :type output
-- output
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape '[ 'Dim ('Sized 2), 'Dim ('Sized 2)])
--
-- At the value level, a single dimension may be '-1',
-- in which case it is inferred from the remaining dimensions and the number of elements in the input:
--
-- >>> output' = reshape @('Shape '[ 'UncheckedDim]) (Sized (-1)) output
-- >>> :type output'
-- output'
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        'UncheckedShape
-- >>> shape output'
-- [Sized 4]
reshape ::
  forall shape' requiresGradient layout device dataType shape shape''.
  ( shape'' ~ ReshapeF shape shape',
    WithShapeC shape' (Tensor requiresGradient layout device dataType shape -> Tensor requiresGradient layout device dataType shape'')
  ) =>
  WithShapeF shape' (Tensor requiresGradient layout device dataType shape -> Tensor requiresGradient layout device dataType shape'')
reshape = withShape @shape' @(Tensor requiresGradient layout device dataType shape -> Tensor requiresGradient layout device dataType shape'') $ \dims' input ->
  unsafePerformIO $ cast2 ATen.reshape_tl input (fmap dimSize dims')

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
-- >>> type SelectDim0 = 'SelectDim ('ByName "batch")
-- >>> type SelectDim1 = 'SelectDim ('ByName "feature")
-- >>> type Dims = '[ 'Dim ('NamedSized "batch" 10), 'Dim ('NamedSized "feature" 8), 'Dim ('NamedSized "anotherFeature" 12)]
-- >>> :kind! TransposeF SelectDim0 SelectDim1 ('Shape Dims)
-- TransposeF SelectDim0 SelectDim1 ('Shape Dims) :: Shape [Dim (Name Symbol) (Size Nat)]
-- = 'Shape '[ 'Dim ('NamedSized "feature" 8), 'Dim ('NamedSized "batch" 10), 'Dim ('NamedSized "anotherFeature" 12)]
-- >>> :kind! TransposeF SelectDim1 SelectDim0 ('Shape Dims)
-- TransposeF SelectDim1 SelectDim0 ('Shape Dims) :: Shape [Dim (Name Symbol) (Size Nat)]
-- = 'Shape '[ 'Dim ('NamedSized "feature" 8), 'Dim ('NamedSized "batch" 10), 'Dim ('NamedSized "anotherFeature" 12)]
type family TransposeF (selectDim0 :: SelectDim (By Symbol Nat)) (selectDim1 :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  TransposeF _ _ 'UncheckedShape = 'UncheckedShape
  TransposeF _ 'UncheckedSelectDim _ = 'UncheckedShape
  TransposeF 'UncheckedSelectDim _ _ = 'UncheckedShape
  TransposeF ( 'SelectDim ( 'ByName name0)) ( 'SelectDim ( 'ByName name1)) ( 'Shape dims) =
    'Shape
      ( TransposeIndexIndexDimsF
          ( FromMaybe
              (TypeError (TransposeBy0Message (ByName name0) dims))
              (GetIndexByNameF name0 dims)
          )
          ( FromMaybe
              (TypeError (TransposeBy1Message (ByName name1) dims))
              (GetIndexByNameF name1 dims)
          )
          dims
      )
  TransposeF ( 'SelectDim ( 'ByIndex index0)) ( 'SelectDim ( 'ByIndex index1)) ( 'Shape dims) = 'Shape (TransposeIndexIndexDimsF index0 index1 dims)
  TransposeF ( 'SelectDim by0) ( 'SelectDim by1) _ =
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
      (TypeError (TransposeBy1Message (ByIndex index1) dims))
      ( ReplaceDimImplF
          (ByIndex index1)
          ( FromMaybe
              (TypeError (TransposeBy0Message (ByIndex index0) dims))
              ( ReplaceDimImplF
                  (ByIndex index0)
                  dims
                  ( FromMaybe
                      (TypeError (TransposeBy1Message (ByIndex index1) dims))
                      (GetDimImplF (ByIndex index1) dims)
                  )
              )
          )
          ( FromMaybe
              (TypeError (TransposeBy0Message (ByIndex index0) dims))
              (GetDimImplF (ByIndex index0) dims)
          )
      )

-- | Returns a tensor that is a transposed version of 'input'.
-- The selected dimensions 'selectDim0' and 'selectDim1' are swapped.
--
-- >>> g <- generator @('Device CPU) 0
-- >>> (input, _) = randn @'Dependent @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('NamedSized "batch" 10), 'Dim ('NamedSized "feature" 5)]) g
-- >>> output = transpose @('SelectDim ('ByName "batch")) @('SelectDim ('ByName "feature")) input
-- >>> :type output
-- output
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        ('Shape
--           '[ 'Dim ('NamedSized "feature" 5), 'Dim ('NamedSized "batch" 10)])
-- >>> output = transpose @'UncheckedSelectDim @('SelectDim ('ByIndex 1)) (ByIndex 0) input
-- >>> :type output
-- output
--   :: Tensor
--        'Dependent
--        ('Layout 'Dense)
--        ('Device 'CPU)
--        ('DataType 'Float)
--        'UncheckedShape
-- >>> shape output
-- [NamedSized "feature" 5,NamedSized "batch" 10]
transpose ::
  forall selectDim0 selectDim1 requiresGradient layout device dataType shape shape'.
  ( WithSelectDimC selectDim0 (WithSelectDimF selectDim1 (Tensor requiresGradient layout device dataType shape -> Tensor requiresGradient layout device dataType shape')),
    WithSelectDimC selectDim1 (Tensor requiresGradient layout device dataType shape -> Tensor requiresGradient layout device dataType shape'),
    shape' ~ TransposeF selectDim0 selectDim1 shape
  ) =>
  WithSelectDimF selectDim0 (WithSelectDimF selectDim1 (Tensor requiresGradient layout device dataType shape -> Tensor requiresGradient layout device dataType shape'))
transpose = withSelectDim @selectDim0 $
  \selectDim0 -> withSelectDim @selectDim1 @(Tensor requiresGradient layout device dataType shape -> Tensor requiresGradient layout device dataType shape') $
    \selectDim1 input ->
      case (selectDim0, selectDim1) of
        (ByName name0, ByName name1) -> unsafePerformIO $ cast3 ATen.transpose_tnn input name0 name1
        (ByIndex index0, ByIndex index1) -> unsafePerformIO $ cast3 ATen.transpose_tll input (fromIntegral index0 :: Int) (fromIntegral index1 :: Int)
        _ ->
          error $
            "Cannot transpose the tensor. "
              <> "The source and target dimensions must be selected either both by name or both by index, "
              <> "but mixed selectors were found: '"
              <> show selectDim0
              <> "' and '"
              <> show selectDim1
              <> "'."