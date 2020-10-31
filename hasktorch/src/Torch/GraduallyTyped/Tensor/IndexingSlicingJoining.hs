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
import Torch.GraduallyTyped.Prelude (FromMaybe, LiftTypeEqMaybe, MapMaybe)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), UnifyRequiresGradientF)
import Torch.GraduallyTyped.Shape (AddDimF, By (..), Dim (..), DimType (..), GetDimDimsImplF, GetDimF, NumelF, ReplaceDimDimsImplF, ReplaceDimF, ReplaceDimImplF, SelectDim (..), Shape (..), UnifyShapeF, WithSelectDimC (..), WithShapeC (..), sizedDims)
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
  CatListImplF selectDim (Tensor requiresGradient layout device dataType shape) = MapMaybe (Tensor requiresGradient layout device dataType) (MapMaybe 'Shape (ReplaceDimImplF selectDim shape 'UncheckedDim))

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
    (acc :: Maybe (RequiresGradient, Layout LayoutType, Device (DeviceType Nat), DataType DType, Shape [Dim (DimType Symbol Nat)])) ::
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
               (UnifyShapeF (ReplaceDimF selectDim shape 'UncheckedDim) (ReplaceDimF selectDim shape' 'UncheckedDim))
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

reshape ::
  forall shape' requiresGradient layout device dataType shape.
  ( LiftTypeEqMaybe (NumelF shape') (NumelF shape),
    WithShapeC shape' (Tensor requiresGradient layout device dataType shape')
  ) =>
  Tensor requiresGradient layout device dataType shape ->
  WithShapeF shape' (Tensor requiresGradient layout device dataType shape')
reshape input = withShape @shape' @(Tensor requiresGradient layout device dataType shape') $ \shape' ->
  case sizedDims shape' of
    Just sizes -> unsafePerformIO $ cast2 ATen.reshape_tl input sizes
    Nothing -> error $ "Invalid tensor shape specification '" <> show shape' <> "'."

type TransposeBy0Message (by0 :: By Symbol Nat) (dims :: [Dim (DimType Symbol Nat)]) =
  "Cannot transpose the tensor with the dimensions"
    % ""
    % "    '" <> dims <> "'"
    % ""
    % "because the specified source dimension"
    % ""
    % "    '" <> by0 <> "'"
    % ""
    % "could not be found."

type TransposeBy1Message (by1 :: By Symbol Nat) (dims :: [Dim (DimType Symbol Nat)]) =
  "Cannot transpose the tensor with the dimensions"
    % ""
    % "    '" <> dims <> "'"
    % ""
    % "because the specified target dimension"
    % ""
    % "    '" <> by1 <> "'"
    % ""
    % "could not be found."

-- | Transpose
--
-- >>> :kind! Transpose '[3,2] 0 1
-- Transpose '[3,2] 0 1 :: [Nat]
-- = '[2, 3]
-- >>> :kind! Transpose '[3,2,1] 1 2
-- Transpose '[3,2,1] 1 2 :: [Nat]
-- = '[3, 1, 2]
type family TransposeF (selectDim0 :: SelectDim (By Symbol Nat)) (selectDim1 :: SelectDim (By Symbol Nat)) (shape :: Shape [Dim (DimType Symbol Nat)]) :: Shape [Dim (DimType Symbol Nat)] where
  TransposeF _ _ 'UncheckedShape = 'UncheckedShape
  TransposeF _ 'UncheckedSelectDim _ = 'UncheckedShape
  TransposeF 'UncheckedSelectDim _ _ = 'UncheckedShape
  TransposeF ( 'SelectDim by0) ( 'SelectDim by1) ( 'Shape dims) =
    'Shape
      ( FromMaybe
          (TypeError (TransposeBy1Message by1 dims))
          ( ReplaceDimDimsImplF
              by1
              ( FromMaybe
                  (TypeError (TransposeBy0Message by0 dims))
                  ( ReplaceDimDimsImplF
                      by0
                      dims
                      ( FromMaybe
                          (TypeError (TransposeBy1Message by1 dims))
                          (GetDimDimsImplF by1 dims)
                      )
                  )
              )
              ( FromMaybe
                  (TypeError (TransposeBy0Message by0 dims))
                  (GetDimDimsImplF by0 dims)
              )
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
--           '[ 'Dim ('NamedSized "batch" 10), 'Dim ('NamedSized "feature" 5)])
-- >>> output = transpose @'UncheckedSelectDim @('SelectDim ('ByIndex 1)) (SelectDim (ByIndex 0)) input
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
              <> "The two dimensions must be selected either both by name or both by index, "
              <> "but mixed selectors where found: '"
              <> show selectDim0
              <> "' and '"
              <> show selectDim1
              <> "'."