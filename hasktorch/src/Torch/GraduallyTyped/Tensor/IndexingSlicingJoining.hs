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
import Torch.GraduallyTyped.Prelude (MapMaybe)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), UnifyRequiresGradientF)
import Torch.GraduallyTyped.Shape (AddDimF, By (..), Dim (..), DimType (..), GetDimF, ReplaceDimF, ReplaceDimImplF, SelectDim (..), Shape (..), UnifyShapeF, WithSelectDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorF)
import Torch.HList (HList)
import Torch.Internal.Cast (cast2)
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
