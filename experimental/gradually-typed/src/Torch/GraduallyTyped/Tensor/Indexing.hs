{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveLift #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns #-}

{-# LANGUAGE PartialTypeSignatures #-}
module Torch.GraduallyTyped.Tensor.Indexing
  ( IndexType (..),
    SIndexType (..),
    Indices (..),
    SIndices (..),
    IndexShape,
    (!),
    slice,
    parseSlice,
    setAt,
    -- toLens,
  )
where

import Control.Arrow ((>>>))
import Control.Lens (Lens, Lens', Traversal)
import Control.Monad (forM_, void, (<=<), join)
import Control.Monad.Catch (MonadThrow)
import Control.Monad.Trans (lift)
import Data.Coerce (coerce)
import Data.Foldable (asum)
import Data.Kind (Constraint, Type)
import Data.Singletons (Demote, SingI, SingKind, SomeSing (..), fromSing, sing, toSing, withSomeSing)
import Data.Singletons.Prelude (Reverse, SBool (..), SList (..), Sing)
import Data.Singletons.TH (genSingletons)
import Data.Type.Bool (Not, type (&&))
import Data.Type.Equality (type (==))
import Data.Void (Void)
import Foreign (ForeignPtr, fromBool)
import GHC.TypeLits (Div, ErrorMessage (..), Nat, Symbol, type (+), type (-), type (<=?))
import Language.Haskell.TH.Quote (QuasiQuoter (..))
import qualified Language.Haskell.TH.Syntax as TH
import Text.Megaparsec (ParsecT, between, empty, eof, errorBundlePretty, optional, runParserT, sepBy, some, try, (<|>))
import qualified Text.Megaparsec.Char as M
import qualified Text.Megaparsec.Char.Lexer as L
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Index.Type (DemotedIndex (..), Index (..), SIndex (..))
import Torch.GraduallyTyped.Prelude (Catch, If, IsChecked (..), Seq, When, forgetIsChecked, type (<?))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, PrependDimF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), TensorLike (sToTensor), fromTensor, toTensor)
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Type.TensorIndex as ATen
import qualified Torch.Internal.Type as ATen
import Type.Errors.Pretty (TypeError, type (%), type (<>))

data IndexType a
  = NewAxis
  | Ellipsis
  | SliceAll
  | SliceAt a
  | SliceBool Bool
  | SliceFrom a
  | SliceUpTo a
  | SliceWithStep a
  | SliceFromUpTo a a
  | SliceFromWithStep a a
  | SliceUpToWithStep a a
  | SliceFromUpToWithStep a a a
  deriving stock (Show, Eq, Functor, TH.Lift)

genSingletons [''IndexType]

deriving stock instance Show (SIndexType (indexType :: IndexType (Index Nat)))

type ReverseShape :: Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)]
type family ReverseShape shape where
  ReverseShape 'UncheckedShape = 'UncheckedShape
  ReverseShape ('Shape dims) = 'Shape (Reverse dims)

type ErrorOnEllipsis :: [IndexType (Index Nat)] -> [IndexType (Index Nat)]
type family ErrorOnEllipsis indices where
  ErrorOnEllipsis '[] = '[]
  ErrorOnEllipsis ('Ellipsis ': ixs) = TypeError ('Text "Indices can only contain a single ellipsis ('...').")
  ErrorOnEllipsis (ix ': ixs) = ix ': ErrorOnEllipsis ixs

type StepZeroErrorMessage = 'Text "Slice step cannot be zero"

-- | Calculate the size of the dimension with step.
--
-- >>> :kind! Stepped 8 1
-- Stepped 8 1 :: Nat
-- = 8
-- >>> :kind! Stepped 5 2
-- Stepped 5 2 :: Nat
-- = 3
-- >>> :kind! Stepped 6 3
-- Stepped 6 3 :: Nat
-- = 2
type Stepped :: Nat -> Nat -> Nat
type family Stepped length step where
  Stepped _ 0 = TypeError StepZeroErrorMessage
  Stepped 0 _ = 0
  Stepped length step = (length - 1) `Div` step + 1

type family CheckUpTo (upTo :: Nat) ok where
  CheckUpTo upTo ok =
    If
      (upTo == 0)
      (TypeError ('Text "Slice 'upTo' type parameter must not be equal to zero"))
      ok

type family CheckFromSize (from :: Nat) (size :: Nat) ok where
  CheckFromSize from size ok =
    If
      (from <? size)
      ok
      ( TypeError
          ( "Slice 'from' type parameter must be smaller than the size of the indexed dimension:"
              % "    " <> "from < size"
              % "but"
              % "    " <> from <> " >= " <> size
          )
      )

type family CheckUpToSize (upTo :: Nat) (size :: Size Nat) ok where
  CheckUpToSize upTo 'UncheckedSize ok = CheckUpTo upTo ok
  CheckUpToSize upTo ('Size size) ok =
    CheckUpTo
      upTo
      ( If
          (upTo <=? size)
          ok
          ( TypeError
              ( "Slice 'upTo' type parameter must be less than or equal to the size of the indexed dimension:"
                  % "    " <> "upTo <= size"
                  % "but"
                  % "    " <> upTo <> " > " <> size
              )
          )
      )

type family CheckFromUpTo (from :: Nat) (upTo :: Nat) ok where
  CheckFromUpTo from upTo ok =
    If
      (from <? upTo)
      ok
      ( TypeError
          ( "Slice 'from' type parameter must be less than the 'upTo' type parameter:"
              % "    " <> "from < upTo"
              % "but"
              % "    " <> from <> " >= " <> upTo
          )
      )

type family CheckFromUpToSize (from :: Nat) (upTo :: Nat) (size :: Size Nat) ok where
  CheckFromUpToSize from upTo size ok = CheckFromUpTo from upTo (CheckUpToSize upTo size ok)

type family CheckSliceAt (at :: Nat) (size :: Nat) ok where
  CheckSliceAt at size ok =
    If
      (at <? size)
      ok
      ( TypeError
          ( "Index of 'SliceAt' must be less than the size of the indexed dimension:"
              % "    " <> "at < size"
              % "but"
              % "    " <> at <> " >= " <> size
          )
      )

type family CheckStep (step :: Index Nat) ok where
  CheckStep ('Index 0) _ = TypeError StepZeroErrorMessage
  CheckStep _ ok = ok

type IndexDimsImpl ::
  [IndexType (Index Nat)] ->
  [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)]
type family IndexDimsImpl indices dims where
  IndexDimsImpl '[] dims = 'Shape dims
  IndexDimsImpl ('NewAxis ': ixs) dims = 'Dim ('Name "*") ('Size 1) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('Ellipsis ': _) '[] = 'Shape '[]
  IndexDimsImpl ('Ellipsis ': ixs) dims = ReverseShape (IndexDimsImpl (Reverse (ErrorOnEllipsis ixs)) (Reverse dims))
  IndexDimsImpl ('SliceAll ': ixs) (dim ': dims) = dim `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceAt ('Index at) ': ixs) ('Dim name ('Size size) ': dims) = CheckSliceAt at size (IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceAt _ ': ixs) ('Dim name _ ': dims) = IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceBool 'False ': ixs) ('Dim name _ ': dims) = 'Dim name ('Size 0) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceBool 'True ': ixs) ('Dim name _ ': dims) = 'Dim name ('Size 1) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFrom ('Index from) ': ixs) ('Dim name ('Size size) ': dims) =
    CheckFromSize from size ('Dim name ('Size (size - from)) `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceFrom _ ': ixs) ('Dim name _ ': dims) =
    'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceUpTo ('Index upTo) ': ixs) ('Dim name size ': dims) =
    CheckUpToSize upTo size ('Dim name ('Size upTo) `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceUpTo _ ': ixs) ('Dim name _ ': dims) =
    'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceWithStep ('Index step) ': ixs) ('Dim name ('Size size) ': dims) =
    'Dim name ('Size (Stepped size step)) `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceWithStep step ': ixs) ('Dim name _ ': dims) =
    CheckStep step ('Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceFromUpTo ('Index from) ('Index upTo) ': ixs) ('Dim name size ': dims) =
    CheckFromUpToSize from upTo size ('Dim name ('Size (upTo - from)) `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceFromUpTo _ _ ': ixs) ('Dim name _ ': dims) =
    'Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims
  IndexDimsImpl ('SliceFromWithStep ('Index from) ('Index step) ': ixs) ('Dim name ('Size size) ': dims) =
    CheckFromSize from size ('Dim name ('Size (Stepped (size - from) step)) `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceFromWithStep _ step ': ixs) ('Dim name _ ': dims) =
    CheckStep step ('Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceUpToWithStep ('Index upTo) ('Index step) ': ixs) ('Dim name size ': dims) =
    CheckUpToSize upTo size ('Dim name ('Size (Stepped upTo step)) `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceUpToWithStep _ step ': ixs) ('Dim name _ ': dims) =
    CheckStep step ('Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceFromUpToWithStep ('Index from) ('Index upTo) ('Index step) ': ixs) ('Dim name size ': dims) =
    CheckFromUpToSize from upTo size ('Dim name ('Size (Stepped (upTo - from) step)) `PrependDimF` IndexDimsImpl ixs dims)
  IndexDimsImpl ('SliceFromUpToWithStep _ _ step ': ixs) ('Dim name _ ': dims) =
    CheckStep step ('Dim name 'UncheckedSize `PrependDimF` IndexDimsImpl ixs dims)

type IndexShape ::
  Indices [IndexType (Index Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)]
type family IndexShape indices shape where
  IndexShape 'UncheckedIndices _ = 'UncheckedShape
  IndexShape _ 'UncheckedShape = 'UncheckedShape
  IndexShape ('Indices indices) ('Shape dims) = IndexDimsImpl indices dims

data Indices (indexTypes :: Type) where
  UncheckedIndices :: forall indexTypes. Indices indexTypes
  Indices :: forall indexTypes. indexTypes -> Indices indexTypes
  deriving (Show)

data SIndices (indices :: Indices [IndexType (Index Nat)]) where
  SUncheckedIndices :: [IndexType Integer] -> SIndices 'UncheckedIndices
  SIndices :: forall indexTypes. SList indexTypes -> SIndices ('Indices indexTypes)

deriving stock instance Show (SIndices (indices :: Indices [IndexType (Index Nat)]))

type instance Sing = SIndices

instance SingI indexTypes => SingI ('Indices (indexTypes :: [IndexType (Index Nat)])) where
  sing = SIndices $ sing @indexTypes

instance SingKind (Indices [IndexType (Index Nat)]) where
  type Demote (Indices [IndexType (Index Nat)]) = IsChecked [IndexType (IsChecked Integer)]
  fromSing (SUncheckedIndices indexTypes) = Unchecked $ fmap Unchecked <$> indexTypes
  fromSing (SIndices indexTypes) = Checked . coerce . fromSing $ indexTypes
  toSing (Unchecked indexTypes) = SomeSing . SUncheckedIndices $ fmap forgetIsChecked <$> indexTypes
  toSing (Checked indexTypes) = withSomeSing ((fmap . fmap . fmap) DemotedIndex indexTypes) $ SomeSing . SIndices

toTensorIndexList :: [IndexType Integer] -> IO (ForeignPtr (ATen.StdVector ATen.TensorIndex))
toTensorIndexList indices = do
  indexList <- ATen.newTensorIndexList
  tensorIndices <- traverse toTensorIndex indices
  forM_ tensorIndices $ ATen.tensorIndexList_push_back indexList
  pure indexList
  where
    toTensorIndex =
      fmap fromIntegral >>> \case
        NewAxis -> ATen.newTensorIndexWithNone
        Ellipsis -> ATen.newTensorIndexWithEllipsis
        SliceAt at -> ATen.newTensorIndexWithInt at
        SliceBool b -> ATen.newTensorIndexWithBool (fromBool b)
        SliceAll -> ATen.newTensorIndexWithSlice 0 maxBound 1
        SliceFrom from -> ATen.newTensorIndexWithSlice from maxBound 1
        SliceUpTo upTo -> ATen.newTensorIndexWithSlice 0 upTo 1
        SliceWithStep step -> ATen.newTensorIndexWithSlice 0 maxBound step
        SliceFromUpTo from upTo -> ATen.newTensorIndexWithSlice from upTo 1
        SliceFromWithStep from step -> ATen.newTensorIndexWithSlice from maxBound step
        SliceUpToWithStep upTo step -> ATen.newTensorIndexWithSlice 0 upTo step
        SliceFromUpToWithStep from upTo step -> ATen.newTensorIndexWithSlice from upTo step

-- | Indexes/slices a tensor.
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> sRandn' = sRandn . TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat)
-- >>> t = sRandn' (SShape $ SName @"*" :&: SSize @3 :|: SName @"*" :&: SSize @5 :|: SNil) g
-- >>> result <- t ! [slice|:2, 3|]
(!) ::
  forall indices gradient layout device dataType shape m.
  MonadThrow m =>
  Tensor gradient layout device dataType shape ->
  SIndices indices ->
  m (Tensor gradient layout device dataType (IndexShape indices shape))
(UnsafeTensor t) ! sIndices = unsafeThrowableIO $ do
  indexList <- toTensorIndexList indices
  UnsafeTensor <$> ATen.index t indexList
  where
    indices = fmap forgetIsChecked <$> forgetIsChecked (fromSing sIndices)

setAt ::
  forall gradient layout device dataType shape shape' indices m.
  ( -- TensorLike a dType dims,
    -- Catch ('DataType dType <+> dataType),
    -- Catch ('Shape dims <+> shape'),
    shape' ~ IndexShape indices shape,
    Catch (shape' <+> BroadcastShapesF shape shape'),
    SingI gradient,
    SingI layout,
    SingI device,
    MonadThrow m
  ) =>
  Tensor gradient layout device dataType shape ->
  SIndices indices ->
  Tensor gradient layout device dataType shape' ->
  m (Tensor gradient layout device dataType shape)
setAt (UnsafeTensor t') sIndices (UnsafeTensor x) = unsafeThrowableIO $ do
  t <- ATen.clone_t t'
  indexList <- toTensorIndexList indices
  -- UnsafeTensor x <- toTensor @gradient @layout @device x'
  UnsafeTensor <$> ATen.index_put_ t indexList x
  where
    indices = fmap forgetIsChecked <$> forgetIsChecked (fromSing sIndices)

-- instance Ixed (Tensor gradient layout device dataType shape)

toLens ::
  forall s a a' gradient layout device dataType shape indices dims dType m.
  ( TensorLike a' dType dims,
    'DataType dType ~ dataType,
    'Shape dims ~ IndexShape indices shape,
    s ~ Tensor gradient layout device dataType shape,
    a ~ Tensor gradient layout device dataType (IndexShape indices shape),
    Catch ('Shape dims <+> BroadcastShapesF shape ('Shape dims)),
    SingI gradient,
    SingI layout,
    SingI device,
    MonadThrow m,
    Traversable m
  ) =>
  SIndices indices ->
  Traversal s (m s) a' (m a')
toLens sIndices (f :: a' -> f (m a')) s =
  let ma' :: m a' = fromTensor @a' <$> (s ! sIndices :: m a)
      mfma' :: m (f (m a')) = f <$> ma'
      set' :: a' -> m s
      set' a' = do
        s' <- toTensor a'
        setAt s sIndices s'
      set :: m (f (m a')) -> f (m s)
      set x = 
        let y :: m (f (m s)) = do
              x' <- x
              pure $ (>>= set') <$> x'
            y'' :: f (m (m s)) = sequenceA y
        in join <$> y''
   in set mfma'

type Parser = ParsecT Void String TH.Q

sc :: Parser ()
sc = L.space M.space1 empty empty

lexeme :: Parser a -> Parser a
lexeme = L.lexeme sc

char :: Char -> Parser Char
char = lexeme . M.char

string :: String -> Parser String
string = lexeme . M.string

parseSlice :: String -> TH.Q TH.Exp
parseSlice = either (fail . errorBundlePretty) pure <=< runParserT indicesP ""
  where
    indicesP :: Parser TH.Exp
    indicesP = do
      indexExps <- sc *> (try sliceP <|> try boolP <|> otherP) `sepBy` char ',' <* eof
      let indicesExp = foldr (TH.AppE . TH.AppE (TH.ConE 'SCons)) (TH.ConE 'SNil) indexExps
      lift [|SIndices $(pure indicesExp)|]
    otherP :: Parser TH.Exp
    otherP =
      asum
        [ lift [|SNewAxis|] <* (string "+" <|> string "NewAxis"),
          lift [|SEllipsis|] <* (string "..." <|> string "Ellipsis")
        ]
    boolP :: Parser TH.Exp
    boolP = do
      sBool <-
        asum
          [ [|STrue|] <$ string "True",
            [|SFalse|] <$ string "False"
          ]
      lift [|SSliceBool $sBool|]
    indexP :: Parser TH.Exp
    indexP =
      asum
        [ do
            index <- L.signed sc $ lexeme L.decimal
            let con = if index < 0 then [|SNegativeIndex|] else [|SIndex|]
                nat = pure $ TH.LitT $ TH.NumTyLit $ abs index
            lift [|$con @($nat)|],
          TH.VarE . TH.mkName <$> lexeme (between (char '{') (char '}') (some M.alphaNumChar))
        ]
    sliceP :: Parser TH.Exp
    sliceP =
      asum . map try $
        [ do
            from <- indexP
            void $ char ':'
            upTo <- indexP
            void $ char ':'
            step <- indexP
            lift [|SSliceFromUpToWithStep $(pure from) $(pure upTo) $(pure step)|],
          do
            void $ char ':'
            upTo <- indexP
            void $ char ':'
            step <- indexP
            lift [|SSliceUpToWithStep $(pure upTo) $(pure step)|],
          do
            from <- indexP
            void $ char ':' <* char ':'
            step <- indexP
            lift [|SSliceFromWithStep $(pure from) $(pure step)|],
          do
            from <- indexP
            void $ char ':'
            upTo <- indexP
            lift [|SSliceFromUpTo $(pure from) $(pure upTo)|],
          do
            void $ char ':' <* char ':'
            step <- indexP
            lift [|SSliceWithStep $(pure step)|],
          do
            void $ char ':'
            upTo <- indexP
            void $ optional $ char ':'
            lift [|SSliceUpTo $(pure upTo)|],
          do
            from <- indexP
            void $ char ':' <* optional (char ':')
            lift [|SSliceFrom $(pure from)|],
          do
            at <- indexP
            lift [|SSliceAt $(pure at)|],
          lift [|SSliceAll|] <* char ':' <* optional (char ':')
        ]

-- | Generate a slice from a [python compatible expression](https://pytorch.org/cppdocs/notes/tensor_indexing.html).
-- When you take the odd-numberPed element of tensor with `tensor[1::2]` in python,
-- you can write `tensor ! [slice|1::2|]` in hasktorch.
slice :: QuasiQuoter
slice =
  QuasiQuoter
    { quoteExp = parseSlice,
      quotePat = notHandled,
      quoteType = notHandled,
      quoteDec = notHandled
    }
  where
    notHandled = const $ fail "'slice' quasiquoter can only be used as an expression."
