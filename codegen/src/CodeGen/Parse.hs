module CodeGen.Parse
  ( Parser
  , parser
  , Parsable(..)
  , Arg(..)
  , Function(..)
  ) where

import CodeGen.Prelude
import CodeGen.Types

import qualified Data.Text as T
import qualified CodeGen.Render.C as C

-- ----------------------------------------
-- File parser for TH templated header files
-- ----------------------------------------

ptr :: Parser ()
ptr = void (space >> char '*')

ptr2 :: Parser ()
ptr2 = ptr >> ptr

-------------------------------------------------------------------------------

ctypePtrPtr :: Parser Parsable
ctypePtrPtr = fmap (Ptr . Ptr . CType) ctypes <* ptr2

tentypePtrPtr, nntypePtrPtr :: LibType -> Parser Parsable
tentypePtrPtr lt = fmap (Ptr . Ptr . TenType) (tentypes lt) <* ptr2
nntypePtrPtr  lt = fmap (Ptr . Ptr . NNType)  (nntypes lt)  <* ptr2

ctypePtr :: Parser Parsable
ctypePtr = fmap (Ptr . CType) ctypes <* ptr

tentypePtr, nntypePtr :: LibType -> Parser Parsable
tentypePtr lt = fmap (Ptr . Ptr . TenType) (tentypes lt) <* ptr
nntypePtr  lt = fmap (Ptr . Ptr . NNType)  (nntypes lt)  <* ptr

ctypes :: Parser CType
ctypes
  = asum
  . flip map [minBound..maxBound::CType] $
  (\ctype -> string (T.unpack $ C.renderCType ctype) >> pure ctype)

tentypes :: LibType -> Parser TenType
tentypes lt
  = asum
  . flip map [minBound..maxBound::TenType] $
  (\ctype -> string (T.unpack $ C.renderTenType lt ctype) >> pure ctype)

nntypes :: LibType -> Parser NNType
nntypes lt
  = asum
  . flip map [minBound..maxBound::NNType] $
  (\ctype -> string (T.unpack $ C.renderNNType lt ctype) >> pure ctype)

-------------------------------------------------------------------------------

parsabletypes :: Parser Parsable
parsabletypes
  = typeModifier
  >> asum (map (\lt ->
    -- search for any double pointer first
        nntypePtrPtr lt
    <|> tentypePtrPtr lt
    <|> ctypePtrPtr

    -- then any pointer
    <|> nntypePtr lt
    <|> tentypePtr lt
    <|> ctypePtr

    -- finally, all of our concrete types and wrap them in the Parsable format
    <|> fmap NNType  (nntypes lt)
    <|> fmap TenType (tentypes lt)
    <|> fmap CType ctypes
  ) supportedLibraries )

 where
  typeModifier :: Parser ()
  typeModifier
    =   void (string "const ")
    <|> void (string "unsigned ")
    <|> void (string "struct ")
    <|> space


-------------------------------------------------------------------------------

-- Landmarks

api :: LibType -> Parser ()
api lt = string (show lt <> "_API") >> space

semicolon :: Parser ()
semicolon = void (char ';')

-- Function signatures

-- functionArgVoid :: Parser Arg
-- functionArgVoid = do
--   string "void"
--   space
--   lookAhead (char ')')
--   pure (Arg (CType CVoid) "")

functionArg :: Parser Arg
functionArg = do
  argType <- parsabletypes
  space
  try (string "volatile" >> space)
  -- e.g. declaration sometimes has no variable name - eg Storage.h
  argName <- some (alphaNumChar <|> char '_') <|> string ""
  space >> try (char ',') >> space
  pure $ Arg argType (T.pack argName)

-- functionArg :: Parser Arg
-- functionArg = thFunctionArgNamed -- <|> thFunctionArgVoid

functionArgs :: Parser [Arg]
functionArgs = char '(' *> some functionArg <* char ')'

genericPrefixes :: Parser ()
genericPrefixes = void $ asum (foldMap go supportedLibraries)
 where
  prefix :: LibType -> String -> Parser String
  prefix lt x = string (show lt <> x <> "_(")

  go :: LibType -> [Parser String]
  go lt = map (prefix lt) ["Tensor", "Blas", "Lapack", "Storage", "Vector", ""]

functionTemplate :: LibType -> Parser (Maybe Function)
functionTemplate lt = do
  api lt >> space
  funReturn' <- parsabletypes <* space

  genericPrefixes
  funName' <- some (alphaNumChar <|> char '_') <* space
  string ")" >> space

  funArgs' <- functionArgs <* semicolon
  void $ optional (try comment)
  pure . pure $ Function (T.pack funName') funArgs' funReturn'

{-
inlineComment :: Parser ()
inlineComment = do
  some space
  string "//"
  some (alphaNumChar <|> char '_' <|> char ' ')
  eol <|> (some (notChar '\n') >> eol)
  pure ()
-}


comment :: Parser ()
comment = space >>
  void (string "/*" *> some (alphaNumChar <|> char '_' <|> char ' ') <* string "*/")

functionConcrete :: Parser (Maybe Function)
functionConcrete = do
  funReturn' <- parsabletypes <* space
  funName'   <- some (alphaNumChar <|> char '_') <* space
  funArgs'   <- functionArgs <* semicolon
  void (optional (try comment))
  pure . Just $ Function (T.pack funName') funArgs' funReturn'


-- TODO - exclude TH_API prefix. Parse should crash if TH_API parse is invalid
skip :: Parser (Maybe Function)
skip =
  (eol <|> (some (notChar '\n') >> eol))
  >> pure Nothing

constant :: Parser (Maybe Function)
constant = do
  -- THLogAdd has constants, these are not surfaced
  string "const" >> space
  parsabletypes  >> space
  some (alphaNumChar <|> char '_') >> semicolon
  pure Nothing

-- thItem :: Parser (Maybe Function)
-- thItem = try thConstant <|> thFunctionTemplate <|> thSkip

-- NOTE: ordering is important for parsers
parser :: LibType -> CodeGenType -> Parser [Maybe Function]
parser lt = \case
  GenericFiles -> go (functionTemplate lt)
  ConcreteFiles -> go functionConcrete
 where
  go :: Parser (Maybe Function) -> Parser [Maybe Function]
  go funpar = some
    $   try (api lt >> pure Nothing)
    <|> try constant
    <|> funpar
    <|> skip
