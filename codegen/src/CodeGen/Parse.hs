{-# LANGUAGE TupleSections #-}
{-# LANGUAGE ScopedTypeVariables #-}
module CodeGen.Parse where
  -- ( Parser
  -- , parser
  -- , functionConcrete
  -- , Parsable(..)
  -- , Arg(..)
  -- , Function(..)
  -- ) where

import CodeGen.Prelude
import CodeGen.Types
import Control.Arrow (second)

import qualified Data.Text as T
import qualified CodeGen.Render.C as C

-- ----------------------------------------
-- File parser for TH templated header files
-- ----------------------------------------

ptr :: Parser ()
ptr = void (space >> char '*')

ptr2 :: Parser ()
ptr2 = ptr >> ptr

{-
nntypes :: Parser Parsable
nntypes = forLibraries go
 where
  go :: LibType -> Parser Parsable
  go = genericParsers NNType . C.renderNNType
-}

tentypes :: Parser Parsable
tentypes = forLibraries go
 where
  go :: LibType -> Parser Parsable
  go lt = genericParsers allTenTypes TenType C.renderTenType


ctypes :: Parser Parsable
ctypes = genericParsers [minBound..maxBound :: CType] CType C.renderCType


-- | build a parser that will try to find the double-pointer- or pointer- variant first.
genericParsers
  :: forall x . [x]
  -> (x -> Parsable)
  -> (x -> Text)
  -> Parser Parsable
genericParsers xs cons render = asum $ map goAll xs
 where
  goAll :: x -> Parser Parsable
  goAll x
    -- search for any double pointer first
    =   try (Ptr . Ptr <$> (go1 x <* ptr2))

    -- then any pointer
    <|> try (Ptr       <$> (go1 x <* ptr))

    -- finally, all of our concrete types and wrap them in the Parsable format
    <|> try (go1 x)

  go1 :: x -> Parser Parsable
  go1 = fmap cons . typeParser render


typeParser :: (x -> Text) -> x -> Parser x
typeParser render t = string (T.unpack $ render t) >> pure t


-- | parse a library-dependent parser across all of our supported libraries
forLibraries :: (LibType -> Parser x) -> Parser x
forLibraries go = asum (map go supportedLibraries)

-------------------------------------------------------------------------------

parsabletypes :: Parser Parsable
parsabletypes
  = do
  typeModifier
  try tentypes {- <|> try nntypes -} <|> ctypes
 where
  typeModifier :: Parser ()
  typeModifier =
        void (try (string "const "))
    <|> void (try (string "unsigned "))
    <|> void (try (string "struct "))
    <|> space


-------------------------------------------------------------------------------

-- Landmarks
api :: Parser ()
api = forLibraries go
 where
  go :: LibType -> Parser ()
  go lt = void $ try (string (show lt <> "_API"))

semicolon :: Parser ()
semicolon = void (char ';')

functionArg :: Parser Arg
functionArg = do
  space
  optional $ try (string "volatile" <|> string "const") <* space1
  argType <- parsabletypes <* space
  -- e.g. declaration sometimes has no variable name - eg Storage.h
  argName <- optional $ some (alphaNumChar <|> char '_') <* space
  try (void (char ',' >> space >> eol)) <|> void (char ',') <|> lookAhead (void $ char ')')
  pure $ Arg argType (maybe "" T.pack argName)

functionArgs :: Parser [Arg]
functionArgs = char '(' *> some functionArg <* char ')'

genericPrefixes :: Parser (LibType, Text)
genericPrefixes = second T.pack <$> asum (foldMap go supportedLibraries)
 where
  prefix :: LibType -> String -> Parser (LibType, String)
  prefix lt x = try (((,) <$> typeParser tshow lt <*> string x) <* string "_(")

  go :: LibType -> [Parser (LibType, String)]
  go lt = map (prefix lt) ["Tensor", "Blas", "Lapack", "Storage", "Vector", ""]


function :: Parser (Maybe Function)
function = do
  optional (api >> space)
  funReturn' <- parsabletypes <* space
  (funPrefix', funName') <- choice [ try genericName, (Nothing,) <$> concreteName ]
  funArgs' <- functionArgs <* space <* semicolon
  optional (try comment)
  pure . pure $ Function funPrefix' funName' funArgs' funReturn'
 where
  genericName :: Parser (Maybe (LibType, Text), Text)
  genericName = do
    pref <- genericPrefixes
    name <- concreteName <* string ")" <* space
    pure (Just pref, name)

  concreteName :: Parser Text
  concreteName = T.pack <$> (some (alphaNumChar <|> char '_') <|> string "") <* space

inlineComment :: Parser ()
inlineComment = do
  space
  string "//"
  some (alphaNumChar <|> char '_' <|> char ' ')
  void $ eol <|> (some (notChar '\n') >> eol)

-- | skip over a _single-line_ of block comment -- something which seems standard in the libTH.
comment :: Parser ()
comment = space >> void (string "/*" *> some (alphaNumChar <|> char '_' <|> char ' ') <* string "*/")

-- | run a parser to find all possible functions, returning one maybe type per-line.
parser :: Parser [Maybe Function]
parser = some (try constant <|> try function <|> skip)

-- | returns a Maybe Function because we actually don't care about constants when generating FFI code.
constant :: Parser (Maybe Function)
constant = do
  -- THLogAdd has constants, these are not surfaced
  string "const" >> space
  parsabletypes >> space
  some (alphaNumChar <|> char '_') >> semicolon
  pure Nothing

-- | Skip a line because we have failed to find a function
skip :: Parser (Maybe Function)
skip = do
  (not <$> atEnd) >>= guard
  void $ many (notChar '\n') <* (void eol <|> eof)
  pure Nothing


