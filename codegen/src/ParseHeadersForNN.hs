{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ParseHeadersForNN where

import Data.Void (Void)
import GHC.Generics
import Text.Megaparsec as M
import Text.Megaparsec.Char as M
import Text.Megaparsec.Char.Lexer as L

{-
Spec:
  spec/THNN.h
  spec/THCUNN.h

Original parser:
  deps/pytorch/aten/src/ATen/common_with_cwrap.py
-}

type Parser = Parsec Void String

data TenType
  = Scalar
  | Tensor
  | IndexTensor
  deriving (Eq, Show)

data CType
  = CBool
  | CVoid
  | CDouble
  | CInt
  | CInt64
  deriving (Eq, Show, Generic, Bounded, Enum)

data Parsable
  = Ptr Parsable
  | StateType
  | GeneratorType
  | TenType TenType
  | CType CType
  deriving (Eq, Show, Generic)

data Parameter  = Parameter
  { ptype :: Parsable
  , pname :: String
  } deriving (Eq, Show)

data Function  = Function
  { name :: String
  , parameters :: [Parameter]
  } deriving (Eq, Show)


sc :: Parser ()
sc = L.space space1 empty empty

lexm :: Parser a -> Parser a
lexm = L.lexeme sc


comment :: Parser ()
comment =
  try (newline >> return ()) <|>
  try (string "#" >> manyTill anySingle newline >> return ()) <|>
  try (string "//" >> manyTill anySingle newline >> return ()) <|>
  (string "/*" >> manyTill anySingle (string "*/") >> return ())

-- | parser of a function
-- >>> parseTest (comment >> string "1") "#define FOO 1\n1"
-- "1"
-- >>> parseTest (comments >> string "1") "#define FOO 1\n1"
-- "1"
-- >>> parseTest (comments >> string "1") "#define FOO 1\n#define BAR 1\n1"
-- "1"
-- >>> parseTest (comments >> string "1") "#define FOO 1\n\n#define BAR 1\n\n1"
-- "1"
-- >>> parseTest (comments >> string "1") "#define FOO 1\n//aaaa\n\n#define BAR 1\n\n1"
-- "1"
comments :: Parser ()
comments = try (comment >> comments) <|> comment

-- | parser of type-identifier
-- >>> parseTest typ "THCIndexTensor"
-- TenType IndexTensor
-- >>> parseTest typ "THCState"
-- StateType
-- >>> parseTest typ "THCState*"
-- Ptr StateType
-- >>> parseTest typ "THCTensor"
-- TenType Tensor
-- >>> parseTest typ "THCTensor*"
-- Ptr (TenType Tensor)
-- >>> parseTest typ "THGenerator"
-- GeneratorType
-- >>> parseTest typ "THIndexTensor"
-- TenType IndexTensor
-- >>> parseTest typ "THNNState"
-- StateType
-- >>> parseTest typ "THTensor"
-- TenType Tensor
-- >>> parseTest typ "THTensor*"
-- Ptr (TenType Tensor)
-- >>> parseTest typ "accreal"
-- TenType Scalar
-- >>> parseTest typ "bool"
-- CType CBool
-- >>> parseTest typ "double"
-- CType CDouble
-- >>> parseTest typ "int"
-- CType CInt
-- >>> parseTest typ "int64_t"
-- CType CInt64
-- >>> parseTest typ "void"
-- CType CVoid
typ :: Parser Parsable
typ = try ptr <|> noPtr
  where
    noPtr =
      generator <|>
      state <|>
      tensor <|>
      ctype
    ptr = do
      t <- noPtr
      _ <- (lexm $ string "*")
      pure $ Ptr t
    generator =
      ((lexm $ string "THGenerator") >> (pure $ GeneratorType))
    state =
      ((lexm $ string "THCState") >> (pure $ StateType)) <|>
      ((lexm $ string "THNNState") >> (pure $ StateType))
    tensor =
      ((lexm $ string "accreal") >> (pure $ TenType Scalar)) <|>
      ((lexm $ string "THTensor") >> (pure $ TenType Tensor)) <|>
      ((lexm $ string "THCTensor") >> (pure $ TenType Tensor)) <|>
      ((lexm $ string "THIndexTensor") >> (pure $ TenType IndexTensor)) <|>
      ((lexm $ string "THCIndexTensor") >> (pure $ TenType IndexTensor))
    ctype =
      ((lexm $ string "bool") >> (pure $ CType CBool)) <|>
      ((lexm $ string "void") >> (pure $ CType CVoid)) <|>
      ((lexm $ string "double") >> (pure $ CType CDouble)) <|>
      try ((lexm $ string "int64_t") >> (pure $ CType CInt64)) <|>
      ((lexm $ string "int") >> (pure $ CType CInt))

identStart :: [Char]
identStart = ['a'..'z'] ++ ['A'..'Z'] ++ ['_']

identLetter :: [Char]
identLetter = ['a'..'z'] ++ ['A'..'Z'] ++ ['_'] ++ ['0'..'9'] ++ [':', '<', '>']

rws :: [String]
rws = []

identifier :: Parser String
identifier = (lexm . try) (p >>= check)
 where
  p = (:) <$> (oneOf identStart) <*> many (oneOf identLetter)
  check x = if x `elem` rws
    then fail $ "keyword " ++ show x ++ " cannot be an identifier"
    else return x

arg :: Parser Parameter
arg = do
  _ <- optional comment
  v <- param
  _ <- optional comment
  pure v
 where
  param = do
    pt <- typ
    pn <- lexm $ identifier
    pure $ Parameter pt pn

-- | parser of a function
-- >>> parseTest func "TH_API void THNN_(GatedLinear_updateOutput)(THNNState *state,THTensor *input,THTensor *output,int dim);"
-- Function {name = "GatedLinear_updateOutput", parameters = [Parameter {ptype = Ptr StateType, pname = "state"},Parameter {ptype = Ptr (TenType Tensor), pname = "input"},Parameter {ptype = Ptr (TenType Tensor), pname = "output"},Parameter {ptype = CType CInt, pname = "dim"}]}
-- >>> parseTest func "TH_API void THNN_(GatedLinear_updateOutput)(THNNState *state,   // library's state\nTHTensor *input,THTensor *output,int dim);"
-- Function {name = "GatedLinear_updateOutput", parameters = [Parameter {ptype = Ptr StateType, pname = "state"},Parameter {ptype = Ptr (TenType Tensor), pname = "input"},Parameter {ptype = Ptr (TenType Tensor), pname = "output"},Parameter {ptype = CType CInt, pname = "dim"}]}
func :: Parser Function
func = do
  _ <- lexm $ string "TH_API void THNN_("
  v <- identifier
  _ <- lexm $ string ")("
  args <- (sepBy arg (lexm (string ",")))
  _ <- string ");"
  pure $ Function v args

-- | parser of a function
-- >>> parseTest functions "#define FOO 1\n\n#define BAR 1\n\nTH_API void THNN_(GatedLinear_updateOutput)(THNNState *state,THTensor *input,THTensor *output,int dim);\n\n"
-- [Function {name = "GatedLinear_updateOutput", parameters = [Parameter {ptype = Ptr StateType, pname = "state"},Parameter {ptype = Ptr (TenType Tensor), pname = "input"},Parameter {ptype = Ptr (TenType Tensor), pname = "output"},Parameter {ptype = CType CInt, pname = "dim"}]}]
functions :: Parser [Function]
functions = ((lexm eof) >> pure []) <|> funcs
  where
    funcs = do
      optional comments
      x <- func
      optional comments
      xs <- functions
      return (x:xs)
