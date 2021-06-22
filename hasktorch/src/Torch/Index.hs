{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Index
  ( pySlice
  ) where

import Torch.Tensor
import Language.Haskell.TH.Quote (QuasiQuoter (..))
import Language.Haskell.TH.Syntax hiding (Unsafe)
import Language.Haskell.TH.Lib
import Text.Megaparsec as M
import Text.Megaparsec.Char.Lexer 
import Text.Megaparsec.Char hiding (space)
import Data.Void
import Control.Monad ((>=>))

type Parser = Parsec Void String

sc :: Parser ()
sc = space space1 empty empty

lexm :: Parser a -> Parser a
lexm = lexeme sc

parseSlice :: String -> Q [Exp]
parseSlice str =
  case M.runParser parse' "pySlice" str of
    Left e -> fail $ show e
    Right v -> return v
    
  where
    parse' :: Parser [Exp]
    parse' = (sc >> (try slice <|> try bool <|> try other <|> number)) `sepBy` char ','
    other :: Parser Exp
    other =
      ( do
          _ <- lexm $ string ("None" :: Tokens String)
          pure $ ConE 'None
      ) <|>
      ( do
          _ <- lexm $ string ("Ellipsis" :: Tokens String)
          pure $ ConE 'Ellipsis
      ) <|>
      ( do
          _ <- lexm $ string ("..." :: Tokens String)
          pure $ ConE 'Ellipsis
      )
    bool :: Parser Exp
    bool =
      ( do
          _ <- lexm $ string "True"
          pure $ ConE 'True
      ) <|>
      ( do
          _ <- lexm $ string "False"
          pure $ ConE 'False
      )
    number :: Parser Exp 
    number = lexm decimal >>= \v -> pure $ LitE (IntegerL v)
    slice =
      try ( do
          a <- number
          lexm $ string ":"
          b <- number
          lexm $ string ":"
          c <- number
          pure $ AppE (ConE 'Slice) (TupE [Just a,Just b,Just c])
        ) <|>
      try ( do
          lexm $ string ":"
          b <- number
          lexm $ string ":"
          c <- number
          pure $ AppE (ConE 'Slice) (TupE [Just (ConE 'None),Just b,Just c])
        ) <|>
      try ( do
          a <- number
          lexm $ string "::"
          c <- number
          pure $ AppE (ConE 'Slice) (TupE [Just a,Just (ConE 'None),Just c])
        ) <|>
      try ( do
          a <- number
          lexm $ string ":"
          b <- number
          pure $ AppE (ConE 'Slice) (TupE [Just a,Just b])
        ) <|>
      try ( do
          lexm $ string "::"
          c <- number
          pure $ AppE (ConE 'Slice) (TupE [Just (ConE 'None),Just (ConE 'None),Just c])
        ) <|>
      try ( do
          lexm $ string ":"
          b <- number
          lexm $ string ":"
          pure $ AppE (ConE 'Slice) (TupE [Just (ConE 'None),Just b])
        ) <|>
      try ( do
          lexm $ string ":"
          b <- number
          pure $ AppE (ConE 'Slice) (TupE [Just (ConE 'None),Just b])
        ) <|>
      try ( do
          a <- number
          lexm $ string "::"
          pure $ AppE (ConE 'Slice) (TupE [Just a,Just (ConE 'None)])
        ) <|>
      try ( do
          a <- number
          lexm $ string ":"
          pure $ AppE (ConE 'Slice) (TupE [Just a,Just (ConE 'None)])
        ) <|>
      try ( do
          _ <- lexm $ string "::"
          pure $ AppE (ConE 'Slice) (ConE '())
        ) <|>
          ( do
          _ <- lexm $ string ":"
          pure $ AppE (ConE 'Slice) (ConE '())
          )

pySlice :: QuasiQuoter
pySlice = QuasiQuoter
  { quoteExp = parseSlice >=> qconcat
  , quotePat = error "quotePat is not implemented for pySlice."
  , quoteDec = error "quoteDec is not implemented for pySlice."
  , quoteType = error "quoteType is not implemented for pySlice."
  }
  where
    qconcat :: [Exp] -> Q Exp
    qconcat [exp] = pure exp
    qconcat exps = pure $ TupE $ map Just exps
