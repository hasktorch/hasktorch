{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}

module Torch.Index
  ( slice,
    lslice,
  )
where

import Control.Monad ((>=>))
import Data.Void
import Language.Haskell.TH.Lib
import Language.Haskell.TH.Quote (QuasiQuoter (..))
import Language.Haskell.TH.Syntax hiding (Unsafe)
import Text.Megaparsec as M
import Text.Megaparsec.Char hiding (space)
import Text.Megaparsec.Char.Lexer
import Torch.Tensor

type Parser = Parsec Void String

sc :: Parser ()
sc = space space1 empty empty

lexm :: Parser a -> Parser a
lexm = lexeme sc

parseSlice :: String -> Q [Exp]
parseSlice str =
  case M.runParser parse' "slice" str of
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
      )
        <|> ( do
                _ <- lexm $ string ("Ellipsis" :: Tokens String)
                pure $ ConE 'Ellipsis
            )
        <|> ( do
                _ <- lexm $ string ("..." :: Tokens String)
                pure $ ConE 'Ellipsis
            )
    bool :: Parser Exp
    bool =
      ( do
          _ <- lexm $ string "True"
          pure $ ConE 'True
      )
        <|> ( do
                _ <- lexm $ string "False"
                pure $ ConE 'False
            )
    number :: Parser Exp
    number =
      ( do
          v <- lexm decimal
          pure $ LitE (IntegerL v)
      )
        <|> ( do
                _ <- lexm $ string "-"
                v <- lexm decimal
                pure $ LitE (IntegerL (- v))
            )
        <|> ( do
                v <- lexm $ between (char '{') (char '}') (some alphaNumChar)
                pure $ VarE (mkName v)
            )
    slice =
      try
        ( do
            a <- number
            lexm $ string ":"
            b <- number
            lexm $ string ":"
            c <- number
            pure $ AppE (ConE 'Slice) (TupE [Just a, Just b, Just c])
        )
        <|> try
          ( do
              lexm $ string ":"
              b <- number
              lexm $ string ":"
              c <- number
              pure $ AppE (ConE 'Slice) (TupE [Just (ConE 'None), Just b, Just c])
          )
        <|> try
          ( do
              a <- number
              lexm $ string "::"
              c <- number
              pure $ AppE (ConE 'Slice) (TupE [Just a, Just (ConE 'None), Just c])
          )
        <|> try
          ( do
              a <- number
              lexm $ string ":"
              b <- number
              pure $ AppE (ConE 'Slice) (TupE [Just a, Just b])
          )
        <|> try
          ( do
              lexm $ string "::"
              c <- number
              pure $ AppE (ConE 'Slice) (TupE [Just (ConE 'None), Just (ConE 'None), Just c])
          )
        <|> try
          ( do
              lexm $ string ":"
              b <- number
              lexm $ string ":"
              pure $ AppE (ConE 'Slice) (TupE [Just (ConE 'None), Just b])
          )
        <|> try
          ( do
              lexm $ string ":"
              b <- number
              pure $ AppE (ConE 'Slice) (TupE [Just (ConE 'None), Just b])
          )
        <|> try
          ( do
              a <- number
              lexm $ string "::"
              pure $ AppE (ConE 'Slice) (TupE [Just a, Just (ConE 'None)])
          )
        <|> try
          ( do
              a <- number
              lexm $ string ":"
              pure $ AppE (ConE 'Slice) (TupE [Just a, Just (ConE 'None)])
          )
        <|> try
          ( do
              _ <- lexm $ string "::"
              pure $ AppE (ConE 'Slice) (ConE '())
          )
        <|> ( do
                _ <- lexm $ string ":"
                pure $ AppE (ConE 'Slice) (ConE '())
            )

-- | Generate a slice from a [python compatible expression](https://pytorch.org/cppdocs/notes/tensor_indexing.html).
-- When you take the odd-numbered element of tensor with `tensor[1::2]` in python,
-- you can write `tensor ! [slice|1::2|]` in hasktorch.
slice :: QuasiQuoter
slice =
  QuasiQuoter
    { quoteExp = parseSlice >=> qconcat,
      quotePat = error "quotePat is not implemented for slice.",
      quoteDec = error "quoteDec is not implemented for slice.",
      quoteType = error "quoteType is not implemented for slice."
    }
  where
    qconcat :: [Exp] -> Q Exp
    qconcat [exp] = pure exp
    qconcat exps = pure $ TupE $ map Just exps

-- | Generate a lens from a [python compatible expression](https://pytorch.org/cppdocs/notes/tensor_indexing.html).
-- When you take the odd-numbered elements of tensor with `tensor[1::2]` in python,
-- you can write `tensor ^. [lslice|1::2|]` in hasktorch.
-- When you put 2 in the odd numbered elements of the tensor,
-- you can write `tensor & [lslice|1::2|] ~. 2`.
lslice :: QuasiQuoter
lslice =
  QuasiQuoter
    { quoteExp = parseSlice >=> qconcat,
      quotePat = error "quotePat is not implemented for slice.",
      quoteDec = error "quoteDec is not implemented for slice.",
      quoteType = error "quoteType is not implemented for slice."
    }
  where
    qconcat :: [Exp] -> Q Exp
    qconcat [exp] = pure $ AppE (VarE 'toLens) exp
    qconcat exps = pure $ AppE (VarE 'toLens) $ TupE $ map Just exps
