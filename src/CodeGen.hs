module Main where

import Control.Monad (void)
import Data.Void
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Expr
import qualified Text.Megaparsec.Char.Lexer as L
import Text.Show.Pretty

data THType =
  THVoid
  | THDescBuff
  | THTensorPtr
  | THTensorPtrPtr
  | THStoragePtr
  | THLongStoragePtr
  | THPtrDiff
  | THLongPtr
  | THLong
  | THInt
  | THChar
  | THRealPtr
  | THReal deriving Show

data THArg = THArg {
  thArgType :: THType,
  thArgName :: String
  } deriving Show

data THItem = THSkip
            | THFunction {
                funName :: String,
                funArgs :: [THArg],
                funReturn :: THType
                } deriving Show

type Parser = Parsec Void String

{- Type Parsers -}

thPtr :: Parser Char
thPtr = char '*'

thVoid :: Parser THType
thVoid = string "void" >> pure THVoid

thDescBuff :: Parser THType
thDescBuff = string "THDescBuff" >> pure THDescBuff

thTensorPtr :: Parser THType
thTensorPtr = string "THTensor" >> space >> thPtr >> pure THTensorPtr

thTensorPtrPtr :: Parser THType
-- thTensorPtrPtr = string "THTensor" >> space >> (count 2 thPtr) >> pure THTensorPtrPtr
thTensorPtrPtr = string "THTensor **" >> pure THTensorPtrPtr
-- TODO : determine a better way to match both double and single pointers

thStoragePtr :: Parser THType
thStoragePtr = string "THStorage" >> space >> thPtr >> pure THStoragePtr

thLongStoragePtr :: Parser THType
thLongStoragePtr = string "THLongStorage" >> space >> thPtr >> pure THStoragePtr

thPtrDiff :: Parser THType
thPtrDiff = string "ptrdiff_t" >> pure THStoragePtr

thLongPtr :: Parser THType
thLongPtr = string "long *" >> pure THLongPtr
-- TODO : determine a better way to match both pointer/non-pointer

thLong :: Parser THType
thLong = string "long" >> pure THLong

thInt :: Parser THType
thInt = string "int" >> pure THInt

thChar :: Parser THType
thChar = string "char" >> pure THChar

thRealPtr :: Parser THType
thRealPtr = string "real *" >> pure THRealPtr
-- TODO : fix pointer/non-pointer match

thReal :: Parser THType
thReal = string "real" >> pure THReal

thType = do
  ((string "const " >> pure ()) <|> space)
  (thVoid
   <|> thDescBuff
   <|> thTensorPtrPtr
   <|> thTensorPtr
   <|> thStoragePtr
   <|> thLongStoragePtr
   <|> thPtrDiff
   <|> thLongPtr
   <|> thLong
   <|> thInt
   <|> thChar
   <|> thRealPtr -- ptr before raw
   <|> thReal)

{- Landmarks -}

thAPI :: Parser String
thAPI = string "TH_API"

thSemicolon :: Parser Char
thSemicolon = char ';'

thFunctionArgVoid = do
  arg <- thVoid
  space
  char ')' :: Parser Char -- TODO move this outside
  pure $ THArg THVoid ""

thFunctionArgNamed = do
  argType <- thType
  space
  argName <- some (alphaNumChar <|> char '_')
  space
  (char ',' :: Parser Char) <|> (char ')' :: Parser Char)
  space
  pure $ THArg argType argName

thFunctionArg = thFunctionArgVoid <|> thFunctionArgNamed

thFunctionArgs = do
  char '(' :: Parser Char
  functionArgs <- some thFunctionArg
  -- close paren consumed by last thFunctionArg (TODO - clean this up)
  pure functionArgs

thFunctionTemplate = do
  thAPI >> space
  funRet <- thType
  space
  string "THTensor_("
  funName <- some alphaNumChar
  space
  string ")"
  space
  funArgs <- thFunctionArgs
  thSemicolon
  pure $ THFunction funName funArgs funRet

thSkip = do
  eol <|> (some (notChar '\n') >> eol)
  pure $ THSkip

thItem = thFunctionTemplate <|> thSkip -- ordering is important

thFile = some thItem

testString inp = case (parse thFile "" inp) of
  Left err -> putStrLn (parseErrorPretty err)
  Right val -> putStrLn $ (ppShow val)

parseFromFile p file = runParser p file <$> readFile file

cleanList (Left _) = []
cleanList (Right lst) = filter f lst
  where
    f THSkip = False
    f _ = True

testFile file = do
  res <- parseFromFile thFile file
  pure $ cleanList res
  -- case res of
  --   Left err -> putStr (parseErrorPretty err)
  --   Right val -> putStr $ (ppShow val) ++ "\n"

test1 = do
  testString ex1
  where
    ex1 = "skip this line\nTH_API void THTensor_(setFlag)(THTensor *self,const char flag);"

makePrefix templateType = "TH" ++ templateType ++ "Tensor"

renderFunName _ THSkip = ""
renderFunName prefix (THFunction name args ret) =
  prefix ++ "_" ++ name

renderFunSig _ THSkip = ""
renderFunSig prefix (THFunction name args ret) =
  prefix ++ "_" ++ name ++ " :: \n"
  -- TODO signature

renderAll templateType lst =
  foldr (\x y -> x ++ ",\n" ++ y) "" (renderFunName prefix <$> lst)
  where prefix = makePrefix templateType

main = do
  res <- testFile "vendor/torch7/lib/TH/generic/THTensor.h"
  putStrLn "First 5 signatures"
  putStrLn $ ppShow (take 5 res)
  -- putStrLn $ ppShow (take 5 (renderFunName "THIntTensor" <$> res))
  writeFile "./render/test.hs" (renderAll "Int" res)
  putStrLn "Done"
