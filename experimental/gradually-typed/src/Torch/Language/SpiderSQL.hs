{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | This module contains bits and pieces to work with the Spider SQL language.
--
-- TODO:
-- * don't accept reserved keywords in names without quoting
-- * aliases have to be defined in scope, otherwise fall back to table id
-- * don't define an alias twice in the same scope
-- * optionally (?) use the schema to constrain column and table names
-- * test the parser(s) on more examples
-- * pretty printing
-- * random generation of 'SpiderSQL' values
module Torch.Language.SpiderSQL where

import Control.Applicative (Alternative (..), liftA2, optional)
import Control.Monad (MonadPlus, guard)
import Control.Monad.Logic.Class (MonadLogic (..))
import Data.Char (isAlphaNum, isDigit, isSpace, toLower)
import Data.Foldable (Foldable (toList))
import Data.Functor (($>))
import Data.List (nub)
import Data.Maybe (fromMaybe)
import Text.Parser.Char (CharParsing (char, notChar, satisfy, string), spaces)
import Text.Parser.Combinators
  ( Parsing ((<?>)),
    between,
    choice,
    many,
    optional,
    sepBy,
    sepBy1,
    some,
  )
import Text.Parser.Token (TokenParsing (someSpace))
import Text.Read (readMaybe)
import Torch.Data.Parser (combine, doubleP, eitherP, intP, isToken, parseString)

data SpiderSQL = SpiderSQL
  { spiderSQLSelect :: Select,
    spiderSQLFrom :: From,
    spiderSQLWhere :: Maybe Cond,
    spiderSQLGroupBy :: [ColUnit],
    spiderSQLOrderBy :: Maybe OrderBy,
    spiderSQLHaving :: Maybe Cond,
    spiderSQLLimit :: Maybe Int,
    spiderSQLIntersect :: Maybe SpiderSQL,
    spiderSQLExcept :: Maybe SpiderSQL,
    spiderSQLUnion :: Maybe SpiderSQL
  }
  deriving (Eq, Show)

data Select
  = Select [Agg]
  | SelectDistinct [Agg]
  deriving (Eq, Show)

data From = From
  { fromTableUnits :: [TableUnit],
    fromCond :: Maybe Cond
  }
  deriving (Eq, Show)

data Cond
  = And Cond Cond
  | Or Cond Cond
  | Not Cond
  | Between ValUnit Val Val
  | Eq ValUnit Val
  | Gt ValUnit Val
  | Lt ValUnit Val
  | Ge ValUnit Val
  | Le ValUnit Val
  | Ne ValUnit Val
  | In ValUnit Val
  | Like ValUnit Val
  deriving (Eq, Show)

data ColUnit
  = ColUnit
      { colUnitAggId :: AggType,
        colUnitTable :: Maybe (Either Alias TableId),
        colUnitColId :: ColumnId
      }
  | DistinctColUnit
      { distinctColUnitAggId :: AggType,
        distinctColUnitTable :: Maybe (Either Alias TableId),
        distinctColUnitColdId :: ColumnId
      }
  deriving (Eq, Show)

data OrderBy = OrderBy OrderByOrder [ValUnit] deriving (Eq, Show)

data OrderByOrder = Asc | Desc deriving (Eq, Show)

data Agg = Agg AggType ValUnit deriving (Eq, Show)

data TableUnit
  = TableUnitSQL SpiderSQL (Maybe Alias)
  | Table TableId (Maybe Alias)
  deriving (Eq, Show)

data ValUnit
  = Column ColUnit
  | Minus ColUnit ColUnit
  | Plus ColUnit ColUnit
  | Times ColUnit ColUnit
  | Divide ColUnit ColUnit
  deriving (Eq, Show)

data Val
  = ValColUnit ColUnit
  | Number Double
  | ValString String
  | ValSQL SpiderSQL
  | Terminal
  deriving (Eq, Show)

data AggType = NoneAggOp | Max | Min | Count | Sum | Avg deriving (Eq, Show)

data ColumnId = Star | ColumnId String deriving (Eq, Show)

newtype TableId = TableId String deriving (Eq, Show)

newtype Alias = Alias String deriving (Eq, Show)

-- | @keyword k@ is a parser that consumes 'Char' tokens and yields them
-- if and only if they assemble the 'String' @s@. The parser is not sensitive to
-- letter casing.
--
-- >>> head $ parseString @[] (isKeyword "mykeyword") "MYKEYWORD"
-- ("MYKEYWORD","")
isKeyword :: CharParsing m => String -> m String
isKeyword s = traverse (satisfy . ((. toLower) . (==) . toLower)) s <?> s

isSelect :: CharParsing m => m String
isSelect = isKeyword "select"

isDistinct :: CharParsing m => m String
isDistinct = isKeyword "distinct"

isStar :: CharParsing m => m String
isStar = pure <$> char '*'

isComma :: CharParsing m => m String
isComma = pure <$> char ','

isDot :: CharParsing m => m String
isDot = pure <$> char '.'

isSemicolon :: CharParsing m => m String
isSemicolon = pure <$> char ';'

isEq :: CharParsing m => m String
isEq = pure <$> char '='

isGt :: CharParsing m => m String
isGt = pure <$> char '>'

isLt :: CharParsing m => m String
isLt = pure <$> char '<'

isGe :: CharParsing m => m String
isGe = string ">="

isLe :: CharParsing m => m String
isLe = string "<="

isNe :: CharParsing m => m String
isNe = string "!="

isIn :: CharParsing m => m String
isIn = isKeyword "in"

isLike :: CharParsing m => m String
isLike = isKeyword "like"

isBetween :: CharParsing m => m String
isBetween = isKeyword "between"

isAnd :: CharParsing m => m String
isAnd = isKeyword "and"

isOr :: CharParsing m => m String
isOr = isKeyword "or"

isNot :: CharParsing m => m String
isNot = isKeyword "not"

isMinus :: CharParsing m => m String
isMinus = string "-"

isPlus :: CharParsing m => m String
isPlus = string "+"

isTimes :: CharParsing m => m String
isTimes = string "*"

isDivide :: CharParsing m => m String
isDivide = string "/"

isMax :: CharParsing m => m String
isMax = isKeyword "max"

isMin :: CharParsing m => m String
isMin = isKeyword "min"

isCount :: CharParsing m => m String
isCount = isKeyword "count"

isSum :: CharParsing m => m String
isSum = isKeyword "sum"

isAvg :: CharParsing m => m String
isAvg = isKeyword "avg"

isFrom :: CharParsing m => m String
isFrom = isKeyword "from"

isJoin :: CharParsing m => m String
isJoin = isKeyword "join"

isAs :: CharParsing m => m String
isAs = isKeyword "as"

isOn :: CharParsing m => m String
isOn = isKeyword "on"

isWhere :: CharParsing m => m String
isWhere = isKeyword "where"

isGroupBy :: CharParsing m => m String
isGroupBy = isKeyword "group by"

isOrderBy :: CharParsing m => m String
isOrderBy = isKeyword "order by"

isAsc :: CharParsing m => m String
isAsc = isKeyword "asc"

isDesc :: CharParsing m => m String
isDesc = isKeyword "desc"

isHaving :: CharParsing m => m String
isHaving = isKeyword "having"

isLimit :: CharParsing m => m String
isLimit = isKeyword "limit"

isIntersect :: CharParsing m => m String
isIntersect = isKeyword "intersect"

isExcept :: CharParsing m => m String
isExcept = isKeyword "except"

isUnion :: CharParsing m => m String
isUnion = isKeyword "union"

betweenParentheses :: CharParsing m => m a -> m a
betweenParentheses = between (spaces *> char '(' <* spaces) (spaces *> char ')' <* spaces)

betweenOptionalParentheses :: CharParsing m => m a -> m a
betweenOptionalParentheses p = betweenParentheses p <|> p

-- | 'Select' parser
--
-- >>> head $ parseString @[] select "select count table.*"
-- (Select [Agg Count (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "table")), colUnitColId = Star}))],"")
--
-- >>> head $ parseString @[] select "SELECT COUNT (DISTINCT t5.title)"
-- (Select [Agg Count (Column (DistinctColUnit {distinctColUnitAggId = NoneAggOp, distinctColUnitTable = Just (Left (Alias "t5")), distinctColUnitColdId = ColumnId "title"}))],"")
select :: (TokenParsing m, Monad m) => m Select
select = do
  isSelect
  someSpace
  distinct <- optional (isDistinct <* someSpace)
  aggs <- sepBy agg isComma
  case distinct of
    Just _ -> pure $ SelectDistinct aggs
    Nothing -> pure $ Select aggs

-- | 'Agg' parser.
--
-- >>> head $ parseString @[] agg "count table.id"
-- (Agg Count (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "table")), colUnitColId = ColumnId "id"})),"")
agg :: (TokenParsing m, Monad m) => m Agg
agg =
  Agg
    <$> ( aggType >>= \case
            NoneAggOp -> pure NoneAggOp
            at -> at <$ someSpace
        )
    <*> valUnit

-- | 'AggType' parser.
--
-- >>> head $ parseString @[] aggType ""
-- (NoneAggOp,"")
aggType :: CharParsing m => m AggType
aggType = choice choices
  where
    choices =
      [ isMax $> Max,
        isMin $> Min,
        isCount $> Count,
        isSum $> Sum,
        isAvg $> Avg,
        pure NoneAggOp
      ]

-- | 'ValUnit' parser.
--
-- >>> head $ parseString @[] valUnit "t1.stadium_id"
-- (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "stadium_id"}),"")
--
-- >>> head . filter (null . snd) $ parseString @[] valUnit "t1.stadium_length * t1.stadium_width"
-- (Times (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "stadium_length"}) (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "stadium_width"}),"")
valUnit :: (TokenParsing m, Monad m) => m ValUnit
valUnit =
  betweenOptionalParentheses
    ( spaces *> choice choices <* spaces
    )
  where
    choices = [column, minus, plus, times, divide]
    column = Column <$> colUnit
    binary f p = f <$> colUnit <*> (someSpace *> p *> someSpace *> colUnit)
    minus = binary Minus isMinus
    plus = binary Plus isPlus
    times = binary Times isTimes
    divide = binary Divide isDivide

-- | 'ColUnit' parser.
--
-- >>> head $ parseString @[] colUnit "count ( distinct my_table.* )"
-- (DistinctColUnit {distinctColUnitAggId = Count, distinctColUnitTable = Just (Left (Alias "my_table")), distinctColUnitColdId = Star},"")
colUnit :: (TokenParsing m, Monad m) => m ColUnit
colUnit = do
  at <- aggType
  (distinct, tabAli, col) <-
    betweenOptionalParentheses $
      (,,) <$> optional (isDistinct <* someSpace)
        <*> optional (eitherP alias tableId <* isDot)
        <*> columnId
  case distinct of
    Just _ -> pure $ DistinctColUnit at tabAli col
    Nothing -> pure $ ColUnit at tabAli col

-- | 'TableId' parser.
tableId :: CharParsing m => m TableId
tableId = TableId <$> name

-- | 'Alias' parser.
alias :: CharParsing m => m Alias
alias = Alias <$> name

-- | 'ColumnId' parser.
--
-- >>> parseString @[] columnId "*"
-- [(Star,"")]
--
-- >>> parseString @[] columnId "c"
-- [(ColumnId "c","")]
columnId :: CharParsing m => m ColumnId
columnId = isStar $> Star <|> ColumnId <$> name

tableUnitAlias :: TableUnit -> Maybe Alias
tableUnitAlias (TableUnitSQL _ malias) = malias
tableUnitAlias (Table _ malias) = malias

tableUnitTableId :: TableUnit -> Maybe TableId
tableUnitTableId (TableUnitSQL _ _) = Nothing
tableUnitTableId (Table tableId _) = Just tableId

condAliases :: Cond -> [Alias]
condAliases = go
  where
    go (And cond cond') = go cond <> go cond'
    go (Or cond cond') = go cond <> go cond'
    go (Not cond) = go cond
    go (Between valUnit val val') =
      valUnitAliases valUnit <> toList (valAlias val) <> toList (valAlias val')
    go (Eq valUnit val) =
      valUnitAliases valUnit <> toList (valAlias val)
    go (Gt valUnit val) =
      valUnitAliases valUnit <> toList (valAlias val)
    go (Lt valUnit val) =
      valUnitAliases valUnit <> toList (valAlias val)
    go (Ge valUnit val) =
      valUnitAliases valUnit <> toList (valAlias val)
    go (Le valUnit val) =
      valUnitAliases valUnit <> toList (valAlias val)
    go (Ne valUnit val) =
      valUnitAliases valUnit <> toList (valAlias val)
    go (In valUnit val) =
      valUnitAliases valUnit <> toList (valAlias val)
    go (Like valUnit val) =
      valUnitAliases valUnit <> toList (valAlias val)

condTableIds :: Cond -> [TableId]
condTableIds = go
  where
    go (And cond cond') = go cond <> go cond'
    go (Or cond cond') = go cond <> go cond'
    go (Not cond) = go cond
    go (Between valUnit val val') =
      valUnitTableIds valUnit <> toList (valTableId val) <> toList (valTableId val')
    go (Eq valUnit val) =
      valUnitTableIds valUnit <> toList (valTableId val)
    go (Gt valUnit val) =
      valUnitTableIds valUnit <> toList (valTableId val)
    go (Lt valUnit val) =
      valUnitTableIds valUnit <> toList (valTableId val)
    go (Ge valUnit val) =
      valUnitTableIds valUnit <> toList (valTableId val)
    go (Le valUnit val) =
      valUnitTableIds valUnit <> toList (valTableId val)
    go (Ne valUnit val) =
      valUnitTableIds valUnit <> toList (valTableId val)
    go (In valUnit val) =
      valUnitTableIds valUnit <> toList (valTableId val)
    go (Like valUnit val) =
      valUnitTableIds valUnit <> toList (valTableId val)

valUnitAliases :: ValUnit -> [Alias]
valUnitAliases (Column colUnit) =
  toList (colUnitAlias colUnit)
valUnitAliases (Minus colUnit colUnit') =
  toList (colUnitAlias colUnit) <> toList (colUnitAlias colUnit')
valUnitAliases (Plus colUnit colUnit') =
  toList (colUnitAlias colUnit) <> toList (colUnitAlias colUnit')
valUnitAliases (Times colUnit colUnit') =
  toList (colUnitAlias colUnit) <> toList (colUnitAlias colUnit')
valUnitAliases (Divide colUnit colUnit') =
  toList (colUnitAlias colUnit) <> toList (colUnitAlias colUnit')

valUnitTableIds :: ValUnit -> [TableId]
valUnitTableIds (Column colUnit) =
  toList (colUnitTableId colUnit)
valUnitTableIds (Minus colUnit colUnit') =
  toList (colUnitTableId colUnit) <> toList (colUnitTableId colUnit')
valUnitTableIds (Plus colUnit colUnit') =
  toList (colUnitTableId colUnit) <> toList (colUnitTableId colUnit')
valUnitTableIds (Times colUnit colUnit') =
  toList (colUnitTableId colUnit) <> toList (colUnitTableId colUnit')
valUnitTableIds (Divide colUnit colUnit') =
  toList (colUnitTableId colUnit) <> toList (colUnitTableId colUnit')

colUnitAlias :: ColUnit -> Maybe Alias
colUnitAlias ColUnit {..} = colUnitTable >>= either Just (const Nothing)
colUnitAlias DistinctColUnit {..} = distinctColUnitTable >>= either Just (const Nothing)

colUnitTableId :: ColUnit -> Maybe TableId
colUnitTableId ColUnit {..} = colUnitTable >>= either (const Nothing) Just
colUnitTableId DistinctColUnit {..} = distinctColUnitTable >>= either (const Nothing) Just

valAlias :: Val -> Maybe Alias
valAlias (ValColUnit colUnit) = colUnitAlias colUnit
valAlias (Number _) = Nothing
valAlias (ValString _) = Nothing
valAlias (ValSQL _) = Nothing
valAlias Terminal = Nothing

valTableId :: Val -> Maybe TableId
valTableId (ValColUnit colUnit) = colUnitTableId colUnit
valTableId (Number _) = Nothing
valTableId (ValString _) = Nothing
valTableId (ValSQL _) = Nothing
valTableId Terminal = Nothing

-- | 'From' parser.
--
-- >>> head $ parseString @[] from "FROM people AS t1 JOIN pets AS t2 ON t1.pet_id = t2.pet_id"
-- (From {fromTableUnits = [Table (TableId "people") (Just (Alias "t1")),Table (TableId "pets") (Just (Alias "t2"))], fromCond = Just (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "pet_id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t2")), colUnitColId = ColumnId "pet_id"})))},"")
--
-- >>> head $ parseString @[] from "FROM organization AS t3 JOIN author AS t1 ON t3.oid = t1.oid JOIN writes AS t4 ON t4.aid = t1.aid JOIN publication AS t5 ON t4.pid = t5.pid JOIN conference AS t2 ON t5.cid = t2.cid"
-- (From {fromTableUnits = [Table (TableId "organization") (Just (Alias "t3")),Table (TableId "author") (Just (Alias "t1")),Table (TableId "writes") (Just (Alias "t4")),Table (TableId "publication") (Just (Alias "t5")),Table (TableId "conference") (Just (Alias "t2"))], fromCond = Just (And (And (And (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t3")), colUnitColId = ColumnId "oid"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "oid"}))) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t4")), colUnitColId = ColumnId "aid"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "aid"})))) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t4")), colUnitColId = ColumnId "pid"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t5")), colUnitColId = ColumnId "pid"})))) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t5")), colUnitColId = ColumnId "cid"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t2")), colUnitColId = ColumnId "cid"}))))},"")
from :: forall m. (TokenParsing m, Monad m) => m From
from = do
  isFrom
  someSpace
  from@From {..} <- uncurry mkFrom <$> p
  let boundAliases = foldMap (toList . tableUnitAlias) fromTableUnits
      aliasReferences = foldMap condAliases fromCond
  guard (boundAliases == nub boundAliases)
  guard (all (`elem` boundAliases) aliasReferences)
  let boundTableIds = foldMap (toList . tableUnitTableId) fromTableUnits
      tableIdReferences = foldMap condTableIds fromCond
  guard (all (`elem` boundTableIds) tableIdReferences)
  pure from
  where
    p :: m (TableUnit, [(TableUnit, Maybe Cond)])
    p =
      (,)
        <$> tableUnit
        <*> many
          ( someSpace
              *> isJoin
              *> someSpace
              *> ( (,)
                     <$> tableUnit
                     <*> optional
                       ( someSpace
                           *> isOn
                           *> someSpace
                           *> cond
                       )
                 )
          )
    mkFrom :: TableUnit -> [(TableUnit, Maybe Cond)] -> From
    mkFrom tu tus =
      From
        (tu : fmap fst tus)
        ( foldl
            ( \a b ->
                case (a, b) of
                  (Just c, Just c') -> Just (And c c')
                  (Just c, Nothing) -> Just c
                  (Nothing, Just c') -> Just c'
                  (Nothing, Nothing) -> Nothing
            )
            Nothing
            (fmap snd tus)
        )

-- | 'TableUnit' parser.
--
-- >>> head $ parseString @[] tableUnit "people as t1"
-- (Table (TableId "people") (Just (Alias "t1")),"")
tableUnit :: (TokenParsing m, Monad m) => m TableUnit
tableUnit =
  let tableUnitSQL =
        TableUnitSQL
          <$> betweenParentheses spiderSQL
            <*> optional (someSpace *> isAs *> someSpace *> alias)
      table =
        Table
          <$> tableId
            <*> optional (someSpace *> isAs *> someSpace *> alias)
   in tableUnitSQL <|> table

-- | 'Cond' parser.
--
-- >>> head $ parseString @[] cond "t1.stadium_id = t2.stadium_id"
-- (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "stadium_id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t2")), colUnitColId = ColumnId "stadium_id"})),"")
--
-- >>> head $ parseString @[] (cond <* isToken ';') "t2.name = \"VLDB\" AND t3.name = \"University of Michigan\";"
-- (And (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t2")), colUnitColId = ColumnId "name"})) (ValString "VLDB")) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t3")), colUnitColId = ColumnId "name"})) (ValString "University of Michigan")),"")
cond :: (TokenParsing m, Monad m) => m Cond
cond =
  let mkCond p' =
        let suffix r' =
              let q = mkCond p'
               in choice
                    [ And r' <$> (someSpace *> isAnd *> someSpace *> q),
                      Or r' <$> (someSpace *> isOr *> someSpace *> q)
                    ]
            suffixRec base = do
              c <- base
              suffixRec (suffix c) <|> pure c
            r =
              choice
                [ Not <$> (isNot *> spaces *> p'),
                  p'
                ]
         in suffixRec r
      p =
        choice
          [ binary Eq isEq,
            binary Gt isGt,
            binary Lt isLt,
            binary Ge isGe,
            binary Le isLe,
            binary Ne isNe,
            binary In isIn,
            binary Like isLike,
            Between <$> valUnit <*> (someSpace *> isBetween *> someSpace *> val) <*> (someSpace *> isAnd *> someSpace *> val)
          ]
      binary f q = f <$> valUnit <*> (spaces *> q *> spaces *> val)
   in mkCond p

-- | 'Val' parser.
--
-- >>> head $ parseString @[] val "count t1.stadium_id"
-- (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Nothing, colUnitColId = ColumnId "count"})," t1.stadium_id")
--
-- >>> head $ parseString @[] val "(select *)"
-- (ValSQL (SpiderSQL {spiderSQLSelect = Select [Agg NoneAggOp (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Nothing, colUnitColId = Star}))], spiderSQLFrom = From {fromTableUnits = [], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}),"")
val :: (TokenParsing m, Monad m) => m Val
val = choice choices
  where
    choices = [valColUnit, number, valString, valSQL, terminal]
    valColUnit = ValColUnit <$> colUnit
    number = Number <$> doubleP
    valString = ValString <$> quotedString
    valSQL = ValSQL <$> betweenParentheses spiderSQL
    terminal = pure Terminal

-- | Parser for quoted strings.
--
-- >>> head $ parseString @[] quotedString "\"hello world\""
-- ("hello world","")
quotedString :: CharParsing m => m String
quotedString =
  let q = char '"'
      s = many (notChar '"')
   in between q q s

-- | Parser for where clauses.
--
-- >>> head $ parseString @[] whereCond "where t1.id = t2.id"
-- (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t2")), colUnitColId = ColumnId "id"})),"")
whereCond :: (TokenParsing m, Monad m) => m Cond
whereCond = isWhere *> someSpace *> cond

-- | Parser for group-by clauses.
--
-- >>> head $ parseString @[] groupBy "group by count t1.id, t2.id"
-- ([ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Nothing, colUnitColId = ColumnId "count"}]," t1.id, t2.id")
groupBy :: (TokenParsing m, Monad m) => m [ColUnit]
groupBy = isGroupBy *> someSpace *> sepBy1 colUnit (isComma <* someSpace)

-- | 'OrderBy' Parser.
--
-- >>> head . filter (null . snd) $ parseString @[] orderBy "order by t1.stadium_id, t2.pet_id desc"
-- (OrderBy Desc [Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "stadium_id"}),Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t2")), colUnitColId = ColumnId "pet_id"})],"")
orderBy :: (TokenParsing m, Monad m) => m OrderBy
orderBy = do
  isOrderBy
  someSpace
  valUnits <- sepBy1 valUnit (isComma <* someSpace)
  order <- optional (someSpace *> (isAsc $> Asc <|> isDesc $> Desc)) >>= maybe (pure Asc) pure
  pure $ OrderBy order valUnits

-- | Parser for having clauses.
--
-- >>> head $ parseString @[] havingCond "having count(t1.customer_id) = 10"
-- (Eq (Column (ColUnit {colUnitAggId = Count, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "customer_id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Nothing, colUnitColId = ColumnId "10"})),"")
havingCond :: (TokenParsing m, Monad m) => m Cond
havingCond = isHaving *> someSpace *> cond

-- | Parser for limit clauses.
--
-- >>> head $ parseString @[] limit "limit 10"
-- (10,"")
limit :: (TokenParsing m, Monad m) => m Int
limit = isLimit *> someSpace *> intP

-- | 'SpiderSQL' parser.
--
-- >>> head $ parseString @[] (spiderSQL <* spaces <* isSemicolon) "select * ;"
-- (SpiderSQL {spiderSQLSelect = Select [Agg NoneAggOp (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Nothing, colUnitColId = Star}))], spiderSQLFrom = From {fromTableUnits = [], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing},"")
--
-- >>> head $ parseString @[] (spiderSQL <* spaces <* isSemicolon) "select * from concert;"
-- (SpiderSQL {spiderSQLSelect = Select [Agg NoneAggOp (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Nothing, colUnitColId = Star}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId "concert") Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing},"")
--
-- >>> head $ parseString @[] (spiderSQL <* spaces <* isSemicolon) "select T2.name, count(*) from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id group by t1.stadium_id;"
-- (SpiderSQL {spiderSQLSelect = Select [Agg NoneAggOp (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "T2")), colUnitColId = ColumnId "name"})),Agg NoneAggOp (Column (ColUnit {colUnitAggId = Count, colUnitTable = Nothing, colUnitColId = Star}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId "concert") (Just (Alias "t1")),Table (TableId "stadium") (Just (Alias "t2"))], fromCond = Just (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "stadium_id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t2")), colUnitColId = ColumnId "stadium_id"})))}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "stadium_id"}], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing},"")
--
-- >>> head $ parseString @[] (spiderSQL <* spaces <* isSemicolon) "SELECT COUNT ( DISTINCT t5.title ) FROM organization AS t3 JOIN author AS t1 ON t3.oid = t1.oid JOIN writes AS t4 ON t4.aid = t1.aid JOIN publication AS t5 ON t4.pid = t5.pid JOIN conference AS t2 ON t5.cid = t2.cid WHERE t2.name = \"VLDB\" AND t3.name = \"University of Michigan\";"
-- (SpiderSQL {spiderSQLSelect = Select [Agg Count (Column (DistinctColUnit {distinctColUnitAggId = NoneAggOp, distinctColUnitTable = Just (Left (Alias "t5")), distinctColUnitColdId = ColumnId "title"}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId "organization") (Just (Alias "t3")),Table (TableId "author") (Just (Alias "t1")),Table (TableId "writes") (Just (Alias "t4")),Table (TableId "publication") (Just (Alias "t5")),Table (TableId "conference") (Just (Alias "t2"))], fromCond = Just (And (And (And (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t3")), colUnitColId = ColumnId "oid"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "oid"}))) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t4")), colUnitColId = ColumnId "aid"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "aid"})))) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t4")), colUnitColId = ColumnId "pid"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t5")), colUnitColId = ColumnId "pid"})))) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t5")), colUnitColId = ColumnId "cid"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t2")), colUnitColId = ColumnId "cid"}))))}, spiderSQLWhere = Just (And (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t2")), colUnitColId = ColumnId "name"})) (ValString "VLDB")) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t3")), colUnitColId = ColumnId "name"})) (ValString "University of Michigan"))), spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing},"")
spiderSQL :: (TokenParsing m, Monad m) => m SpiderSQL
spiderSQL = do
  sel <- select
  fro <- fromMaybe (From [] Nothing) <$> optional (spaces *> from)
  whe <- optional (someSpace *> whereCond)
  grp <- fromMaybe [] <$> optional (someSpace *> groupBy)
  ord <- optional (someSpace *> orderBy)
  hav <- optional (someSpace *> havingCond)
  lim <- optional (someSpace *> limit)
  int <- optional (someSpace *> isIntersect *> someSpace *> spiderSQL)
  exc <- optional (someSpace *> isExcept *> someSpace *> spiderSQL)
  uni <- optional (someSpace *> isUnion *> someSpace *> spiderSQL)
  pure $ SpiderSQL sel fro whe grp ord hav lim int exc uni

-- | Auxiliary parser for table names, column names, and aliases.
name :: CharParsing m => m String
name =
  let p = satisfy ((||) <$> isAlphaNum <*> (== '_'))
   in some p -- liftA2 (:) p (atMost 16 p)

-- space1' :: MonadPlus b => Parser b Char String
-- space1' = pure <$> satisfy isSpace

-- digits1' :: MonadPlus b => Parser b Char String
-- digits1' =
--   let p = satisfy isDigit
--    in liftA2 (:) p (atMost 8 p)

-- intP' :: MonadPlus b => Parser b Char Int
-- intP' = digits1' >>= maybe empty pure . readMaybe

-- doubleP' :: MonadPlus b => Parser b Char Double
-- doubleP' =
--   let p = satisfy (not . isSpace)
--    in liftA2 (:) p (atMost 8 p) >>= maybe empty pure . readMaybe

-- sepBy' :: MonadPlus m => m a -> m sep -> m [a]
-- sepBy' p sep = (p `sepBy1'` sep) <|> pure []

-- sepBy1' :: MonadPlus m => m a -> m sep -> m [a]
-- sepBy1' p sep = (:) <$> p <*> atMost 4 (sep *> p)
