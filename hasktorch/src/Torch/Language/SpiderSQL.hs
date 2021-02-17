{-# LANGUAGE RankNTypes #-}
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

import Control.Applicative (Alternative (..))
import Control.Monad (MonadPlus)
import Data.Char (isAlphaNum, toLower)
import Data.Functor (($>))
import Data.Maybe (maybeToList)
import Text.Read (readMaybe)
import Torch.Data.Parser

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
        colUnitTable :: Maybe (Either TableId Alias),
        colUnitColId :: ColumnId
      }
  | DistinctColUnit
      { distinctColUnitAggId :: AggType,
        distinctColUnitTable :: Maybe (Either TableId Alias),
        distinctColUnitColdId :: ColumnId
      }
  deriving (Eq, Show)

data OrderBy = OrderBy OrderByOrder [ValUnit] deriving (Eq, Show)

data OrderByOrder = Asc | Desc deriving (Eq, Show)

data Agg = Agg AggType ValUnit deriving (Eq, Show)

data TableUnit
  = TableUnitSql SpiderSQL (Maybe Alias)
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
  | ValSql SpiderSQL
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
isKeyword :: MonadPlus b => String -> Parser b Char String
isKeyword = traverse (satisfy . ((. toLower) . (==) . toLower))

isSelect :: MonadPlus b => Parser b Char String
isSelect = isKeyword "select"

isDistinct :: MonadPlus b => Parser b Char String
isDistinct = isKeyword "distinct"

isStar :: MonadPlus b => Parser b Char String
isStar = pure <$> is '*'

isComma :: MonadPlus b => Parser b Char String
isComma = pure <$> is ','

isDot :: MonadPlus b => Parser b Char String
isDot = pure <$> is '.'

isSemicolon :: MonadPlus b => Parser b Char String
isSemicolon = pure <$> is ';'

isEq :: MonadPlus b => Parser b Char String
isEq = pure <$> is '='

isGt :: MonadPlus b => Parser b Char String
isGt = pure <$> is '>'

isLt :: MonadPlus b => Parser b Char String
isLt = pure <$> is '<'

isGe :: MonadPlus b => Parser b Char String
isGe = isString ">="

isLe :: MonadPlus b => Parser b Char String
isLe = isString "<="

isNe :: MonadPlus b => Parser b Char String
isNe = isString "!="

isIn :: MonadPlus b => Parser b Char String
isIn = isKeyword "in"

isLike :: MonadPlus b => Parser b Char String
isLike = isKeyword "like"

isAnd :: MonadPlus b => Parser b Char String
isAnd = isKeyword "and"

isOr :: MonadPlus b => Parser b Char String
isOr = isKeyword "or"

isNotKeyword :: MonadPlus b => Parser b Char String
isNotKeyword = isKeyword "not"

isCount :: MonadPlus b => Parser b Char String
isCount = isKeyword "count"

isFrom :: MonadPlus b => Parser b Char String
isFrom = isKeyword "from"

isJoin :: MonadPlus b => Parser b Char String
isJoin = isKeyword "join"

isAs :: MonadPlus b => Parser b Char String
isAs = isKeyword "as"

isOn :: MonadPlus b => Parser b Char String
isOn = isKeyword "on"

isWhere :: MonadPlus b => Parser b Char String
isWhere = isKeyword "where"

isGroupBy :: MonadPlus b => Parser b Char String
isGroupBy = isKeyword "group" `combine` space `combine` isKeyword "by"

isOrderBy :: MonadPlus b => Parser b Char String
isOrderBy = isKeyword "order" `combine` space `combine` isKeyword "by"

isHaving :: MonadPlus b => Parser b Char String
isHaving = isKeyword "having"

isLimit :: MonadPlus b => Parser b Char String
isLimit = isKeyword "limit"

betweenParentheses :: MonadPlus b => Parser b Char a -> Parser b Char a
betweenParentheses = between (is '(') (is ')')

betweenOptionalParentheses :: MonadPlus b => Parser b Char a -> Parser b Char a
betweenOptionalParentheses p = betweenParentheses p <|> p

-- | 'Select' parser
--
-- >>> head $ parseString @[] select "select count table.*"
-- (Select [Agg Count (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "table")), colUnitColId = ColumnId "*"}))],"")
select :: MonadPlus b => Parser b Char Select
select =
  space
    *> isSelect
    *> space
    *> ( (Select <$> sepBy agg isComma)
           <|> (isDistinct *> space *> (SelectDistinct <$> sepBy agg isComma))
       )
    <* space

-- | 'Agg' parser.
--
-- >>> head $ parseString @[] agg "count table.id"
-- (Agg Count (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "table")), colUnitColId = ColumnId "id"})),"")
agg :: MonadPlus b => Parser b Char Agg
agg = space *> (Agg <$> aggType <*> valUnit) <* space

-- | 'AggType' parser.
--
-- >>> head $ parseString @[] aggType ""
-- (NoneAggOp,"")
aggType :: MonadPlus b => Parser b Char AggType
aggType = space *> ((isCount $> Count) <|> pure NoneAggOp) <* space

-- | 'ValUnit' parser.
--
-- >>> head $ parseString @[] valUnit "t1.stadium_id"
-- (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_id"}),"")
valUnit :: MonadPlus b => Parser b Char ValUnit
valUnit =
  space
    *> betweenOptionalParentheses
      ( space *> choice choices <* space
      )
    <* space
  where
    choices = [column] --, minus, plus, times, divide]
    column = Column <$> colUnit

-- | 'ColUnit' parser.
--
-- >>> head $ parseString @[] colUnit "count my_table.*"
-- (ColUnit {colUnitAggId = Count, colUnitTable = Just (Left (TableId "my_table")), colUnitColId = Star},"")
colUnit :: MonadPlus b => Parser b Char ColUnit
colUnit =
  space
    *> ( ColUnit
           <$> aggType
           <*> maybeP (eitherP tableId alias <* isDot)
           <*> columnId
       )
    <* space

-- | Auxiliary parser for table names, column names, and aliases.
name :: MonadPlus b => Parser b Char String
name = many1 $ satisfy ((||) <$> isAlphaNum <*> (== '_'))

-- | 'TableId' parser.
tableId :: MonadPlus b => Parser b Char TableId
tableId = TableId <$> name

-- | 'Alias' parser.
alias :: MonadPlus b => Parser b Char Alias
alias = Alias <$> name

-- | 'ColumnId' parser.
--
-- >>> parseString @[] columnId "*"
-- [(Star,"")]
-- >>> parseString @[] columnId "c"
-- [(ColumnId "c","")]
columnId :: MonadPlus b => Parser b Char ColumnId
columnId = isStar $> Star <|> ColumnId <$> name

-- | 'From' parser.
--
-- >>> head $ parseString @[] from "FROM people AS t1 JOIN pets AS t2 ON t1.pet_id = t2.pet_id"
-- (From {fromTableUnits = [Table (TableId "people") (Just (Alias "t1")),Table (TableId "pets") (Just (Alias "t2"))], fromCond = Just (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "pet_id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t2")), colUnitColId = ColumnId "pet_id"})))},"")
from :: forall b. MonadPlus b => Parser b Char From
from = space *> isFrom *> space *> (uncurry mkFrom <$> p) <* space
  where
    p :: Parser b Char (TableUnit, [(TableUnit, Maybe Cond)])
    p =
      (,)
        <$> tableUnit
        <*> many
          ( space
              *> isJoin
              *> space
              *> ( (,)
                     <$> tableUnit
                     <*> maybeP
                       ( space
                           *> isOn
                           *> space
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
tableUnit :: MonadPlus b => Parser b Char TableUnit
tableUnit = space *> (Table <$> tableId <*> maybeP (space *> isAs *> space *> alias)) <* space

-- | 'Cond' parser.
--
-- >>> head $ parseString @[] cond "t1.stadium_id = t2.stadium_id"
-- (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t2")), colUnitColId = ColumnId "stadium_id"})),"")
-- >>> head $ parseString @[] cond "t2.name = \"VLDB\" AND t3.name = \"University of Michigan\""
-- ProgressCancelledException
cond :: MonadPlus b => Parser b Char Cond
cond =
  let and =
        And
          <$> betweenOptionalParentheses eq
          <*> (space *> isAnd *> space *> betweenOptionalParentheses eq)
      -- or =
      --   Or
      --     <$> betweenOptionalParentheses cond
      --     <*> (space *> isOr *> space *> betweenOptionalParentheses cond)
      -- not = Not <$> betweenOptionalParentheses cond
      binary f q = f <$> valUnit <*> (space *> q *> space *> val)
      eq = binary Eq isEq
      gt = binary Gt isGt
      lt = binary Lt isLt
      ge = binary Ge isGe
      le = binary Le isLe
      ne = binary Ne isNe
      in' = binary In isIn
      like = binary Like isLike
   in space *> choice [and, eq, gt, lt, ge, le, ne, in', like] <* space

-- | 'Val' parser.
--
-- >>> head $ parseString @[] val "count t1.stadium_id"
-- (ValColUnit (ColUnit {colUnitAggId = Count, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_id"}),"")
val :: MonadPlus b => Parser b Char Val
val = space *> choice choices <* space
  where
    choices = [valColUnit, valString, terminal]
    -- choices = [valColUnit, number, valString, valSql, terminal]
    valColUnit = ValColUnit <$> colUnit
    -- number = undefined
    valString = ValString <$> quotedString
    -- valSql = undefined
    terminal = pure Terminal

quotedString :: MonadPlus b => Parser b Char String
quotedString =
  let q = is '"'
      s = many (isNot '"')
   in between q q s

whereCond :: MonadPlus b => Parser b Char (Maybe Cond)
whereCond = space *> maybeP (isWhere *> space *> cond) <* space

-- | Parser for group-by clauses.
--
-- >>> head $ parseString @[] groupBy "group by count t1.id"
-- ([ColUnit {colUnitAggId = Count, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "id"}],"")
groupBy :: MonadPlus b => Parser b Char [ColUnit]
groupBy = space *> (maybeToList <$> maybeP (isGroupBy *> space *> colUnit)) <* space

-- | 'OrderBy' Parser.
--
-- >>> head $ parseString @[] orderBy "order by t1.stadium_id desc"
-- (Just (OrderBy Ascending []),"t1.stadium_id desc")
orderBy :: forall b. MonadPlus b => Parser b Char (Maybe OrderBy)
orderBy =
  space *> maybeP (isOrderBy *> space *> p) <* space
  where
    p :: Parser b Char OrderBy
    p = pure $ OrderBy Asc []

-- | Parser for having clauses.
--
-- >>> head $ parseString @[] havingCond "having count(t1.customer_id) = 10"
-- (Nothing,"having count(t1.customer_id) = 10")
havingCond :: MonadPlus b => Parser b Char (Maybe Cond)
havingCond = space *> maybeP (isHaving *> space *> cond) <* space

-- | Parser for limit clauses.
--
-- >>> head $ parseString @[] limit "limit 10"
-- (Just 10,".5")
limit :: MonadPlus b => Parser b Char (Maybe Int)
limit = space *> maybeP (isLimit *> space *> digits1 >>= maybe empty pure . readMaybe) <* space

-- | 'SpiderSQL' parser.
--
-- >>> head $ parseString @[] spiderSQL "select T2.name, count(*) from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id group by t1.stadium_id"
-- (SpiderSQL {spiderSQLSelect = Select [Agg NoneAggOp (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "T2")), colUnitColId = ColumnId "name"})),Agg Count (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Nothing, colUnitColId = ColumnId "*"}))], spiderSQFrom = From {fromTableUnits = [Table (TableId "concert") (Just (Alias "t1")),Table (TableId "stadium") (Just (Alias "t2"))], fromCond = Just (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t2")), colUnitColId = ColumnId "stadium_id"})))}, spiderSQWhere = Nothing, spiderSQGroupBy = [ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_id"}], spiderSQOrderBy = Nothing, spiderSQHaving = Nothing, spiderSQLimit = Nothing, spiderSQIntersect = Nothing, spiderSQExcept = Nothing, spiderSQUnion = Nothing},"")
-- >>> head $ parseString @[] spiderSQL "SELECT COUNT ( DISTINCT t5.title ) FROM organization AS t3 JOIN author AS t1 ON t3.oid  =  t1.oid JOIN writes AS t4 ON t4.aid  =  t1.aid JOIN publication AS t5 ON t4.pid  =  t5.pid JOIN conference AS t2 ON t5.cid  =  t2.cid WHERE t2.name  =  \"VLDB\" AND t3.name  =  \"University of Michigan\";"
-- Prelude.head: empty list
spiderSQL :: MonadPlus b => Parser b Char SpiderSQL
spiderSQL =
  SpiderSQL
    <$> select
      <*> from
      <*> whereCond
      <*> groupBy
      <*> orderBy
      <*> havingCond
      <*> limit
      <*> pure Nothing
      <*> pure Nothing
      <*> pure Nothing
    <* (isSemicolon <|> pure [])
