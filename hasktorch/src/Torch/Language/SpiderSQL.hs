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
import Text.Read (readMaybe)
import Torch.Data.Parser
  ( Parser,
    atMost,
    between,
    choice,
    combine,
    eitherP,
    is,
    isNot,
    isString,
    maybeP,
    satisfy,
    space,
    parseString
  )

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

isBetween :: MonadPlus b => Parser b Char String
isBetween = isKeyword "between"

isAnd :: MonadPlus b => Parser b Char String
isAnd = isKeyword "and"

isOr :: MonadPlus b => Parser b Char String
isOr = isKeyword "or"

isNotKeyword :: MonadPlus b => Parser b Char String
isNotKeyword = isKeyword "not"

isMinus :: MonadPlus b => Parser b Char String
isMinus = isString "-"

isPlus :: MonadPlus b => Parser b Char String
isPlus = isString "+"

isTimes :: MonadPlus b => Parser b Char String
isTimes = isString "*"

isDivide :: MonadPlus b => Parser b Char String
isDivide = isString "/"

isMax :: MonadPlus b => Parser b Char String
isMax = isKeyword "max"

isMin :: MonadPlus b => Parser b Char String
isMin = isKeyword "min"

isCount :: MonadPlus b => Parser b Char String
isCount = isKeyword "count"

isSum :: MonadPlus b => Parser b Char String
isSum = isKeyword "sum"

isAvg :: MonadPlus b => Parser b Char String
isAvg = isKeyword "avg"

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
isGroupBy = isKeyword "group" `combine` space1' `combine` isKeyword "by"

isOrderBy :: MonadPlus b => Parser b Char String
isOrderBy = isKeyword "order" `combine` space1' `combine` isKeyword "by"

isAsc :: MonadPlus b => Parser b Char String
isAsc = isKeyword "asc"

isDesc :: MonadPlus b => Parser b Char String
isDesc = isKeyword "desc"

isHaving :: MonadPlus b => Parser b Char String
isHaving = isKeyword "having"

isLimit :: MonadPlus b => Parser b Char String
isLimit = isKeyword "limit"

isIntersect :: MonadPlus b => Parser b Char String
isIntersect = isKeyword "intersect"

isExcept :: MonadPlus b => Parser b Char String
isExcept = isKeyword "except"

isUnion :: MonadPlus b => Parser b Char String
isUnion = isKeyword "union"

betweenParentheses :: MonadPlus b => Parser b Char a -> Parser b Char a
betweenParentheses = between (space *> is '(' <* space) (space *> is ')' <* space)

betweenOptionalParentheses :: MonadPlus b => Parser b Char a -> Parser b Char a
betweenOptionalParentheses p = betweenParentheses p <|> p

-- | 'Select' parser
--
-- >>> head $ parseString @[] select "select count table.*"
-- (Select [Agg Count (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "table")), colUnitColId = Star}))],"")
select :: MonadPlus b => Parser b Char Select
select = do
  isSelect
  space1'
  distinct <- optional (isDistinct <* space1')
  aggs <- sepBy' agg isComma
  case distinct of
    Just _ -> pure $ SelectDistinct aggs
    Nothing -> pure $ Select aggs

-- | 'Agg' parser.
--
-- >>> head $ parseString @[] agg "count table.id"
-- (Agg Count (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "table")), colUnitColId = ColumnId "id"})),"")
agg :: MonadPlus b => Parser b Char Agg
agg =
  Agg
    <$> ( aggType >>= \case
            NoneAggOp -> pure NoneAggOp
            at -> at <$ space1'
        )
    <*> valUnit

-- | 'AggType' parser.
--
-- >>> head $ parseString @[] aggType ""
-- (NoneAggOp,"")
aggType :: MonadPlus b => Parser b Char AggType
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
-- (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_id"}),"")
-- >>> head . filter (null . snd) $ parseString @[] valUnit "t1.stadium_length * t1.stadium_width"
-- (Times (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_length"}) (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_width"}),"")
valUnit :: MonadPlus b => Parser b Char ValUnit
valUnit =
  betweenOptionalParentheses
    ( space *> choice choices <* space
    )
  where
    choices = [column, minus, plus, times, divide]
    column = Column <$> colUnit
    binary f p = f <$> colUnit <*> (space1' *> p *> space1' *> colUnit)
    minus = binary Minus isMinus
    plus = binary Plus isPlus
    times = binary Times isTimes
    divide = binary Divide isDivide

-- | 'ColUnit' parser.
--
-- >>> head $ parseString @[] colUnit "count ( distinct my_table.* )"
-- (DistinctColUnit {distinctColUnitAggId = Count, distinctColUnitTable = Just (Left (TableId "my_table")), distinctColUnitColdId = Star},"")
colUnit :: MonadPlus b => Parser b Char ColUnit
colUnit = do
  at <- aggType
  (distinct, tabAli, col) <-
    betweenOptionalParentheses $
      (,,) <$> optional (isDistinct <* space1')
        <*> optional (eitherP alias tableId <* isDot)
        <*> columnId
  case distinct of
    Just _ -> pure $ DistinctColUnit at tabAli col
    Nothing -> pure $ ColUnit at tabAli col

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
from :: forall b. MonadLogic b => Parser b Char From
from = do
  isFrom
  space1'
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
    p :: Parser b Char (TableUnit, [(TableUnit, Maybe Cond)])
    p =
      (,)
        <$> tableUnit
        <*> many
          ( space1'
              *> isJoin
              *> space1'
              *> ( (,)
                     <$> tableUnit
                     <*> maybeP
                       ( space1'
                           *> isOn
                           *> space1'
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
tableUnit :: MonadLogic b => Parser b Char TableUnit
tableUnit =
  let tableUnitSQL =
        TableUnitSQL
          <$> betweenParentheses spiderSQL
            <*> optional (space1' *> isAs *> space1' *> alias)
      table =
        Table
          <$> tableId
            <*> optional (space1' *> isAs *> space1' *> alias)
   in tableUnitSQL <|> table

-- | 'Cond' parser.
--
-- >>> head $ parseString @[] cond "t1.stadium_id = t2.stadium_id"
-- (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t2")), colUnitColId = ColumnId "stadium_id"})),"")
-- >>> head $ parseString @[] (cond <* is ';') "t2.name = \"VLDB\" AND t3.name = \"University of Michigan\";"
-- (And (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t2")), colUnitColId = ColumnId "name"})) (ValString "VLDB")) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t3")), colUnitColId = ColumnId "name"})) (ValString "University of Michigan")),"")
cond :: MonadLogic b => Parser b Char Cond
cond =
  let and q =
        And
          <$> betweenOptionalParentheses q
          <*> (space1' *> isAnd *> space1' *> betweenOptionalParentheses q)
      or q =
        Or
          <$> betweenOptionalParentheses q
          <*> (space1' *> isOr *> space1' *> betweenOptionalParentheses q)
      not q = Not <$> (isNotKeyword *> space1' *> betweenOptionalParentheses q)
      binary f q = f <$> valUnit <*> (space1' *> q *> space1' *> val)
      eq = binary Eq isEq
      gt = binary Gt isGt
      lt = binary Lt isLt
      ge = binary Ge isGe
      le = binary Le isLe
      ne = binary Ne isNe
      in' = binary In isIn
      like = binary Like isLike
      between = Between <$> valUnit <*> (space1' *> isBetween *> space1' *> val) <*> (space1' *> isAnd *> space1' *> val)
      p 0 = empty
      p n =
        let q = p (n - 1)
         in choice [eq, gt, lt, ge, le, ne, in', like, between, and q, or q, not q]
   in betweenOptionalParentheses (p 4)

-- | 'Val' parser.
--
-- >>> head $ parseString @[] val "count t1.stadium_id"
-- (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Nothing, colUnitColId = ColumnId "count"})," t1.stadium_id")
-- >>> head $ parseString @[] val "(select *)"
-- (ValSQL (SpiderSQL {spiderSQLSelect = Select [Agg NoneAggOp (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Nothing, colUnitColId = Star}))], spiderSQLFrom = From {fromTableUnits = [], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}),"")
val :: MonadLogic b => Parser b Char Val
val = choice choices
  where
    choices = [valColUnit, number, valString, valSQL, terminal]
    valColUnit = ValColUnit <$> colUnit
    number = Number <$> doubleP'
    valString = ValString <$> quotedString
    valSQL = ValSQL <$> betweenParentheses spiderSQL
    terminal = pure Terminal

-- | Parser for quoted strings.
--
-- >>> head $ parseString @[] quotedString "\"hello world\""
-- ("hello world","")
quotedString :: MonadPlus b => Parser b Char String
quotedString =
  let q = is '"'
      s = many (isNot '"')
   in between q q s

-- | Parser for where clauses.
--
-- >>> head $ parseString @[] whereCond "where t1.id = t2.id"
-- (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t2")), colUnitColId = ColumnId "id"})),"")
whereCond :: MonadLogic b => Parser b Char Cond
whereCond = isWhere *> space1' *> cond

-- | Parser for group-by clauses.
--
-- >>> head $ parseString @[] groupBy "group by count t1.id, t2.id"
-- ([ColUnit {colUnitAggId = Count, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "id"},ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t2")), colUnitColId = ColumnId "id"}],"")
groupBy :: MonadPlus b => Parser b Char [ColUnit]
groupBy = isGroupBy *> space1' *> sepBy1' colUnit (isComma <* space1')

-- | 'OrderBy' Parser.
--
-- >>> head . filter (null . snd) $ parseString @[] orderBy "order by t1.stadium_id, t2.pet_id desc"
-- (OrderBy Desc [Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_id"}),Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t2")), colUnitColId = ColumnId "pet_id"})],"")
orderBy :: forall b. MonadPlus b => Parser b Char OrderBy
orderBy = do
  isOrderBy
  space1'
  valUnits <- sepBy1' valUnit (isComma <* space1')
  order <- optional (space1' *> (isAsc $> Asc <|> isDesc $> Desc)) >>= maybe (pure Asc) pure
  pure $ OrderBy order valUnits

-- | Parser for having clauses.
--
-- >>> head $ parseString @[] havingCond "having count(t1.customer_id) = 10"
-- (Nothing,"having count(t1.customer_id) = 10")
havingCond :: MonadLogic b => Parser b Char Cond
havingCond = isHaving *> space1' *> cond

-- | Parser for limit clauses.
--
-- >>> head $ parseString @[] limit "limit 10"
-- (Just 10,".5")
limit :: MonadPlus b => Parser b Char Int
limit = isLimit *> space1' *> intP'

-- | 'SpiderSQL' parser.
--
-- >>> head $ parseString @[] (spiderSQL <* space <* isSemicolon) "select * ;"
-- (SpiderSQL {spiderSQLSelect = Select [Agg NoneAggOp (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Nothing, colUnitColId = Star}))], spiderSQLFrom = From {fromTableUnits = [], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing},"")
-- >>> head $ parseString @[] (spiderSQL <* space <* isSemicolon) "select * from concert;"
-- (SpiderSQL {spiderSQLSelect = Select [Agg NoneAggOp (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Nothing, colUnitColId = Star}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId "concert") Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing},"")
-- >>> head $ parseString @[] (spiderSQL <* space <* isSemicolon) "select T2.name, count(*) from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id group by t1.stadium_id;"
-- (SpiderSQL {spiderSQLSelect = Select [Agg NoneAggOp (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "T2")), colUnitColId = ColumnId "name"})),Agg NoneAggOp (Column (ColUnit {colUnitAggId = Count, colUnitTable = Nothing, colUnitColId = Star}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId "concert") (Just (Alias "t1")),Table (TableId "stadium") (Just (Alias "t2"))], fromCond = Just (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "stadium_id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t2")), colUnitColId = ColumnId "stadium_id"})))}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "stadium_id"}], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing},"")
-- >>> head $ parseString @[] (spiderSQL <* space <* isSemicolon) "SELECT COUNT ( DISTINCT t5.title ) FROM organization AS t3 JOIN author AS t1 ON t3.oid = t1.oid JOIN writes AS t4 ON t4.aid = t1.aid JOIN publication AS t5 ON t4.pid = t5.pid JOIN conference AS t2 ON t5.cid = t2.cid WHERE t2.name = \"VLDB\" AND t3.name = \"University of Michigan\";"
-- (SpiderSQL {spiderSQLSelect = Select [Agg Count (Column (DistinctColUnit {distinctColUnitAggId = NoneAggOp, distinctColUnitTable = Just (Left (Alias "t5")), distinctColUnitColdId = ColumnId "title"}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId "organization") (Just (Alias "t3")),Table (TableId "author") (Just (Alias "t1")),Table (TableId "writes") (Just (Alias "t4")),Table (TableId "publication") (Just (Alias "t5")),Table (TableId "conference") (Just (Alias "t2"))], fromCond = Just (And (And (And (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t3")), colUnitColId = ColumnId "oid"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "oid"}))) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t4")), colUnitColId = ColumnId "aid"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t1")), colUnitColId = ColumnId "aid"})))) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t4")), colUnitColId = ColumnId "pid"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t5")), colUnitColId = ColumnId "pid"})))) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t5")), colUnitColId = ColumnId "cid"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t2")), colUnitColId = ColumnId "cid"}))))}, spiderSQLWhere = Just (And (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t2")), colUnitColId = ColumnId "name"})) (ValString "VLDB")) (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (Alias "t3")), colUnitColId = ColumnId "name"})) (ValString "University of Michigan"))), spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing},"")
spiderSQL :: MonadLogic b => Parser b Char SpiderSQL
spiderSQL = do
  sel <- select
  fro <- fromMaybe (From [] Nothing) <$> optional (space1' *> from)
  whe <- optional (space1' *> whereCond)
  grp <- fromMaybe [] <$> optional (space1' *> groupBy)
  ord <- optional (space1' *> orderBy)
  hav <- optional (space1' *> havingCond)
  lim <- optional (space1' *> limit)
  int <- optional (space1' *> isIntersect *> space1' *> spiderSQL)
  exc <- optional (space1' *> isExcept *> space1' *> spiderSQL)
  uni <- optional (space1' *> isUnion *> space1' *> spiderSQL)
  pure $ SpiderSQL sel fro whe grp ord hav lim int exc uni

-- | Auxiliary parser for table names, column names, and aliases.
name :: MonadPlus b => Parser b Char String
name =
  let p = satisfy ((||) <$> isAlphaNum <*> (== '_'))
   in liftA2 (:) p (atMost 16 p)

space1' :: MonadPlus b => Parser b Char String
space1' = pure <$> satisfy isSpace

digits1' :: MonadPlus b => Parser b Char String
digits1' =
  let p = satisfy isDigit
   in liftA2 (:) p (atMost 8 p)

intP' :: MonadPlus b => Parser b Char Int
intP' = digits1' >>= maybe empty pure . readMaybe

doubleP' :: MonadPlus b => Parser b Char Double
doubleP' =
  let p = satisfy (not . isSpace)
   in liftA2 (:) p (atMost 8 p) >>= maybe empty pure . readMaybe

sepBy' :: MonadPlus m => m a -> m sep -> m [a]
sepBy' p sep = (p `sepBy1'` sep) <|> pure []

sepBy1' :: MonadPlus m => m a -> m sep -> m [a]
sepBy1' p sep = (:) <$> p <*> atMost 4 (sep *> p)
