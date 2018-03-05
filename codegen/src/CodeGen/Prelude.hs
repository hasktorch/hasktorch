module CodeGen.Prelude
  ( module X
  , impossible
  , tshow
  , tputStrLn
  ) where

import Prelude         as X
import Control.Monad   as X (guard)

import Data.Char       as X (toLower)
import Data.Either     as X (either)
import Data.Hashable   as X (Hashable)
import Data.HashMap.Strict as X (HashMap)
import Data.HashSet        as X (HashSet)
import Data.List       as X (nub, intercalate)
import Data.Maybe      as X (fromMaybe, mapMaybe, catMaybes, isJust, Maybe)
import Data.Monoid     as X ((<>))
import Data.Text       as X (Text)
import Data.Void       as X (Void)
import Debug.Trace     as X
import Text.Megaparsec as X (ParseError, runParser, Parsec)
import GHC.Exts        as X (IsString(..))
import GHC.Generics    as X (Generic)

import qualified Data.Text as T

impossible :: Show msg => msg -> a
impossible x = error (show x)

tshow :: Show t => t -> Text
tshow = T.pack . show

tputStrLn :: Text -> IO ()
tputStrLn = putStrLn . T.unpack
