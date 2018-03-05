module CodeGen.Prelude
  ( module X
  , impossible
  , tshow
  , tputStrLn
  ) where

import Prelude         as X
import Control.Monad   as X (guard)
import Data.List       as X (nub)
import Data.Maybe      as X (fromMaybe, mapMaybe, catMaybes, isJust, Maybe)
import Data.Either     as X (either)
import Data.Monoid     as X ((<>))
import Data.Text       as X (Text)
import Data.Void       as X (Void)
import Text.Megaparsec as X (ParseError, runParser, Parsec)
import Debug.Trace     as X
import Data.Hashable   as X (Hashable)
import GHC.Generics    as X (Generic)
import GHC.Exts        as X (IsString(..))
import Data.HashMap.Strict as X (HashMap)
import Data.HashSet        as X (HashSet)

import qualified Data.Text as T

impossible :: Show msg => msg -> a
impossible x = error (show x)

tshow :: Show t => t -> Text
tshow = T.pack . show

tputStrLn :: Text -> IO ()
tputStrLn = putStrLn . T.unpack
