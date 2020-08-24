module Torch.Data ( module Torch.Data.Pipeline
                  , module Torch.Data.StreamedPipeline
                  , module Torch.Data.Utils
                  , module Pipes
                  ) where

import Torch.Data.Pipeline
import Torch.Data.StreamedPipeline
import Torch.Data.Utils
import Pipes hiding (cat, embed, Proxy)
