

-- -- |randomly initialize a tensor with uniform random values from a range
-- -- TODO - finish implementation to handle sizes correctly
-- randInit sz lower upper = do
--   gen <- c_THGenerator_new
--   t <- fromJust $ tensorNew sz
--   mapM_ (\x -> do
--             c_THDoubleTensor_uniform t gen lower upper
--             disp t
--         ) [0..3]
