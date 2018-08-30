-- Output of:
-- # valgrind --tool=memcheck ./dist-newstyle/build/x86_64-linux/ghc-8.4.3/hasktorch-core-0.0.1.0/x/memcheck/noopt/build/memcheck/memcheck

{-

==19639== Memcheck, a memory error detector
==19639== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==19639== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==19639== Command: ./dist-newstyle/build/x86_64-linux/ghc-8.4.3/hasktorch-core-0.0.1.0/x/memcheck/noopt/build/memcheck/memcheck
==19639==
==19639== Warning: set address range perms: large range [0x51da000, 0x17bd0000) (defined)
==19639== Warning: set address range perms: large range [0x26eee000, 0x3b56e000) (defined)
==19639== Warning: set address range perms: large range [0x4200000000, 0x14200100000) (noaccess)
(0x0000000000000000,0x000000004a488ac0)
[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,48.0,49.0,50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0,61.0,62.0,63.0,64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,80.0,81.0,82.0,83.0,84.0,85.0,86.0,87.0,88.0,89.0,90.0,91.0,92.0,93.0,94.0,95.0,96.0,97.0,98.0,99.0,100.0]
==19639== Invalid free() / delete / delete[] / realloc()
==19639==    at 0x4C30D3B: free (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==19639==    by 0x6167A94: THFree (in /home/stites/git/hasktorch/vendor/aten/build/lib/libATen.so.1)
==19639==    by 0x616800B: THDefaultAllocator_free (in /home/stites/git/hasktorch/vendor/aten/build/lib/libATen.so.1)
==19639==    by 0x6169855: THDoubleStorage_free (in /home/stites/git/hasktorch/vendor/aten/build/lib/libATen.so.1)
==19639==    by 0x43C4B9: free_DoubleStorage (finalizers.c:77)
==19639==    by 0x560735: runCFinalizers (in /home/stites/git/hasktorch/dist-newstyle/build/x86_64-linux/ghc-8.4.3/hasktorch-core-0.0.1.0/x/memcheck/noopt/build/memcheck/memcheck)
==19639==    by 0x560833: scheduleFinalizers (in /home/stites/git/hasktorch/dist-newstyle/build/x86_64-linux/ghc-8.4.3/hasktorch-core-0.0.1.0/x/memcheck/noopt/build/memcheck/memcheck)
==19639==    by 0x567BA8: GarbageCollect (in /home/stites/git/hasktorch/dist-newstyle/build/x86_64-linux/ghc-8.4.3/hasktorch-core-0.0.1.0/x/memcheck/noopt/build/memcheck/memcheck)
==19639==    by 0x5612DB: scheduleDoGC.constprop.23 (in /home/stites/git/hasktorch/dist-newstyle/build/x86_64-linux/ghc-8.4.3/hasktorch-core-0.0.1.0/x/memcheck/noopt/build/memcheck/memcheck)
==19639==    by 0x561475: performGC_ (in /home/stites/git/hasktorch/dist-newstyle/build/x86_64-linux/ghc-8.4.3/hasktorch-core-0.0.1.0/x/memcheck/noopt/build/memcheck/memcheck)
==19639==    by 0x4F800C: ??? (in /home/stites/git/hasktorch/dist-newstyle/build/x86_64-linux/ghc-8.4.3/hasktorch-core-0.0.1.0/x/memcheck/noopt/build/memcheck/memcheck)
==19639==  Address 0x420000c010 is in a rw- anonymous segment
==19639==
==19639==
==19639== HEAP SUMMARY:
==19639==     in use at exit: 2,295 bytes in 15 blocks
==19639==   total heap usage: 73,694 allocs, 73,680 frees, 6,638,289 bytes allocated
==19639==
==19639== LEAK SUMMARY:
==19639==    definitely lost: 0 bytes in 0 blocks
==19639==    indirectly lost: 0 bytes in 0 blocks
==19639==      possibly lost: 0 bytes in 0 blocks
==19639==    still reachable: 2,295 bytes in 15 blocks
==19639==         suppressed: 0 bytes in 0 blocks
==19639== Rerun with --leak-check=full to see details of leaked memory
==19639==
==19639== For counts of detected and suppressed errors, rerun with: -v
==19639== ERROR SUMMARY: 1 errors from 1 contexts (suppressed: 0 from 0)

-}

{-# LANGUAGE DataKinds #-}
module Main where

import Torch.Double.Storage
import System.Mem

-- TODO: test GC with different formats:
main = do
  -- let Just s = vector [0..100] :: Maybe (Tensor '[101])
  -- let s = vector [0..100] :: Dynamic
  -- print s

  s <- fromList [0..100]
  print $ storageState s
  tensordata s >>= print
  performGC

  -- fromList [10..1990] >>= tensordata >>= print
  -- performMajorGC


