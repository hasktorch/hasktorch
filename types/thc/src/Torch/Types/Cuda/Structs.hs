{-# OPTIONS_GHC -fno-warn-unused-imports #-}


module Torch.Types.Cuda.Structs where
import Foreign.Ptr
import Foreign.Ptr (Ptr,FunPtr,plusPtr)
import Foreign.Ptr (wordPtrToPtr,castPtrToFunPtr)
import Foreign.Storable
import Foreign.C.Types
import Foreign.C.String (CString,CStringLen,CWString,CWStringLen)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (peekArray,pokeArray)
import Data.Int
import Data.Word


{- typedef struct cudaDeviceProp {
            char name[256];
            size_t totalGlobalMem;
            size_t sharedMemPerBlock;
            int regsPerBlock;
            int warpSize;
            size_t memPitch;
            int maxThreadsPerBlock;
            int maxThreadsDim[3];
            int maxGridSize[3];
            int clockRate;
            size_t totalConstMem;
            int major;
            int minor;
            size_t textureAlignment;
            size_t texturePitchAlignment;
            int deviceOverlap;
            int multiProcessorCount;
            int kernelExecTimeoutEnabled;
            int integrated;
            int canMapHostMemory;
            int computeMode;
            int maxTexture1D;
            int maxTexture1DMipmap;
            int maxTexture1DLinear;
            int maxTexture2D[2];
            int maxTexture2DMipmap[2];
            int maxTexture2DLinear[3];
            int maxTexture2DGather[2];
            int maxTexture3D[3];
            int maxTexture3DAlt[3];
            int maxTextureCubemap;
            int maxTexture1DLayered[2];
            int maxTexture2DLayered[3];
            int maxTextureCubemapLayered[2];
            int maxSurface1D;
            int maxSurface2D[2];
            int maxSurface3D[3];
            int maxSurface1DLayered[2];
            int maxSurface2DLayered[3];
            int maxSurfaceCubemap;
            int maxSurfaceCubemapLayered[2];
            size_t surfaceAlignment;
            int concurrentKernels;
            int ECCEnabled;
            int pciBusID;
            int pciDeviceID;
            int pciDomainID;
            int tccDriver;
            int asyncEngineCount;
            int unifiedAddressing;
            int memoryClockRate;
            int memoryBusWidth;
            int l2CacheSize;
            int maxThreadsPerMultiProcessor;
            int streamPrioritiesSupported;
            int globalL1CacheSupported;
            int localL1CacheSupported;
            size_t sharedMemPerMultiprocessor;
            int regsPerMultiprocessor;
            int managedMemSupported;
            int isMultiGpuBoard;
            int multiGpuBoardGroupID;
            int singleToDoublePrecisionPerfRatio;
            int pageableMemoryAccess;
            int concurrentManagedAccess;
            int computePreemptionSupported;
            int canUseHostPointerForRegisteredMem;
            int cooperativeLaunch;
            int cooperativeMultiDeviceLaunch;
        } cudaDeviceProp; -}






































































data C'cudaDeviceProp = C'cudaDeviceProp{
  c'cudaDeviceProp'name :: [CChar],
  c'cudaDeviceProp'totalGlobalMem :: CSize,
  c'cudaDeviceProp'sharedMemPerBlock :: CSize,
  c'cudaDeviceProp'regsPerBlock :: CInt,
  c'cudaDeviceProp'warpSize :: CInt,
  c'cudaDeviceProp'memPitch :: CSize,
  c'cudaDeviceProp'maxThreadsPerBlock :: CInt,
  c'cudaDeviceProp'maxThreadsDim :: [CInt],
  c'cudaDeviceProp'maxGridSize :: [CInt],
  c'cudaDeviceProp'clockRate :: CInt,
  c'cudaDeviceProp'totalConstMem :: CSize,
  c'cudaDeviceProp'major :: CInt,
  c'cudaDeviceProp'minor :: CInt,
  c'cudaDeviceProp'textureAlignment :: CSize,
  c'cudaDeviceProp'texturePitchAlignment :: CSize,
  c'cudaDeviceProp'deviceOverlap :: CInt,
  c'cudaDeviceProp'multiProcessorCount :: CInt,
  c'cudaDeviceProp'kernelExecTimeoutEnabled :: CInt,
  c'cudaDeviceProp'integrated :: CInt,
  c'cudaDeviceProp'canMapHostMemory :: CInt,
  c'cudaDeviceProp'computeMode :: CInt,
  c'cudaDeviceProp'maxTexture1D :: CInt,
  c'cudaDeviceProp'maxTexture1DMipmap :: CInt,
  c'cudaDeviceProp'maxTexture1DLinear :: CInt,
  c'cudaDeviceProp'maxTexture2D :: [CInt],
  c'cudaDeviceProp'maxTexture2DMipmap :: [CInt],
  c'cudaDeviceProp'maxTexture2DLinear :: [CInt],
  c'cudaDeviceProp'maxTexture2DGather :: [CInt],
  c'cudaDeviceProp'maxTexture3D :: [CInt],
  c'cudaDeviceProp'maxTexture3DAlt :: [CInt],
  c'cudaDeviceProp'maxTextureCubemap :: CInt,
  c'cudaDeviceProp'maxTexture1DLayered :: [CInt],
  c'cudaDeviceProp'maxTexture2DLayered :: [CInt],
  c'cudaDeviceProp'maxTextureCubemapLayered :: [CInt],
  c'cudaDeviceProp'maxSurface1D :: CInt,
  c'cudaDeviceProp'maxSurface2D :: [CInt],
  c'cudaDeviceProp'maxSurface3D :: [CInt],
  c'cudaDeviceProp'maxSurface1DLayered :: [CInt],
  c'cudaDeviceProp'maxSurface2DLayered :: [CInt],
  c'cudaDeviceProp'maxSurfaceCubemap :: CInt,
  c'cudaDeviceProp'maxSurfaceCubemapLayered :: [CInt],
  c'cudaDeviceProp'surfaceAlignment :: CSize,
  c'cudaDeviceProp'concurrentKernels :: CInt,
  c'cudaDeviceProp'ECCEnabled :: CInt,
  c'cudaDeviceProp'pciBusID :: CInt,
  c'cudaDeviceProp'pciDeviceID :: CInt,
  c'cudaDeviceProp'pciDomainID :: CInt,
  c'cudaDeviceProp'tccDriver :: CInt,
  c'cudaDeviceProp'asyncEngineCount :: CInt,
  c'cudaDeviceProp'unifiedAddressing :: CInt,
  c'cudaDeviceProp'memoryClockRate :: CInt,
  c'cudaDeviceProp'memoryBusWidth :: CInt,
  c'cudaDeviceProp'l2CacheSize :: CInt,
  c'cudaDeviceProp'maxThreadsPerMultiProcessor :: CInt,
  c'cudaDeviceProp'streamPrioritiesSupported :: CInt,
  c'cudaDeviceProp'globalL1CacheSupported :: CInt,
  c'cudaDeviceProp'localL1CacheSupported :: CInt,
  c'cudaDeviceProp'sharedMemPerMultiprocessor :: CSize,
  c'cudaDeviceProp'regsPerMultiprocessor :: CInt,
  c'cudaDeviceProp'managedMemSupported :: CInt,
  c'cudaDeviceProp'isMultiGpuBoard :: CInt,
  c'cudaDeviceProp'multiGpuBoardGroupID :: CInt,
  c'cudaDeviceProp'singleToDoublePrecisionPerfRatio :: CInt,
  c'cudaDeviceProp'pageableMemoryAccess :: CInt,
  c'cudaDeviceProp'concurrentManagedAccess :: CInt,
  c'cudaDeviceProp'computePreemptionSupported :: CInt,
  c'cudaDeviceProp'canUseHostPointerForRegisteredMem :: CInt,
  c'cudaDeviceProp'cooperativeLaunch :: CInt,
  c'cudaDeviceProp'cooperativeMultiDeviceLaunch :: CInt
} deriving (Eq,Show)
p'cudaDeviceProp'name p = plusPtr p 0
p'cudaDeviceProp'name :: Ptr (C'cudaDeviceProp) -> Ptr (CChar)
p'cudaDeviceProp'totalGlobalMem p = plusPtr p 256
p'cudaDeviceProp'totalGlobalMem :: Ptr (C'cudaDeviceProp) -> Ptr (CSize)
p'cudaDeviceProp'sharedMemPerBlock p = plusPtr p 264
p'cudaDeviceProp'sharedMemPerBlock :: Ptr (C'cudaDeviceProp) -> Ptr (CSize)
p'cudaDeviceProp'regsPerBlock p = plusPtr p 272
p'cudaDeviceProp'regsPerBlock :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'warpSize p = plusPtr p 276
p'cudaDeviceProp'warpSize :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'memPitch p = plusPtr p 280
p'cudaDeviceProp'memPitch :: Ptr (C'cudaDeviceProp) -> Ptr (CSize)
p'cudaDeviceProp'maxThreadsPerBlock p = plusPtr p 288
p'cudaDeviceProp'maxThreadsPerBlock :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxThreadsDim p = plusPtr p 292
p'cudaDeviceProp'maxThreadsDim :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxGridSize p = plusPtr p 304
p'cudaDeviceProp'maxGridSize :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'clockRate p = plusPtr p 316
p'cudaDeviceProp'clockRate :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'totalConstMem p = plusPtr p 320
p'cudaDeviceProp'totalConstMem :: Ptr (C'cudaDeviceProp) -> Ptr (CSize)
p'cudaDeviceProp'major p = plusPtr p 328
p'cudaDeviceProp'major :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'minor p = plusPtr p 332
p'cudaDeviceProp'minor :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'textureAlignment p = plusPtr p 336
p'cudaDeviceProp'textureAlignment :: Ptr (C'cudaDeviceProp) -> Ptr (CSize)
p'cudaDeviceProp'texturePitchAlignment p = plusPtr p 344
p'cudaDeviceProp'texturePitchAlignment :: Ptr (C'cudaDeviceProp) -> Ptr (CSize)
p'cudaDeviceProp'deviceOverlap p = plusPtr p 352
p'cudaDeviceProp'deviceOverlap :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'multiProcessorCount p = plusPtr p 356
p'cudaDeviceProp'multiProcessorCount :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'kernelExecTimeoutEnabled p = plusPtr p 360
p'cudaDeviceProp'kernelExecTimeoutEnabled :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'integrated p = plusPtr p 364
p'cudaDeviceProp'integrated :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'canMapHostMemory p = plusPtr p 368
p'cudaDeviceProp'canMapHostMemory :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'computeMode p = plusPtr p 372
p'cudaDeviceProp'computeMode :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTexture1D p = plusPtr p 376
p'cudaDeviceProp'maxTexture1D :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTexture1DMipmap p = plusPtr p 380
p'cudaDeviceProp'maxTexture1DMipmap :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTexture1DLinear p = plusPtr p 384
p'cudaDeviceProp'maxTexture1DLinear :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTexture2D p = plusPtr p 388
p'cudaDeviceProp'maxTexture2D :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTexture2DMipmap p = plusPtr p 396
p'cudaDeviceProp'maxTexture2DMipmap :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTexture2DLinear p = plusPtr p 404
p'cudaDeviceProp'maxTexture2DLinear :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTexture2DGather p = plusPtr p 416
p'cudaDeviceProp'maxTexture2DGather :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTexture3D p = plusPtr p 424
p'cudaDeviceProp'maxTexture3D :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTexture3DAlt p = plusPtr p 436
p'cudaDeviceProp'maxTexture3DAlt :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTextureCubemap p = plusPtr p 448
p'cudaDeviceProp'maxTextureCubemap :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTexture1DLayered p = plusPtr p 452
p'cudaDeviceProp'maxTexture1DLayered :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTexture2DLayered p = plusPtr p 460
p'cudaDeviceProp'maxTexture2DLayered :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxTextureCubemapLayered p = plusPtr p 472
p'cudaDeviceProp'maxTextureCubemapLayered :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxSurface1D p = plusPtr p 480
p'cudaDeviceProp'maxSurface1D :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxSurface2D p = plusPtr p 484
p'cudaDeviceProp'maxSurface2D :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxSurface3D p = plusPtr p 492
p'cudaDeviceProp'maxSurface3D :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxSurface1DLayered p = plusPtr p 504
p'cudaDeviceProp'maxSurface1DLayered :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxSurface2DLayered p = plusPtr p 512
p'cudaDeviceProp'maxSurface2DLayered :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxSurfaceCubemap p = plusPtr p 524
p'cudaDeviceProp'maxSurfaceCubemap :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxSurfaceCubemapLayered p = plusPtr p 528
p'cudaDeviceProp'maxSurfaceCubemapLayered :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'surfaceAlignment p = plusPtr p 536
p'cudaDeviceProp'surfaceAlignment :: Ptr (C'cudaDeviceProp) -> Ptr (CSize)
p'cudaDeviceProp'concurrentKernels p = plusPtr p 544
p'cudaDeviceProp'concurrentKernels :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'ECCEnabled p = plusPtr p 548
p'cudaDeviceProp'ECCEnabled :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'pciBusID p = plusPtr p 552
p'cudaDeviceProp'pciBusID :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'pciDeviceID p = plusPtr p 556
p'cudaDeviceProp'pciDeviceID :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'pciDomainID p = plusPtr p 560
p'cudaDeviceProp'pciDomainID :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'tccDriver p = plusPtr p 564
p'cudaDeviceProp'tccDriver :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'asyncEngineCount p = plusPtr p 568
p'cudaDeviceProp'asyncEngineCount :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'unifiedAddressing p = plusPtr p 572
p'cudaDeviceProp'unifiedAddressing :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'memoryClockRate p = plusPtr p 576
p'cudaDeviceProp'memoryClockRate :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'memoryBusWidth p = plusPtr p 580
p'cudaDeviceProp'memoryBusWidth :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'l2CacheSize p = plusPtr p 584
p'cudaDeviceProp'l2CacheSize :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'maxThreadsPerMultiProcessor p = plusPtr p 588
p'cudaDeviceProp'maxThreadsPerMultiProcessor :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'streamPrioritiesSupported p = plusPtr p 592
p'cudaDeviceProp'streamPrioritiesSupported :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'globalL1CacheSupported p = plusPtr p 596
p'cudaDeviceProp'globalL1CacheSupported :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'localL1CacheSupported p = plusPtr p 600
p'cudaDeviceProp'localL1CacheSupported :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'sharedMemPerMultiprocessor p = plusPtr p 608
p'cudaDeviceProp'sharedMemPerMultiprocessor :: Ptr (C'cudaDeviceProp) -> Ptr (CSize)
p'cudaDeviceProp'regsPerMultiprocessor p = plusPtr p 616
p'cudaDeviceProp'regsPerMultiprocessor :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'managedMemSupported p = plusPtr p 620
p'cudaDeviceProp'managedMemSupported :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'isMultiGpuBoard p = plusPtr p 624
p'cudaDeviceProp'isMultiGpuBoard :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'multiGpuBoardGroupID p = plusPtr p 628
p'cudaDeviceProp'multiGpuBoardGroupID :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'singleToDoublePrecisionPerfRatio p = plusPtr p 632
p'cudaDeviceProp'singleToDoublePrecisionPerfRatio :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'pageableMemoryAccess p = plusPtr p 636
p'cudaDeviceProp'pageableMemoryAccess :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'concurrentManagedAccess p = plusPtr p 640
p'cudaDeviceProp'concurrentManagedAccess :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'computePreemptionSupported p = plusPtr p 644
p'cudaDeviceProp'computePreemptionSupported :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'canUseHostPointerForRegisteredMem p = plusPtr p 648
p'cudaDeviceProp'canUseHostPointerForRegisteredMem :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'cooperativeLaunch p = plusPtr p 652
p'cudaDeviceProp'cooperativeLaunch :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
p'cudaDeviceProp'cooperativeMultiDeviceLaunch p = plusPtr p 656
p'cudaDeviceProp'cooperativeMultiDeviceLaunch :: Ptr (C'cudaDeviceProp) -> Ptr (CInt)
instance Storable C'cudaDeviceProp where
  sizeOf _ = 664
  alignment _ = 8
  peek _p = do
    v0 <- let s0 = div 256 $ sizeOf $ (undefined :: CChar) in peekArray s0 (plusPtr _p 0)
    v1 <- peekByteOff _p 256
    v2 <- peekByteOff _p 264
    v3 <- peekByteOff _p 272
    v4 <- peekByteOff _p 276
    v5 <- peekByteOff _p 280
    v6 <- peekByteOff _p 288
    v7 <- let s7 = div 12 $ sizeOf $ (undefined :: CInt) in peekArray s7 (plusPtr _p 292)
    v8 <- let s8 = div 12 $ sizeOf $ (undefined :: CInt) in peekArray s8 (plusPtr _p 304)
    v9 <- peekByteOff _p 316
    v10 <- peekByteOff _p 320
    v11 <- peekByteOff _p 328
    v12 <- peekByteOff _p 332
    v13 <- peekByteOff _p 336
    v14 <- peekByteOff _p 344
    v15 <- peekByteOff _p 352
    v16 <- peekByteOff _p 356
    v17 <- peekByteOff _p 360
    v18 <- peekByteOff _p 364
    v19 <- peekByteOff _p 368
    v20 <- peekByteOff _p 372
    v21 <- peekByteOff _p 376
    v22 <- peekByteOff _p 380
    v23 <- peekByteOff _p 384
    v24 <- let s24 = div 8 $ sizeOf $ (undefined :: CInt) in peekArray s24 (plusPtr _p 388)
    v25 <- let s25 = div 8 $ sizeOf $ (undefined :: CInt) in peekArray s25 (plusPtr _p 396)
    v26 <- let s26 = div 12 $ sizeOf $ (undefined :: CInt) in peekArray s26 (plusPtr _p 404)
    v27 <- let s27 = div 8 $ sizeOf $ (undefined :: CInt) in peekArray s27 (plusPtr _p 416)
    v28 <- let s28 = div 12 $ sizeOf $ (undefined :: CInt) in peekArray s28 (plusPtr _p 424)
    v29 <- let s29 = div 12 $ sizeOf $ (undefined :: CInt) in peekArray s29 (plusPtr _p 436)
    v30 <- peekByteOff _p 448
    v31 <- let s31 = div 8 $ sizeOf $ (undefined :: CInt) in peekArray s31 (plusPtr _p 452)
    v32 <- let s32 = div 12 $ sizeOf $ (undefined :: CInt) in peekArray s32 (plusPtr _p 460)
    v33 <- let s33 = div 8 $ sizeOf $ (undefined :: CInt) in peekArray s33 (plusPtr _p 472)
    v34 <- peekByteOff _p 480
    v35 <- let s35 = div 8 $ sizeOf $ (undefined :: CInt) in peekArray s35 (plusPtr _p 484)
    v36 <- let s36 = div 12 $ sizeOf $ (undefined :: CInt) in peekArray s36 (plusPtr _p 492)
    v37 <- let s37 = div 8 $ sizeOf $ (undefined :: CInt) in peekArray s37 (plusPtr _p 504)
    v38 <- let s38 = div 12 $ sizeOf $ (undefined :: CInt) in peekArray s38 (plusPtr _p 512)
    v39 <- peekByteOff _p 524
    v40 <- let s40 = div 8 $ sizeOf $ (undefined :: CInt) in peekArray s40 (plusPtr _p 528)
    v41 <- peekByteOff _p 536
    v42 <- peekByteOff _p 544
    v43 <- peekByteOff _p 548
    v44 <- peekByteOff _p 552
    v45 <- peekByteOff _p 556
    v46 <- peekByteOff _p 560
    v47 <- peekByteOff _p 564
    v48 <- peekByteOff _p 568
    v49 <- peekByteOff _p 572
    v50 <- peekByteOff _p 576
    v51 <- peekByteOff _p 580
    v52 <- peekByteOff _p 584
    v53 <- peekByteOff _p 588
    v54 <- peekByteOff _p 592
    v55 <- peekByteOff _p 596
    v56 <- peekByteOff _p 600
    v57 <- peekByteOff _p 608
    v58 <- peekByteOff _p 616
    v59 <- peekByteOff _p 620
    v60 <- peekByteOff _p 624
    v61 <- peekByteOff _p 628
    v62 <- peekByteOff _p 632
    v63 <- peekByteOff _p 636
    v64 <- peekByteOff _p 640
    v65 <- peekByteOff _p 644
    v66 <- peekByteOff _p 648
    v67 <- peekByteOff _p 652
    v68 <- peekByteOff _p 656
    return $ C'cudaDeviceProp v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68
  poke _p (C'cudaDeviceProp v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68) = do
    let s0 = div 256 $ sizeOf $ (undefined :: CChar)
    pokeArray (plusPtr _p 0) (take s0 v0)
    pokeByteOff _p 256 v1
    pokeByteOff _p 264 v2
    pokeByteOff _p 272 v3
    pokeByteOff _p 276 v4
    pokeByteOff _p 280 v5
    pokeByteOff _p 288 v6
    let s7 = div 12 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 292) (take s7 v7)
    let s8 = div 12 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 304) (take s8 v8)
    pokeByteOff _p 316 v9
    pokeByteOff _p 320 v10
    pokeByteOff _p 328 v11
    pokeByteOff _p 332 v12
    pokeByteOff _p 336 v13
    pokeByteOff _p 344 v14
    pokeByteOff _p 352 v15
    pokeByteOff _p 356 v16
    pokeByteOff _p 360 v17
    pokeByteOff _p 364 v18
    pokeByteOff _p 368 v19
    pokeByteOff _p 372 v20
    pokeByteOff _p 376 v21
    pokeByteOff _p 380 v22
    pokeByteOff _p 384 v23
    let s24 = div 8 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 388) (take s24 v24)
    let s25 = div 8 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 396) (take s25 v25)
    let s26 = div 12 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 404) (take s26 v26)
    let s27 = div 8 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 416) (take s27 v27)
    let s28 = div 12 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 424) (take s28 v28)
    let s29 = div 12 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 436) (take s29 v29)
    pokeByteOff _p 448 v30
    let s31 = div 8 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 452) (take s31 v31)
    let s32 = div 12 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 460) (take s32 v32)
    let s33 = div 8 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 472) (take s33 v33)
    pokeByteOff _p 480 v34
    let s35 = div 8 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 484) (take s35 v35)
    let s36 = div 12 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 492) (take s36 v36)
    let s37 = div 8 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 504) (take s37 v37)
    let s38 = div 12 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 512) (take s38 v38)
    pokeByteOff _p 524 v39
    let s40 = div 8 $ sizeOf $ (undefined :: CInt)
    pokeArray (plusPtr _p 528) (take s40 v40)
    pokeByteOff _p 536 v41
    pokeByteOff _p 544 v42
    pokeByteOff _p 548 v43
    pokeByteOff _p 552 v44
    pokeByteOff _p 556 v45
    pokeByteOff _p 560 v46
    pokeByteOff _p 564 v47
    pokeByteOff _p 568 v48
    pokeByteOff _p 572 v49
    pokeByteOff _p 576 v50
    pokeByteOff _p 580 v51
    pokeByteOff _p 584 v52
    pokeByteOff _p 588 v53
    pokeByteOff _p 592 v54
    pokeByteOff _p 596 v55
    pokeByteOff _p 600 v56
    pokeByteOff _p 608 v57
    pokeByteOff _p 616 v58
    pokeByteOff _p 620 v59
    pokeByteOff _p 624 v60
    pokeByteOff _p 628 v61
    pokeByteOff _p 632 v62
    pokeByteOff _p 636 v63
    pokeByteOff _p 640 v64
    pokeByteOff _p 644 v65
    pokeByteOff _p 648 v66
    pokeByteOff _p 652 v67
    pokeByteOff _p 656 v68
    return ()


{- typedef enum cudaError_t {
            cudaSuccess = 0,
            cudaErrorMissingConfiguration = 1,
            cudaErrorMemoryAllocation = 2,
            cudaErrorInitializationError = 3,
            cudaErrorLaunchFailure = 4,
            cudaErrorPriorLaunchFailure = 5,
            cudaErrorLaunchTimeout = 6,
            cudaErrorLaunchOutOfResources = 7,
            cudaErrorInvalidDeviceFunction = 8,
            cudaErrorInvalidConfiguration = 9,
            cudaErrorInvalidDevice = 10,
            cudaErrorInvalidValue = 11,
            cudaErrorInvalidPitchValue = 12,
            cudaErrorInvalidSymbol = 13,
            cudaErrorMapBufferObjectFailed = 14,
            cudaErrorUnmapBufferObjectFailed = 15,
            cudaErrorInvalidHostPointer = 16,
            cudaErrorInvalidDevicePointer = 17,
            cudaErrorInvalidTexture = 18,
            cudaErrorInvalidTextureBinding = 19,
            cudaErrorInvalidChannelDescriptor = 20,
            cudaErrorInvalidMemcpyDirection = 21,
            cudaErrorAddressOfConstant = 22,
            cudaErrorTextureFetchFailed = 23,
            cudaErrorTextureNotBound = 24,
            cudaErrorSynchronizationError = 25,
            cudaErrorInvalidFilterSetting = 26,
            cudaErrorInvalidNormSetting = 27,
            cudaErrorMixedDeviceExecution = 28,
            cudaErrorCudartUnloading = 29,
            cudaErrorUnknown = 30,
            cudaErrorNotYetImplemented = 31,
            cudaErrorMemoryValueTooLarge = 32,
            cudaErrorInvalidResourceHandle = 33,
            cudaErrorNotReady = 34,
            cudaErrorInsufficientDriver = 35,
            cudaErrorSetOnActiveProcess = 36,
            cudaErrorInvalidSurface = 37,
            cudaErrorNoDevice = 38,
            cudaErrorECCUncorrectable = 39,
            cudaErrorSharedObjectSymbolNotFound = 40,
            cudaErrorSharedObjectInitFailed = 41,
            cudaErrorUnsupportedLimit = 42,
            cudaErrorDuplicateVariableName = 43,
            cudaErrorDuplicateTextureName = 44,
            cudaErrorDuplicateSurfaceName = 45,
            cudaErrorDevicesUnavailable = 46,
            cudaErrorInvalidKernelImage = 47,
            cudaErrorNoKernelImageForDevice = 48,
            cudaErrorIncompatibleDriverContext = 49,
            cudaErrorPeerAccessAlreadyEnabled = 50,
            cudaErrorPeerAccessNotEnabled = 51,
            cudaErrorDeviceAlreadyInUse = 54,
            cudaErrorProfilerDisabled = 55,
            cudaErrorProfilerNotInitialized = 56,
            cudaErrorProfilerAlreadyStarted = 57,
            cudaErrorProfilerAlreadyStopped = 58,
            cudaErrorAssert = 59,
            cudaErrorTooManyPeers = 60,
            cudaErrorHostMemoryAlreadyRegistered = 61,
            cudaErrorHostMemoryNotRegistered = 62,
            cudaErrorOperatingSystem = 63,
            cudaErrorPeerAccessUnsupported = 64,
            cudaErrorLaunchMaxDepthExceeded = 65,
            cudaErrorLaunchFileScopedTex = 66,
            cudaErrorLaunchFileScopedSurf = 67,
            cudaErrorSyncDepthExceeded = 68,
            cudaErrorLaunchPendingCountExceeded = 69,
            cudaErrorNotPermitted = 70,
            cudaErrorNotSupported = 71,
            cudaErrorHardwareStackError = 72,
            cudaErrorIllegalInstruction = 73,
            cudaErrorMisalignedAddress = 74,
            cudaErrorInvalidAddressSpace = 75,
            cudaErrorInvalidPc = 76,
            cudaErrorIllegalAddress = 77,
            cudaErrorInvalidPtx = 78,
            cudaErrorInvalidGraphicsContext = 79,
            cudaErrorNvlinkUncorrectable = 80,
            cudaErrorJitCompilerNotFound = 81,
            cudaErrorCooperativeLaunchTooLarge = 82,
            cudaErrorStartupFailure = 0x7f,
            cudaErrorApiFailureBase = 10000
        } cudaError_t; -}
type C'cudaError_t = CUInt

c'cudaSuccess = 0
c'cudaSuccess :: (Num a) => a

c'cudaErrorMissingConfiguration = 1
c'cudaErrorMissingConfiguration :: (Num a) => a

c'cudaErrorMemoryAllocation = 2
c'cudaErrorMemoryAllocation :: (Num a) => a

c'cudaErrorInitializationError = 3
c'cudaErrorInitializationError :: (Num a) => a

c'cudaErrorLaunchFailure = 4
c'cudaErrorLaunchFailure :: (Num a) => a

c'cudaErrorPriorLaunchFailure = 5
c'cudaErrorPriorLaunchFailure :: (Num a) => a

c'cudaErrorLaunchTimeout = 6
c'cudaErrorLaunchTimeout :: (Num a) => a

c'cudaErrorLaunchOutOfResources = 7
c'cudaErrorLaunchOutOfResources :: (Num a) => a

c'cudaErrorInvalidDeviceFunction = 8
c'cudaErrorInvalidDeviceFunction :: (Num a) => a

c'cudaErrorInvalidConfiguration = 9
c'cudaErrorInvalidConfiguration :: (Num a) => a

c'cudaErrorInvalidDevice = 10
c'cudaErrorInvalidDevice :: (Num a) => a

c'cudaErrorInvalidValue = 11
c'cudaErrorInvalidValue :: (Num a) => a

c'cudaErrorInvalidPitchValue = 12
c'cudaErrorInvalidPitchValue :: (Num a) => a

c'cudaErrorInvalidSymbol = 13
c'cudaErrorInvalidSymbol :: (Num a) => a

c'cudaErrorMapBufferObjectFailed = 14
c'cudaErrorMapBufferObjectFailed :: (Num a) => a

c'cudaErrorUnmapBufferObjectFailed = 15
c'cudaErrorUnmapBufferObjectFailed :: (Num a) => a

c'cudaErrorInvalidHostPointer = 16
c'cudaErrorInvalidHostPointer :: (Num a) => a

c'cudaErrorInvalidDevicePointer = 17
c'cudaErrorInvalidDevicePointer :: (Num a) => a

c'cudaErrorInvalidTexture = 18
c'cudaErrorInvalidTexture :: (Num a) => a

c'cudaErrorInvalidTextureBinding = 19
c'cudaErrorInvalidTextureBinding :: (Num a) => a

c'cudaErrorInvalidChannelDescriptor = 20
c'cudaErrorInvalidChannelDescriptor :: (Num a) => a

c'cudaErrorInvalidMemcpyDirection = 21
c'cudaErrorInvalidMemcpyDirection :: (Num a) => a

c'cudaErrorAddressOfConstant = 22
c'cudaErrorAddressOfConstant :: (Num a) => a

c'cudaErrorTextureFetchFailed = 23
c'cudaErrorTextureFetchFailed :: (Num a) => a

c'cudaErrorTextureNotBound = 24
c'cudaErrorTextureNotBound :: (Num a) => a

c'cudaErrorSynchronizationError = 25
c'cudaErrorSynchronizationError :: (Num a) => a

c'cudaErrorInvalidFilterSetting = 26
c'cudaErrorInvalidFilterSetting :: (Num a) => a

c'cudaErrorInvalidNormSetting = 27
c'cudaErrorInvalidNormSetting :: (Num a) => a

c'cudaErrorMixedDeviceExecution = 28
c'cudaErrorMixedDeviceExecution :: (Num a) => a

c'cudaErrorCudartUnloading = 29
c'cudaErrorCudartUnloading :: (Num a) => a

c'cudaErrorUnknown = 30
c'cudaErrorUnknown :: (Num a) => a

c'cudaErrorNotYetImplemented = 31
c'cudaErrorNotYetImplemented :: (Num a) => a

c'cudaErrorMemoryValueTooLarge = 32
c'cudaErrorMemoryValueTooLarge :: (Num a) => a

c'cudaErrorInvalidResourceHandle = 33
c'cudaErrorInvalidResourceHandle :: (Num a) => a

c'cudaErrorNotReady = 34
c'cudaErrorNotReady :: (Num a) => a

c'cudaErrorInsufficientDriver = 35
c'cudaErrorInsufficientDriver :: (Num a) => a

c'cudaErrorSetOnActiveProcess = 36
c'cudaErrorSetOnActiveProcess :: (Num a) => a

c'cudaErrorInvalidSurface = 37
c'cudaErrorInvalidSurface :: (Num a) => a

c'cudaErrorNoDevice = 38
c'cudaErrorNoDevice :: (Num a) => a

c'cudaErrorECCUncorrectable = 39
c'cudaErrorECCUncorrectable :: (Num a) => a

c'cudaErrorSharedObjectSymbolNotFound = 40
c'cudaErrorSharedObjectSymbolNotFound :: (Num a) => a

c'cudaErrorSharedObjectInitFailed = 41
c'cudaErrorSharedObjectInitFailed :: (Num a) => a

c'cudaErrorUnsupportedLimit = 42
c'cudaErrorUnsupportedLimit :: (Num a) => a

c'cudaErrorDuplicateVariableName = 43
c'cudaErrorDuplicateVariableName :: (Num a) => a

c'cudaErrorDuplicateTextureName = 44
c'cudaErrorDuplicateTextureName :: (Num a) => a

c'cudaErrorDuplicateSurfaceName = 45
c'cudaErrorDuplicateSurfaceName :: (Num a) => a

c'cudaErrorDevicesUnavailable = 46
c'cudaErrorDevicesUnavailable :: (Num a) => a

c'cudaErrorInvalidKernelImage = 47
c'cudaErrorInvalidKernelImage :: (Num a) => a

c'cudaErrorNoKernelImageForDevice = 48
c'cudaErrorNoKernelImageForDevice :: (Num a) => a

c'cudaErrorIncompatibleDriverContext = 49
c'cudaErrorIncompatibleDriverContext :: (Num a) => a

c'cudaErrorPeerAccessAlreadyEnabled = 50
c'cudaErrorPeerAccessAlreadyEnabled :: (Num a) => a

c'cudaErrorPeerAccessNotEnabled = 51
c'cudaErrorPeerAccessNotEnabled :: (Num a) => a

c'cudaErrorDeviceAlreadyInUse = 54
c'cudaErrorDeviceAlreadyInUse :: (Num a) => a

c'cudaErrorProfilerDisabled = 55
c'cudaErrorProfilerDisabled :: (Num a) => a

c'cudaErrorProfilerNotInitialized = 56
c'cudaErrorProfilerNotInitialized :: (Num a) => a

c'cudaErrorProfilerAlreadyStarted = 57
c'cudaErrorProfilerAlreadyStarted :: (Num a) => a

c'cudaErrorProfilerAlreadyStopped = 58
c'cudaErrorProfilerAlreadyStopped :: (Num a) => a

c'cudaErrorAssert = 59
c'cudaErrorAssert :: (Num a) => a

c'cudaErrorTooManyPeers = 60
c'cudaErrorTooManyPeers :: (Num a) => a

c'cudaErrorHostMemoryAlreadyRegistered = 61
c'cudaErrorHostMemoryAlreadyRegistered :: (Num a) => a

c'cudaErrorHostMemoryNotRegistered = 62
c'cudaErrorHostMemoryNotRegistered :: (Num a) => a

c'cudaErrorOperatingSystem = 63
c'cudaErrorOperatingSystem :: (Num a) => a

c'cudaErrorPeerAccessUnsupported = 64
c'cudaErrorPeerAccessUnsupported :: (Num a) => a

c'cudaErrorLaunchMaxDepthExceeded = 65
c'cudaErrorLaunchMaxDepthExceeded :: (Num a) => a

c'cudaErrorLaunchFileScopedTex = 66
c'cudaErrorLaunchFileScopedTex :: (Num a) => a

c'cudaErrorLaunchFileScopedSurf = 67
c'cudaErrorLaunchFileScopedSurf :: (Num a) => a

c'cudaErrorSyncDepthExceeded = 68
c'cudaErrorSyncDepthExceeded :: (Num a) => a

c'cudaErrorLaunchPendingCountExceeded = 69
c'cudaErrorLaunchPendingCountExceeded :: (Num a) => a

c'cudaErrorNotPermitted = 70
c'cudaErrorNotPermitted :: (Num a) => a

c'cudaErrorNotSupported = 71
c'cudaErrorNotSupported :: (Num a) => a

c'cudaErrorHardwareStackError = 72
c'cudaErrorHardwareStackError :: (Num a) => a

c'cudaErrorIllegalInstruction = 73
c'cudaErrorIllegalInstruction :: (Num a) => a

c'cudaErrorMisalignedAddress = 74
c'cudaErrorMisalignedAddress :: (Num a) => a

c'cudaErrorInvalidAddressSpace = 75
c'cudaErrorInvalidAddressSpace :: (Num a) => a

c'cudaErrorInvalidPc = 76
c'cudaErrorInvalidPc :: (Num a) => a

c'cudaErrorIllegalAddress = 77
c'cudaErrorIllegalAddress :: (Num a) => a

c'cudaErrorInvalidPtx = 78
c'cudaErrorInvalidPtx :: (Num a) => a

c'cudaErrorInvalidGraphicsContext = 79
c'cudaErrorInvalidGraphicsContext :: (Num a) => a

c'cudaErrorNvlinkUncorrectable = 80
c'cudaErrorNvlinkUncorrectable :: (Num a) => a

c'cudaErrorJitCompilerNotFound = 81
c'cudaErrorJitCompilerNotFound :: (Num a) => a

c'cudaErrorCooperativeLaunchTooLarge = 82
c'cudaErrorCooperativeLaunchTooLarge :: (Num a) => a

c'cudaErrorStartupFailure = 127
c'cudaErrorStartupFailure :: (Num a) => a

c'cudaErrorApiFailureBase = 10000
c'cudaErrorApiFailureBase :: (Num a) => a


