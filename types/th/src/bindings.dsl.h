/******
 * Copyright © 2008–2011 Maurício C. Antunes
 * This file is distributed under the BSD license.
 * Check LICENSE file in distribution package for
 * details.
******/

#ifndef __BINDINGS_DSL_H__
#define __BINDINGS_DSL_H__

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#ifdef __cplusplus
#include <cinttypes>
#else
#include <inttypes.h>
#endif

#define hsc_strict_import(dummy) printf( \
    "import Foreign.Ptr (Ptr,FunPtr,plusPtr)\n" \
    "import Foreign.Ptr (wordPtrToPtr,castPtrToFunPtr)\n" \
    "import Foreign.Storable\n" \
    "import Foreign.C.Types\n" \
    "import Foreign.C.String (CString,CStringLen,CWString,CWStringLen)\n" \
    "import Foreign.Marshal.Alloc (alloca)\n" \
    "import Foreign.Marshal.Array (peekArray,pokeArray)\n" \
    "import Data.Int\n" \
    "import Data.Word\n" \
    ); \

#define bc_word(name) \
    { \
     char *p, *q, buffer_w[strlen(name)+1]; \
     strcpy(buffer_w,name); \
     for (p=strtok(buffer_w," \t");(q=strtok(NULL," \t"));p=q); \
     printf("%s",p); \
    } \

#define bc_glue(type,field) \
    { \
     bc_word(type); \
     printf("'"); \
     char *p, buffer_g[strlen(field)+1]; \
     strcpy(buffer_g,field); \
     for (p=buffer_g;*p;p++) \
        *p = *p=='.' ? '\'' : ispunct(*p) ? '_' : *p; \
     bc_word(buffer_g); \
    } \

#define bc_typemarkup(name) \
    { \
     char buffer_t[strlen(name)+1]; \
     strcpy(buffer_t,name); \
     char *p1,*p2,*p3; \
     p1 = buffer_t; \
     while (*p1) \
        { \
         for (p2=p1;*p2 && *p2!='<';p2++); \
         for (p3=p2;*p3 && *p3!='>';p3++); \
         if (*p2 == '<') *p2++ = '\0'; \
         if (*p3 == '>') *p3++ = '\0'; \
         printf("%s",p1); \
         if (*p2) bc_conid(p2); \
         p1 = p3; \
        } \
    } \

#define bc_varid(name) {printf("c'");bc_word(name);}; \

#define bc_conid(name) {printf("C'");bc_word(name);}; \

#define bc_ptrid(name) {printf("p'");bc_word(name);}; \

#define bc_wrapper(name) {printf("mk'");bc_word(name);}; \

#define bc_dynamic(name) {printf("mK'");bc_word(name);}; \

#define bc_decimal(name) (name) > 0 \
    ? printf("%" PRIuMAX,(uintmax_t)(name)) \
    : printf("%" PRIdMAX,(intmax_t)(name)) \

#define bc_wordptr(name) printf("%" PRIuPTR,(uintptr_t)(name)) \

#define bc_float(name) printf("%Le",(long double)(name)) \

#if __GLASGOW_HASKELL__ >= 800
/* GHC has supported pattern synonym type signatures since 7.10, but due to a
 * bug in 7.10, GHC will reject the type signature we try to give to our
 * pattern synonyms, even though they are valid. As a result, we only produce
 * explicit type signatures on GHC 8.0 or later, where the feature works
 * as intended.
 */
# define bc_patsig(name,constr) \
    printf("pattern ");bc_conid(name); \
    printf(" :: (Eq a, %s a) => a",constr);
#else
# define bc_patsig(name,constr)
#endif

#define hsc_num(name) \
    bc_varid(# name);printf(" = ");bc_decimal(name);printf("\n"); \
    bc_varid(# name);printf(" :: (Num a) => a\n"); \

#define hsc_fractional(name) \
    bc_varid(# name);printf(" = ");bc_float(name);printf("\n"); \
    bc_varid(# name);printf(" :: (Fractional a) => a\n"); \

#if __GLASGOW_HASKELL__ >= 710
# define hsc_num_pattern(name) \
     printf("pattern ");bc_conid(# name);printf(" <- ((== ("); \
     bc_decimal(name);printf(")) -> True) where\n    "); \
     bc_conid(# name);printf(" = ");bc_decimal(name);printf("\n"); \
     bc_patsig(# name,"Num");

# define hsc_fractional_pattern(name) \
     printf("pattern ");bc_conid(# name);printf(" <- ((== ("); \
     bc_float(name);printf(")) -> True) where\n    "); \
     bc_conid(# name);printf(" = ");bc_float(name);printf("\n"); \
     bc_patsig(# name,"Fractional");
#endif

#define hsc_pointer(name) \
    bc_varid(# name);printf(" = wordPtrToPtr "); \
    bc_wordptr(name);printf("\n"); \
    bc_varid(# name);printf(" :: Ptr a\n"); \

#define hsc_function_pointer(name) \
    bc_varid(# name);printf(" = (castPtrToFunPtr . wordPtrToPtr) "); \
    bc_wordptr(name);printf("\n"); \
    bc_varid(# name);printf(" :: FunPtr a\n"); \

#ifdef BINDINGS_STDCALLCONV
#define hsc_ccall(name,type) hsc_callconv(name,stdcall,type)
#else
#define hsc_ccall(name,type) hsc_callconv(name,ccall,type)
#endif

#define hsc_callconv(name,conv,type) \
    printf("foreign import "# conv" \"%s\" ",# name); \
    bc_varid(# name);printf("\n"); \
    printf("  :: ");bc_typemarkup(# type);printf("\n"); \
    printf("foreign import "# conv" \"&%s\" ",# name); \
    bc_ptrid(# name);printf("\n"); \
    printf("  :: FunPtr (");bc_typemarkup(# type);printf(")\n"); \

/* experimental support for unsafe calls */
#define hsc_ccall_unsafe(name,type) \
    printf("foreign import ccall unsafe \"%s\" unsafe'",# name); \
    bc_varid(# name);printf("\n"); \
    printf("  :: ");bc_typemarkup(# type);printf("\n"); \
    printf("foreign import ccall unsafe \"&%s\" unsafe'",# name); \
    bc_ptrid(# name);printf("\n"); \
    printf("  :: FunPtr (");bc_typemarkup(# type);printf(")\n"); \

/* experimental support for interruptible calls */
#define hsc_ccall_interruptible(name,type) \
    printf("foreign import ccall interruptible \"%s\" interruptible'",# name); \
    bc_varid(# name);printf("\n"); \
    printf("  :: ");bc_typemarkup(# type);printf("\n"); \
    printf("foreign import ccall interruptible \"&%s\" interruptible'",# name); \
    bc_ptrid(# name);printf("\n"); \
    printf("  :: FunPtr (");bc_typemarkup(# type);printf(")\n"); \

#define hsc_cinline(name,type) \
    printf("foreign import ccall \"inline_%s\" ",# name); \
    bc_varid(# name);printf("\n"); \
    printf("  :: ");bc_typemarkup(# type);printf("\n"); \

#define hsc_globalvar(name,type) \
    printf("foreign import ccall \"&%s\" ",# name); \
    bc_ptrid(# name);printf("\n"); \
    printf("  :: Ptr (");bc_typemarkup(# type);printf(")\n"); \

#define hsc_globalarray(name,type) \
    printf("foreign import ccall \"array_%s\" ",# name); \
    bc_varid(# name);printf("\n"); \
    printf("  :: Ptr (");bc_typemarkup(# type);printf(")\n"); \

#define hsc_integral_t(name) \
    printf("type ");bc_conid(# name);printf(" = "); \
    { \
     int sign = (name)(-1)<0; \
     size_t size = sizeof(name); \
     if (size==sizeof(int)) printf("%s",sign?"CInt":"CUInt"); \
     else if (size==sizeof(char)) printf("%s", \
       (char)(-1)<0?(sign?"CChar":"CUChar"):(sign?"CSChar":"CChar")); \
     else printf("%s%" PRIuMAX,sign?"Int":"Word",(uintmax_t)(8*size)); \
     printf("\n"); \
    } \

#define hsc_opaque_t(name) \
    printf("data ");bc_conid(# name); \
    printf(" = "); \
    bc_conid(# name);printf("\n"); \

#define hsc_synonym_t(name,type) \
    printf("type ");bc_conid(# name); \
    printf(" = "); \
    bc_typemarkup(# type); \
    printf("\n"); \

#ifdef BINDINGS_STDCALLCONV
#define hsc_callback(name,type) hsc_callbackconv(name,stdcall,type)
#define hsc_callback_t(name,type) hsc_callbackconv(name,stdcall,type)
#else
#define hsc_callback(name,type) hsc_callbackconv(name,ccall,type)
#define hsc_callback_t(name,type) hsc_callbackconv(name,ccall,type)
#endif

#define hsc_callbackconv(name,conv,type) \
    printf("type ");bc_conid(# name);printf(" = FunPtr ("); \
    bc_typemarkup(# type);printf(")\n"); \
    printf("foreign import "# conv" \"wrapper\" "); \
    bc_wrapper(# name);printf("\n"); \
    printf("  :: (");bc_typemarkup(# type); \
    printf(") -> IO ");bc_conid(# name);printf("\n"); \
    printf("foreign import "# conv" \"dynamic\" "); \
    bc_dynamic(# name);printf("\n"); \
    printf("  :: ");bc_conid(# name); \
    printf(" -> (");bc_typemarkup(# type);printf(")\n"); \

static struct {
	int n, is_union[500], is_fam[500];
	uintmax_t array_size[500], offset[500];
	char fname[500][1000], ftype[500][1000];
} bc_fielddata;

#define bc_fieldname(type,field) {printf("c'");bc_glue(type,field);}; \

#define bc_unionupdate(type,field) {printf("u'");bc_glue(type,field);}; \

#define bc_fieldoffset(type,field) {printf("p'");bc_glue(type,field);}; \

#define hsc_starttype(name) \
    { \
     struct {char _; name v;} bc_refdata; \
     size_t typesize = sizeof bc_refdata.v; \
     ptrdiff_t typealign = (char*)&bc_refdata.v - (char*)&bc_refdata; \
     bc_fielddata.n = 0; \
     char typename[] = # name; \
     int index; \
     int standalone_deriving = 0; \

#define bc_basicfield(name,type,u,f) \
     index = bc_fielddata.n++; \
     bc_fielddata.offset[index] = (uintmax_t) \
         ((char*)&bc_refdata.v.name - (char*)&bc_refdata.v); \
     bc_fielddata.array_size[index] = 0; \
     bc_fielddata.is_union[index] = u; \
     bc_fielddata.is_fam[index] = f; \
     strcpy(bc_fielddata.fname[index],# name); \
     strcpy(bc_fielddata.ftype[index],type); \

#define hsc_field(name,type) \
     bc_basicfield(name,# type,0,0); \

#define hsc_union_field(name,type) \
     bc_basicfield(name,# type,1,0); \

#define hsc_flexible_array_member(name,type) \
     bc_basicfield(name,# type,0,1); \

#define hsc_array_field(name,type) \
     bc_basicfield(name,# type,0,0); \
     bc_fielddata.array_size[index] = sizeof bc_refdata.v.name \

#define hsc_union_array_field(name,type) \
     bc_basicfield(name,# type,1,0); \
     bc_fielddata.array_size[index] = sizeof bc_refdata.v.name \

#define hsc_stoptype(dummy) \
     printf("data ");bc_conid(typename);printf(" = "); \
     bc_conid(typename);printf("{\n"); \
     int i; \
     for (i=0; i < bc_fielddata.n; i++) \
        { \
         printf("  "); \
         bc_fieldname(typename,bc_fielddata.fname[i]); \
         printf(" :: "); \
         if (bc_fielddata.array_size[i] > 0 || bc_fielddata.is_fam[i]) \
             printf("["); \
         bc_typemarkup(bc_fielddata.ftype[i]); \
         if (bc_fielddata.array_size[i] > 0 || bc_fielddata.is_fam[i]) \
             printf("]"); \
         if (i+1 < bc_fielddata.n) printf(","); \
         printf("\n"); \
        } \
     if (!standalone_deriving) \
         printf("} deriving (Eq,Show)\n"); \
     else \
        { \
         printf("}\n"); \
         printf("deriving instance Eq ");bc_conid(typename);printf("\n"); \
         printf("deriving instance Show ");bc_conid(typename);printf("\n"); \
        } \
     for (i=0; i < bc_fielddata.n; i++) \
        { \
         bc_fieldoffset(typename,bc_fielddata.fname[i]); \
         printf(" p = plusPtr p %" PRIuMAX "\n",bc_fielddata.offset[i]); \
         bc_fieldoffset(typename,bc_fielddata.fname[i]); \
         printf(" :: Ptr (");bc_conid(typename);printf(") -> "); \
         printf("Ptr (");bc_typemarkup(bc_fielddata.ftype[i]);printf(")\n"); \
        } \
     for (i=0; i < bc_fielddata.n; i++) if (bc_fielddata.is_union[i]) \
        { \
         bc_unionupdate(typename,bc_fielddata.fname[i]); \
         printf(" :: ");bc_conid(typename);printf(" -> "); \
         if (bc_fielddata.array_size[i] > 0) printf("["); \
         bc_typemarkup(bc_fielddata.ftype[i]); \
         if (bc_fielddata.array_size[i] > 0) printf("]"); \
         printf(" -> IO ");bc_conid(typename); \
         printf("\n"); \
         bc_unionupdate(typename,bc_fielddata.fname[i]); \
         printf(" v vf = alloca $ \\p -> do\n"); \
         printf("  poke p v\n"); \
         if (bc_fielddata.array_size[i] > 0) \
            { \
             printf("  let s%d = div %" PRIuMAX " $ sizeOf $ (undefined :: ", \
               i, bc_fielddata.array_size[i]); \
             bc_typemarkup(bc_fielddata.ftype[i]); \
             printf(")\n  pokeArray (plusPtr p %" PRIuMAX ") $ take s%d vf", \
               bc_fielddata.offset[i], i); \
            } \
         else \
           printf("  pokeByteOff p %" PRIuMAX " vf", \
               bc_fielddata.offset[i]); \
         printf("\n"); \
         printf("  vu <- peek p\n"); \
         printf("  return $ v\n"); \
         int j; \
         for (j=0; j < bc_fielddata.n; j++) if (bc_fielddata.is_union[j]) \
            { \
             printf("    {"); bc_fieldname(typename,bc_fielddata.fname[j]); \
             printf(" = "); bc_fieldname(typename,bc_fielddata.fname[j]); \
             printf(" vu}\n"); \
            } \
        } \
     printf("instance Storable "); \
     bc_conid(typename);printf(" where\n"); \
     printf("  sizeOf _ = %" PRIuMAX "\n  alignment _ = %" PRIuMAX "\n", \
       (uintmax_t)(typesize),(uintmax_t)(typealign)); \
     printf("  peek _p = do\n"); \
     for (i=0; i < bc_fielddata.n; i++) \
        { \
         printf("    v%d <- ",i); \
         if (bc_fielddata.is_fam[i]) \
            printf("return []"); \
         else if (bc_fielddata.array_size[i] > 0) \
           { \
            printf ("let s%d = div %" PRIuMAX " $ sizeOf $ (undefined :: ", \
              i, bc_fielddata.array_size[i]); \
            bc_typemarkup(bc_fielddata.ftype[i]); \
            printf(") in peekArray s%d (plusPtr _p %" PRIuMAX ")", \
              i, bc_fielddata.offset[i]); \
           } \
         else \
            printf("peekByteOff _p %" PRIuMAX "", bc_fielddata.offset[i]); \
         printf("\n"); \
        } \
     printf("    return $ ");bc_conid(typename); \
     for (i=0; i < bc_fielddata.n; i++) printf(" v%d",i); \
     printf("\n"); \
     printf("  poke _p (");bc_conid(typename); \
     for (i=0; i < bc_fielddata.n; i++) printf(" v%d",i); \
     printf(") = do\n"); \
     for (i=0; i < bc_fielddata.n; i++) \
        { \
         if (bc_fielddata.is_fam[i]) \
            printf("    pokeArray (plusPtr _p %" PRIuMAX ") v%d", \
              bc_fielddata.offset[i],i); \
         else if (bc_fielddata.array_size[i] > 0) \
           { \
            printf("    let s%d = div %" PRIuMAX " $ sizeOf $ (undefined :: ", \
              i, bc_fielddata.array_size[i]); \
            bc_typemarkup(bc_fielddata.ftype[i]); \
            printf(")\n    pokeArray (plusPtr _p %" PRIuMAX ") (take s%d v%d)", \
              bc_fielddata.offset[i], i, i); \
           } \
         else \
            printf("    pokeByteOff _p %" PRIuMAX " v%d", \
              bc_fielddata.offset[i],i); \
         printf("\n"); \
        } \
     printf("    return ()\n"); \
    } \

#define hsc_gobject_notclassed(prefix,object,CamelCase) \
    hsc_opaque_t(CamelCase) \
    hsc_cinline(prefix##_TYPE_##object,<GType>) \
    hsc_cinline(prefix##_##object,Ptr a -> Ptr <CamelCase>) \
    hsc_cinline(prefix##_IS_##object,Ptr a -> <gboolean>) \

#define hsc_gobject(prefix,object,CamelCase) \
    hsc_opaque_t(CamelCase##Class) \
    hsc_gobject_notclassed(prefix,object,CamelCase) \
    hsc_cinline(prefix##_##object##_CLASS,Ptr a -> Ptr <CamelCase##Class>) \
    hsc_cinline(prefix##_IS_##object##_CLASS,Ptr a -> <gboolean>) \
    hsc_cinline(prefix##_##object##_GET_CLASS,Ptr a -> Ptr <CamelCase##Class>) \

#endif /* __BINDINGS_DSL_H__ */

