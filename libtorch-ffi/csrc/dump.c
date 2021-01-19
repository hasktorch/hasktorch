#include "Rts.h"

extern void showObject(void* ptr, void* fptr);
extern void showWeakPtrList();

void
showCFinalizers(StgCFinalizerList *list)
{
  StgCFinalizerList *head;
  for (head = list;
       (StgClosure *)head != &stg_NO_FINALIZER_closure;
       head = (StgCFinalizerList *)head->link)
    {
      showObject(head->ptr,head->fptr);
    }
}

void
showAllCFinalizers(StgWeak *list)
{
  StgWeak *w;
  for (w = list; w; w = w->link) {
    // We need to filter out DEAD_WEAK objects, because it's not guaranteed
    // that the list will not have them when shutting down.
    // They only get filtered out during GC for the generation they
    // belong to.
    // If there's no major GC between the time that the finalizer for the
    // object from the oldest generation is manually called and shutdown
    // we end up running the same finalizer twice. See #7170.
    const StgInfoTable *winfo = w->header.info;
    if (winfo != &stg_DEAD_WEAK_info) {
      showCFinalizers((StgCFinalizerList *)w->cfinalizers);
    }
  }

}

void
showWeakPtrList(){
  //  runAllCFinalizers(StgWeak *w)
  /* run C finalizers for all active weak pointers */
  //for (uint32_t i = 0; i < n_capabilities; i++) {
  // showAllCFinalizers(capabilities[i]->weak_ptr_list_hd);
  //}
  ACQUIRE_LOCK(sm_mutex);
  for (uint32_t g = 0; g < RtsFlags.GcFlags.generations; g++) {
    showAllCFinalizers(generations[g].weak_ptr_list);
  }
  RELEASE_LOCK(sm_mutex);
}
