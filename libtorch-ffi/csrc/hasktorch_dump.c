#include "Rts.h"
#include "hasktorch_dump.h"

void
showCFinalizers(int flag, StgCFinalizerList *list)
{
  StgCFinalizerList *head;
  for (head = list;
       (StgClosure *)head != &stg_NO_FINALIZER_closure;
       head = (StgCFinalizerList *)head->link)
    {
      showObject(flag, head->ptr, head->fptr);
    }
}

void
showAllCFinalizers(int flag, StgWeak *list)
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
      showCFinalizers(flag,(StgCFinalizerList *)w->cfinalizers);
    }
  }

}

void
showWeakPtrList(int flag){
  ACQUIRE_LOCK(sm_mutex);
  shiftObjectMap();
  for (uint32_t g = 0; g < RtsFlags.GcFlags.generations; g++) {
    showAllCFinalizers(flag,generations[g].weak_ptr_list);
  }
  RELEASE_LOCK(sm_mutex);
}
