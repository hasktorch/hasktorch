clean:

install:
	if [ ! -d  $(DESTDIR)/usr/lib/ ] ; then mkdir -p $(DESTDIR)/usr/lib/; fi
	if [ ! -d  $(DESTDIR)/usr/include/ ] ; then mkdir -p $(DESTDIR)/usr/include/; fi
	cp -af /hasktorch/ffi/deps/aten/build/lib/* $(DESTDIR)/usr/lib/
	cp -arf /hasktorch/ffi/deps/aten/build/include/* $(DESTDIR)/usr/include/
	cp -af /hasktorch/ffi/deps/aten/build/include/TH/*.h $(DESTDIR)/usr/include/
	cp -af /hasktorch/ffi/deps/aten/build/include/THNN/*.h $(DESTDIR)/usr/include/
	cp -af /hasktorch/ffi/deps/aten/build/include/THC/*.h $(DESTDIR)/usr/include/
	cp -af /hasktorch/ffi/deps/aten/build/include/THCUNN/*.h $(DESTDIR)/usr/include/
