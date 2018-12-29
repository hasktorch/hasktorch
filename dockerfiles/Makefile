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

cpu-deb:
	rm -f debian/changelog
	dch --create --package libaten-dev -v 0.1.0 'This package is built with CPU-only support'
	docker build -f deb-cpu.Dockerfile -t hasktorch-deb
	docker cp hoge:/tmp/libaten-dev_0.1.0_amd64.deb .
	docker rm hasktorch-deb

gpu-deb:
	rm -f debian/changelog
	dch --create --package libaten-dev -v 0.1.0 'This package is built with GPU support'
	docker build -f deb-gpu.Dockerfile -t hasktorch-deb
	docker cp hoge:/tmp/libaten-dev_0.1.0_amd64.deb .
	docker rm hasktorch-deb
