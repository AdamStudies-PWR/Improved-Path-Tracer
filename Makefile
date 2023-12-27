compile-release:
	cmake -B build -S . -DIS_RELEASE:BOOL=TRUE -DImageMagick_EXECUTABLE_DIR=/usr/local/bin
	$(MAKE) -C build
	rm -f tracer
	mv ./build/src/tracer .

compile-debug:
	cmake -B build -S . -DIS_RELEASE:BOOL=FALSE -DImageMagick_EXECUTABLE_DIR=/usr/local/bin
	$(MAKE) -C build
	rm -f tracer
	mv ./build/src/tracer .

clean:
	rm -f tracer
	rm -rf build
