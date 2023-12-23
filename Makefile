compile-release:
	cmake -B build -S . -DIS_RELEASE:BOOL=TRUE
	$(MAKE) -C build
	rm -f tracer
	mv ./build/src/tracer .

compile-debug:
	cmake -B build -S . -DIS_RELEASE:BOOL=FALSE
	$(MAKE) -C build
	rm -f tracer
	mv ./build/src/tracer .

clean:
	rm -f tracer
	rm -rf build
