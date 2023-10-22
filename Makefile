compile:
	cmake -B build -S .
	$(MAKE) -C build
	rm -f tracer
	mv ./build/src/tracer .

clean:
	rm -f tracer
	rm -rf build
