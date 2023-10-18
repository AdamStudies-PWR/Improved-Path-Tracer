compile:
	rm -f tracer
	cmake -B build -S .
	$(MAKE) -C build
	mv ./build/src/tracer .

clean:
	rm -f tracer
	rm -rf build
