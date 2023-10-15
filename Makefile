CC = g++
CFLAGS = -O3

tracer: main.o
	$(CC) $(CFLAGS) -o tracer main.o

main.o:
	$(CC) $(CFLAGS) -c main.cpp

clean:
	rm main.o tracer
