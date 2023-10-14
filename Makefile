CC = g++
CFLAGS = -g -Wall

tracer: main.o
	$(CC) $(CFLAGS) -o tracer main.o

main.o:
	$(CC) $(CFLAGS) -c main.cpp

clean:
	rm main.o tracer
