FC = gfortran
executables := $(patsubst %.f,%.out,$(wildcard *.f))


all:
	make $(executables)

%.out: %.f
	$(FC) -O2 $*.f -o $*.out

clean:
	-rm *.out
