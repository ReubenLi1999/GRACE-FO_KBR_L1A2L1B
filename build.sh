#!/bin/zsh

#
#  Simple build script for fortran.
#
#  Requires: FoBiS and Ford
#

MODCODE='kbr.f90'               # module file name
LIBOUT='kbr.a'                  # name of library
DOCDIR='./doc/'                 # build directory for documentation
SRCDIR='./src/mod/'             # library source directory
TESTSRCDIR='./src/'             # unit test source directory
BINDIR='./bin/'                 # build directory for unit tests
LIBDIR='./lib/'                 # build directory for library
FORDMD='readme.md'         # FORD config file name

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	source ~/intel/parallel_studio_xe_2019.0.045/bin/psxevars.sh intel64
elif [[ "$OSTYPE" == "msys"* ]]; then
	"E:\Program Files (x86)\IntelSWTools\parallel_studio_xe_2020\bin\psxevars.bat" intel64
fi

#compiler flags:
# FCOMPILER='gnu' #Set compiler to gfortran
# FCOMPILERFLAGS='-c -O2 -std=f2008'
FCOMPILER='intel' #Set compiler to intel
FCOMPILERFLAGS='-c -O2 -stand f08 -traceback -heap-arrays 256000000'

#build using FoBiS:
if hash FoBiS.py 2>/dev/null; then

	echo "Building library..."

	FoBiS.py build -compiler ${FCOMPILER} -cflags "${FCOMPILERFLAGS}" -dbld ${LIBDIR} -s ${SRCDIR} -dmod ./ -dobj ./ -t ${MODCODE} -o ${LIBOUT} -mklib static -colors

	echo "Building test programs..."

	FoBiS.py build -compiler ${FCOMPILER} -cflags "${FCOMPILERFLAGS}" -dbld ${BINDIR} -s ${TESTSRCDIR} -dmod ./ -dobj ./ -colors -libs ${LIBDIR}${LIBOUT} --include ${LIBDIR} 

else
	echo "FoBiS.py not found! Cannot build library. Install using: sudo pip install FoBiS.py"
fi

# build the documentation using FORD:

if hash ford 2>/dev/null; then

	echo "Building documentation..."

    ford ${FORDMD}

else
	echo "Ford not found! Cannot build documentation. Install using: sudo pip install ford"
fi

# run this programme 
cd ${BINDIR}
./phase_wrap
