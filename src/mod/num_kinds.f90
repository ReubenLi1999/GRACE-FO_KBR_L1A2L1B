module num_kinds

    use, intrinsic :: iso_fortran_env

    implicit none

    private

    integer, parameter, public :: wp = real128 !! Using "double precision" real kinds
    integer, parameter, public :: ip = int32   !! Integer working precision

end module num_kinds