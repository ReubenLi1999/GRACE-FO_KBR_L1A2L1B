program main
!!***************************************************************************************************
!!> author: Hao-si Li
!!  date: 1/12/2020
!!  Institution: Institute of Mechanics, Academy of China
!!
!!  This is a test programme for processing the GRACE-FO KBR data product from Level-1A to Level-1B.
!!  
!!  This programme is capable of processing one-day data, so the example is carried out by 2019-01-01
!!
!!  Date                Version             Progammer              
!!  ==========          ==========          ==========
!!  2020.11.21          1.0                 Hao-si Li
!! 
!!  Input: The GRACE-FO KBR1A, USO1B, CLK1B, PCI1A, PLT1A for this specific day.
!!
!!  Output: The KBR1B data product including the CRN-filtered range, range-rate, range accelaration,
!!          light-of-flight correction, light-of-flight rate, time-of-flight accelaration, and the an-
!!          tenna centre offset.
!!
!!  Copy right. All the source codes are open-resource.
!!***************************************************************************************************

    use gracefo
    use num_kinds
    implicit none

    type, extends(preprocess):: kbr_preprocess
        ! nothing extended
    end type kbr_preprocess

    type(kbr_preprocess)                                    :: c, d, both

    character(len = 100)                                    :: pwd
    CHARACTER(len = 100)                                    :: filenames_c(6), filenames_d(6)
    CHARACTER(len = 100)                                    :: flags(6)

    real(wp)                                                :: start_time, end_time

    integer(ip)                                             :: err
    integer(ip)                                             :: nheaders(6)
    
    ! start time for this programme
    call CPU_TIME(start_time)

    date = '2019-06-22'
    call get_environment_variable('PWD', pwd)
    !DEC$ IF DEFINED(_WIN32)
        call execute_command_line('D:\Users\LHS10\anaconda3\python.exe ../preFORfortran.py '//date)
    !DEC$ ELSEIF DEFINED(__linux)
        call execute_command_line('~/anaconda3/bin/python ../src/preFORfortran.py '//date)
    !DEC$ ELSE
        print *, 'oops'
    !DEC$ ENDIF
    
    ! assign some parameters
    filenames_c = ['..//..//..//..//gracefo_dataset//gracefo_1A_'//trim(date)//'_RL04.ascii.noLRI//KBR1A_'//trim(date)//'_C_04.txt', &
                   '..//..//..//..//gracefo_dataset//gracefo_1A_'//trim(date)//'_RL04.ascii.noLRI//CLK1A_'//trim(date)//'_C_05.txt', &
                   '..//..//..//..//gracefo_dataset//gracefo_1B_'//trim(date)//'_RL04.ascii.noLRI//USO1B_'//trim(date)//'_C_05.txt', &
                   '..//..//..//..//gracefo_dataset//gracefo_1A_'//trim(date)//'_RL04.ascii.noLRI//PCI1A_'//trim(date)//'_C_04.txt', &
                   '..//..//..//..//gracefo_dataset//gracefo_1A_'//trim(date)//'_RL04.ascii.noLRI//PLT1A_'//trim(date)//'_Y_04.txt', &
                   '..//..//..//..//gracefo_dataset//gracefo_1B_'//trim(date)//'_RL04.ascii.noLRI//GNI1B_'//trim(date)//'_C_04.txt']
    filenames_d = ['..//..//..//..//gracefo_dataset//gracefo_1A_'//trim(date)//'_RL04.ascii.noLRI//KBR1A_'//trim(date)//'_D_04.txt', &
                   '..//..//..//..//gracefo_dataset//gracefo_1A_'//trim(date)//'_RL04.ascii.noLRI//CLK1A_'//trim(date)//'_D_05.txt', &
                   '..//..//..//..//gracefo_dataset//gracefo_1B_'//trim(date)//'_RL04.ascii.noLRI//USO1B_'//trim(date)//'_D_05.txt', &
                   '..//..//..//..//gracefo_dataset//gracefo_1A_'//trim(date)//'_RL04.ascii.noLRI//PCI1A_'//trim(date)//'_D_04.txt', &
                   '..//..//..//..//gracefo_dataset//gracefo_1A_'//trim(date)//'_RL04.ascii.noLRI//PLT1A_'//trim(date)//'_Y_04.txt', &
                   '..//..//..//..//gracefo_dataset//gracefo_1B_'//trim(date)//'_RL04.ascii.noLRI//GNI1B_'//trim(date)//'_D_04.txt']

    nheaders = [235_ip, 0_ip, 0_ip, 84_ip, 99_ip, 148_ip]
    ! assign the flags array one by one so that the warning will not be rasied
    flags(1) = 'kbr_phase'; flags(2) = 'kbr_time'; flags(3) = 'kbr_freq'; flags(4) = 'ant_centr_corr'
    flags(5) = 'tof'; flags(6) = 'gni1b'

    ! initialisation
    call c%initialise(filenames_c, nheaders, flags, 'C')
    call d%initialise(filenames_d, nheaders, flags, 'D')

    ! phase wrap section
    call c%phase_wrap_kbr('C')
    call d%phase_wrap_kbr('D')

    ! kbrt2gpst
    call c%kbrt2gpst()
    call d%kbrt2gpst()

    ! phase interpolation using lagrange method
    call c%phase_lagrange_kbr()
    call d%phase_lagrange_kbr()

    ! calculate the kbr dual-one-way range and output to file
    call both%kbr_dowr(c, d)

    ! calculate the double differenced one-way range
    call both%dd_range(c, d, .true.)

    ! calculate the time of flight
    call both%cal_tof(c, d)

    ! crn filter
    call both%crn_filter()
    call both%output_crn()

    ! ant_centr_corr
    call both%ant_centr_corr(c, d)

    ! check shadow
    call c%check_shadow_wrap(.true., 'C')
    call d%check_shadow_wrap(.true., 'D')

    ! destructor
    call c%destruct_x()
    call d%destruct_x()
    call both%destruct_x()

    call CPU_TIME(end_time)

    print *, 'The elapsed time is', end_time - start_time, 's.'
end program main
