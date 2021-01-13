module gracefo
!!***************************************************************************************************
!!> author: Hao-si Li
!!  date: 4/7/2018
!!  Institution: Institute of Mechanics, Academy of China
!!
!!  This is a integrated module for processing the GRACE-FO KBR data product from Level-1A to Level-1B.
!!  
!!  Summary: this module using OOP
!!
!!  Date                Version             Progammer              
!!  ==========          ==========          ==========
!!  2020.11.21          1.0                 Hao-si Li
!!
!!  Copy right. All the source codes are open-resource.
!!***************************************************************************************************
    use num_kinds
    implicit none

    
    TYPE, public:: grace_fo
        real(wp)                                        :: utc_time
        !! the standard utc time for the satellite
        real(wp)                                        :: gps_time
        !! the standard gps time for the satellite
        real(wp)                                        :: kbr_time
        !! the receive time of the kbr
        real(wp)                                        :: l1b_time
        !! the gracefo level-1b data product time
        real(wp)                                        :: jdtt
        !! julian day for utct
        real(wp)                                        :: eps_time_kbr     
        !! the eps time of kbr time from gps time
        real(wp)                                        :: k_phase          
        !! the k phase for this satellite in level-1a
        real(wp)                                        :: ka_phase         
        !! the ka phase for this satellite in level-1a
        real(wp)                                        :: k_freq           
        !! the k frequency for this satellite
        real(wp)                                        :: ka_freq          
        !! the ka frequency for this satellite
        real(wp)                                        :: tau_plt          
        !! Travel time of signal between satellites with identifiers rcv_id and trx_id.
        real(wp)                                        :: tof              
        !! time of flight
        real(wp)                                        :: crn_tof          
        !! the time of flight filtered by CRN filter
        real(wp)                                        :: crn_tofr         
        !! the first time devirative time of flight
        real(wp)                                        :: crn_tofa         
        !! the second time derivative time of flight
        real(wp)                                        :: pos_n(3)
        !! position vector in the new-built frame, x axis pointing to the Sun
        real(wp)                                        :: pos_i(3)         
        !! position vector in the inertial frame
        real(wp)                                        :: pos_e(3)
        !! position vector in the earth-fixed frame
        real(wp)                                        :: vel_i(3)
        !! velocity vector in the inertial frame
        real(wp)                                        :: rho              
        !！ the intersatellite range calculating with GPS data
        real(wp)                                        :: ant_centr_range  
        !！ antenna phase centre offset correction to inter-satellite range
        real(wp)                                        :: ant_centr_rate   
        !！ first derivative of ant_centr_range
        real(wp)                                        :: ant_centr_accl   
        !！ second derivative of ant_centr_range
        real(wp)                                        :: right_ascension
        !! the right ascension of the Sun on the point
        real(wp)                                        :: negative_declination
        !! the negative declination of the Sun on the point
        real(wp)                                        :: dd_range
        !! double differenced one-way range
        real(wp)                                        :: dd_phase
        !! double differenced one-way phase

        real(wp)                                        :: arg1, arg2
        real(wp)                                        :: reg
        
        real(wp)                                        :: kbr_dow_range
        !! the dual-one-way range
        real(wp)                                        :: crn_range
        real(wp)                                        :: crn_rate
        real(wp)                                        :: crn_accl

        integer(ip)                                     :: flag_solar_pressure
    contains
        procedure                                       :: check_shadow                 =>      check_shadow
    end type grace_fo

    type, public:: io_file
        character(len = 100)                            :: name
        integer(ip)                                     :: unit
        integer(ip)                                     :: nrow
        integer(ip)                                     :: nheader
    contains
        procedure                                       :: read_in_data                 =>      read_in_data
        procedure                                       :: get_file_n                   =>      get_file_n
    end type io_file

    type, public:: preprocess
        CLASS(grace_fo), ALLOCATABLE                    :: x(:)
        class(io_file) , ALLOCATABLE                    :: i_f(:)
    contains
        procedure                                       :: initialise                   =>      initialise
        procedure                                       :: destruct_x                   =>      destruct_x
        procedure                                       :: phase_wrap_kbr               =>      phase_wrap_kbr
        procedure                                       :: phase_lagrange_kbr           =>      phase_lagrange_kbr
        procedure                                       :: kbrt2gpst                    =>      kbrt2gpst
        procedure                                       :: crn_filter                   =>      crn_filter
        procedure                                       :: kbr_dowr                     =>      kbr_dowr
        procedure                                       :: output_crn                   =>      output_crn
        procedure                                       :: ant_centr_corr               =>      ant_centr_corr
        procedure                                       :: cal_tof                      =>      cal_tof
        procedure                                       :: l1bt2utct                    =>      l1bt2utct
        procedure                                       :: gpst2utct                    =>      gpst2utct
        procedure                                       :: utct2jdtt                    =>      utct2jdtt
        procedure                                       :: date2RADECL                  =>      date2RADECL
        procedure                                       :: check_shadow_wrap            =>      check_shadow_wrap
        procedure                                       :: dd_range                     =>      dd_range
    end type preprocess

    real(wp), PARAMETER                                 :: speed_light = 299792458.0_wp ! speed of light
    real(wp), PARAMETER                                 :: pi = acos(-1.0_wp)
    real(wp), PARAMETER                                 :: radius_earth = 6371000.0_wp     ! the radius of the earth in m
    real(wp), PARAMETER                                 :: au_unit = 149597871.0_wp     ! Astronomical unit

    CHARACTER(len = 100)                                :: date

contains

    subroutine dd_range(self, lead, track, tofile)
    !>----------------------------------------------------------------------------------------------
    !@      Arthur      => Hao-si Li
    !@      Date        => 2020-12-19
    !@      Revision    => 1.0      :       2020-12-19 for writing the subroutine
    !>
    !>      Description:
    !>              This subroutine is designed to compute the double differenced one-way range.
    !>
    !>              Double differenced one-way range is raised to evaluate the GRACE-like satellites'
    !>              KBR noise level. Since the ratio between the K and Ka is exactly 3/4, this 
    !>              combination of the four one-way measurements remove the gravity signal component
    !>              and reflects part of noise components, mainly ionosphere difference between the 
    !>              two satellites.
    !>
    !>      Input arguments:
    !>              lead: the <preprocess> class object, which is the leading satellite. In this pa-
    !>                    rticular subtoutine, the wrappeded k and ka phase date are used.
    !>              track: the <preprocess> class object, which is the tracking satellite. In this  
    !>                    particular subroutine, the wrapped k and ka phase data are used.
    !>              tofile: the flag to check whether output the data to file or not.
    !>
    !>      Output arguments
    !>              self: the <preprocess> class object, which indicates the so-called "both". In t-
    !>                    he particular subroutine, the double differenced one-way range is computed
    !>
    !>      Methodology:
    !>              double_differenced_one_way_range = speed_of_light / nominal_frequency_of_k * 
    !>              [(phase_k_c2d - 3/4*phase_ka_c2d) - (phase_k_d2c - 3/4*phase_ka_d2c)]
    !>
    !>      Literature:
    !>              J. Kim, S.W. Lee, Flight performance analysis of GRACE K-band ranging instrument
    !>              with simulation data, Acta Astronaut. 65 (11–12) (2009) 1571–1581.
    !>----------------------------------------------------------------------------------------------
        class(preprocess), intent(inout)                :: self
        class(preprocess), INTENT(IN   )                :: lead, track
        logical, OPTIONAL, value                        :: tofile
        real(wp)                                        :: nominal_freq_k = 26.0e9_wp ! Hz
        type(io_file)                                   :: dd_file
        integer(ip)                                     :: ios, i

        !>------------------------------------------------------------------------------------------
        !> assign the double differenced one-way range file properties
        !>------------------------------------------------------------------------------------------
        dd_file%name = '..//..//..//gracefo_dataset//gracefo_1A_'//trim(date)//'_RL04.ascii.noLRI//DDR1A_'//trim(date)//'_Y_04.txt'
        dd_file%unit = 712
        !>------------------------------------------------------------------------------------------
        if (.not. present(tofile)) tofile = .false.

        open(unit=dd_file%unit, file=dd_file%name, iostat=ios, action="write")
        if ( ios /= 0 ) stop "Error opening file name"

        ! compute the double differenced one-way range section
        dd_range_loop: do i = 1, size(self%x), 1
            self%x(i)%dd_phase = (lead%x(i)%k_phase - lead%x(i)%k_freq / lead%x(i)%ka_freq * lead%x(i)%ka_phase) - &
                                 (track%x(i)%k_phase - track%x(i)%k_freq / track%x(i)%ka_freq * track%x(i)%ka_phase)
            self%x(i)%dd_range = speed_light * self%x(i)%dd_phase / nominal_freq_k

            write(dd_file%unit, *) self%x(i)%dd_range
        end do dd_range_loop

        close(unit=dd_file%unit, iostat=ios)
        if ( ios /= 0 ) stop "Error closing file unit dd_file%unit"

    end subroutine dd_range

    subroutine gpst2utct(me)
        class(preprocess), INTENT(INOUT)                :: me

        me%x%utc_time = me%x%gps_time - 18.0_wp

    end subroutine gpst2utct

    subroutine l1bt2utct(me)
        CLASS(preprocess), intent(inout)                :: me

        me%x%utc_time = me%x%l1b_time - 18.0_wp
    end subroutine l1bt2utct

    subroutine utct2jdtt(me)
        class(preprocess), INTENT(INOUT)                :: me

        me%x%jdtt = (me%x%utc_time + 32.184_wp + 37.0_wp) / 86400.0_wp + 2458118.833333333_wp

    end subroutine utct2jdtt

    subroutine date2RADECL(me)
        implicit none
        class(preprocess), INTENT(INOUT)                :: me

        real(wp)                                        :: jdstart, t(1: size(me%x))
        real(wp)                                        :: L0(1: size(me%x)), M(1: size(me%x))
        real(wp)                                        :: Ci(1: size(me%x)), THETA(1: size(me%x))
        real(wp)                                        :: OMIGA(1: size(me%x)), lamda(1: size(me%x))
        real(wp)                                        :: year(1: size(me%x)), THETA2000(1: size(me%x))
        real(wp)                                        :: eph(1: size(me%x))
        real(wp)                                        :: s1(1: size(me%x)), s2(1: size(me%x))
        real(wp)                                        :: s3(1: size(me%x)), s4(1: size(me%x))

        jdstart = 2451545.0_wp
        t = (me%x%jdtt - jdstart) / 36525.0_wp
        
        L0 = 280.46645_wp + 36000.76983_wp * t + 0.0003032_wp * t**2
        
        M = 357.52910_wp + 35999.05030_wp * t - 0.0001559_wp *t**2 - 0.00000048_wp * t**3
        
        Ci = (1.914600_wp - 0.004817_wp * t - 0.000014_wp * t**2) * sind(M) + &
                (0.019993_wp - 0.000101_wp * t) * sind(2.0_wp * M) + 0.000290_wp * sind(3.0_wp * M)
        
        THETA = L0 + Ci
        
        OMIGA = 125.04_wp - 1934.136_wp * t
        lamda = THETA - 0.00569_wp - 0.00478_wp * sind(OMIGA)
        
        year = floor((me%x%jdtt - jdstart) / 365.25_wp) + 2000.0_wp
        THETA2000 = lamda - 0.01397_wp * (year - 2000.0_wp)
        
        eph = 23.439291_wp - 0.013004_wp * t - 0.00059_wp * t / 3600.0_wp + 0.001813_wp*t**3 / 3600.0_wp + 0.00256_wp * cosd(OMIGA)
        
        s1 = cosd(eph) * sind(THETA2000)
        s2 = cosd(THETA2000)
        s3 = sind(eph)
        s4 = sind(THETA2000)
        
        me%x%right_ascension = atan2d(cosd(eph) * sind(THETA2000), cosd(THETA2000))
        
        me%x%negative_declination = asind(sind(eph) * sind(THETA2000))  
    end subroutine date2RADECL

    subroutine check_shadow(me)
        !!> ----------------------------------------------------------------------------------------
        !! The method is particularly special because of the intermediate variable
        !!> ----------------------------------------------------------------------------------------
        class(grace_fo), INTENT(INOUT)                          :: me
        real(wp)                                                :: r(3, 3, 3)

        ! the second rotation matrix
        r(2, 1, 1) =  cosd(-1.0_wp * me%negative_declination)
        r(2, 1, 2) =  0.0_wp
        r(2, 1, 3) = -sind(-1.0_wp * me%negative_declination)
        r(2, 2, 1) =  0.0_wp
        r(2, 2, 2) =  1.0_wp
        r(2, 2, 3) =  0.0_wp
        r(2, 3, 1) =  sind(-1.0_wp * me%negative_declination)
        r(2, 3, 2) =  0.0_wp
        r(2, 3, 3) =  cosd(-1.0_wp * me%negative_declination)

        ! the third rotation matrix
        r(3, 1, 1) =  cosd(me%right_ascension)
        r(3, 1, 2) =  sind(me%right_ascension)
        r(3, 1, 3) =  0.0_wp
        r(3, 2, 1) = -sind(me%right_ascension)
        r(3, 2, 2) =  cosd(me%right_ascension)
        r(3, 2, 3) =  0.0_wp
        r(3, 3, 1) =  0.0_wp
        r(3, 3, 2) =  0.0_wp
        r(3, 3, 3) =  1.0_wp

        me%pos_n = matmul(r(2, :, :), matmul(r(3, :, :), me%pos_i))
        
        me%flag_solar_pressure = 1_ip
        if (me%pos_n(1) < 0.0_wp) then
            if (me%pos_n(2)**2 + me%pos_n(3)**2 <= radius_earth**2) then
                me%flag_solar_pressure = 0_ip
                ! print *, 1
            end if
        end if
    end subroutine check_shadow

    subroutine check_shadow_wrap(self, tofile, id)
        class(preprocess)       , INTENT(INOUT)         :: self
        logical, OPTIONAL, value                        :: tofile
        integer(ip)                                     :: i, ios
        type(io_file)                                   :: shadow_file
        CHARACTER(len = 1)                              :: id

        !> -----------------------------------------------------------------------------------------
        !> assign shadow file properties
        shadow_file%name = '..//..//..//gracefo_dataset//gracefo_1A_'//trim(date)//'_RL04.ascii.noLRI//SHA1A_'//trim(date)//'_'//id//'_04.txt'
        shadow_file%unit = 350
        !> -----------------------------------------------------------------------------------------

        if (.not. present(tofile)) tofile = .false.

        open(unit=shadow_file%unit, file=shadow_file%name, iostat=ios, action="write")
        if ( ios /= 0 ) stop "Error opening file name"

        call self%l1bt2utct()
        call self%date2RADECL()
        check_shadow_loop: do i= 1, size(self%x), 1
            call self%x(i)%check_shadow()
            if (tofile) then
                if (self%x(i)%l1b_time > 0.0_wp) then
                    write(shadow_file%unit, *) self%x(i)%l1b_time, self%x(i)%flag_solar_pressure
                end if
            end if
        end do check_shadow_loop

        close(unit=shadow_file%unit, iostat=ios)
        if ( ios /= 0 ) stop "Error closing file unit shadow_file%unit"

    end subroutine check_shadow_wrap

    subroutine cal_tof(self, c, d)
        class(preprocess), INTENT(INOUT)                :: self, c, d

        integer(ip)                                     :: i

        real(wp)                                        :: reg

        ! assign
        c%x(1)%k_freq = c%x(2)%k_freq; d%x(1)%k_freq = d%x(2)%k_freq;
        c%x(1)%ka_freq = c%x(2)%ka_freq; d%x(1)%ka_freq = d%x(2)%ka_freq;

        ! gps interdistance
        ASSOCIATE(                              &
            f_k_c        => c%x%k_freq         ,&
            f_ka_c       => c%x%ka_freq        ,&
            f_k_d        => d%x%k_freq         ,&
            f_ka_d       => d%x%ka_freq        ,&
            x_c          => c%x%pos_e(1)       ,&
            x_d          => d%x%pos_e(1)       ,&
            y_c          => c%x%pos_e(2)       ,&
            y_d          => d%x%pos_e(2)       ,&
            z_c          => c%x%pos_e(3)       ,&
            z_d          => d%x%pos_e(3)       ,&
            rho          => self%x%rho         ,&
            tau_c        => c%x%tau_plt        ,&
            tau_d        => d%x%tau_plt        ,&
            tof          => self%x%tof          )
            gps_range_loop: do i = 1, size(self%x), 1
                rho(i) = norm2([x_c(i) - x_d(i), y_c(i) - y_d(i), z_c(i) - z_d(i)])
                reg = (16.0_wp / 7.0_wp * (f_ka_c(i) * tau_c(i) + f_ka_d(i) * tau_d(i)) / (f_ka_c(i) + f_ka_d(i)) &
                    - 9.0_wp  / 7.0_wp * (f_k_c(i) * tau_c(i) + f_k_d(i) * tau_d(i)) / (f_k_c(i) + f_k_d(i))) * speed_light
                tof(i) = rho(i) + reg
            end do gps_range_loop
        end ASSOCIATE
    end subroutine cal_tof

    subroutine ant_centr_corr(self, c, d)
        class(preprocess), INTENT(INOUT)                :: self, c, d
        
        self%x%ant_centr_range = c%x%ant_centr_range - d%x%ant_centr_range
        self%x%ant_centr_rate = c%x%ant_centr_rate - d%x%ant_centr_rate
        self%x%ant_centr_accl = c%x%ant_centr_accl - d%x%ant_centr_accl
    end subroutine ant_centr_corr

    subroutine output_crn(self)
        CLASS(preprocess), INTENT(INOUT)                :: self
        type(io_file)                                   :: o_crn
        integer(ip)                                     :: i, ios

        o_crn%name = '..//output//KBR1B_'//trim(date)//'_Y_03.txt'
        o_crn%unit = 652
        
        open(unit=o_crn%unit, file=o_crn%name, iostat=ios, action="write")
        if ( ios /= 0 ) stop "Error opening file name"
        
        output_crn_loop: do i = 1, size(self%x), 1
            write(o_crn%unit, '(f40.20)') self%x(i)%tof
        end do output_crn_loop

        close(unit=o_crn%unit, iostat=ios)
        if ( ios /= 0 ) stop "Error closing file unit o_crn%unit"
        
    end subroutine output_crn

    subroutine crn_filter(self)
        class(preprocess), INTENT(Inout)                :: self
        integer(ip)                                     :: ios, i, err
        real(wp), ALLOCATABLE                           :: crn_coeffs(:, :)
        type(io_file)                                   :: crn

        crn%name = '..//output//crn_filter_coeffs.txt'
        crn%unit = 434
        crn%nrow = crn%get_file_n()
        crn%nheader = 0

        allocate(crn_coeffs(crn%nrow - crn%nheader, 3), stat=err)
        if (err /= 0) print *, "crn_coeffs: Allocation request denied"

        self%x%crn_range = 0.0_wp; self%x%crn_rate = 0.0_wp; self%x%crn_accl = 0.0_wp

        open(unit=crn%unit, file=crn%name, iostat=ios, status="old", action="read")
        if ( ios /= 0 ) stop "Error opening file name"
        
        read_crn_coeffs_loop: do i = 1, crn%nrow, 1
            read(crn%unit, *) crn_coeffs(i, :)
        end do read_crn_coeffs_loop

        close(unit=crn%unit, iostat=ios)
        if ( ios /= 0 ) stop "Error closing file unit 434"

        self%x%crn_range = convolution(self%x(:)%kbr_dow_range, crn_coeffs(:, 1))
        self%x%crn_rate = convolution(self%x(:)%kbr_dow_range, crn_coeffs(:, 2))
        self%x%crn_accl = convolution(self%x(:)%kbr_dow_range, crn_coeffs(:, 3))
        self%x%crn_tof = convolution(self%x(:)%tof, crn_coeffs(:, 1))
        self%x%crn_tofr = convolution(self%x(:)%tof, crn_coeffs(:, 2))
        self%x%crn_tofa = convolution(self%x(:)%tof, crn_coeffs(:, 3))
        
        if (allocated(crn_coeffs)) deallocate(crn_coeffs, stat=err)
        if (err /= 0) print *, "crn_coeffs: Deallocation request denied"

    end subroutine crn_filter

    function convolution(x, h) result(convolve)
        !x is the signal array
        !h is the noise/impulse array
        real(wp), dimension(:), INTENT(IN   )           :: x, h
        real(wp)                                        :: convolve(size(x)), y(size(x))
        integer(ip)                                     :: kernelsize, datasize
        integer(ip)                                     :: i, j, k
        
        datasize = size(x)
        kernelsize = size(h)

        !last part
        do i = kernelsize, datasize, 1
            y(i) = 0.0_wp
            j = i
            do k = 1, kernelsize, 1
                y(i) = y(i) + x(j) * h(k)
                j = j - 1
            end do
        end do
        
        !first part
        do i = 1, kernelsize, 1
            y(i) = 0.0_wp
            j = i
            k = 1
            do while (j > 0)
                y(i) = y(i) + x(j) * h(k)
                j = j - 1
                k = k + 1
            end do
        end do
        
        convolve = y
    end function convolution

    subroutine kbr_dowr(self, c, d)
        class(preprocess), INTENT(IN   )                :: c, d
        class(preprocess), INTENT(INOUT)                :: self

        real(wp)                                        :: dowr_k(size(c%x))
        real(wp)                                        :: dowr_ka(size(c%x))
        real(wp)                                        :: dowr(size(c%x))

        type(io_file)                                   :: dowr_file
        type(io_file)                                   :: iono_file

        integer(ip)                                     :: ios, i, err

        dowr_k = speed_light * (c%x%k_phase + d%x%k_phase) / (c%x%k_freq + d%x%k_freq)

        dowr_ka = speed_light * (c%x%ka_phase + d%x%ka_phase) / (c%x%ka_freq + d%x%ka_freq)

        dowr = 16.0_wp / 7.0_wp * dowr_ka - 9.0_wp / 7.0_wp * dowr_k

        dowr_file%name = '..//..//..//gracefo_dataset//gracefo_1A_'//trim(date)//'_RL04.ascii.noLRI//DOWR1A_'//trim(date)//'_Y_04.txt'
        dowr_file%unit = 234

        !<------------------------------------------------------------------------------------------
        !< write dowr to file, unfiltered, 10Hz
        open(unit=dowr_file%unit, file=dowr_file%name, iostat=ios, action="write")
        if ( ios /= 0 ) stop "Error opening file name"

        out_dowr_loop: do i = 2, size(c%x) - 1, 1
            write(dowr_file%unit,'(3f50.25)') c%x(i)%gps_time, dowr_k(i), dowr_ka(i)
        end do out_dowr_loop
        
        close(unit=dowr_file%unit, iostat=ios)
        if ( ios /= 0 ) stop "Error closing file unit dowr_file%unit"
        !<------------------------------------------------------------------------------------------

        iono_file%name = '..//output//IONO1A_'//trim(date)//'_Y_03.txt'
        iono_file%unit = 512
        OPEN(unit=iono_file%unit, file=iono_file%name, iostat=ios, action='write')
        if ( ios /= 0 ) stop "Error opening ionosphere correction file"
        ! output ionosphere correction
        output_ionosphere_corr: do i = 1, size(c%x) - 1, 1
            write(iono_file%unit, *) dowr_ka(i) - dowr(i)
        end do output_ionosphere_corr
        close(unit=iono_file%unit, iostat=ios)
        if ( ios /= 0 ) stop "Error closing file unit iono_file%unit"

        allocate(self%x(size(c%x)), stat=err)
        if (err /= 0) print *, "self%kbr_dow_range: Allocation request denied"
        self%x%kbr_dow_range = dowr

    end subroutine kbr_dowr

    subroutine kbrt2gpst(self)
        class(preprocess), INTENT(INout)                :: self

        integer(ip)                                     :: i

        self%x%kbr_time = 0.0_wp
        kbrt2gpst_loop: do i = 0, size(self%x) - 1, 1
            self%x(i + 1)%kbr_time = real(i, kind=16) * 0.1_wp + 0.05_wp + self%x(i + 1)%eps_time_kbr
            self%x(i + 1)%gps_time = real(i, kind=16) * 0.1_wp
        end do kbrt2gpst_loop

    end subroutine kbrt2gpst

    subroutine phase_lagrange_kbr(self)
        class(preprocess), intent(inout)                :: self

        real(wp)                                        :: reg_k(1: size(self%x) - 2, 1)
        real(wp)                                        :: reg_ka(1: size(self%x) - 2, 1)
        real(wp)                                        :: time1(4), phase1(4), time2(8), phase2(8), temp(1)

        integer(ip)                                     :: i, m

        m = size(self%x) - 2

        former_3_loop: do i = 1, 3, 1
            ! k
            time1 = self%x(1: 4)%kbr_time; phase1 = self%x(1: 4)%k_phase; temp = [self%x(i + 1)%gps_time]
            reg_k(i, :) = lagrange_value_1d(4, time1, phase1, 1, temp)
            ! ka
            phase1 = self%x(1: 4)%ka_phase
            reg_ka(i, :) = lagrange_value_1d(4, time1, phase1, 1, temp)
        end do former_3_loop

        between_loop: do i = 4, m - 4, 1
            ! k
            time2 = self%x(i - 3: i + 4)%kbr_time; phase2 = self%x(i - 3: i + 4)%k_phase; temp = [self%x(i + 1)%gps_time]
            reg_k(i, :) = lagrange_value_1d(8, time2, phase2, 1, temp)
            ! ka
            phase2 = self%x(i - 3: i + 4)%ka_phase
            reg_ka(i, :) = lagrange_value_1d(8, time2, phase2, 1, temp)
        end do between_loop

        latter_3_loop: do i = m - 3, m, 1
            ! k
            time1 = self%x(m - 3: m)%kbr_time; phase1 = self%x(m - 3: m)%k_phase; temp = [self%x(i + 1)%gps_time]
            reg_k(i, :) = lagrange_value_1d(4, time1, phase1, 1, temp)
            ! ka
            phase1 = self%x(m - 3: m)%ka_phase
            reg_ka(i, :) = lagrange_value_1d(4, time1, phase1, 1, temp)
        end do latter_3_loop
        
        ! assign 0 to necessary positions
        self%x(1)%k_phase = 0; self%x(m + 2)%k_phase = 0; self%x(1)%ka_phase = 0; self%x(m + 2)%ka_phase = 0
        self%x(2: m + 1)%k_phase = reg_k(:, 1); self%x(2: m + 1)%ka_phase = reg_ka(:, 1)
        self%x(1)%k_freq = 0; self%x(m + 2)%k_freq = 0; self%x(1)%ka_freq = 0; self%x(m + 2)%ka_freq = 0

    end subroutine phase_lagrange_kbr

    function lagrange_value_1d (nd, xd, yd, ni, xi) result(yi)

        !*****************************************************************************80
        !
        !! LAGRANGE_VALUE_1D evaluates the Lagrange interpolant.
        !
        !  Discussion:
        !
        !    The Lagrange interpolant L(ND,XD,YD)(X) is the unique polynomial of
        !    degree ND-1 which interpolates the points (XD(I),YD(I)) for I = 1
        !    to ND.
        !
        !    The Lagrange interpolant can be constructed from the Lagrange basis
        !    polynomials.  Given ND distinct abscissas, XD(1:ND), the I-th Lagrange 
        !    basis polynomial LB(ND,XD,I)(X) is defined as the polynomial of degree 
        !    ND - 1 which is 1 at  XD(I) and 0 at the ND - 1 other abscissas.
        !
        !    Given data values YD at each of the abscissas, the value of the
        !    Lagrange interpolant may be written as
        !
        !      L(ND,XD,YD)(X) = sum ( 1 <= I <= ND ) LB(ND,XD,I)(X) * YD(I)
        !
        !  Licensing:
        !
        !    This code is distributed under the GNU LGPL license.
        !
        !  Modified:
        !
        !    11 September 2012
        !
        !  Author:
        !
        !    John Burkardt
        !
        !  Parameters:
        !
        !    Input, integer ( kind = 4 ) ND, the number of data points.
        !    ND must be at least 1.
        !
        !    Input, real ( kind = 8 ) XD(ND), the data points.
        !
        !    Input, real ( kind = 8 ) YD(ND), the data values.
        !
        !    Input, integer ( kind = 4 ) NI, the number of interpolation points.
        !
        !    Input, real ( kind = 8 ) XI(NI), the interpolation points.
        !
        !    Output, real ( kind = 8 ) YI(NI), the interpolated values.
        !
        implicit none

        integer(ip), INTENT(IN   )                          :: nd
        integer(ip), INTENT(IN   )                          :: ni

        real(wp)                                            :: lb(ni, nd)
        real(wp)   , INTENT(IN   )                          :: xd(nd)
        real(wp)   , INTENT(IN   )                          :: yd(nd)
        real(wp)   , INTENT(IN   )                          :: xi(ni)
        real(wp)                                            :: yi(ni)

        call lagrange_basis_1d (nd, xd, ni, xi, lb)

        yi = matmul (lb, yd)

        return
    end function lagrange_value_1d

    subroutine lagrange_basis_1d (nd, xd, ni, xi, lb)

        !*****************************************************************************80
        !
        !! LAGRANGE_BASIS_1D evaluates a 1D Lagrange basis.
        !
        !  Licensing:
        !
        !    This code is distributed under the GNU LGPL license.
        !
        !  Modified:
        !
        !    09 October 2012
        !
        !  Author:
        !
        !    John Burkardt
        !
        !  Parameters:
        !
        !    Input, integer ( kind = 4 ) ND, the number of data points.
        !
        !    Input, real ( kind = 8 ) XD(ND), the interpolation nodes.
        !
        !    Input, integer ( kind = 4 ) NI, the number of evaluation points.
        !
        !    Input, real ( kind = 8 ) XI(NI), the evaluation points.
        !
        !    Output, real ( kind = 8 ) LB(NI,ND), the value, at the I-th point XI, 
        !    of the Jth basis function.
        !
        implicit none

        integer(ip), INTENT(IN   )                          :: nd
        integer(ip), INTENT(IN   )                          :: ni
        integer(ip)                                         :: i
        integer(ip)                                         :: j
        real(wp)   , INTENT(  OUT)                          :: lb(ni, nd)
        real(wp)   , INTENT(IN   )                          :: xd(nd)
        real(wp)   , INTENT(IN   )                          :: xi(ni)

        do i = 1, ni, 1
            do j = 1, nd, 1
                lb(i, j) = product ( ( xi(i) - xd(1: j - 1)  ) / ( xd(j) - xd(1: j - 1)  ) ) &
                         * product ( ( xi(i) - xd(j + 1: nd) ) / ( xd(j) - xd(j + 1: nd) ) )
            end do
        end do

        return
    end subroutine lagrange_basis_1d

    subroutine phase_wrap_kbr(me, id)
        class(preprocess) , intent(inout)               :: me
        CHARACTER(len = 1), INTENT(IN   )               :: id

        me%x(:)%k_phase = phase_wrap(me%x(:)%k_phase, id)
        me%x(:)%ka_phase = phase_wrap(me%x(:)%ka_phase, id)

    end subroutine phase_wrap_kbr

    subroutine destruct_x(me)
        class(preprocess) , INTENT(INOUT)               :: me
        
        integer(ip)                                     :: err
        
        if (allocated(me%x)) deallocate(me%x, stat=err)
        if (err /= 0) print *, "array: Deallocation request denied"

    end subroutine destruct_x

    subroutine initialise(me, filenames, nheaders, flags, id)
        class(preprocess) , INTENT(INOUT)               :: me

        CHARACTER(len = *), INTENT(IN   )               :: filenames(:), flags(:)
        CHARACTER(len = 1), INTENT(IN   )               :: id
        INTEGER(ip)       , intent(in   )               :: nheaders(:)

        integer(ip)                                     :: i, m, err, ios

        if (size(filenames) /= size(nheaders)) stop "Error occurred in initialising"

        m = size(filenames)

        allocate(me%i_f(m), stat=err)
        if (err /= 0) print *, "me%i_f: Allocation request denied"

        assign_if_loop: do i = 1, m, 1
            me%i_f(i)%name = filenames(i)
            me%i_f(i)%nheader = nheaders(i)
            me%i_f(i)%unit = random_int(20, 200)
            me%i_f(i)%nrow = me%i_f(i)%get_file_n()
            if (.not. allocated(me%x)) then
                allocate(me%x(me%i_f(i)%nrow - me%i_f(i)%nheader), stat=err)
                if (err /= 0) print *, "me%x: Allocation request denied"
            end if
            open(unit=me%i_f(i)%unit, file=me%i_f(i)%name, iostat=ios, status="old", action="read")
            if ( ios /= 0 ) stop "Error opening file name"

            call me%i_f(i)%read_in_data(flags(i), me%x, id)

            close(unit=me%i_f(i)%unit, iostat=ios)
            if ( ios /= 0 ) stop "Error closing file unit me%i_f(i)%unit"
        end do assign_if_loop

        if (allocated(me%i_f)) deallocate(me%i_f, stat=err)
        if (err /= 0) print *, "me%i_f: Deallocation request denied"
    end subroutine initialise

    function random_int(from, too) result(outi)
        integer(ip), intent(in   )                      :: from, too
        integer(ip)                                     :: outi

        real(wp)                                        :: u

        call RANDOM_NUMBER(u)
        outi = from + floor((too + 1 - from) * u)
    end function random_int

    function phase_wrap(phase_in, id) result(phase_out)

        real(wp), INTENT(IN   )                         :: phase_in(:)
        real(wp)                                        :: phase_out(size(phase_in))
        real(wp)                                        :: phase_diff(size(phase_in) - 1)

        integer(ip)                                     :: m
        integer(ip)                                     :: index, i, sign

        character(len = 1)                              :: id

        sign = 0.0_wp
        if (id == 'C') then
            sign = -1.0_wp
        else
            sign = 1.0_wp
        end if

        m = size(phase_in)
        phase_out = phase_in
        phase_diff = abs(phase_in(1: m - 1) - phase_in(2: m))

        index = 1_ip
        detect_wrap_index_loop: do i = 1, size(phase_diff), 1
            index = index + 1
            if (phase_diff(i) > 10000000.0_wp) then
                phase_out(index:) = phase_out(index:) + sign * 100000000.0_wp
            end if
        end do detect_wrap_index_loop

    end function phase_wrap

    function get_file_n(me) result(nrows)
        !! from www.fcode.cn
        implicit none
        CLASS(io_file), intent(in   )                   :: me
        character(len = 1)                              :: cDummy
        integer(ip)                                     :: ierr, ios
        integer(ip)                                     :: nrows

        open(unit=me%unit, file=me%name, iostat=ios, status="old", action="read")
        if ( ios /= 0 ) stop "Error opening file name"

        nrows = 0

        do
            read(me%unit, *, ioStat=ierr) cDummy
            if(ierr /= 0) exit
            nrows = nrows + 1
        end do

        close(unit=me%unit, iostat=ios)
        if ( ios /= 0 ) stop "Error closing file unit me%unit"
    end function get_file_n

    subroutine read_in_data(me, flag, s, id)
        class(io_file)      , INTENT(IN   )             :: me
        class(grace_fo)     , INTENT(INOUT)             :: s(:)

        CHARACTER(len = *  ), intent(in   )             :: flag, id
        character(len = 100)                            :: temp_c

        real(wp)                                        :: temp_r

        integer(ip)                                     :: ios, i, err, J, K

        open(unit=me%unit, file=me%name, iostat=ios, status="old", action="read")
        if ( ios /= 0 ) stop "Error opening file name"
        
        read_header_loop: do i = 1, me%nheader, 1
            read(me%unit, *)
        end do read_header_loop

        read_data_switch: select case (flag)
        
            case ('kbr_phase') ! phase
            read_kbr_phase_loop: do i = 1, me%nrow - me%nheader, 1
                read(me%unit, *) temp_r, temp_r, temp_c, temp_r, temp_r, temp_r, temp_r, s(i)%k_phase, s(i)%ka_phase
            end do read_kbr_phase_loop

            case ('kbr_time') ! time
            read_kbr_time_loop: do i = 1, me%nrow - me%nheader, 1
                read(me%unit, *) s(i)%eps_time_kbr
            end do read_kbr_time_loop

            case ('kbr_freq')
            read_kbr_freq_loop: do i = 1, me%nrow - me%nheader, 1
                read(me%unit, *) s(i)%k_freq, s(i)%ka_freq 
            end do read_kbr_freq_loop

            case ('ant_centr_corr') 
            s%ant_centr_range = 0.0_wp;
            s%ant_centr_rate = 0.0_wp;
            s%ant_centr_accl = 0.0_wp;
            read_ant_centr_corr: do i = 1, me%nrow - me%nheader, 1
                read(me%unit, *) temp_r, temp_c, s(i)%ant_centr_range, s(i)%ant_centr_rate, s(i)%ant_centr_accl
            end do read_ant_centr_corr

            case ('tof')
            s%pos_e(1) = 0.0_wp; s%pos_e(2) = 0.0_wp; s%pos_e(3) = 0.0_wp;
            s%tau_plt = 0.0_wp;
            read_tau_plt_loop: do i = 1, int((me%nrow - me%nheader) / 2.0_wp), 1
                if (id == 'C') then
                    read(me%unit, *)
                    read(me%unit, *) temp_r, temp_c, temp_c, s(i)%tau_plt, s(i)%pos_e
                else
                    read(me%unit, *) temp_r, temp_c, temp_c, s(i)%tau_plt, s(i)%pos_e
                    read(me%unit, *)
                end if
            end do read_tau_plt_loop

            case ('gni1b')
            s%pos_i(1) = 0.0_wp; s%l1b_time = 0.0_wp; s%pos_i(2) = 0.0_wp; s%pos_i(3) = 0.0_wp
            read_gni1b_loop: do i = 1, me%nrow - me%nheader, 1
                read(me%unit, *) s(i)%l1b_time, temp_c, temp_c, s(i)%pos_i, temp_r, temp_r, temp_r, &
                                    s(i)%vel_i
            end do read_gni1b_loop
        end select read_data_switch

        close(unit=me%unit, iostat=ios)
        if ( ios /= 0 ) stop "Error closing file unit me%unit"
        
    end subroutine read_in_data
    
end module gracefo
