c         PROGRAM FOR reading drf data

          PARAMETER (isiz=35,jsiz=33)

          real XGRID(366,ISIZ,JSIZ),MASK(ISIZ,JSIZ)
          REAL LAT(JSIZ),LON(ISIZ)
          character*6 aday(366)

          OPEN(8,FILE='MAPGRID35X33.TXT',STATUS='OLD')
          OPEN(10,FILE='drf_1956.prt',status='old')
          OPEN(12,FILE='dumb.prt',status='unknown')

          DO J=1,JSIZ
            READ(8,20)(mask(i,j),i=1,isiz)
          ENDDO
20      format(35f4.1)

          nday=366  ! for leap year change 365 to 366

          LAT(1)=6.5
          DO K=2,JSIZ
            LAT(K)=LAT(K-1)+1.0
c            WRITE(*,*)LAT(K)
          ENDDO

          LON(1)=66.5
          DO I= 2,ISIZ
            LON(I)=LON(I-1)+1.0
c            WRITE(*,*)I,' ',LON(I)
          ENDDO

          do iday=1,nday
            do i=1,isiz
              do j=1,jsiz
                xgrid(iday,i,j)=-99.9
              enddo
            enddo
          enddo

          do iday=1,nday
            read(10,'(6x,a6)')aday(iday)
            read(10,40)(LON(JJ),JJ=1,ISIZ-3)

            do j=jsiz-1,3,-1
               read(10,50)alat,(xgrid(iday,i,j),i=1,isiz-3)
            enddo
          read(10,*)
            do i=1,isiz
              do j=1,jsiz
               if(mask(i,j).eq.0.0) then
                xgrid(iday,i,j)=-99.9
               endif
             enddo
            enddo

            write(12,*)' day= ',aday(iday)
            WRITE(12,45)(LON(JJ),JJ=1,ISIZ)
            do j=jsiz,1,-1
               write(12,60)LAT(J),(xgrid(iday,i,j),i=1,isiz)
            enddo

          enddo

40      format(7X,35(f5.1,1x))
45      format(6X,35(f5.1,'E'))

50      format(1X,F4.1,1x,35(1x,f5.1))
60      format(1X,F4.1,'N',35(1x,f5.1))

          CLOSE(10)


          STOP
          END

