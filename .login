tset -I -Q
setenv EXINIT 'set redraw wm=8'
setenv EDITOR vi
date

if ( `uname -p` == 'sparc' ) then
    biff n
endif
