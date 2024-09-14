# .bachrc

export USER=$LOGNAME
export PATH=$HOME/bin:/bin:/usr/bin:/usr/local/bin:/opt/bin
export MAIL=/var/spool/mail/$USER

if [ $(uname -s) = 'SunOS' ]; then
    export PATH=$PATH:/usr/ucb:/usr/ccs/bin:/usr/local/workshop/bin
    export PATH=$PATH:/usr/X11R6/bin:/usr/X11R5/bin:/usr/openwin/bin

    export MANPATH=/usr/man:/usr/local/man:/usr/X11R6/man:/usr/X11R5/man:/usr/motif1.2/man:/usr/share/catman:/opt/SUNWspro/man

    # for CXterm
    export HZINPUTDIR=/usr/X11R6/lib/X11/cxterm.dic
    export HBFPATH=/usr/local/chinese/fonts/cnprint:/usr/X11R6/lib/X11/fonts/chpower
    alias b5hztty='hztty -O hz2gb:gb2big -I big2gb:gb2hz'
fi

export PS1='\h:\w> '

alias ls='ls -aF'
alias cp='cp -i'
alias mv='mv -i'
alias rm='rm -i'
ulimit -c 0
umask 077

#Cache Server
export http_proxy=http://proxy.cse.cuhk.edu.hk:8000/
export https_proxy=http://proxy.cse.cuhk.edu.hk:8000/
export ftp_proxy=http://proxy.cse.cuhk.edu.hk:8000/
export gopher_proxy=http://proxy.cse.cuhk.edu.hk:8000/
export WWW_HOME=http://www.cse.cuhk.edu.hk/

##
## put command run after interactive login in ~/.bash_profile
##
