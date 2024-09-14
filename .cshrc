if ($?prompt) then
	set prompt="`uname -n`> "
	set notify
	set history = 50
	alias pd pushd
	alias pop popd
	alias vt52 "set term = vt52"
	alias ti745 "set term = ti745 ; stty -tabs"
	alias ti785 "set term = ti745 ; stty -tabs"
	alias vt100 "set term = vt100"
	stty echoe
endif

umask 077
setenv MAIL /var/spool/mail/$USER

set path=($HOME/bin /bin /usr/bin /usr/local/bin /opt/bin)

if ( `uname -s` == 'SunOS' ) then
    set path=($path /usr/ucb /usr/ccs/bin /usr/local/workshop/bin )
    set path=($path /usr/X11R6/bin /usr/X11R5/bin /usr/openwin/bin)
  
    setenv MANPATH /usr/man:/usr/local/man:/usr/X11R6/man:/usr/X11R5/man:/usr/motif1.2/man:/usr/share/catman
    set mail=$MAIL

    # for CXterm
    setenv HZINPUTDIR /usr/X11R6/lib/X11/cxterm.dic
    setenv HBFPATH /usr/local/chinese/fonts/cnprint:/usr/X11R6/lib/X11/fonts/chpower
    alias b5hztty 'hztty -O hz2gb:gb2big -I big2gb:gb2hz'
endif

set filec
set noclobber
set prompt="`uname -n`:${cwd}> "
set prompt='%S %s%m:%~> '

alias ls 'ls -aF'
alias cp 'cp -i'
alias mv 'mv -i'
alias rm 'rm -i'
alias pwd 'echo $cwd'
# alias cd 'cd \!*; set prompt = "`uname -n`:${cwd}> "'
limit coredumpsize 0

#Cache Server
setenv http_proxy http://proxy.cse.cuhk.edu.hk:8000/
setenv ftp_proxy http://proxy.cse.cuhk.edu.hk:8000/
setenv gopher_proxy http://proxy.cse.cuhk.edu.hk:8000/
setenv WWW_HOME http://www.cse.cuhk.edu.hk/

# set path=($path /local/sas/SAS_8.2)
# Setting for CSCI2100A
# set path=($path /uac/cact/csci2100a/bin)
