tty -s
if test $? = 0
then
	stty dec crt
fi
PATH=$HOME/bin:/usr/ucb:/bin:/usr/bin:/usr/local/bin:/usr/X11R5/bin:/opt/bin
MAIL=/var/spool/mail/$USER
tset -n -I
export TERM MAIL PATH
biff n

umask 077
