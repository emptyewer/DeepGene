echo "***** Environment Variables *****"
echo PATH: $PATH
echo ""

IP=`ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'`
echo "***** IP Address *****"
echo -e "$IP"
echo ""

USER_ID=${LOCAL_USER_ID:-9001}
GROUP_ID=${LOCAL_GROUP_ID:-100}

echo "User ID: $USER_ID and Group ID: $GROUP_ID"
echo ""
echo "***** STDOUT *****"

useradd --shell /bin/bash -u $USER_ID -g $GROUP_ID -o -c "" -m user
export HOME=/home/user
exec /usr/local/bin/gosu user "$@"

