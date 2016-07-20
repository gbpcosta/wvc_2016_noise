#!/bin/bash

# read the options
TEMP=`getopt -o f:m:l:n:h --long folder:,method:,level:,noise:,help -n 'denoisingAll.sh' -- "$@"`
eval set -- "$TEMP"

function PrintHelp() {
  echo "Parameters:"
  echo "    -f | --folder       : folder with dataset"
  echo "    -m | --method       : denoising method [NLM, Bilateral or Median]"
  echo "    -l | --level        : level of denoising method"
  echo "    -n | --noise        : noise type (subfolder) [gaussian, poisson or sp]"
  exit 1
}

if [ \( "$#" -lt 9 \) -a \( "$#" -ne 2 \) ]
then
	echo -e "Missing pararameters! \nSee -h or --help";
	exit 1
else
	while true ; do

		case "$1" in
	    -f|--folder)
	        case "$2" in
	            "") shift 2 ;;
	            *) FOLDER=$2 ; shift 2 ;;
	        esac ;;
	    -m|--method)
	        case "$2" in
	            "") shift 2 ;;
	            *) METHOD=$2 ; shift 2 ;;
	        esac ;;
      -l|--level)
          case "$2" in
              "") shift 2 ;;
              *) LEVEL=$2 ; shift 2 ;;
          esac ;;
      -n|--noise)
          case "$2" in
              "") shift 2 ;;
              *) NOISE=$2 ; shift 2 ;;
          esac ;;
			-h|--help) PrintHelp ;;
		    --) shift ; break ;;
		    *) echo "Internal error!" ; exit 1 ;;
		esac
	done
fi

if [ $NOISE == "gaussian" ]; then
  list="10 20 30 40 50"
elif [ $NOISE == "poisson" ]; then
  list="10 10.5 11 11.5 12"
elif [ $NOISE == "sp" ]; then
  list="0.1 0.2 0.3 0.4 0.5"
fi

echo "Denoising in the folder '"$FOLDER"' with filter '"$METHOD"' in noise '"${NOISE^^}"'"
for i in $list
do
  echo "Applying denoising in $NOISE-"$i" images"
  python denoising.py -in $FOLDER/$NOISE-$i -l $LEVEL -m $METHOD
done
