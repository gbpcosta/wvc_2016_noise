#!/bin/bash

# read the options
TEMP=`getopt -o f:d:h --long folder:,descriptor:,help -n 'extractAll.sh' -- "$@"`
eval set -- "$TEMP"

function PrintHelp() {
  echo "Parameters:"
  echo "    -f | --folder       : folder with dataset"
  echo "    -d | --descriptor   : descriptor"
  exit 1
}

if [ \( "$#" -lt 5 \) -a \( "$#" -ne 2 \) ]
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
	    -d|--descriptor)
	        case "$2" in
	            "") shift 2 ;;
	            *) DESC=$2 ; shift 2 ;;
	        esac ;;
			-h|--help) PrintHelp ;;
		    --) shift ; break ;;
		    *) echo "Internal error!" ; exit 1 ;;
		esac
	done
fi

echo "Extracting in the folder '"$FOLDER"' with descriptor '"$DESC"'"
echo "Running in original images"
python extract_features.py -d $FOLDER/original -f $DESC

for h in {10,20,30,40,50}
do
  echo "Running in gaussian-"$h" images"
  python extract_features.py -d $FOLDER/gaussian-$h -f $DESC
done

for h in {10,10.5,11,11.5,12}
do
  echo "Running in poisson-"$h" images"
  python extract_features.py -d $FOLDER/poisson-$h -f $DESC
done

for h in {0.1,0.2,0.3,0.4,0.5}
do
  echo "Running in sp-"$h" images"
  python extract_features.py -d $FOLDER/sp-$h -f $DESC
done
