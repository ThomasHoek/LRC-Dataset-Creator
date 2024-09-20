rm -r temp/
FILES=`ls -r */*/sick.log`

for i in $FILES
do
    # echo $i
    filename=$(echo $i | cut -f 1 -d '/')
    model=$(echo $i | cut -f 2 -d '/')
    # echo $model / $filename
    mkdir -p temp/$model
    tail -16 "$i" > temp/$model/$filename.txt
done