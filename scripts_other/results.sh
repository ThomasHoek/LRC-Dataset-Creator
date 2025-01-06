rm -r temp/
FILES=`find . -name "*.log"`

for i in $FILES
do
    filename=$(echo $i |  rev | cut -f 1 -d '/' | rev)
    model=$(echo $i | rev | cut -f 2 -d '/' | rev)
    echo $model / $filename
    mkdir -p temp/models/$model
    mkdir -p temp/parts/$filename
    tail -16 "$i" > temp/models/$model/$filename.txt
    tail -16 "$i" > temp/parts/$filename/$model.txt
done