URL=https://github.com/pytorch/fairseq
FOLDER=fairseq

if [ ! -d "$FOLDER" ] ; then
    git clone "$URL" "$FOLDER"
fi

cd "$FOLDER" || exit
pip install --editable ./
