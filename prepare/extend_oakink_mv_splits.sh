# mode_split and data_split has following options:
# mode_split:
#    "subject"            SP1, subject split, subjects recorded in the test will not appear in the train split;
#    "object"             SP2, objects split, objects recorded in the test will not appear in the train split;
# --------------------
# data_split:
#    all, train+val, test, train, val

echo "Extend the OakInk data splits for multi-view setting"

python thirdparty/OakInk/dev/extend_split.py --data_dir data/OakInk --mode_split subject --data_split train+val
python thirdparty/OakInk/dev/extend_split.py --data_dir data/OakInk --mode_split subject --data_split test

python thirdparty/OakInk/dev/extend_split.py --data_dir data/OakInk --mode_split object --data_split train+val
python thirdparty/OakInk/dev/extend_split.py --data_dir data/OakInk --mode_split object --data_split test

echo "Done, results are saved in data/OakInk/image/anno_mv"