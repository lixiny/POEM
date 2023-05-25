echo "Pack the OakInk-MV annotations into a single archive for each sample"
echo "This may take a while ..."

echo "1/4"
python thirdparty/OakInk/dev/pack_oakink_image_mv.py --data_dir data/OakInk --mode_split subject --data_split train+val
echo "2/4"
python thirdparty/OakInk/dev/pack_oakink_image_mv.py --data_dir data/OakInk --mode_split subject --data_split test

echo "3/4"
python thirdparty/OakInk/dev/pack_oakink_image_mv.py --data_dir data/OakInk --mode_split object --data_split train+val
echo "4/4"
python thirdparty/OakInk/dev/pack_oakink_image_mv.py --data_dir data/OakInk --mode_split object --data_split test

echo "Done, results are saved in data/OakInk/image/anno_packed_mv"