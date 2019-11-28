#this script creates sequences of frames from the videos in clips directory
#using FFMPEG
#If you don't have ffmpeg installed on your system
#run sudo apt-get install ffmpeg

CLIPS_DIR=PIE_clips  #path to the directory with mp4 videos
FRAMES_DIR=images  #path to the directory for frames

################################################################


for set_dir in ${CLIPS_DIR}/set*
do
    mkdir -p ${FRAMES_DIR}/${set_dir}
    for file in ${FRAMES_DIR}/${set_dir}/*.mp4
    do
        filename=$(basename "$file")
        fname="${filename%.*}"
        echo $fname

        #create a directory for each frame sequence
        mkdir -p ${FRAMES_DIR}/${set_dir}/$fname
        #FFMPEG will overwrite any existing images in that directory
        ffmpeg  -y -i $file -start_number 0 -f image2 -qscale 1 ${FRAMES_DIR}/${set_dir}/$fname/%05d.png
        fi
    done
done