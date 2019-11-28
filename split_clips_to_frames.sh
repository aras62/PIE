#this script creates sequences of frames from the videos in clips directory
#using FFMPEG
#If you don't have ffmpeg installed on your system
#run sudo apt-get install ffmpeg

CLIPS_DIR=PIE_clips  #path to the directory with mp4 videos
FRAMES_DIR=images  #path to the directory for frames

################################################################


for set_dir in set01 set02 set03 set04 set05 set06
do
    for video in ${CLIPS_DIR}/${set_dir}/*
    do
        filename=$(basename "$video")
        fname="${filename%.*}"

        #create a directory for each frame sequence
        mkdir -p ${FRAMES_DIR}/${set_dir}/$fname
        #FFMPEG will overwrite any existing images in that directory
        echo ffmpeg  -y -i $file -start_number 0 -f image2 -qscale 1 ${FRAMES_DIR}/${set_dir}/$fname/%05d.png
        
    done
done