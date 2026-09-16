python /data1/wangzeyu/st_ot/GapFilter/gapfilter/gapfilter_4/train.py dataset=visiumhd_hbc
latest_dir=$(ls -dt /data1/wangzeyu/st_ot/GapFilter/models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
python /data1/wangzeyu/st_ot/GapFilter/gapfilter/gapfilter_4/predict.py dataset=visiumhd_hbc experiment=predict "model.ckpt_dir=${latest_dir}"

python /data1/wangzeyu/st_ot/GapFilter/gapfilter/gapfilter_4/train.py dataset=visiumhd_hpc
latest_dir=$(ls -dt /data1/wangzeyu/st_ot/GapFilter/models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
python /data1/wangzeyu/st_ot/GapFilter/gapfilter/gapfilter_4/predict.py dataset=visiumhd_hpc experiment=predict "model.ckpt_dir=${latest_dir}"

python /data1/wangzeyu/st_ot/GapFilter/gapfilter/gapfilter_4/train.py dataset=xenium_hbc_s1r1 
latest_dir=$(ls -dt /data1/wangzeyu/st_ot/GapFilter/models/gapfilter_4/Xenium/HBC_S1R1/*/ | head -n 1)
python /data1/wangzeyu/st_ot/GapFilter/gapfilter/gapfilter_4/predict.py dataset=xenium_hbc_s1r1 experiment=predict "model.ckpt_dir=${latest_dir}"

python /data1/wangzeyu/st_ot/GapFilter/gapfilter/gapfilter_4/train.py dataset=xenium_hbc_s1r2 
latest_dir=$(ls -dt /data1/wangzeyu/st_ot/GapFilter/models/gapfilter_4/Xenium/HBC_S1R2/*/ | head -n 1)
python /data1/wangzeyu/st_ot/GapFilter/gapfilter/gapfilter_4/predict.py dataset=xenium_hbc_s1r2 experiment=predict "model.ckpt_dir=${latest_dir}"

python /data1/wangzeyu/st_ot/GapFilter/gapfilter/gapfilter_4/train.py dataset=visium_dlpfc 
latest_dir=$(ls -dt /data1/wangzeyu/st_ot/GapFilter/models/gapfilter_4/Visium/DLPFC/*/ | head -n 1)
python /data1/wangzeyu/st_ot/GapFilter/gapfilter/gapfilter_4/predict.py dataset=visium_dlpfc experiment=predict "model.ckpt_dir=${latest_dir}"