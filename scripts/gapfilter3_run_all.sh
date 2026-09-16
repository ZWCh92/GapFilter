# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py dataset=visiumhd_hbc
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HBC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py dataset=visiumhd_hbc experiment=predict "model.ckpt_dir=${latest_dir}"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py dataset=visiumhd_hpc
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py dataset=visiumhd_hpc experiment=predict "model.ckpt_dir=${latest_dir}"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py dataset=xenium_hbc_s1r1 
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/Xenium/HBC_S1R1/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py dataset=xenium_hbc_s1r1 experiment=predict "model.ckpt_dir=${latest_dir}"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py dataset=xenium_hbc_s1r2 
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/Xenium/HBC_S1R2/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py dataset=xenium_hbc_s1r2 experiment=predict "model.ckpt_dir=${latest_dir}"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py dataset=visium_dlpfc 
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/Visium/DLPFC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py dataset=visium_dlpfc experiment=predict "model.ckpt_dir=${latest_dir}"




