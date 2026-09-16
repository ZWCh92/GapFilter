# d_model调整
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.d_model=256"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.d_model=256"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.d_model=512"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.d_model=512" 

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.d_model=1024" # great
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.d_model=1024"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.d_model=2048"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.d_model=2048"


# # cycle_weight调整
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.cycle_weight=0.0" # great
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.cycle_weight=0.0"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.cycle_weight=0.5"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.cycle_weight=0.5"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.cycle_weight=1.0"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.cycle_weight=1.0"


# n_neighbors调整
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.n_neighbors=3" "model.predict_n_neighbors=3"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.n_neighbors=3" "model.predict_n_neighbors=3"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.n_neighbors=6" "model.predict_n_neighbors=6"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.n_neighbors=6" "model.predict_n_neighbors=6"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.n_neighbors=9" "model.predict_n_neighbors=9"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.n_neighbors=9" "model.predict_n_neighbors=9"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.n_neighbors=12" "model.predict_n_neighbors=12"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.n_neighbors=12" "model.predict_n_neighbors=12"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.n_neighbors=15" "model.predict_n_neighbors=15"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.n_neighbors=15" "model.predict_n_neighbors=15"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.n_neighbors=18" "model.predict_n_neighbors=18" # great
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.n_neighbors=18" "model.predict_n_neighbors=18"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.n_neighbors=21" "model.predict_n_neighbors=21"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.n_neighbors=21" "model.predict_n_neighbors=21"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.n_neighbors=24" "model.predict_n_neighbors=24"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.n_neighbors=24" "model.predict_n_neighbors=24"

# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/train.py "model.n_neighbors=27" "model.predict_n_neighbors=27"
# latest_dir=$(ls -dt /mnt/zzf_nas/A_xff/GapFilter/models/gapfilter_3/VisiumHD/HPC/*/ | head -n 1)
# python /mnt/zzf_nas/A_xff/GapFilter/gapfilter/gapfilter_3/predict.py experiment=predict "model.ckpt_dir=${latest_dir}" "model.n_neighbors=27" "model.predict_n_neighbors=27"