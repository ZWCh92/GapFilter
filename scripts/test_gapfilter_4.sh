# 所有数据上测试
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=xenium_hbc_s1r1
# latest_dir=$(ls -dt models/gapfilter_4/Xenium/HBC_S1R1/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=xenium_hbc_s1r1 experiment=predict "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=xenium_hbc_s1r2
# latest_dir=$(ls -dt models/gapfilter_4/Xenium/HBC_S1R2/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=xenium_hbc_s1r2 experiment=predict "model.ckpt_dir=${latest_dir}"


# # 参数调整
# # model.d_model=512
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.d_model=512"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.d_model=512" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.d_model=512"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.d_model=512" "model.ckpt_dir=${latest_dir}"

# # model.d_model=2048
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.d_model=2048"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.d_model=2048" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.d_model=2048"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.d_model=2048" "model.ckpt_dir=${latest_dir}"



# # model.residual_rank=32
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_rank=32"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_rank=32" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_rank=32"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_rank=32" "model.ckpt_dir=${latest_dir}"

# # model.residual_rank=64
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_rank=64"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_rank=64" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_rank=64"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_rank=64" "model.ckpt_dir=${latest_dir}"

# # model_residual_rank=96
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_rank=96"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_rank=96" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_rank=96"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_rank=96" "model.ckpt_dir=${latest_dir}"



# # model.residual_gamma=0.2
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_gamma=0.2"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_gamma=0.2" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_gamma=0.2"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_gamma=0.2" "model.ckpt_dir=${latest_dir}"

# # model.residual_gamma=0.3
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_gamma=0.3"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_gamma=0.3" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_gamma=0.3"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_gamma=0.3" "model.ckpt_dir=${latest_dir}"

# # model.residual_gamma=0.4
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_gamma=0.4"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_gamma=0.4" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_gamma=0.4"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_gamma=0.4" "model.ckpt_dir=${latest_dir}"




# # model.pearson_weight=0.25
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.pearson_weight=0.25"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.pearson_weight=0.25" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.pearson_weight=0.25"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.pearson_weight=0.25" "model.ckpt_dir=${latest_dir}"

# # model.pearson_weight=0.75
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.pearson_weight=0.75"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.pearson_weight=0.75" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.pearson_weight=0.75"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.pearson_weight=0.75" "model.ckpt_dir=${latest_dir}"



# # model.cosine_weight=0.4
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.cosine_weight=0.4"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.cosine_weight=0.4" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.cosine_weight=0.4"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.cosine_weight=0.4" "model.ckpt_dir=${latest_dir}"

# # model.cosine_weight=0.6
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.cosine_weight=0.6"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.cosine_weight=0.6" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.cosine_weight=0.6"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.cosine_weight=0.6" "model.ckpt_dir=${latest_dir}"

# # model.cosine_weight=0.8
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.cosine_weight=0.8"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.cosine_weight=0.8" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.cosine_weight=0.8"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.cosine_weight=0.8" "model.ckpt_dir=${latest_dir}"



# 二轮调参
# # model_residual_rank=128
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_rank=128"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_rank=128" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_rank=128"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_rank=128" "model.ckpt_dir=${latest_dir}"

# # model_residual_rank=192
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_rank=192"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_rank=192" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_rank=192"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_rank=192" "model.ckpt_dir=${latest_dir}"

# # model_residual_rank=256
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_rank=256"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_rank=256" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_rank=256"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_rank=256" "model.ckpt_dir=${latest_dir}"



# # model.residual_gamma=0.6
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_gamma=0.6"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_gamma=0.6" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_gamma=0.6"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_gamma=0.6" "model.ckpt_dir=${latest_dir}"

# # model.residual_gamma=0.8
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_gamma=0.8"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_gamma=0.8" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_gamma=0.8"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_gamma=0.8" "model.ckpt_dir=${latest_dir}"

# # model.residual_gamma=1.0
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_gamma=1.0"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_gamma=1.0" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_gamma=1.0"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_gamma=1.0" "model.ckpt_dir=${latest_dir}"


# # model.pearson_weight=1.0
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.pearson_weight=1.0"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.pearson_weight=1.0" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.pearson_weight=1.0"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.pearson_weight=1.0" "model.ckpt_dir=${latest_dir}"


# # model.cosine_weight=1.0
# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.cosine_weight=1.0"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.cosine_weight=1.0" "model.ckpt_dir=${latest_dir}"

# python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.cosine_weight=1.0"
# latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
# python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.cosine_weight=1.0" "model.ckpt_dir=${latest_dir}"




# 三轮调参
# model_residual_rank=512
python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_rank=512"
latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_rank=512" "model.ckpt_dir=${latest_dir}"

python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_rank=512"
latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_rank=512" "model.ckpt_dir=${latest_dir}"

# model_residual_rank=768
python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_rank=768"
latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_rank=768" "model.ckpt_dir=${latest_dir}"

python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_rank=768"
latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_rank=768" "model.ckpt_dir=${latest_dir}"

# model_residual_rank=1024
python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hpc "model.residual_rank=1024"
latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HPC/*/ | head -n 1)
python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hpc experiment=predict "model.residual_rank=1024" "model.ckpt_dir=${latest_dir}"

python gapfilter/gapfilter_4/train.py model=gapfilter_4 dataset=visiumhd_hbc "model.residual_rank=1024"
latest_dir=$(ls -dt models/gapfilter_4/VisiumHD/HBC/*/ | head -n 1)
python gapfilter/gapfilter_4/predict.py model=gapfilter_4 dataset=visiumhd_hbc experiment=predict "model.residual_rank=1024" "model.ckpt_dir=${latest_dir}"