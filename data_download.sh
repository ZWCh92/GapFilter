RAW_DIR=/NAS_zzf/A_xff/GapFilter/data/raw
VISIUM_DIR=${RAW_DIR}/Visium
VISIUMHD_DIR=${RAW_DIR}/VisiumHD
XENIUM_DIR=${RAW_DIR}/Xenium

mkdir -p ${VISIUM_DIR} ${VISIUMHD_DIR} ${XENIUM_DIR}


# Visium
mkdir -p ${VISIUM_DIR}/DLPFC/h5 ${VISIUM_DIR}/DLPFC/images ${VISIUM_DIR}/DLPFC/metadata
DLPFC dataset (https://github.com/LieberInstitute/HumanPilot)
for id in 151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676
do
    wget -c -P ${VISIUM_DIR}/DLPFC/h5 \
    https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/${id}_filtered_feature_bc_matrix.h5
    wget -c -P ${VISIUM_DIR}/DLPFC/images \
    https://spatial-dlpfc.s3.us-east-2.amazonaws.com/images/${id}_full_image.tif
done
TMP_DIR="${VISIUM_DIR}/DLPFC/HumanPilot_tmp"
rm -rf "${TMP_DIR}"
git clone https://github.com/LieberInstitute/HumanPilot.git "${TMP_DIR}"
for id in 151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676
do
    mkdir -p "${VISIUM_DIR}/DLPFC/metadata/${id}"
    cp -a "${TMP_DIR}/10X/${id}/." "${VISIUM_DIR}/DLPFC/metadata/${id}/"
done
rm -rf "${TMP_DIR}"
echo "DLPFC Visium metadata downloaded successfully."


# VisiumHD
mkdir -p ${VISIUMHD_DIR}/HBC ${VISIUMHD_DIR}/HPC
# Human Breast Cancer (https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-breast-cancer-ff-ultima)
wget -c -P ${VISIUMHD_DIR}/HBC https://cf.10xgenomics.com/samples/spatial-exp/3.1.2/Visium_HD_FF_Human_Breast_Cancer/Visium_HD_FF_Human_Breast_Cancer_tissue_image.tif
wget -c -P ${VISIUMHD_DIR}/HBC https://cf.10xgenomics.com/samples/spatial-exp/3.1.2/Visium_HD_FF_Human_Breast_Cancer/Visium_HD_FF_Human_Breast_Cancer_feature_slice.h5
wget -c -P ${VISIUMHD_DIR}/HBC https://cf.10xgenomics.com/samples/spatial-exp/3.1.2/Visium_HD_FF_Human_Breast_Cancer/Visium_HD_FF_Human_Breast_Cancer_metrics_summary.csv
wget -c -P ${VISIUMHD_DIR}/HBC https://cf.10xgenomics.com/samples/spatial-exp/3.1.2/Visium_HD_FF_Human_Breast_Cancer/Visium_HD_FF_Human_Breast_Cancer_spatial.tar.gz
wget -c -P ${VISIUMHD_DIR}/HBC https://s3-us-west-2.amazonaws.com/10x.files/samples/spatial-exp/3.1.2/Visium_HD_FF_Human_Breast_Cancer/Visium_HD_FF_Human_Breast_Cancer_binned_outputs.tar.gz
# Human Prostate Cancer (https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-prostate-cancer-ffpe)
wget -c -P ${VISIUMHD_DIR}/HPC https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Human_Prostate_Cancer_FFPE/Visium_HD_Human_Prostate_Cancer_FFPE_tissue_image.tif
wget -c -P ${VISIUMHD_DIR}/HPC https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Human_Prostate_Cancer_FFPE/Visium_HD_Human_Prostate_Cancer_FFPE_binned_outputs.tar.gz
wget -c -P ${VISIUMHD_DIR}/HPC https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Human_Prostate_Cancer_FFPE/Visium_HD_Human_Prostate_Cancer_FFPE_feature_slice.h5
wget -c -P ${VISIUMHD_DIR}/HPC https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Human_Prostate_Cancer_FFPE/Visium_HD_Human_Prostate_Cancer_FFPE_metrics_summary.csv
wget -c -P ${VISIUMHD_DIR}/HPC https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Human_Prostate_Cancer_FFPE/Visium_HD_Human_Prostate_Cancer_FFPE_spatial.tar.gz


# Xenium
mkdir -p ${XENIUM_DIR}/HBC_S1R1 ${XENIUM_DIR}/HBC_S1R2
# Human Breast Cancer (Sample 1, Replicate 1)(https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast)
wget -c -P ${XENIUM_DIR}/HBC_S1R1 https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif
wget -c -P ${XENIUM_DIR}/HBC_S1R1 https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif
wget -c -P ${XENIUM_DIR}/HBC_S1R1 https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_imagealignment.csv
wget -c -P ${XENIUM_DIR}/HBC_S1R1 https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_gene_groups.csv
wget -c -P ${XENIUM_DIR}/HBC_S1R1 https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip
# Human Breast Cancer (Sample 1, Replicate 2)(https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast)
wget -c -P ${XENIUM_DIR}/HBC_S1R2 https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.tif
wget -c -P ${XENIUM_DIR}/HBC_S1R2 https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif
wget -c -P ${XENIUM_DIR}/HBC_S1R2 https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_imagealignment.csv
wget -c -P ${XENIUM_DIR}/HBC_S1R2 https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_gene_groups.csv
wget -c -P ${XENIUM_DIR}/HBC_S1R2 https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_outs.zip
