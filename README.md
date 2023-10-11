# Model Overview
The study keenly recognizes the limitations of [SwinUNETR](https://arxiv.org/pdf/2201.01266.pdf) [1] and presents a creative method by integrating the [Metaformer](https://doi.org/10.48550/arXiv.2111.11418) [2]'s idea of replacing attention blocks with average pooling layers. This approach efficiently reduces computational requirements while maintaining performance. Additionally, the inclusion of the Squeeze-and-Excitation (SE) Block skillfully captures crucial global information in a flexible way. This repository contains the code for ```MetaSwin``` for the task of brain tumor segmentation using the [BraTS 23](https://www.synapse.org/#!Synapse:syn51156910/wiki/621282) challenge dataset [3,4]. The architecture of MetaSwin is demonstrated as below
![figure 1](https://github.com/soyeon1608/MetaSwin/assets/100922793/3eda3191-00b4-4242-a9f9-06328f9bc2eb)

# Installing Dependencies
MONAI installation
To install the current milestone release:
``` bash
pip install monai
```
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

# Data Description
## ```BraTS 2023```

Modality: MRI
Size: 1470 3D volumes (1251 Training + 219 Validation)
Challenge: RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge

- Register and download the official BraTS 23 dataset from the link below and place then into "TrainingData" in the dataset folder:

  https://www.synapse.org/#!Synapse:syn51514105


- Download the json file from this [link](https://github.com/soyeon1608/MetaSwin/tree/main/assets) and placed in the same folder as the dataset.

All BraTS mpMRI scans are available as NIfTI files (.nii.gz) and describe (a) native (T1) and (b) post-contrast T1-weighted (T1Gd), (c) T2-weighted (T2), and (d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple data contributing institutions.
All the imaging datasets have been annotated manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 3), the peritumoral edematous/invaded tissue (ED — label 2), and the necrotic tumor core (NCR — label 1), as described in the latest BraTS summarizing paper. The ground truth data were created after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution (1 mm3) and skull-stripped.

The sub-regions considered for evaluation are the "enhancing tumor" (ET), the "tumor core" (TC), and the "whole tumor" (WT) [see figure below]. The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (NCR) parts of the tumor. The appearance of NCR is typically hypo-intense in T1-Gd when compared to T1. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edematous/invaded tissue (ED), which is typically depicted by hyper-intense signal in FLAIR.

The provided segmentation labels have values of 1 for NCR, 2 for ED, 3 for ET, and 0 for everything else.

![brats2023](https://github.com/soyeon1608/MetaSwin/assets/100922793/e3ecb4c0-3a22-4776-bed4-d7ad029fdbb0)

Figure from [Adewole et al.](https://arxiv.org/ftp/arxiv/papers/2305/2305.19369.pdf) [2]

## ```BTCV```
![image](https://lh3.googleusercontent.com/pw/AM-JKLX0svvlMdcrchGAgiWWNkg40lgXYjSHsAAuRc5Frakmz2pWzSzf87JQCRgYpqFR0qAjJWPzMQLc_mmvzNjfF9QWl_1OHZ8j4c9qrbR6zQaDJWaCLArRFh0uPvk97qAa11HtYbD6HpJ-wwTCUsaPcYvM=w1724-h522-no?authuser=0)

The training data is from the [BTCV challenge dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752).

- Target: 13 abdominal organs including 1. Spleen 2. Right Kidney 3. Left Kideny 4.Gallbladder 5.Esophagus 6. Liver 7. Stomach 8.Aorta 9. IVC 10. Portal and Splenic Veins 11. Pancreas 12.Right adrenal gland 13.Left adrenal gland.
- Task: Segmentation
- Modality: CT
- Size: 30 3D volumes (24 Training + 6 Testing)

Please download the json file from this link.

We provide the json file that is used to train our models in the following <a href="https://drive.google.com/file/d/1t4fIQQkONv7ArTSZe4Nucwkk1KfdUDvW/view?usp=sharing"> link</a>.

Once the json file is downloaded, please place it in the same folder as the dataset. Note that you need to provide the location of your dataset directory by using ```--data_dir```.

# Training
## ```BraTS 2023```

A MetaSwin network with standard hyper-parameters for brain tumor semantic segmentation (BraTS dataset) is be defined as:

``` bash
model = MetaSwin(img_size=(128,128,128),
                  in_channels=4,
                  out_channels=3,
                  feature_size=48,
                  use_checkpoint=True,
                  )
```


The above MetaSwin model is used for multi-modal MR images (4-channel input) with input image size ```(128, 128, 128)``` and for ```3``` class segmentation outputs and feature size of  ```48```.

To train a `MetaSwin` from scratch on a single GPU:

```bash
python main_MetaSwin.py --json_list=<json-path> --data_dir=<data-path> --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --feature_size=48
```

## ```BTCV```

A MetaSwin network with standard hyper-parameters for multi-organ semantic segmentation (BTCV dataset) is be defined as:

``` bash
model = MetaSwin(img_size=(96,96,96),
                  in_channels=1,
                  out_channels=14,
                  feature_size=48,
                  use_checkpoint=True,
                  )
```

The above MetaSwin model is used for CT images (1-channel input) with input image size ```(96, 96, 96)``` and for ```14``` class segmentation outputs and feature size of  ```48```.

# Evaluation

To evaluate a `MetaSwin` on a single GPU, the model path using `pretrained_dir` and model
name using `--pretrained_model_name` need to be provided:

```bash
python test.py --json_list=<json-path> --data_dir=<data-path> --feature_size=<feature-size>\
--infer_overlap=0.7 --pretrained_model_name=<model-name> --pretrained_dir=<model-dir>
```

# Segmentation Output

By following the commands for evaluating `MetaSwin` in the above, `test.py` saves the segmentation outputs
in the original spacing in a new folder based on the name of the experiment which is passed by `--exp_name`.

# Visualization
![Visualization](https://github.com/soyeon1608/MetaSwin/assets/100922793/0e91d938-8286-41ed-a543-557b55e1f04d)

# References
[1]: Hatamizadeh, Ali, et al. "Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images." International MICCAI Brainlesion Workshop. Cham: Springer International Publishing, 2021.

[2]: Yu, Weihao, et al. "Metaformer is actually what you need for vision." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[3]: Adewole, Maruf, et al. "The Brain Tumor Segmentation (BraTS) Challenge 2023: Glioma Segmentation in Sub-Saharan Africa Patient Population (BraTS-Africa)." arXiv preprint arXiv:2305.19369 (2023).

[4]: Kazerooni, Anahita Fathi, et al. "The Brain Tumor Segmentation (BraTS) Challenge 2023: Focus on Pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs)." arXiv preprint arXiv:2305.17033 (2023).
