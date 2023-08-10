# iEnhancer-TransformerEncoder

The project is a Transformer Encoder-based model used for classifying enhancer and non-enhancer regions, as well as identifying the strength of enhancers.

### contents

+ first_step: Directory for the model and training code for classifying enhancer and non-enhancer regions.
+ second_step: Directory for the model and training code for identifying enhancer strength.
+ train_datas: Training data.
+ WGAN-GP: Model, training, and data augmentation code for WGAN-GP.

### command

#### GAN

```shell
# Train GAN to generate weak enhancers.
nohup python wgan_gp.py --gpu_id 0 --batch_size 8 --model_name gan_weak_enhancer --data_dir ../train_datas/gan/weak_enhancer.txt > out/gan_weak_enhancer.out &

# Train GAN to generate strong enhancers.
nohup python wgan_gp.py --gpu_id 1 --batch_size 8 --model_name gan_strong_enhancer --data_dir ../train_datas/gan/strong_enhancer.txt > out/gan_strong_enhancer.out &

# Train GAN to generate non-enhancers.
nohup python wgan_gp.py --gpu_id 0 --batch_size 16 --model_name gan_non_enhancer --data_dir ../train_datas/gan/non_enhancer.txt > out/gan_non_enhancer.out &

# Generate data (to validate GAN).
python generate_fake_data.py --model_name gan_strong_enhancer --type strong --amount 10
```

#### Classify enhancers and non-enhancers.

```shell
# Train the model for classifying enhancers (without GAN).
PYTHONUNBUFFERED=1 nohup python classification_without_gan.py --model_name class_without_gan --num_gpus 2  > out/class_without_gan.out &

# Train the model for classifying enhancers (with GAN).
PYTHONUNBUFFERED=1 nohup python classification.py --model_name class_with_gan --num_gpus 2  > out/class_with_gan.out &
```

#### Predict the strength of enhancers.

```shell
# Train the model for predicting the strength of enhancers (without GAN).
PYTHONUNBUFFERED=1 nohup python prediction_without_gan.py --model_name predict_without_gan --num_gpus 2 > out/predict_without_gan.out &

# Train the model for predicting the strength of enhancers (with GAN).
PYTHONUNBUFFERED=1 nohup python prediction.py --model_name predict_with_gan --num_gpus 2 > out/predict_with_gan.out &
```

