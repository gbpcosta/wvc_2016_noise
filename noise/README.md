### Scripts for insert noise in images (Poisson and Gaussian) ###

Run python script:

* **-in** or **--input**: Input PATH with the input images (Color images)
* **-type** or **--noiseType**: Choose which noise will apply (gaussian or poisson) **(Poisson noise doesn't work yet :) )**
* **-sig** or **--sigma**: Input the parameter Sigma (Noise value for Gaussian noise)
* **-lam** or **--lambdav**: Input the parameter Lambda (Noise value for Poisson noise)

```
#!zsh
python2.7 generateNoise.py -in /home/user/datasets/dtd/images/ -type gaussian -sig 10

```

Run MATLAB function:
addNoise(**pathImages**, **extImages**, **noiseType**, **noiseLevel**)

* **pathImages** : Path with input images 
* **extImages**  : Image type (jpg or png)
* **noiseType**  : Noise type (gaussian or poisson)
* **noiseLevel** : Noise level (gaussian 0-50 ---- poisson 1-12)


```
#!matlab
addNoise('/home/user/datasets/dtd/images/','jpg','gaussian',25)
or
addNoise('/home/user/datasets/dtd/images/','jpg','poisson',10)
```
