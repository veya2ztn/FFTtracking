# FFT Machine Learning Problem

-----------

This project is based on a series idea about parse image information in Fourier Space rather than Real Space.

For quantum physics problem, scientists find that the behavior of particles in real space is so complicated to study. (Usually need to solve highly entangled PDE equation in extramely high dimenstion). 

However, it emerges a turnning point when we force on other analysis domain like Fourier Space which physicists more like to call it 'Momentum Space'. Not only Fouirer Space, physicists gradually find other useful analysis domain such as ' Fork Sapce ', ' Vector Representation ' ... 

A spirit cross through these domain: analysis base on the symmetry. 

- Due to the space translation symmetry, the entangle between particles usaually can explict show as momentum coupling.
- Due to the Symmetry/Anti-symmetry of identical particle, fork space is proposed to draw the quanta interaction.

Back to this project, for machine learning on image problem (Compute Vision Domain), we want to analysis the map between graphic elements to human knowledge. In this question, graphic elements have intrinsic local correlation and space translation/rotation symmetry. From most CNN project, we see using the convolution operation can help us parse the local correlation. Autually, do convolution operation is same do multiple in Fourier Space. So why not direct change the analysis domain to Fourier Space.

[Recently researches](https://www.quantamagazine.org/where-we-see-shapes-ai-sees-textures-20190701/) find that the convolution can not help use parse the information about shape (which is the overall character ). So the machine classifier mechanism is a bit different from human brain. Beyond convolution (which is powerful to deal with local information), we need a new tool to parse the shape information. Indeed, fourier space cannot parse shape directly. For shape recogniztion, maybe we can rely on Conformal transformation (so the basic number system need change to complex number)or some geometry group.

-----------

A well-knowledge Fourier Implement in Compute Vision tracking problem is the KCF algorithm which is cannot regarded strictly as Machine Learning Method. 

In this repository, a pytorch version KCF is build with the help of [torch.fft](https://pytorch.org/docs/stable/torch.html). Meanwhile, a pytorch complex system is build to help code.

#### KCF：

input pair image: (x,y)

- x.shape=  (Batch,Channels,w,h)
- y.shape=  (Batch,1,w,h)

input target image: z

-  z.shape =  (Batch,Channels,w2,h2)

output response field: r

- r.shape =  (Batch,1,w2,h2)

$$
\hat{r}=\hat{k^{xz}}.*\frac{\hat{y}}{\hat{k^{xx}}+\lambda}
$$

Here $\hat{r} = \mathcal{F}(r)$ is the Fourier Transformation of r.

Here the $k^{xz}=\kappa(x,z) $ is the kernel function satisfy
$$
\kappa(x,z) = g[||x||^2,||z||^2,\mathcal{F}^{-1}(\hat{x}^*.*\hat{z})]
$$
where g is any funtion, usually Poly function g(x)=x^n and Exponential function g(x)=exp(x)

#### ML-KCF

in order to get better porfermance, using Nerual Network N present function g. The requirment is that $k^{xx}$ get the same shape as input data y. 

The consistent shape between $k^{xz}$ and x(y) is not necessary. Because the real reponse can be achieved by convolution between real $k^{xz}$ and real $\alpha=\mathcal{F}^{-1}(\frac{\hat{y}}{\hat{k^{xx}}+\lambda})$. So the dimenstion can be different.

Of course, the key part $\mathcal{F}^{-1}(\hat{x}^*.*\hat{z})$ can still be some convolution between $x$ and $z$. 

##### Tracking dataset

Showing the tracked template: a image x and a location y

Showing the tracking image: a image z and a ground_truth r

-------

Input {x|(B,C,W,H)} into Network ---> {$k^{xx}$|(B,C2,W,H)}

Using {y|(B,1,W,H)} and  {$k^{xx}$|(B,C2,W,H)} get {$\alpha$|(B,C2,W,H)}

Input  {z|(B,C,W2,H2)} into Network ---> {$k^{xz}$|(B,C2,W2,H2)}

Using {$k^{xz}$|(B,C2,W2,H2)} and {$\alpha$|(B,C2,W,H)} ---> {r|(B,C2,W2,H2}---> {r|(B,1,W2,H2}

----

#### DML-KCF

Using Feature Extract Network parse image to this feature slices: 

x -> {fx1|(B,C,w,h)} ->{fx2|(B,C2,w/2,h/2)} ->{fx3|(B,C3,w/4,h/4)}

z -> {fz1|(B,C,W,H)}->{fz2|(B,C2,W/2,H/2)}->{fz3|(B,C3,W/4,H/4)}

For every slices, do KCF or ML-KCF.

----------

##### Acknowledge

The backbone part code base on [DaSiamRPN](https://github.com/foolwood/DaSiamRPN) and [Yolact](https://github.com/dbolya/yolact).

-----



##### Trick

1.  Using Exp Kernel has a good performance (maybe universary good) but can not explain why.

   Trainning a kernel may has a good performance but only guarantee its performance on dataset. (fix class, fix size)

   Our hope is to find a universary Model.

3.