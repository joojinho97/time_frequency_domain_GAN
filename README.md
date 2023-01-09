we focuse to propose not mentioned problem in generate model(GAN,AutoEncoder,Autoregressive model, etc) and how to solve.

we think 

1. we can imagine 3-Dimension, watching the picutre. because we live in 3-Dimension world

2. if the people(generate model) who live in 2-Dimension world watching the picture(2d data), can they imagine(3d-data)?

<br/><br/>

data from AI HUB(https://www.aihub.or.kr/).
<br/><br/>


for checking our idea, we designed to verify problem in this order


1.  we make x = model(x).
* any model can use(∵ linear training), x is only time domain
2. we make generated_data = model(test_data).
* make test_data
3.  we postprocess(Bandpass, Fourier_transform) in generated_data, test_data.
* time-amplitude : 2-Dimension Image, time-frequency-amplitude : 3-Dimension Image, we train only time domain data, and check in frequency domain
4.  check
<br/><br/>
***
<br/><br/>

<div align="center"> 

Time domain

</div>
<br/>

![voice_timedomain](https://user-images.githubusercontent.com/81897022/211230617-fb9ee75d-636f-43ef-90e8-0a6b75804951.png)

<div align="center"> 

Time domain (post-preprocessing with bandpass filter)

</div>
<br/>

![12](https://user-images.githubusercontent.com/81897022/211232095-92e25bb8-7ab6-4dae-a7fc-c9584c20b4f4.png)


