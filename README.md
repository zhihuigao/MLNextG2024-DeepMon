### Step 1: Install Necessary Python Libraries

You may need to install the following Python libraries if you haven't done so.

1. `argparser`.
2. `math`
3. `numpy`
4. `os`
5. `scipy`
6. `torch`, where `PyTorch 2.1.1` is recommended
7. `tqdm`

### Step 2: Download Dataset from DropBox

The dataset is available on the DropBox link as 
https://www.dropbox.com/scl/fi/s6jga6alw0ika8yfw9r1z/Data.zip?rlkey=0qd79gfzbztp79j2g3h0smp46&st=cs2bohqs&dl=0

### Step 3: Train and Test DeepMon

Go to the Code folder by `cd Code/`

Run DeepMon by `python main.py`.

You can customize the arguments by running `python main.py --help` for more instructions.

The detailed results are saved in `../Result/`.

## Reference
If you find our work useful in your research, please consider citing our paper:\
[DeepMon: Wi-Fi Monitoring Using Sub-Nyquist Sampling Rate Receivers with Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3636534.3698250)

```console
@inproceedings{gao2024deepmon,
  title={DeepMon: {Wi-Fi} monitoring using sub-Nyquist sampling rate receivers with deep learning},
  author={Gao, Zhihui and Zhang, Yunjia and Chen, Tingjun},
  booktitle={Proc. ACM MLNextG'24},
  year={2024}
}
```

If you have any further questions, please feel free to contact us at :D\
zhihui.gao@duke.edu
