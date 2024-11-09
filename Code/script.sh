#!/bin/bash

# tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut prot --prot Non-HT,HT,HTGF,VHT,others --rate 1000000 --token protocol' &
# tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut prot --prot Non-HT,HT,HTGF,VHT,others --rate 2000000 --token protocol' &
# tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut prot --prot Non-HT,HT,HTGF,VHT,others --rate 3000000 --token protocol' &
# tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut prot --prot Non-HT,HT,HTGF,VHT,others --rate 5000000 --token protocol' &



tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut LSIG --prot Non-HT --rate 1000000 --token Non-HT2LSIG' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut LSIG --prot Non-HT --rate 2000000 --token Non-HT2LSIG' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut LSIG --prot Non-HT --rate 3000000 --token Non-HT2LSIG' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut LSIG --prot Non-HT --rate 5000000 --token Non-HT2LSIG' &

tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut psdu --scale 4096 --prot Non-HT --rate 1000000 --token Non-HT2PSDU' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut psdu --scale 4096 --prot Non-HT --rate 2000000 --token Non-HT2PSDU' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut psdu --scale 4096 --prot Non-HT --rate 3000000 --token Non-HT2PSDU' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut psdu --scale 4096 --prot Non-HT --rate 5000000 --token Non-HT2PSDU' &



tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut LSIG --prot HT --rate 1000000 --token HT2LSIG' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut LSIG --prot HT --rate 2000000 --token HT2LSIG' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut LSIG --prot HT --rate 3000000 --token HT2LSIG' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut LSIG --prot HT --rate 5000000 --token HT2LSIG' &

tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut HTSIG --prot HT --rate 1000000 --token HT2HTSIG' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut HTSIG --prot HT --rate 2000000 --token HT2HTSIG' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut HTSIG --prot HT --rate 3000000 --token HT2HTSIG' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut HTSIG --prot HT --rate 5000000 --token HT2HTSIG' &

tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut time --scale 5.484e-3 --prot HT --rate 1000000 --token HT2Time' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut time --scale 5.484e-3 --prot HT --rate 2000000 --token HT2Time' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut time --scale 5.484e-3 --prot HT --rate 3000000 --token HT2Time' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut time --scale 5.484e-3 --prot HT --rate 5000000 --token HT2Time' &

tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut psdu --scale 65536 --prot HT --rate 1000000 --token HT2PSDU' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut psdu --scale 65536 --prot HT --rate 2000000 --token HT2PSDU' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut psdu --scale 65536 --prot HT --rate 3000000 --token HT2PSDU' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=0 python main.py --taskOut psdu --scale 65536 --prot HT --rate 5000000 --token HT2PSDU' &



tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut LSIG --prot VHT --rate 1000000 --token VHT2LSIG' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut LSIG --prot VHT --rate 2000000 --token VHT2LSIG' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut LSIG --prot VHT --rate 3000000 --token VHT2LSIG' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut LSIG --prot VHT --rate 5000000 --token VHT2LSIG' &

tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut time --scale 5.484e-3 --prot VHT --rate 1000000 --token VHT2Time' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut time --scale 5.484e-3 --prot VHT --rate 2000000 --token VHT2Time' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut time --scale 5.484e-3 --prot VHT --rate 3000000 --token VHT2Time' &
tmux new-session -d 'CUDA_VISIBLE_DEVICES=1 python main.py --taskOut time --scale 5.484e-3 --prot VHT --rate 5000000 --token VHT2Time' &
