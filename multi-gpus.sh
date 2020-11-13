CUDA_VISIBLE_DEVICES=1,2,0 python -u train_search.py \
	--lookup_path="./latency_pkl/latency_gpu_seg.pkl" \
	--save="runs" \
	--print_freq=20 \
	--workers=4 \
	--epochs=1000 \
	--batch_size=8 \
	--w_lr=0.045 \
	--w_mom=0.9 \
	--w_wd=4e-5 \
	--a_lr=0.02 \
	--a_wd=5e-4 \
	--grad_clip=5.0 \
	--T=5.0 \
	--T_decay=0.96 \
	--num_classes=19 \
	--lambda_lat=0.1 \
	--target_lat=15.0 \
	--num_gpus=3 \
    --distributed=1 \
    --word_size=1 \
	--note "TF-NAS-fastSCNN-gpus"