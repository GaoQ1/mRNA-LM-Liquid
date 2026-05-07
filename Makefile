train-hf:
	python finetune_all.py --task halflife --output models/halflife --device 0 --head_type liquid --liquid_hidden 128

train-hf-mlp:
	python finetune_all.py --task halflife --output models/halflife_mlp --device 0 --head_type mlp --liquid_hidden 128





predict-604:
	# python predict_all.py --task 604 --output result/604 --model_path outputs/tr/finetuned_model.pt --device 0
	python predict_all.py --task 604 --output result/604 --model_path outputs/hek293t/finetuned_model.pt --device 0

predict-auro_rsv:
	# python predict_all.py --task auro_rsv --output result/auro_rsv --model_path outputs/tr/finetuned_model.pt --device 0
	python predict_all.py --task auro_rsv --output result/auro_rsv --model_path outputs/hek293t/finetuned_model.pt --device 0

predict-zheda_homo:
	# python predict_all.py --task zheda_homo --output result/zheda_homo --model_path outputs/tr/finetuned_model.pt --device 0
	python predict_all.py --task zheda_homo --output result/zheda_homo --model_path outputs/hek293t/finetuned_model.pt --device 0

predict-zheda_mouse:
	python predict_all.py --task zheda_mouse --output result/zheda_mouse --model_path outputs/tr/finetuned_model.pt --device 0
	# python predict_all.py --task zheda_mouse --output result/zheda_mouse --model_path outputs/hek293t/finetuned_model.pt --device 0

