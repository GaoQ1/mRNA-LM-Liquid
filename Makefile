train-hek293t:
	python finetune_all3.py --task hek293t --output models/hek293t --device 0

train-rabies:
	python finetune_all3.py --task rabies --output models/rabies --device 0

train-gfp:
	python finetune_all3.py --task gfp --output models/gfp --device 0




train-tr:
	python finetune_all.py --task tr --output models/tr --device 0

train-hf:
	python finetune_all.py --task halflife --output models/halflife --device 0

train-liver:
	python finetune_all2.py --task liver --output models/liver --device 1

train-5class:
	python finetune_all2.py --task 5class --output models/5class --device 1




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
