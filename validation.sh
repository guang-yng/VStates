CUDA_VISIBLE_DEVICES=0 python main_dist.py "OSE-pixel_OME-test" --mdl.mdl_name="sf_ec_cat" \
    --mdl.load_sf_pretrained=True  --only_val\
    --train.resume=True --train.resume_path="tmp/model_epochs/OSE-pixel_OME/mdl_ep_0.pth" \
    --train.bsv=16 --train.nwv=4\
    --do_dist=False
