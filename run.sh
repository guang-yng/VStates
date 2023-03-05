CUDA_VISIBLE_DEVICES=0 python main_dist.py "OSE-pixel_OME" --mdl.mdl_name="sf_ec_cat" \
    --train.bs=2 --train.gradient_accumulation=4 --train.nw=2 --train.bsv=4 --train.lr=3e-5\
    --train.resume=False --mdl.load_sf_pretrained=True  \
    --do_dist=False

# CUDA_VISIBLE_DEVICES=0 python main_dist.py "OSE-pixel-disp_OME" --mdl.mdl_name="sf_ec_cat" \
#     --train.bs=2 --train.gradient_accumulation=1 --train.nw=8 --train.bsv=4 --train.lr=3e-5 --mdl.C=128\
#     --train.resume=False --mdl.load_sf_pretrained=True  \
#     --do_dist=False

# CUDA_VISIBLE_DEVICES=0 python main_dist.py "OSE-pixel-disp_OME_OIE" --mdl.mdl_name="sf_ec_rel" \
#     --train.bs=2 --train.gradient_accumulation=1 --train.nw=8 --train.bsv=4 --train.lr=3e-5 --mdl.C=128\
#     --train.resume=False --mdl.load_sf_pretrained=True  \
#     --do_dist=False
