# python3.9 preprocess_image.py /mnt/cache_sail/views_release/4a03d2eceba847ea897f0944e8a57ab3/010.png ./ds_tests --recenter False
# CUDA_VISIBLE_DEVICES=7 python3.9 training_carryon.py &
# python3.9 main.py -O --image /mnt/cache_sail/liulj/stable-dreamfusion/ds_tests/010_rgba.png \
#         --workspace dreamscene --iters 5000 --dreamscene --posefile "" \
#         --dreamscene_ckpt /home/liulj/latest.ckpt --save_guidance

# # Phase 1
# python3.9 main.py -O --image /mnt/cache_sail/liulj/stable-dreamfusion/data/teddy_rgba.png \
#         --workspace dreamscene/teddy_phrase1_param1 --iters 10000 --dreamscene --posefile "" \
#         --dreamscene_ckpt /home/liulj/latest.ckpt --save_guidance --save_guidance_interval 10 \
#         --ckpt scratch --batch_size 1 --h 96 --w 96 --fovy_range 19 19 --default_fovy 19 --guidance_scale 5 \
#         --lambda_3d_normal_smooth 10 --dont_override_stuff #\
#         # --default_radius 2.4 --radius_range 2.0 2.6

# Phase 2
# 20X smaller lambda_3d_normal_smooth, --known_view_interval 2, 3X LR
# Much higher jitter to increase disparity (and eliminate some of the flatness)... not too high either (to avoid cropping the face)
# python3.9 main.py -O --image /mnt/cache_sail/liulj/stable-dreamfusion/data/teddy_rgba.png \
#         --workspace dreamscene/teddy_phrase2_param1 --text "a brown teddy bear sitting on a ground" \
#         --iters 12500 --ckpt dreamscene/teddy_phrase1_param1/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
#         --h 128 --w 128 --albedo_iter_ratio 0.0 --t_range 0.2 0.6 --batch_size 1 --radius_range 2.2 2.6 --test_interval 2 \
#         --vram_O --guidance_scale 10 --jitter_pose --jitter_center 0.1 --jitter_target 0.1 --jitter_up 0.05 \
#         --known_view_noise_scale 0 --lambda_depth 0 --lr 0.003 --progressive_view --known_view_interval 2 --dont_override_stuff --lambda_3d_normal_smooth 1 \
#         --exp_start_iter 10000 --exp_end_iter 12500

# Phase 3
python3.9 main.py -O --image /mnt/cache_sail/liulj/stable-dreamfusion/data/teddy_rgba.png \
        --workspace dreamscene/teddy_phrase3_param1 --text "a brown teddy bear sitting on a ground" \
        --iters 25000 --ckpt dreamscene/teddy_phrase2_param1/checkpoints/df_ep0125.pth  --save_guidance --save_guidance_interval 1 \
        --h 256 --w 256 --albedo_iter_ratio 0.0 --t_range 0.0 0.5 --batch_size 1 --radius_range 3.2 3.6 --test_interval 2 \
        --vram_O --guidance_scale 10 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.05 \
        --known_view_noise_scale 0 --lambda_depth 0 --lr 0.003 --known_view_interval 2 --dont_override_stuff --lambda_3d_normal_smooth 0.5 --textureless_ratio 0.0 --min_ambient_ratio 0.3 \
        --exp_start_iter 12500 --exp_end_iter 25000

