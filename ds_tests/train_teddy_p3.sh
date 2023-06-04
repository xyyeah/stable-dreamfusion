# python3.9 preprocess_image.py /mnt/cache_sail/views_release/4a03d2eceba847ea897f0944e8a57ab3/010.png ./ds_tests --recenter False
# CUDA_VISIBLE_DEVICES=7 python3.9 training_carryon.py &
# python3.9 main.py -O --image /mnt/cache_sail/liulj/stable-dreamfusion/ds_tests/010_rgba.png \
#         --workspace dreamscene --iters 5000 --dreamscene --posefile "" \
#         --dreamscene_ckpt /home/liulj/latest.ckpt --save_guidance

 python3.9 main.py -O --image ./data/teddy_rgba.png \
         --workspace /workspace/teddy_vsd --iters 10000 --dreamscene --posefile "" \
         --save_guidance --save_guidance_interval 50 \
         --ckpt scratch --batch_size 1 --h 96 --w 96 --fovy_range 10 30 --default_fovy 20 --guidance_scale 100.0 \
         --dont_override_stuff --test_interval 2 \
         --text "a brown teddy bear sitting on a ground" --negative "low quality" \
         --lambda_2d_normal_smooth 0.0 --lambda_depth 0.0 --lambda_rgb 0.0  --lambda_mask 0.0
         # --lambda_2d_normal_smooth 0.0 --lambda_depth 1.0 --lambda_rgb 1.0  --lambda_mask 1.0 \
         --seed 1
         # --default_radius 2.0 --radius_range 1.6 2.4

   python3.9 main.py -O --image ./data/teddy_rgba.png  --dreamscene\
   --text "a brown teddy bear sitting on a ground" --negative "low quality" \
   --workspace /workspace/teddy_vsd --test_interval 2 --save_guidance --save_guidance_interval 50 --ckpt scratch \
   --lambda_3d_normal_smooth 0.0 --lambda_2d_normal_smooth 0.0 --lambda_depth 0.0 --lambda_rgb 0.0  --lambda_mask 0.0

  python3.9 main.py -O --image ./data/teddy_rgba.png  --dreamscene\
   --text "a brown teddy bear sitting on a ground" --negative "low quality" \
   --workspace /workspace/teddy_vsd --test_interval 2 --save_guidance --save_guidance_interval 50 --ckpt scratch \
   --lambda_3d_normal_smooth 0.0 --lambda_2d_normal_smooth 0.0 --lambda_depth 0.0 --lambda_rgb 0.0  --lambda_mask 0.0

   python3.9 main.py -O --image ./data/teddy_rgba.png\
   --text "a brown teddy bear sitting on a ground" --negative "low quality" \
   --workspace /workspace/teddy_vsd --test_interval 2 --save_guidance --save_guidance_interval 50 --ckpt scratch \
   --lambda_2d_normal_smooth 0.0 --lambda_depth 0.0 --lambda_rgb 0.0  --lambda_mask 0.0

# # # Phase 1
# python3.9 main.py -O --image /mnt/cache_sail/liulj/stable-dreamfusion/data/teddy_rgba.png \
#         --workspace dreamscene/teddy_phrase1_param3 --iters 10000 --dreamscene --posefile "" \
#         --dreamscene_ckpt /home/liulj/latest.ckpt --save_guidance --save_guidance_interval 50 \
#         --ckpt scratch --batch_size 1 --h 96 --w 96 --fovy_range 20 20 --default_fovy 20 --guidance_scale 50 \
#         --lambda_3d_normal_smooth 10 --dont_override_stuff --test_interval 2 \
#         --default_radius 3.2 --radius_range 3.2 3.6 --optim adam --lr 0.001

# Phase 2
# 20X smaller lambda_3d_normal_smooth, --known_view_interval 2, 3X LR
# Much higher jitter to increase disparity (and eliminate some of the flatness)... not too high either (to avoid cropping the face)
python3.9 main.py -O --image /mnt/cache_sail/liulj/stable-dreamfusion/data/teddy_rgba.png \
        --workspace dreamscene/teddy_phrase2_param3 --text "a brown teddy bear sitting on a ground" \
        --iters 12500 --ckpt dreamscene/teddy_phrase1_param3/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
        --h 128 --w 128 --albedo_iter_ratio 0.0 --t_range 0.2 0.6 --batch_size 1 --radius_range 2.2 2.6 --test_interval 2 \
        --vram_O --guidance_scale 10 --jitter_pose --jitter_center 0.1 --jitter_target 0.1 --jitter_up 0.05 \
        --known_view_noise_scale 0 --lambda_depth 0 --lr 0.003 --progressive_view --known_view_interval 2 --dont_override_stuff --lambda_3d_normal_smooth 1 \
        --exp_start_iter 10000 --exp_end_iter 12500 --optim adam --lr 0.001

# # Phase 3
# python3.9 main.py -O --image /mnt/cache_sail/liulj/stable-dreamfusion/data/teddy_rgba.png \
#         --workspace dreamscene/teddy_phrase3_param2 --text "a brown teddy bear sitting on a ground" \
#         --iters 25000 --ckpt dreamscene/teddy_phrase2_param2/checkpoints/df_ep0125.pth  --save_guidance --save_guidance_interval 1 \
#         --h 256 --w 256 --albedo_iter_ratio 0.0 --t_range 0.0 0.5 --batch_size 1 --radius_range 3.2 3.6 --test_interval 2 \
#         --vram_O --guidance_scale 10 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.05 \
#         --known_view_noise_scale 0 --lambda_depth 0 --lr 0.001 --known_view_interval 2 --dont_override_stuff --lambda_3d_normal_smooth 0.5 --textureless_ratio 0.0 --min_ambient_ratio 0.3 \
#         --exp_start_iter 12500 --exp_end_iter 25000 --optim adam

