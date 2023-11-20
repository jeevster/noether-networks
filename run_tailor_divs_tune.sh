DATADIR=/data/divyam123/advection_log_space_res_1024
#task1 (pure data loss): --emb_type pde_const_emb  2341 --
#task2 (pinn): --emb_type pde_const_emb  2343 --
#              --pinn_outer_loss \
#task3 (true pde loss): --no_data_loss 2345 
#                       --emb_type pde_const_emb \
#                       --pinn_outer_loss \
#task4 (data, learned): 135 --
#                        --tailor \
#                        --emb_type pde_emb \
#task5 (data,true ): 146 || 700 --
#                       --tailor \
#                       --emb_type pde_const_emb
#task6 (data true, learned ):  150 || 722 ?? session 0 --
#                       --tailor \
#                       --pinn_outer_loss \
#                       --emb_type pde_emb \
#task7 (data learned, learned ): 151
#                       --tailor \
#                       --pinn_outer_loss \
#                       --learned_pinn_loss \
#                       --emb_type pde_emb \
#task8 (data true, true ): 152 || 647 --
#                       --tailor \
#                       --pinn_outer_loss \
#                       --emb_type pde_const_emb \
#task9 (true, learned): 153 || 633 ?? session 1
#                       --tailor \
#                       --pinn_outer_loss \
#                       --no_data_loss \
#                       --emb_type pde_emb \
#'1d_burgers_multiparam':
#'1d_advection_multiparam':
#'1d_diffusion_reaction_multiparam':
#'2d_reacdiff_multiparam':
#--warmstart_emb_path best_ckpt_model_advection.pt \
#--add_general_learnable \
# 'inner_lr': 0.04325130736234737, 'val_inner_lr': 0.061224104182765714, 'outer_lr': 0.02919379110578439}
python train_noether_net_checkpointing_tuning_conditioning.py \