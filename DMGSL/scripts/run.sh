cd ..

 python main.py -ntrials 5 -sparse 0 -epochs_cls 1000 -lr_cls 0.01 -w_decay_cls 0.0005 -hidden_dim_cls 32 -dropout_cls 0.5 -dropedge_cls 0.75 -nlayers_cls 2 -patience_cls 10 -epochs 1500 -lr 0.01 -w_decay 0.0 -hidden_dim 512 -rep_dim 256 -proj_dim 256 -dropout 0.5 -dropedge_rate 0.5 -nlayers 2 -type_learner fgp -k 2 -sim_function cosine -activation_learner relu -gsl_mode structure_refinement -eval_freq 10 -tau 0.99 -maskfeat_rate_learner 0.8 -maskfeat_rate_anchor 0.4 -contrast_batch_size 0 -c 0 -snapshots 6 -edge 3 -epochs_adj 2000 -lr_adj 0.001 -nlayers_adj 2 -hidden_adj 512 -dropout1 0.5 -dropout_adj1 0.5 -lambda_ 0 -nr 5 -ratio 10 -gen_mode 0 -non_linearity elu -epoch_d 5 -mlp_act relu -w_decay_adj 0.0005
 #python main.py -ntrials 10 -sparse 0 -epochs_cls 1000 -lr_cls 0.01 -w_decay_cls 0.0005 -hidden_dim_cls 32 -dropout_cls 0.5 -dropedge_cls 0.75 -nlayers_cls 2 -patience_cls 10 -epochs 1500 -lr 0.01 -w_decay 0.0 -hidden_dim 512 -rep_dim 256 -proj_dim 256 -dropout 0.5 -dropedge_rate 0.5 -nlayers 4 -type_learner fgp -k 2 -sim_function cosine -activation_learner relu -gsl_mode structure_refinement -eval_freq 10 -tau 0.99 -maskfeat_rate_learner 0.8 -maskfeat_rate_anchor 0.4 -contrast_batch_size 0 -c 0 -snapshots 6 -edge 3 -epochs_adj 2000 -lr_adj 0.001 -nlayers_adj 2 -hidden_adj 512 -dropout1 0.5 -dropout_adj1 0.5 -lambda_ 0 -nr 5 -ratio 10 -gen_mode 0 -non_linearity elu -epoch_d 5 -mlp_act relu -w_decay_adj 0.0005
 #python main.py -ntrials 10 -sparse 0 -epochs_cls 1000 -lr_cls 0.01 -w_decay_cls 0.0005 -hidden_dim_cls 32 -dropout_cls 0.5 -dropedge_cls 0.75 -nlayers_cls 2 -patience_cls 10 -epochs 1500 -lr 0.01 -w_decay 0.0 -hidden_dim 512 -rep_dim 256 -proj_dim 256 -dropout 0.5 -dropedge_rate 0.5 -nlayers 6 -type_learner fgp -k 2 -sim_function cosine -activation_learner relu -gsl_mode structure_refinement -eval_freq 10 -tau 0.99 -maskfeat_rate_learner 0.8 -maskfeat_rate_anchor 0.4 -contrast_batch_size 0 -c 0 -snapshots 6 -edge 3 -epochs_adj 2000 -lr_adj 0.001 -nlayers_adj 2 -hidden_adj 512 -dropout1 0.5 -dropout_adj1 0.5 -lambda_ 0 -nr 5 -ratio 10 -gen_mode 0 -non_linearity elu -epoch_d 5 -mlp_act relu -w_decay_adj 0.0005
 #python main.py -ntrials 10 -sparse 0 -epochs_cls 1000 -lr_cls 0.01 -w_decay_cls 0.0005 -hidden_dim_cls 32 -dropout_cls 0.5 -dropedge_cls 0.75 -nlayers_cls 2 -patience_cls 10 -epochs 1500 -lr 0.01 -w_decay 0.0 -hidden_dim 512 -rep_dim 256 -proj_dim 256 -dropout 0.5 -dropedge_rate 0.5 -nlayers 8 -type_learner fgp -k 2 -sim_function cosine -activation_learner relu -gsl_mode structure_refinement -eval_freq 10 -tau 0.99 -maskfeat_rate_learner 0.8 -maskfeat_rate_anchor 0.4 -contrast_batch_size 0 -c 0 -snapshots 6 -edge 3 -epochs_adj 2000 -lr_adj 0.001 -nlayers_adj 2 -hidden_adj 512 -dropout1 0.5 -dropout_adj1 0.5 -lambda_ 0 -nr 5 -ratio 10 -gen_mode 0 -non_linearity elu -epoch_d 5 -mlp_act relu -w_decay_adj 0.0005
 #python main.py -ntrials 15 -sparse 0 -epochs_cls 1000 -lr_cls 0.01 -w_decay_cls 0.0005 -hidden_dim_cls 32 -dropout_cls 0.5 -dropedge_cls 0.75 -nlayers_cls 2 -patience_cls 10 -epochs 1500 -lr 0.1 -w_decay 0.0 -hidden_dim 512 -rep_dim 256 -proj_dim 256 -dropout 0.5 -dropedge_rate 0.5 -nlayers 2 -type_learner fgp -k 2 -sim_function cosine -activation_learner relu -gsl_mode structure_refinement -eval_freq 10 -tau 0.99 -maskfeat_rate_learner 0.8 -maskfeat_rate_anchor 0.4 -contrast_batch_size 0 -c 0 -snapshots 6 -edge 3 -epochs_adj 2000 -lr_adj 0.001 -nlayers_adj 2 -hidden_adj 512 -dropout1 0.5 -dropout_adj1 0.5 -lambda_ 0 -nr 5 -ratio 10 -gen_mode 0 -non_linearity elu -epoch_d 5 -mlp_act relu -w_decay_adj 0.0005