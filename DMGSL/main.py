import argparse
import copy
from data_loader import load_data

from model import GCN, GCL, GCN_DAE
from graph_learners import *
from utils import *
from sklearn.cluster import KMeans
import dgl

from edgeattention import Edgeattention
from edgeattention import LSTMLearnerLayer
from edgeattention import TemporalAttentionLayer
from edgeattention import PositionwiseFeedForward

import random

import pandas as pd
from imblearn.over_sampling import SMOTE

from datetime import datetime


EOS = 1e-10

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()


    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)

    def loss_cls_train(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)    # 对行作归一化
        smo = SMOTE(k_neighbors=1)
        X_resampled_smote, y_resampled_smote = smo.fit_resample(logp[mask].detach().cpu().numpy(), labels[mask].detach().cpu().numpy())
        loss = F.nll_loss(torch.tensor(X_resampled_smote), torch.tensor(y_resampled_smote), reduction='mean')
        accu = accuracy(torch.tensor(X_resampled_smote), torch.tensor(y_resampled_smote))
        return loss, accu

    def loss_cls_test(self, model, mask, features, labels):
        logits = model(features)
        #print(logits.shape)
        logp = F.log_softmax(logits, 1)    # 对行作归一化
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu, precision1, precision2, precision3, recall1, recall2, recall3, f1_score1, f1_score2, f1_score3 = accuracy_test(logp[mask], labels[mask])

        return loss, accu, precision1, precision2, precision3, recall1, recall2, recall3, f1_score1, f1_score2, f1_score3

    def loss_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)    # 对行作归一化
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu


    def loss_gcl(self, model, learned_adj, g_l, g_a, anchor_adj): # 返回的是学习器学习到的邻接矩阵以及最后对比学习得到的损失大小
        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(g_a, args.maskfeat_rate_anchor)
            features_v1 = g_a * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(g_a.data)

        z1, embedding1 = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(g_l, args.maskfeat_rate_learner)
            features_v2 = g_l * (1 - mask)
        else:
            features_v2 = copy.deepcopy(g_l.data)

        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse)

        z2, embedding2 = model(features_v2, learned_adj, 'learner')

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(g_a.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / g_a.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss, learned_adj, z1, z2, embedding1, embedding2

    def get_loss_masked_features(self, model, features, mask, ogb, noise, loss_t, Adj):
        if ogb:
            if noise == 'mask':
                masked_features = features * (1 - mask)
            elif noise == "normal":
                noise = torch.normal(0.0, 1.0, size=features.shape).cuda()
                masked_features = features + (noise * mask)

            logits, Adj = model(Adj, masked_features)
            indices = mask > 0

            if loss_t == 'bce':
                features_sign = torch.sign(features).cuda() * 0.5 + 0.5
                loss = F.binary_cross_entropy_with_logits(logits[indices], features_sign[indices], reduction='mean')
            elif loss_t == 'mse':
                loss = F.mse_loss(logits[indices], features[indices], reduction='mean')
        else:
            masked_features = features * (1 - mask)
            logits, Adj = model(Adj, masked_features)
            indices = mask > 0
            loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')
        return loss, Adj


    def evaluate_adj_by_cls(self, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args):
        nfeats = features.shape[1]
        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

        bad_counter = 0
        best_val = 0
        best_model = None

        if torch.cuda.is_available():
            model = model.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            test_mask = test_mask.cuda()
            features = features.cuda()
            labels = labels.cuda()

        for epoch in range(1, args.epochs_cls + 1):
            model = model.train()
            loss, train_accu = self.loss_cls(model, train_mask, features, labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                val_loss, val_accu = self.loss_cls(model, val_mask, features, labels)
                if val_accu > best_val:
                    bad_counter = 0
                    best_val = val_accu
                    best_model = copy.deepcopy(model)
                else:
                    bad_counter += 1

                if bad_counter >= args.patience_cls:
                    break

        best_model.eval()
        train_loss, train_accu = self.loss_cls(best_model, train_mask, features, labels)
        test_loss, test_accu, precision1, precision2, precision3, recall1, recall2, recall3, f1_score1, f1_score2, f1_score3 = self.loss_cls_test(best_model, test_mask, features, labels)
        val_loss, val_accu = self.loss_cls(best_model, val_mask, features, labels)
        return best_val, test_accu, train_accu, best_model, train_loss, test_loss, val_loss, precision1, precision2, precision3, recall1, recall2, recall3, f1_score1, f1_score2, f1_score3


    def train(self, args):

        torch.cuda.set_device(args.gpu)

        if args.gsl_mode == 'structure_refinement':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = load_data(args)
        elif args.gsl_mode == 'structure_inference':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, _ = load_data(args)

        if args.downstream_task == 'classification':
            test_accuracies = []
            validation_accuracies = []
            train_accuracies = []
            precision_tp1 = []
            precision_tp2 = []
            precision_tp3 = []
            recall_tp1 = []
            recall_tp2 = []
            recall_tp3 = []
            f1_score_tp1 = []
            f1_score_tp2 = []
            f1_score_tp3 = []
        elif args.downstream_task == 'clustering':
            n_clu_trials = copy.deepcopy(args.ntrials)
            args.ntrials = 1

        for trial in range(args.ntrials):
            self.setup_seed(trial)

            if args.gsl_mode == 'structure_refinement':
                if args.sparse:
                    anchor_adj_raw = adj_original
                else:
                    anchor_adj_raw = torch.from_numpy(adj_original)

            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)

            list_adj_original = []
            list_anchor_adj_raw = []
            if args.gsl_mode == 'structure_refinement':
                for id_edge in range(1, args.edge+1):
                    list_adj_original.append(np.where(adj_original != id_edge, 0, 1))

                if args.sparse:
                    anchor_adj_raw = list_adj_original
                else:
                    for id_edge in range(0, args.edge):
                        list_anchor_adj_raw.append(torch.from_numpy(list_adj_original[id_edge]))

            list_anchor_adj = []
            for id_edge in range(0, args.edge):
                list_anchor_adj.append(normalize(list_anchor_adj_raw[id_edge], 'sym', args.sparse))

            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

            pad = nn.ZeroPad2d(padding=(0, args.snapshots - features.shape[1] % args.snapshots, 0, 0))
            features_pad = pad(features)
            split_feature = torch.split(features_pad, features_pad.shape[1] // args.snapshots, dim=1)

            list_graph_learner = []
            list_optimizer_learner = []
            for id_snap in range(0, args.snapshots):
                if args.type_learner == 'fgp':
                    graph_learner = FGP_learner(split_feature[id_snap].cpu(), args.k, args.sim_function, 6, args.sparse)
                elif args.type_learner == 'mlp':
                    graph_learner = MLP_learner(2, split_feature[id_snap][1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner)
                elif args.type_learner == 'att':
                    graph_learner = ATT_learner(2, split_feature[id_snap][1], args.k, args.sim_function, 6, args.sparse,
                                          args.activation_learner)
                elif args.type_learner == 'gnn':
                    graph_learner = GNN_learner(2, split_feature[id_snap][1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner, anchor_adj)
                list_graph_learner.append(graph_learner)
                optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)
                list_optimizer_learner.append(optimizer_learner)


            model = GCL(nlayers=args.nlayers, in_dim=256, hidden_dim=args.hidden_dim,
                        emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                        dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)  # 对两个视图进行编码映射的模型

            optimizer_cl = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            edgeaggregation = Edgeattention(split_feature[0].shape[1]+adj_original.shape[1], 256)
            optimizer_edgeagg = torch.optim.Adam(edgeaggregation.parameters(), lr=args.lr, weight_decay=args.w_decay)

            LSTMLayer = LSTMLearnerLayer(256, args.snapshots)
            TemporalAggregation = TemporalAttentionLayer(256, n_heads=4, num_time_steps=args.snapshots, attn_drop=0.0)
            optimizer_LSTM = torch.optim.Adam(LSTMLayer.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_Tem = torch.optim.Adam(TemporalAggregation.parameters(), lr=args.lr, weight_decay=args.w_decay)

            Positionwise = PositionwiseFeedForward(256, 512)
            optimizer_Pos = torch.optim.Adam(Positionwise.parameters(), lr=args.lr, weight_decay=args.w_decay)

            model_DAE = GCN_DAE(nlayers=args.nlayers_adj, in_dim=nfeats, hidden_dim=args.hidden_adj,
                                nclasses=nfeats,
                                dropout=args.dropout1, dropout_adj=args.dropout_adj1,
                                features=features.cpu(), k=args.k, knn_metric=args.knn_metric, i_=args.i,
                                non_linearity=args.non_linearity, normalization=args.normalization,
                                mlp_h=args.mlp_h,
                                mlp_epochs=args.mlp_epochs, gen_mode=args.gen_mode, sparse=args.sparse,
                                mlp_act=args.mlp_act)
            optimizer_DAE = torch.optim.Adam(model_DAE.parameters(), lr=args.lr_adj, weight_decay=args.w_decay_adj)

            list_split_feature = []
            if torch.cuda.is_available():
                model_DAE = model_DAE.cuda()
                model = model.cuda()
                edgeaggregation = edgeaggregation.cuda()
                LSTMLayer = LSTMLayer.cuda()
                TemporalAggregation = TemporalAggregation.cuda()
                Positionwise = Positionwise.cuda()
                for id_snap in range(0, args.snapshots):
                    list_graph_learner[id_snap] = list_graph_learner[id_snap].cuda()
                    list_split_feature.append(split_feature[id_snap].cuda())
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                labels = labels.cuda()
                features = features.cuda()
                if not args.sparse:
                    for id_edge in range(0, args.edge):
                        list_anchor_adj[id_edge] = list_anchor_adj[id_edge].cuda()
                    anchor_adj = anchor_adj.cuda()

            if args.downstream_task == 'classification':
                best_val = 0
                best_val_test = 0
                best_val_train = 0
                best_epoch = 0

            h_at = []
            for id_snap in range(0, args.snapshots):
                h_ae = []
                for id_edge in range(0, args.edge):
                    h_ae_ = torch.cat((list_anchor_adj[id_edge], list_split_feature[id_snap]), 1)    #(Ai, X)
                    h_ae.append(h_ae_)  # list:3 X (Ai, X)
                h_at_ = torch.stack(h_ae)
                h_at.append(h_at_)  # list: 5 X tensor(3 X (Ai, X))

            for epoch in range(1, args.epochs + 1):
                edgeaggregation.train()
                LSTMLayer.train()
                TemporalAggregation.train()
                Positionwise.train()
                model.train()
                model_DAE.train()
                list_h_aedge = []
                list_h_ledge = []

                # h_lt = []
                # list_h_lt = []
                learned_adj = torch.zeros_like(list_anchor_adj[0])

                for id_snap in range(0, args.snapshots):
                    h_aedge, _ = edgeaggregation(h_at[id_snap])
                    list_h_aedge.append(h_aedge)

                    list_graph_learner[id_snap].train()
                    adj_learned = list_graph_learner[id_snap]()
                    learned_adj = learned_adj + adj_learned
                    for id_edge in range(0, args.edge):
                        h_le = []
                        h_le_ = torch.cat((adj_learned, list_split_feature[id_snap]), 1)
                        h_le.append(h_le_)
                    h_lt_ = torch.stack(h_le)
                    h_ledge, _ = edgeaggregation(h_lt_)
                    list_h_ledge.append(h_ledge)

                learned_adj = learned_adj / args.snapshots
                tensor_h_aedge = torch.stack(list_h_aedge)
                tensor_h_ledge = torch.stack(list_h_ledge)
                h_edge_aagg = LSTMLayer(tensor_h_aedge)
                h_edge_lagg = LSTMLayer(tensor_h_ledge)
                h_atem = TemporalAggregation(h_edge_aagg)
                h_ltem = TemporalAggregation(h_edge_lagg)

                h_a = Positionwise(h_atem)
                h_l = Positionwise(h_ltem)

                loss1, Adj, z1, z2, embedding1, embedding2 = self.loss_gcl(model, learned_adj, h_l, h_a, anchor_adj)

                mask = get_random_mask(features, args.ratio, args.nr).cuda()
                ogb = False
                loss2, Adj = self.get_loss_masked_features(model_DAE, features, mask, ogb, args.noise, args.loss, Adj)
                loss = loss2 * args.lambda_ + loss1

                for i in range(0, args.snapshots):
                    list_optimizer_learner[i].zero_grad()
                optimizer_cl.zero_grad()
                optimizer_edgeagg.zero_grad()
                optimizer_LSTM.zero_grad()
                optimizer_Tem.zero_grad()
                optimizer_Pos.zero_grad()
                optimizer_DAE.zero_grad()
                loss.backward()

                for i in range(0, args.snapshots):
                    list_optimizer_learner[i].step()
                optimizer_cl.step()
                optimizer_edgeagg.step()
                optimizer_LSTM.step()
                optimizer_Tem.step()
                optimizer_Pos.step()
                optimizer_DAE.step()

                if (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                    if args.sparse:
                        learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                        anchor_adj_torch_sparse = anchor_adj_torch_sparse * args.tau \
                                                  + learned_adj_torch_sparse * (1 - args.tau)
                        anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse)
                    else:
                        anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)

                print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()))

                if epoch % args.eval_freq == 0:
                    if args.downstream_task == 'classification':
                        for i in range(0, args.snapshots):
                            list_graph_learner[i].eval()
                        model.eval()
                        edgeaggregation.eval()
                        LSTMLayer.eval()
                        TemporalAggregation.eval()
                        Positionwise.eval()
                        model_DAE.eval()
                        f_adj = Adj

                        if args.sparse:
                            f_adj.edata['w'] = f_adj.edata['w'].detach()
                        else:
                            f_adj = f_adj.detach()

                        val_accu, test_accu, train_accu, _, train_loss, test_loss, val_loss, precision1, precision2, precision3, recall1, recall2, recall3, f1_score1, f1_score2, f1_score3 = self.evaluate_adj_by_cls(f_adj, features, nfeats, labels, nclasses, train_mask,
                                                                                  val_mask, test_mask, args)

                        if val_accu > best_val:
                            best_val = val_accu
                            best_val_test = test_accu
                            best_val_train = train_accu
                            best_precision1 = precision1
                            best_precision2 = precision2
                            best_precision3 = precision3
                            best_recall1 = recall1
                            best_recall2 = recall2
                            best_recall3 = recall3
                            best_f1_score1 = f1_score1
                            best_f1_score2 = f1_score2
                            best_f1_score3 = f1_score3
                            best_epoch = epoch
                            best_f_adj = f_adj

                        list = [epoch, loss1.detach().cpu().numpy(), loss2.detach().cpu().numpy(),
                                loss.detach().cpu().numpy(), train_accu.detach().cpu().numpy(),
                                val_accu.detach().cpu().numpy(),
                                test_accu.detach().cpu().numpy(), train_loss.detach().cpu().numpy(),
                                test_loss.detach().cpu().numpy(), val_loss.detach().cpu().numpy()]
                        data = pd.DataFrame([list])
                        #data.to_csv(
                        #    "D:/work_sci_res/小论文/results_proposed/24110620/test_{}_{}/basic_{}.csv".format(args.snapshots, args.edge, trial),
                        #    mode='a', header=False, index=False)

                    elif args.downstream_task == 'clustering':
                        for i in range(0, args.snapshots):
                            list_graph_learner[i].eval()
                        model.eval()
                        edgeaggregation.eval()
                        LSTMLayer.eval()
                        TemporalAggregation.eval()
                        Positionwise.eval()
                        model_DAE.eval()


                        list_h_ledge_eva = []
                        learned_adj_eva = torch.zeros_like(Adj)
                        for id_snap in range(0, args.snapshots):
                            adj_learned = list_graph_learner[id_snap]()
                            learned_adj_eva = learned_adj_eva + adj_learned
                            h_ledge_eva = torch.cat((adj_learned, list_split_feature[id_snap]), 1)
                            list_h_ledge_eva.append(h_ledge_eva)

                        learned_adj_eva = learned_adj_eva / args.snapshots
                        tensor_h_ledge_eva = torch.stack(list_h_ledge_eva)
                        h_edge_lagg_eva = LSTMLayer(tensor_h_ledge_eva)
                        h_ltem_eva = TemporalAggregation(h_edge_lagg_eva)
                        h_l_eva = Positionwise(h_ltem_eva)

                        _, embedding = model(h_l_eva, learned_adj_eva)

                        acc_mr, nmi_mr, f1_mr, ari_mr = [], [], [], []
                        for clu_trial in range(n_clu_trials):
                            kmeans = KMeans(n_clusters=nclasses, random_state=clu_trial).fit(
                            embedding)  ######################################
                            # kmeans = KMeans(n_clusters=10, random_state=clu_trial).fit(embedding)
                            predict_labels = kmeans.predict(embedding)
                            cm_all = clustering_metrics(labels.cpu().numpy(), predict_labels)
                            acc_, nmi_, f1_, ari_ = cm_all.evaluationClusterModelFromLabel(print_results=False)
                            acc_mr.append(acc_)
                            nmi_mr.append(nmi_)
                            f1_mr.append(f1_)
                            ari_mr.append(ari_)

                        acc, nmi, f1, ari = np.mean(acc_mr), np.mean(nmi_mr), np.mean(f1_mr), np.mean(ari_mr)

                        if acc > best_acc:
                            best_acc = acc
                            best_nmi = nmi
                            best_F = f1
                            best_ari = ari
                            best_turn = epoch


            if args.downstream_task == 'classification':
                validation_accuracies.append(best_val.item())
                test_accuracies.append(best_val_test.item())
                train_accuracies.append(best_val_train.item())
                precision_tp1.append(best_precision1.item())
                precision_tp2.append(best_precision2.item())
                precision_tp3.append(best_precision3.item())
                recall_tp1.append(best_recall1.item())
                recall_tp2.append(best_recall2.item())
                recall_tp3.append(best_recall3.item())
                f1_score_tp1.append(best_f1_score1.item())
                f1_score_tp2.append(best_f1_score2.item())
                f1_score_tp3.append(best_f1_score3.item())

                print("Trial: ", trial + 1)
                print("Best val ACC: ", best_val.item())
                print("Best test ACC: ", best_val_test.item())

            elif args.downstream_task == 'clustering':
                print("Final ACC: ", best_acc)
                print("Final NMI: ", best_nmi)
                print("Final F-score: ", best_F)
                print("Final ARI: ", best_ari)
                print("Best_epoch", best_turn)

            best_f_adj = pd.DataFrame(best_f_adj.detach().cpu().numpy())
            best_f_adj.to_csv("D:/work_sci_res/小论文/results_proposed/24110620/test_{}_{}/adj_{}.csv".format(args.snapshots, args.edge, trial))

        if args.downstream_task == 'classification' and trial != 0:
            self.print_results(validation_accuracies, test_accuracies, precision_tp1, precision_tp2, precision_tp3, recall_tp1, recall_tp2, recall_tp3, f1_score_tp1, f1_score_tp2, f1_score_tp3)


    def print_results(self, validation_accu, test_accu, precision_tp1, precision_tp2, precision_tp3, recall_tp1, recall_tp2, recall_tp3, f1_score_tp1, f1_score_tp2, f1_score_tp3):
        s_val = "Val accuracy: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
        s_test = "Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu),np.std(test_accu))
        s_p1 = "precision_score_average_micro: {:.4f} +/- {:.4f}".format(np.mean(precision_tp1), np.std(precision_tp1))
        s_p2 = "precision_score_average_macro: {:.4f} +/- {:.4f}".format(np.mean(precision_tp2), np.std(precision_tp2))
        s_p3 = "precision_score_average_weighted: {:.4f} +/- {:.4f}".format(np.mean(precision_tp3), np.std(precision_tp3))
        s_r1 = "recall_score_average_micro: {:.4f} +/- {:.4f}".format(np.mean(recall_tp1), np.std(recall_tp1))
        s_r2 = "recall_score_average_macro: {:.4f} +/- {:.4f}".format(np.mean(recall_tp2), np.std(recall_tp2))
        s_r3 = "recall_score_average_weighted: {:.4f} +/- {:.4f}".format(np.mean(recall_tp3), np.std(recall_tp3))
        s_f11 = "f1_score_average_micro: {:.4f} +/- {:.4f}".format(np.mean(f1_score_tp1), np.std(f1_score_tp1))
        s_f12 = "f1_score_average_macro: {:.4f} +/- {:.4f}".format(np.mean(f1_score_tp2), np.std(f1_score_tp2))
        s_f13 = "f1_score_average_weighted: {:.4f} +/- {:.4f}".format(np.mean(f1_score_tp3), np.std(f1_score_tp3))

        print(s_val)
        print(s_test)
        print(s_p1)
        print(s_p2)
        print(s_p3)
        print(s_r1)
        print(s_r2)
        print(s_r3)
        print(s_f11)
        print(s_f12)
        print(s_f13)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-gsl_mode', type=str, default="structure_inference",
                        choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('-eval_freq', type=int, default=5)
    parser.add_argument('-downstream_task', type=str, default='classification',
                        choices=['classification', 'clustering'])
    parser.add_argument('-gpu', type=int, default=0)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('-dropedge_rate', type=float, default=0.5)

    # GSL Module
    parser.add_argument('-type_learner', type=str, default='fgp', choices=["fgp", "att", "mlp", "gnn"])
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=32)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.25)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)

    # Attention
    parser.add_argument('-snapshots', type=int, default=1)
    parser.add_argument('-edge', type=int, default=1)

    # Denoised
    parser.add_argument('-epochs_adj', type=int, default=2000, help='Number of epochs to learn the adjacency.')
    parser.add_argument('-lr_adj', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('-w_decay_adj', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-hidden_adj', type=int, default=512, help='Number of hidden units.')
    parser.add_argument('-dropout1', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dropout_adj1', type=float, default=0.25, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-nlayers_adj', type=int, default=2, help='#layers')
    parser.add_argument('-lambda_', type=float, default=1, help='ratio of ones to take')
    parser.add_argument('-knn_metric', type=str, default='cosine', help='See choices', choices=['cosine', 'minkowski'])
    parser.add_argument('-i', type=int, default=6)
    parser.add_argument('-epoch_d', type=float, default=5,
                        help='epochs_adj / epoch_d of the epochs will be used for training only with DAE.')
    parser.add_argument('-non_linearity', type=str, default='elu')
    parser.add_argument('-mlp_act', type=str, default='relu', choices=["relu", "tanh"])
    parser.add_argument('-normalization', type=str, default='sym')
    parser.add_argument('-mlp_h', type=int, default=50)
    parser.add_argument('-mlp_epochs', type=int, default=100)
    parser.add_argument('-gen_mode', type=int, default=0)
    #parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-noise', type=str, default="mask", choices=['mask', 'normal'])
    parser.add_argument('-loss', type=str, default="mse", choices=['mse', 'bce'])
    parser.add_argument('-nr', type=int, default=5, help='ratio of zeros to ones')
    parser.add_argument('-ratio', type=int, default=20, help='ratio of ones to select for each mask')

    parser.add_argument('-index', type=int, default=0, help='running times')
    args = parser.parse_args()

    experiment = Experiment()
    experiment.train(args)
