import torch as t
from torch import nn
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from params import args
from model import OpenGraph, ALRS
from data_handler import DataHandler, MultiDataHandler
import numpy as np
import pickle
import os
import setproctitle
import time
import sys

class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'a')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)
        self.stdout.flush()
    def flush(self):
        self.file.flush()
        self.stdout.flush()

class Exp:
    def __init__(self, multi_handler):
        print("开始初始化实验对象...")
        self.multi_handler = multi_handler
        self.metrics = dict()
        trn_mets = ['Loss', 'preLoss']
        tst_mets = ['MRR', 'Hits@1', 'Hits@10']
        mets = trn_mets + tst_mets
        for met in mets:
            if met in trn_mets:
                self.metrics['Train' + met] = list()
            else:
                for handler in self.multi_handler.tst_handlers:
                    self.metrics['Test' + handler.data_name + met] = list()
        print("实验对象初始化完成！")
        
    def make_print(self, name, ep, reses, save, data_name=None):
        if data_name is None:
            ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        else:
            ret = 'Epoch %d/%d, %s %s: ' % (ep, args.epoch, data_name, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric if data_name is None else name + data_name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '      '
        return ret
    
    def run(self):
        self.prepare_model()
        log('Model Prepared')
        stloc = 0
        if args.load_model != None:
            self.load_model()
            stloc = len(self.metrics['TrainLoss']) * args.tst_epoch - (args.tst_epoch - 1)
        
        print(f"\n开始训练，总共 {args.epoch} 个 epoch...")
        
        for ep in range(stloc, args.epoch):
            print(f"\n{'='*60}")
            print(f"Epoch {ep+1}/{args.epoch}")
            print(f"{'='*60}")
            
            tst_flag = (ep % args.tst_epoch == 0)
            reses = self.train_epoch()
            log(self.make_print('Train', ep, reses, tst_flag))
            if ep % 1 == 0:
                self.multi_handler.remake_initial_projections()
            if tst_flag:
                for handler in self.multi_handler.tst_handlers:
                    print(f"\n开始验证 {handler.data_name} 数据集...")
                    reses = self.test_epoch(handler.val_loader, handler)
                    # Note that this is the validation performance
                    log(self.make_print('Test', ep, reses, tst_flag, handler.data_name))
                self.save_history()
            print()
        
        print(f"\n{'='*60}")
        print("开始最终测试...")
        print(f"{'='*60}")
        
        for handler in self.multi_handler.tst_handlers:
            res_summary = dict()
            times = 10
            st = time.time()
            print(f"\n测试 {handler.data_name} 数据集，运行 {times} 次取平均值...")
            for i in range(times):
                print(f"\n第 {i+1}/{times} 次测试...")
                reses = self.test_epoch(handler.tst_loader, handler)
                log(self.make_print('Test', args.epoch, reses, False, handler.data_name))
                self.add_res_to_summary(res_summary, reses)
                self.multi_handler.remake_initial_projections()
            for key in res_summary:
                res_summary[key] /= times
            log(self.make_print('AVG', args.epoch, res_summary, False, handler.data_name))
            print(f"测试耗时: {time.time() - st:.2f} 秒")
        self.save_history()

    def add_res_to_summary(self, summary, res):
        for key in res:
            if key not in summary:
                summary[key] = 0
            summary[key] += res[key]

    def print_model_size(self):
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        for param in self.model.parameters():
            tem = np.prod(param.size())
            total_params += tem
            if param.requires_grad:
                trainable_params += tem
            else:
                non_trainable_params += tem
        print(f'Total params: {total_params/1e6}')
        print(f'Trainable params: {trainable_params/1e6}')
        print(f'Non-trainable params: {non_trainable_params/1e6}')

    def prepare_model(self):
        self.model = OpenGraph()
        t.cuda.empty_cache()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.lr_scheduler = ALRS(self.opt)
        self.print_model_size()

    def train_epoch(self):
        self.model.train()
        trn_loader = self.multi_handler.trn_loader
        trn_loader.dataset.data_shuffling()
        ep_loss, ep_preloss, ep_regloss = 0, 0, 0
        steps = len(trn_loader)
        tot_samp_num = 0
        counter = [0] * len(self.multi_handler.trn_handlers)
        
        print(f"\n开始训练，总共 {steps} 个步骤...")
        
        for i, batch_data in enumerate(trn_loader):
            if args.epoch_max_step > 0 and i >= args.epoch_max_step:
                break
                
            # 显示训练进度条
            progress = (i + 1) / steps * 100
            print(f"\r训练进度: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}% ({i+1}/{steps})", end='', flush=True)
            
            ancs, poss, negs, adj_idx = batch_data
            adj_idx = adj_idx[0]
            ancs = ancs[0].long()
            poss = poss[0].long()
            negs = negs[0].long()
            adj = self.multi_handler.trn_handlers[adj_idx].torch_adj
            if args.cache_adj == 0:
                adj = adj.to(args.devices[0])
            initial_projector = self.multi_handler.trn_handlers[adj_idx].initial_projector
            if args.cache_proj == 0:
                initial_projector = initial_projector.to(args.devices[0])
            loss, loss_dict = self.model.cal_loss((ancs, poss, negs), adj, initial_projector)
            self.opt.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            self.opt.step()

            sample_num = ancs.shape[0]
            tot_samp_num += sample_num
            ep_loss += loss.item() * sample_num
            ep_preloss += loss_dict['preloss'].item() * sample_num
            ep_regloss += loss_dict['regloss'].item()
            
            # 每10个步骤显示一次详细进度
            if (i + 1) % 10 == 0 or (i + 1) == steps:
                print(f"\n步骤 {i+1}/{steps}: loss = {loss:.3f}, pre = {loss_dict['preloss']:.3f}, reg = {loss_dict['regloss']:.3f}, pos = {loss_dict['posloss']:.3f}, neg = {loss_dict['negloss']:.3f}")
            
            counter[adj_idx] += 1
            if args.proj_trn_steps > 0 and counter[adj_idx] >= args.proj_trn_steps:
                counter[adj_idx] = 0
                dice = np.random.uniform()
                if dice < 999:
                    self.multi_handler.remake_one_initial_projection(adj_idx)
                else:
                    self.multi_handler.make_one_self_initialization(self.model, adj_idx)
        
        print(f"\n训练完成！")
        
        ret = dict()
        ret['Loss'] = ep_loss / tot_samp_num
        ret['preLoss'] = ep_preloss / tot_samp_num
        ret['regLoss'] = ep_regloss / steps
        t.cuda.empty_cache()
        self.lr_scheduler.step(ret['Loss'])
        return ret
    
    def test_epoch(self, tst_loader, tst_handler):
        with t.no_grad():
            self.model.eval()
            ep_mrr, ep_hits1, ep_hits10 = 0, 0, 0
            ep_tstnum = len(tst_loader.dataset)
            steps = max(ep_tstnum // args.tst_batch, 1)
            
            print(f"\n开始评估，总共 {ep_tstnum} 个用户，{steps} 个批次...")
            
            for i, batch_data in enumerate(tst_loader):
                usrs = batch_data
                numpy_usrs = usrs.numpy()
                usrs = usrs.long().to(args.devices[1])
                
                # 进度条
                progress = (i + 1) / steps * 100
                print(f"\r评估进度: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}% ({i+1}/{steps})", end='', flush=True)
                
                # 批内训练掩码与候选规模（全候选）
                trn_masks = tst_loader.dataset.csrmat[numpy_usrs].tocoo()
                cand_size = trn_masks.shape[1]
                trn_masks_t = t.from_numpy(np.stack([trn_masks.row, trn_masks.col], axis=0)).long().cuda()
                
                # 模型与必要张量
                adj = tst_handler.torch_adj
                if args.cache_adj == 0:
                    adj = adj.to(args.devices[0])
                initial_projector = tst_handler.initial_projector
                if args.cache_proj == 0:
                    initial_projector = initial_projector.to(args.devices[0])
                
                # 批量全候选打分: [B, cand_size]
                all_preds = self.model.pred_for_test((usrs, trn_masks_t), adj, initial_projector, cand_size, rerun_embed=False if i!=0 else True)
                all_preds_cpu = all_preds.detach().cpu().numpy()
                
                # 取预生成的正负样本（按用户对齐）
                # 将批内用户映射到 per-user 索引
                user_to_idx = dict(zip(tst_loader.dataset.user_list.tolist(), range(len(tst_loader.dataset.user_list))))
                batch_pos_idx = np.array([tst_loader.dataset.user_pos[user_to_idx[u]] for u in numpy_usrs], dtype=np.int64)
                batch_neg_idx = np.stack([tst_loader.dataset.user_neg[user_to_idx[u]] for u in numpy_usrs], axis=0)
                
                # 计算本批 MRR/Hits，仅在正+负集合内排名
                batch_mrr = batch_hits1 = batch_hits10 = 0.0
                bsz = len(numpy_usrs)
                for j in range(bsz):
                    row_scores = all_preds_cpu[j]
                    pos_idx = int(batch_pos_idx[j])
                    neg_idxs = batch_neg_idx[j]
                    pos_score = row_scores[pos_idx]
                    neg_scores = row_scores[neg_idxs]
                    combined = np.concatenate([[pos_score], neg_scores])
                    rank = np.sum(combined >= pos_score)
                    batch_mrr += 1.0 / rank
                    batch_hits1 += 1.0 if rank <= 1 else 0.0
                    batch_hits10 += 1.0 if rank <= 10 else 0.0
                
                # 累积
                ep_mrr += batch_mrr
                ep_hits1 += batch_hits1
                ep_hits10 += batch_hits10
                
                if (i + 1) % 10 == 0 or (i + 1) == steps:
                    print(f"\n步骤 {i+1}/{steps}: MRR = {batch_mrr/bsz:.4f}, H@1 = {batch_hits1/bsz:.4f}, H@10 = {batch_hits10/bsz:.4f}")
            
            print(f"\n评估完成！")
            print(f"最终结果: MRR = {ep_mrr/ep_tstnum:.4f}, H@1 = {ep_hits1/ep_tstnum:.4f}, H@10 = {ep_hits10/ep_tstnum:.4f}")
        ret = dict()
        ret['MRR'] = ep_mrr / ep_tstnum
        ret['Hits@1'] = ep_hits1 / ep_tstnum
        ret['Hits@10'] = ep_hits10 / ep_tstnum
        t.cuda.empty_cache()
        return ret
    
    def calc_recall_ndcg(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg
    
    def calc_mrr_hits(self, pos_preds, neg_preds_list, target_nodes, neg_samples):
        """计算 MRR 和 Hits@K 指标"""
        batch_mrr, batch_hits1, batch_hits10 = 0, 0, 0
        batch_size = len(target_nodes)
        
        for i in range(batch_size):
            pos_score = pos_preds[i].item()
            neg_scores = neg_preds_list[i].cpu().numpy()
            
            # 计算正样本在所有候选中的排名
            all_scores = np.concatenate([[pos_score], neg_scores])
            
            # 计算排名（从1开始）
            rank = np.sum(all_scores >= pos_score)
            
            # 计算 MRR
            mrr = 1.0 / rank
            batch_mrr += mrr
            
            # 计算 Hits@1
            hits1 = 1.0 if rank <= 1 else 0.0
            batch_hits1 += hits1
            
            # 计算 Hits@10
            hits10 = 1.0 if rank <= 10 else 0.0
            batch_hits10 += hits10
        
        # 返回批次平均值
        return batch_mrr / batch_size, batch_hits1 / batch_size, batch_hits10 / batch_size
    
    def calc_mrr_hits_batch(self, all_scores_list, all_labels_list):
        """批量计算 MRR 和 Hits@K 指标"""
        batch_mrr, batch_hits1, batch_hits10 = 0, 0, 0
        batch_size = len(all_scores_list)
        
        for i in range(batch_size):
            scores = all_scores_list[i]
            labels = all_labels_list[i]
            
            # 找到正样本的索引
            pos_idx = np.where(labels == 1)[0][0]
            pos_score = scores[pos_idx]
            
            # 计算正样本的排名
            rank = np.sum(scores >= pos_score)
            
            # 计算 MRR
            mrr = 1.0 / rank
            batch_mrr += mrr
            
            # 计算 Hits@1
            hits1 = 1.0 if rank <= 1 else 0.0
            batch_hits1 += hits1
            
            # 计算 Hits@10
            hits10 = 1.0 if rank <= 10 else 0.0
            batch_hits10 += hits10
        
        # 返回批次平均值
        return batch_mrr / batch_size, batch_hits1 / batch_size, batch_hits10 / batch_size
    
    def save_history(self):
        if args.epoch == 0:
            return
        with open('../History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content, '../Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def load_model(self):
        ckp = t.load('../Models/' + args.load_model + '.mod')
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('../History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if len(args.gpu.split(',')) > 1:
        args.devices = ['cuda:0', 'cuda:1']
    else:
        args.devices = ['cuda:0', 'cuda:0']
    args.devices = list(map(lambda x: t.device(x), args.devices))
    
    # 若指定日志文件，则tee输出
    if args.log_file is not None and len(args.log_file) > 0:
        sys.stdout = Tee(args.log_file)
        sys.stderr = sys.stdout
    
    logger.saveDefault = True
    setproctitle.setproctitle('OpenGraph')

    log('Start')
    trn_datasets = ['gen1']
    tst_datasets = ['ml1m', 'ml10m', 'collab']

    # trn_datasets = ['gen2']
    # tst_datasets = ['ddi']

    # trn_datasets = ['gen0']
    # tst_datasets = ['amazon-book']

    if len(args.tstdata) != 0:
        tst_datasets = [args.tstdata]
    if len(args.trndata) != 0:
        trn_datasets = [args.trndata]
    trn_datasets = list(set(trn_datasets))
    tst_datasets = list(set(tst_datasets))
    multi_handler = MultiDataHandler(trn_datasets, tst_datasets)
    log('Load Data')
    log('数据加载完成，准备开始训练...')
    print("正在创建实验对象...")

    exp = Exp(multi_handler)
    print("实验对象创建完成，开始运行...")
    exp.run()
