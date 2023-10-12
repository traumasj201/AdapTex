from dataset.dataset import Im2LatexDataset
import os
import argparse
import logging
import yaml

import torch
from munch import Munch
from tqdm.auto import tqdm
import wandb
from eval import evaluate
from models import get_model, CosineAnnealingWarmUpRestarts
from utils import parse_args, seed_everything, get_optimizer, get_scheduler, gpu_memory_check
import torch.backends.cudnn as cudnn


def train(args):
    dataloader = Im2LatexDataset().load(args.data)
    dataloader.update(**args, test=False)
    valdataloader = Im2LatexDataset().load(args.valdata)
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
    valdataloader.update(**valargs)
    cudnn.benchmark = True

    device = args.device

    model = get_model(args)
    msg = model.load_state_dict(torch.load(args.load_chkpt, map_location=device), strict=False)
    if args.is_af:
        for name, p in model.named_parameters():
            if name in msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False  # True is all weight training

    if args.is_01_lr:
        opt = get_optimizer(args.optimizer)(model.optim_parameters(args), args.lr, betas=args.betas,
                                            weight_decay=args.weight_decay)
    else:
        opt = get_optimizer(args.optimizer)(model.parameters(), args.lr, betas=args.betas,
                                            weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmUpRestarts(opt, T_0=args.T_0, T_mult=args.T_mult, eta_max=args.eta_max,
                                              T_up=args.T_up, gamma=args.gamma)

    if torch.cuda.is_available():
        gpu_memory_check(model, args)

    max_bleu, max_token_acc = 0, 0
    out_path = os.path.join(args.model_path, args.name)
    os.makedirs(out_path, exist_ok=True)

    def save_models(e, step=0):
        torch.save(model.state_dict(), os.path.join(out_path, '%s_e%02d_step%02d.pth' % (args.name, e + 1, step)))
        args.load_chkpt = os.path.join(out_path, '%s_e%02d_step%02d.pth')
        yaml.dump(dict(args), open(os.path.join(out_path, 'config.yaml'), 'w+'))

    microbatch = args.get('micro_batchsize', -1)
    if microbatch == -1:
        microbatch = args.batchsize
    train_log = open('./train_' + str(args.name) + '.txt', 'w')
    train_loss_log = open('./train_' + str(args.name) + '2.txt', 'w')
    train_log.write('epoch,lr,total_loss,belu,edit_distance,token_accuracy\n')
    train_log.flush()
    try:
        for e in range(args.epoch, args.epochs):
            train_log.write(str(e))
            args.epoch = e
            train_log.write('%d,' % e)
            train_log.flush()
            lr = opt.param_groups[0]['lr']
            if args.wandb:
                wandb.log({'train/lr': lr})
            train_log.write('%.5f,' % lr)
            train_log.flush()

            dset = tqdm(iter(dataloader))
            total_loss = 0
            for i, (seq, im) in enumerate(dset):
                if seq is not None and im is not None:
                    opt.zero_grad()
                    step_loss = 0

                    for j in range(0, len(im), microbatch):
                        tgt_seq, tgt_mask = seq['input_ids'][j:j + microbatch].to(device), seq['attention_mask'][
                                                                                           j:j + microbatch].bool().to(
                            device)
                        loss = model.data_parallel(im[j:j + microbatch].to(device), device_ids=args.gpu_devices,
                                                   tgt_seq=tgt_seq, mask=tgt_mask) * microbatch / args.batchsize
                        loss.backward()  # data parallism loss is a vector
                        step_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                    total_loss += step_loss
                    opt.step()
                    lr = opt.param_groups[0]['lr']
                    dset.set_description('Loss: %.4f, lr: %.4f' % (step_loss, lr))
                    train_loss_log.write('%d, %.5f\n' % (e, total_loss))
                    train_loss_log.flush()

                    if args.wandb:
                        wandb.log({'train/loss': step_loss})

            scheduler.step()
            train_log.write('%.5f,' % total_loss)
            train_log.flush()

            bleu_score, edit_distance, token_accuracy = evaluate(model, valdataloader, args,
                                                                 num_batches=int(args.valbatches),
                                                                 # num_batches=int(args.valbatches * e / args.epochs),
                                                                 name='val')
            train_log.write('%.5f, %.5f, %.5f\n' % (bleu_score, edit_distance, token_accuracy))
            train_log.flush()
            if bleu_score > max_bleu and token_accuracy > max_token_acc:
                max_bleu, max_token_acc = bleu_score, token_accuracy
                save_models(e, step=-9)

            if (e + 1) % args.save_freq == 0:
                save_models(e, step=len(dataloader))
            if args.wandb:
                wandb.log({'train/epoch': e + 1})

    except KeyboardInterrupt:
        if e >= 2:
            save_models(e, step=i)
        raise KeyboardInterrupt
    # save_models(e, step=len(dataloader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', default=None, help='path to yaml config file', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--debug', action='store_true', help='DEBUG')
    parser.add_argument('--resume', help='path to checkpoint folder', action='store_true')
    parsed_args = parser.parse_args()

    if parsed_args.config is None:
        print('insert config yaml file')
        raise Exception

    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    args = parse_args(Munch(params), **vars(parsed_args))
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    seed_everything(args.seed)

    if args.wandb:
        if not parsed_args.resume:
            args.id = wandb.util.generate_id()

        wandb.init(config=dict(args), resume='allow', name=args.name, id=args.id)
        args = Munch(wandb.config)
    train(args)
