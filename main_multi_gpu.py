import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from util import Accuracy, read_configuration, save
from model import ClassifyNet, BertWrapper
from transformers import AdamW, get_scheduler, logging, BertModel, BertTokenizer
from dataset import get_train_sampler_data_loader, get_test_data_loader, get_label_num


def init(config):
    logging.set_verbosity_error()
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.set_device(device=config['device'])


def build_train_model(config):
    print("get label number")
    label_number = get_label_num(config['dataset'])

    print("build bert model")
    bert = BertModel.from_pretrained(f"./checkpoint/{config['dataset']}/{config['checkpoint_epoch']}" if config['use_checkpoint'] else config['bert_version'])
    tokenizer = BertTokenizer.from_pretrained(f"./checkpoint/{config['dataset']}/{config['checkpoint_epoch']}" if config['use_checkpoint'] else config['bert_version'])
    bert_hidden_size = bert.config.hidden_size

    print("build BertWrapper model")
    bertWrapper = BertWrapper(bert).to(config['device'])
    bertWrapper = torch.nn.parallel.DistributedDataParallel(bertWrapper, device_ids=[config['device_id'][config['local_rank']]], output_device=config['device_id'][config['local_rank']])

    print("build ClassifyNet model")
    classify_net = ClassifyNet(label_number=label_number, feature_layers=10, bert_hidden_size=bert_hidden_size).to(config['device'])
    if config['use_checkpoint']:
        classify_net.load_state_dict(torch.load(f"./checkpoint/{config['dataset']}/{config['checkpoint_epoch']}/student_model_file.bin"))
    classify_net = torch.nn.parallel.DistributedDataParallel(classify_net, device_ids=[config['device_id'][config['local_rank']]], output_device=config['device_id'][config['local_rank']])

    print("build train data loader")
    train_data_loader, train_sampler = get_train_sampler_data_loader(config['dataset'], tokenizer, batch_size=config['train_batch_size'], num_workers=config['num_workers'])

    print("build test data loader")
    test_data_loader = get_test_data_loader(config['dataset'], tokenizer, batch_size=config['train_batch_size'], num_workers=config['num_workers'])

    print("build BCE loss")
    classify_loss_function = torch.nn.BCEWithLogitsLoss()

    print("build MSE loss")
    mse_loss_function = torch.nn.MSELoss()

    print("build optimizer")
    bert_train_parameters = [parameter for parameter in bertWrapper.parameters() if parameter.requires_grad]
    classify_net_train_parameters = [parameter for parameter in classify_net.parameters() if parameter.requires_grad]
    parameters = bert_train_parameters + classify_net_train_parameters

    optimizer = AdamW(parameters, lr=config['lr'])

    print("build lr_scheduler")
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=config['epochs'] * len(train_data_loader))

    return train_data_loader, train_sampler, test_data_loader, bertWrapper, classify_net, tokenizer, classify_loss_function, mse_loss_function, optimizer, lr_scheduler



def run_train_batch(device, data, bertWrapper, classify_net, classify_loss_function, mse_loss_function, optimizer, lr_scheduler):
    batch_text_input_ids, batch_text_padding_mask, batch_text_token_type_ids, batch_label_input_ids, batch_label_padding_mask, batch_label_token_type_ids, batch_label_one_hot = data

    batch_text_input_ids = batch_text_input_ids.to(device)
    batch_text_padding_mask = batch_text_padding_mask.to(device)
    batch_text_token_type_ids = batch_text_token_type_ids.to(device)
    batch_label_input_ids = batch_label_input_ids.to(device)
    batch_label_padding_mask = batch_label_padding_mask.to(device)
    batch_label_token_type_ids = batch_label_token_type_ids.to(device)
    batch_label_one_hot = batch_label_one_hot.to(device)


    text_bert_out, label_bert_out = bertWrapper(input_ids=[batch_text_input_ids, batch_label_input_ids],
                                attention_mask=[batch_text_padding_mask, batch_label_padding_mask],
                                token_type_ids=[batch_text_token_type_ids, batch_label_token_type_ids])

    out = classify_net(text_bert_out)
    text_classify_loss = classify_loss_function(out, batch_label_one_hot)


    mse_loss = mse_loss_function(text_bert_out.pooler_output, label_bert_out.pooler_output)

    out = classify_net(label_bert_out)
    label_classify_loss = classify_loss_function(out, batch_label_one_hot)

    loss = text_classify_loss + mse_loss + label_classify_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    return text_classify_loss, mse_loss, label_classify_loss


def run_test_batch(device, data, bertWrapper, classify_net, loss_function, accuracy):
    batch_text_input_ids, batch_text_padding_mask, batch_text_token_type_ids, \
    batch_label_input_ids, batch_label_padding_mask, batch_label_token_type_ids, \
    batch_label_one_hot = data

    batch_text_input_ids = batch_text_input_ids.to(device)
    batch_text_padding_mask = batch_text_padding_mask.to(device)
    batch_text_token_type_ids = batch_text_token_type_ids.to(device)
    batch_label_one_hot = batch_label_one_hot.to(device)

    bert_out = bertWrapper(input_ids=[batch_text_input_ids],
                           attention_mask=[batch_text_padding_mask],
                           token_type_ids=[batch_text_token_type_ids])

    out = classify_net(bert_out)

    loss = loss_function(out, batch_label_one_hot)

    accuracy.calc(out, batch_label_one_hot)

    return loss


def main_worker(local_rank, number_of_gpus_per_node, config):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23233', world_size=number_of_gpus_per_node, rank=local_rank)

    config['device'] = torch.device(f"cuda:{config['device_id'][local_rank]}")
    config['local_rank'] = local_rank

    train(config)


def launch_worker(local_rank, config):
    dist.init_process_group(backend='nccl')

    config['device'] = torch.device(f"cuda:{config['device_id'][local_rank]}")
    config['local_rank'] = local_rank

    train(config)


def train(config):
    print(f"start dataset {config['dataset']}")

    init(config)

    accuracy = Accuracy()

    train_data_loader, train_sampler, test_data_loader, bertWrapper, classify_net, tokenizer, loss_function, mse_loss_function, optimizer, lr_scheduler = build_train_model(config)

    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)
        bertWrapper.train()
        classify_net.train()
        with tqdm(train_data_loader, ncols=200) as batch:
            for data in batch:
                text_classify_loss, mse_loss, label_classify_loss = run_train_batch(config['device'], data, bertWrapper, classify_net, loss_function, mse_loss_function, optimizer, lr_scheduler)
                batch.set_description(f"train device_id:{config['device_id'][config['local_rank']]}, epoch:{epoch + 1}/{config['epochs']}, text_classify_loss:{text_classify_loss.item()}, mse_loss:{mse_loss.item()}, label_classify_loss:{label_classify_loss.item()}")

        with torch.no_grad():
            bertWrapper.eval()
            classify_net.eval()
            accuracy.reset_acc()
            loss = 0
            with tqdm(test_data_loader, ncols=200) as batch:
                for data in batch:
                    _loss = run_test_batch(config['device'], data, bertWrapper, classify_net, loss_function, accuracy)
                    loss += _loss.item()
                    batch.set_description(f"test device_id:{config['device_id'][config['local_rank']]}, epoch:{epoch + 1}/{config['epochs']}, loss:{_loss.item()}")

            if config['local_rank'] == 0:
                save(config['dataset'], epoch, mark, bertWrapper.module.get_bert(), tokenizer, classify_net)

                print(f"test epoch:{epoch + 1}/{config['epochs']} loss:{loss / accuracy.get_total()} p1:{accuracy.get_acc1()} p3:{accuracy.get_acc3()} p5:{accuracy.get_acc5()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help="DDP parameter, do not modify")
    args = parser.parse_args()

    config = read_configuration("./config.yaml")

    if args.local_rank == -1:
        mp.spawn(main_worker, nprocs=len(config['device_id']), args=(len(config['device_id']), config))
    else:
        launch_worker(args.local_rank, config)
